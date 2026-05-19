# Phase E: Stateless Deployment

**Status**: TODO
**Depends on**: Phase D
**Blocks**: None

## Goal

Package the Hermes gateway as a stateless Docker container. No local filesystem state survives a container restart. All state is external (Neon, S3, Redis). Deploy on ECS Fargate behind an ALB. Health check endpoint confirms all dependencies are reachable.

## Context

With Phases 0–D complete, Hermes has no local state requirements in SaaS mode: sessions → Neon, skills → S3, hot cache (optional) → Upstash Redis. The container is a pure compute unit — crash it, replace it, scale it freely.

## Specifications

### S1: Dockerfile

Python 3.12 slim base. Installs Hermes gateway dependencies. Sets `HERMES_MODE=saas`. `HERMES_HOME=/tmp/hermes-runtime` (ephemeral scratch only — no persistent local writes).

### S2: Health check endpoint

`GET /health` returns `{"status": "ok"|"degraded", "storage": "neon"|"error", "skills": "s3"|"error"}`. Used by ALB target group health checks and ECS container health.

### S3: ECS task definition

2 tasks minimum. Secrets from AWS Secrets Manager (not env var literals). No EFS mount points — stateless. Auto-scaling based on CPU + SQS queue depth (Slack events queued).

### S4: Slack gateway: socket mode vs webhook

- **Socket mode** (recommended for internal beta): no public URL needed, works behind private VPC, lower ops overhead
- **Webhook mode** (for GA): public ALB, faster for high volume, required for Slack app distribution

Phase E ships socket mode. Webhook mode is a follow-on.

### S5: Optional Redis hot cache

Upstash Redis for hot session context (active conversation, tool state). TTL-based eviction (24h). Agent checks Redis before hitting Neon for recent history. Opt-in — not required for correctness.

## Steps

| # | Action | File | Expected Result |
|---|--------|------|-----------------|
| 1 | Write `Dockerfile` for Hermes gateway | `Dockerfile.saas` | `docker build` succeeds, container starts |
| 2 | Implement `GET /health` endpoint | `gateway/health.py` | Returns JSON status for Neon + S3 + Redis |
| 3 | Test health check: kill Neon connection, verify "degraded" response | `tests/test_health.py` | Degraded reported cleanly, no crash |
| 4 | Write `docker-compose.saas.yml` for local SaaS mode testing | `docker-compose.saas.yml` | Full stack runs locally with Neon + real S3 |
| 5 | Write ECS task definition JSON | `terraform/ecs_task.json` or `terraform/main.tf` | Task definition valid, secrets from Secrets Manager |
| 6 | Deploy to ECS (staging environment) | CLI / Terraform | Container healthy, Slack events processed |
| 7 | Load test: 10 concurrent users, verify no cross-tenant data leaks | `tests/test_load.py` | Zero leaks, linear throughput |
| 8 | Commit + push | git | `feat: phase-E stateless deployment` |

## Acceptance Criteria

- [ ] `docker build -f Dockerfile.saas -t hermes-saas .` succeeds
- [ ] Container starts with `HERMES_MODE=saas`, connects to Neon and S3
- [ ] `GET /health` returns `{"status": "ok"}` when all dependencies reachable
- [ ] `GET /health` returns `{"status": "degraded", "storage": "error"}` when Neon unreachable
- [ ] Container restart: conversation history retrieved from Neon (no data loss)
- [ ] 10 concurrent users from 3 different team_ids: zero cross-tenant data leaks in 1000 messages
- [ ] ECS task shows healthy in target group within 60s of deploy
- [ ] No EFS mount points in task definition (stateless confirmed)

## Key Files

```dockerfile
# Dockerfile.saas
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -e ".[gateway]"

ENV HERMES_MODE=saas
ENV HERMES_HOME=/tmp/hermes-runtime

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

CMD ["python", "-m", "gateway.run", "--platform", "slack"]
```

```python
# gateway/health.py
from hermes_storage import get_backend
import boto3

async def health_check() -> dict:
    results = {"status": "ok"}
    try:
        db = await get_backend()
        await db.ping()
        results["storage"] = "neon"
    except Exception:
        results["storage"] = "error"
        results["status"] = "degraded"
    try:
        s3 = boto3.client("s3")
        s3.head_bucket(Bucket="hermes-skills")
        results["skills"] = "s3"
    except Exception:
        results["skills"] = "error"
        results["status"] = "degraded"
    return results
```

## Open Questions

- **Q-E.1**: Socket mode (private VPC) vs webhook mode (public ALB) for Slack? (Recommended: socket mode for beta — no TLS termination ops, no ALB cost, easier to iterate)
- **Q-E.2**: Single ECS cluster or separate clusters per tier (free vs pro)? (Recommended: single cluster with task count auto-scaling; tier enforcement is at the application layer, not infra)

## Sub-tasks from #product-feedback (triage: infra)

### ST-E.1 — Replace NAT Gateway with S3 VPC Endpoint
- **Source**: Slack `#product-feedback` ts `1779128521.434179` (2026-05-18)
- **Triage**: infra — cost reduction
- **Detail**: Current ECS Fargate tasks route S3 traffic (skills bucket, artifacts) through the NAT Gateway, incurring data-processing charges per GB. Replacing with a Gateway-type S3 VPC Endpoint eliminates all data-processing costs for S3 traffic within the VPC (endpoint itself is free).
- **Action**: Add `aws_vpc_endpoint` resource to Terraform for S3 (`com.amazonaws.<region>.s3`, type: Gateway). Update route tables for private subnets. Verify Fargate tasks resolve S3 via the endpoint (check VPC Flow Logs / S3 access logs for `vpce-*` source).
- **Effort**: S
- **Acceptance**: ECS task S3 calls do not traverse NAT Gateway (confirmed via CloudWatch VPC Flow Logs or Cost Explorer: NAT data-processing charge drops to ~$0 for S3 traffic).

### ST-E.2 — Replace Claude (Bedrock) with OSS Models for Hermes Deployment
- **Source**: Slack `#product-feedback` ts `1779128576.753829` (2026-05-18)
- **Triage**: infra — cost reduction
- **Detail**: Hermes is currently routed to Claude models via AWS Bedrock, which are expensive relative to OSS alternatives on Bedrock (Llama 3, Mistral, etc.) with 70%+ lower token costs. Non-interactive or background tasks (cron, triage, summarisation) should default to a cheaper OSS model; interactive sessions can remain on Claude.
- **Action**: In `~/.hermes/config.yaml` (or SaaS override), introduce a `model.background` field pointing to a cost-efficient Bedrock model (e.g., `meta.llama3-70b-instruct-v1:0` or `mistral.mistral-large-2402-v1:0`). Wire cron job runner and batch_runner.py to use `model.background` when `HERMES_MODE=saas`. Keep `model.default` pointing to Claude for interactive CLI sessions. Document in `docs/deployment/cost-optimisation.md`.
- **Effort**: M
- **Acceptance**: Nightly cron jobs invoke non-Claude Bedrock model; `hermes chat` interactive sessions still use configured default. Monthly token cost for background tasks reduced by ≥50% vs all-Claude baseline.
- **Depends on**: Phase 001-E S3/ECS setup complete (this is a config-layer change on top of the deployed stack)

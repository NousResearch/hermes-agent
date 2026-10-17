---
name: sre-error-budget-solo
description: "SRE error budgets, SLOs, incident response for solo devs."
version: 1.0.0
author: Rafael Zendron (rafaumeu)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [sre, error-budget, slo, incident-response, reliability, dora, postmortem, solo-dev]
    related_skills: [plan, requesting-code-review]
    requires_toolsets: [terminal]
---

# SRE Error Budgets for Solo Developers

Adapts Google SRE reliability engineering (error budgets, SLOs, incident response, DORA metrics) for solo developers and small teams. No dedicated SRE team required — just systematic reliability tracking.

**What it does:** Provides SLO templates, error budget policies, rollback playbooks, and incident response procedures.
**What it doesn't do:** Replace monitoring infrastructure. You still need Sentry, Grafana, or equivalent.

## When to Use

- Defining reliability targets for a SaaS product
- After an incident — for postmortem and prevention
- Setting up CI quality gates tied to reliability
- When deciding whether to build features or fix reliability
- Measuring DORA metrics (lead time, deploy frequency, change failure rate, MTTR)

## Prerequisites

- Monitoring/observability stack (Sentry + Uptime Kuma is a good free baseline)
- Deployment pipeline with rollback capability
- Git repository with commit/deploy timestamps

## Procedure

### Step 1: Define SLOs per service tier

## SLO Templates

Define SLOs per service tier:

| Service | Availability Target | Allowed Downtime/month |
|---------|-------------------|----------------------|
| Payment API | 99.9% | 43 minutes |
| Core Feature | 99.5% | 3.6 hours |
| Authentication | 99.9% | 43 minutes |
| Static Assets | 99.0% | 7.3 hours |

**Latency targets:**
- p95 < 2 seconds
- p99 < 5 seconds

Adjust targets based on actual usage. SLOs that are always exceeded may be too tight.

## Error Budget Policy

**Window:** 4 weeks (rolling window).

| Event | Threshold | Action |
|-------|-----------|--------|
| Error budget exhausted | 100% consumed | STOP new features, focus on reliability until budget restored |
| Single incident | > 20% of budget | Postmortem required within 24 hours |
| Recurring outage class | > 20% in 1 quarter | P0 in next planning cycle |
| SLO consistently exceeded | 3 windows in a row | Review SLO (too tight?) |

**Practical consequence:** When error budget is exhausted, ALL feature work stops. Only reliability, bug fixes, and security patches until the budget recovers. No exceptions.

**Monthly review:** Reassess SLOs. Adjust if the target is consistently unrealistic or consistently too easy.

## DORA Metrics (2025 Update)

DORA 2025 is now "State of AI-assisted Software Development." AI amplifies existing engineering culture.

| Metric | Target | How to Measure |
|--------|--------|---------------|
| Lead Time | < 1 hour | `git log` timestamp vs deploy timestamp |
| Deploy Frequency | >= 2/week | `git log --oneline main` count |
| Change Failure Rate | < 15% | postmortems count / total deploys |
| MTTR | < 30 minutes | incident start > fix deployed |

## Incident Response

```
DETECT > ISOLATE (feature flag OFF or redeploy) > DIAGNOSE > FIX > VERIFY > POSTMORTEM (24h)
```

### Rollback Playbook

**Standard path (reversible):**
1. Detect via SLO alerts or incident report
2. `vercel rollback [deployment-url]` or `git revert HEAD && git push`
3. Verify health check and SLOs within 5 minutes
4. Open postmortem draft immediately

**Decision: rollback vs forward fix:**

| Situation | Decision |
|-----------|----------|
| Bug in code, data intact | Rollback |
| Schema change, data migrated | Forward fix |
| Feature flag available | Disable flag, investigate |
| < 1% users affected | Forward fix with hotfix priority |
| > 10% users affected | Rollback immediately |

**When rollback is impossible:** Breaking schema already consumed. Sensitive data exposed. External dependency notified. These require coordinated forward fixes.

### Zero-downtime migrations (expand-contract pattern)

1. Add new column nullable — deploy
2. Migrate data in background
3. App reads new column — deploy
4. Drop old column — separate deploy

Never rename or drop in a single migration with live traffic.

## Postmortem Template

Save in project docs or wiki:

```markdown
# Postmortem: [Title]

**Date:** YYYY-MM-DD
**Severity:** P1/P2/P3
**Duration:** Xm
**Impact:** [users affected, revenue impact]

## Timeline
- HH:MM — [event]
- HH:MM — [detection]
- HH:MM — [fix deployed]
- HH:MM — [verified]

## Root Cause (5 Whys)
1. Why did this happen?
2. Why did that condition exist?
3. Why wasn't it caught earlier?
4. Why didn't monitoring catch it?
5. Why was the system vulnerable to this?

## Action Items
- [ ] [Preventive action] (owner, due date)
- [ ] [Monitoring improvement] (owner, due date)

## Lesson
[One sentence takeaway]
```

**Blameless:** Postmortems never blame individuals. They identify systemic gaps.

## Chaos Scenarios (for SaaS with payments)

Test these as integration or E2E tests:

- **Payment:** webhook delayed, duplicated, failed, provider 500, browser closes at checkout
- **Database:** cold start 3s, slow query 5s, pool exhausted, deadlock, mid-request migration
- **Auth:** token expires mid-action, concurrent sessions, simultaneous logout + action
- **Concurrency:** 2 checkouts same user, feature + state update in parallel
- **Frontend:** JS crash during payment, network offline at submit, browser back mid-flow

## Observability Stack (free tier)

**Solo dev minimum:**
- Errors: Sentry (free tier)
- Logs: structured logging (Pino, structlog)
- Uptime: Uptime Kuma (self-hosted)
- Latency: Vercel Analytics or CloudFlare Analytics

**Small team:**
- Metrics + dashboards: Grafana Cloud (free tier)
- Logs: BetterStack (free tier)
- Traces: OpenTelemetry (self-hosted)

**Logging minimum for every API route:**
```
Before: logger.info({ event: 'api_request', method, path, userId })
After:  logger.info({ event: 'api_response', status, durationMs })
Error:  logger.error({ event: 'api_error', path, error: String(error) })
```

Never log stack traces in production. Never log PII.

## Feature Flag Lifecycle

```
CREATE > ENABLE > VALIDATE (30 days) > HARDCODE > REMOVE FLAG
```

1. **CREATE:** Flag created, default OFF, CI tests both paths
2. **ENABLE:** Flag on in staging, then production. Monitor SLOs.
3. **VALIDATE:** Feature stable for 30 days without incident. If incident, back to ENABLE.
4. **HARDCODE:** Remove conditional, ON path becomes the only path
5. **REMOVE FLAG:** Delete flag from code and environment variables

**Monthly review:** Flags active > 30 days without incident should be removed.

## PRR Checklist (pre-launch, for money features)

- **Reliability:** SLO defined, fallback plan documented, feature flags configured, rollback tested, backup verified
- **Security:** Pentest completed, auth on all routes, no debug endpoints, security headers, audit clean
- **Observability:** Structured logging, health checks, error handler, deploy log
- **Testing:** Coverage 100%, mutation >= 90%, E2E for money flows, integration for webhooks+DB
- **Data:** Privacy compliance, consent flow, right to deletion, encryption at rest

## Pitfalls

- **Setting SLOs at 100%.** Impossible to maintain. Error budget = 0 means no room for deployments. Target 99.0-99.9%.
- **Skipping postmortem after "small" incidents.** Small incidents cluster. Every incident > 20% budget gets a postmortem.
- **Never adjusting SLOs.** If you consistently exceed SLO by a huge margin, it's too loose. If you always blow it, it's too tight. Review monthly.
- **Breaking schema in a single migration.** Use expand-contract pattern for zero-downtime.
- **Feature flags accumulating forever.** Flags active > 30 days without incident should be removed. Flag debt slows CI.
- **Blameful postmortems.** Postmortems identify systemic gaps, never individuals. Blameless culture is non-negotiable.
- **Measuring DORA without acting on it.** Metrics without improvement targets are vanity numbers.

## Verification

```bash
# Check deploy frequency
git log --oneline --since="4 weeks ago" main | wc -l

# Check MTTR (from postmortem records)
# Manual: review postmortem folder for time-to-fix

# Check error budget (from monitoring dashboard)
# Manual: review Sentry/Uptime Kuma for incident count vs SLO target
```

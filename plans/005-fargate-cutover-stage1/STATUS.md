# Status — Plan 005: Hermes Fargate Cutover (Stage 1)

**Status:** IN PROGRESS — BLOCKED at Phase E on infrastructure debt
**Last updated:** 2026-05-26
**Blocked by:** **Plan 009 (Cloud LLM Routing Infrastructure)** — cloud LLM routing was never finished by Plan 001-E. Plan 005-E proved cloud Hermes can't make LLM calls (HTTP 401 on Portkey path; Portkey image was missing, Rooben image is missing, all LLM secrets are placeholders).
**Blocks:** Plan 008 (MCPGateway production wiring) — Plan 008 dispatches after Plan 009 + Plan 005-E/F/G complete

## Phase Progress

| Phase | Title | Status | Notes |
|---|---|---|---|
| 005-A | Land + verify PR #13 on main | **Complete (2026-05-26)** | PR #13 admin-merged (squash); main has all Plan 004 + Plan 007 commits |
| 005-B | Build + push v8 Docker image | **Complete (2026-05-26)** | `plan-001-E-amd64-v8` in ECR (digest `sha256:5ba47504...`). Build script patched for --platform linux/amd64. |
| 005-C | Pre-flight: Secrets Manager + IAM + networking | **Complete (2026-05-26)** | All 4 secrets present + accessible; Neon reachable; Bedrock + Slack tokens wired. Discovered: real task family is `agentic-stack-hermes` not `hermes-saas`. |
| 005-D | Flip Fargate desiredCount=1; verify cold-start | **Complete (2026-05-26)** | Surfaced + fixed 3 pre-existing infrastructure bugs (missing stderr log handler in saas mode, PYTHONUNBUFFERED, allowlist defaults). v8 + Plan 007 startup confirmed live in CloudWatch on task def :12. |
| 005-E | Cloud-side smoke test (Slack → Neon round-trip) | **BLOCKED (2026-05-26)** | Cloud Hermes received Slack message + reaction but couldn't reply: LLM call → HTTP 401 (Portkey image was missing, Rooben image missing, all LLM secrets are placeholders). See Plan 009. Cloud rolled back to desired=0. Local Hermes restored as sole listener. |
| 005-F | Unload local launchd; assert single-listener invariant | Deferred until Plan 009 unblocks 005-E | — |
| 005-G | 48h soak + cutover sign-off | Deferred until 005-E green | — |
| 005-H | Document + open Plan 008 carry-forward | Deferred until 005-G | — |

## Resumption context

- Next phase: 005-A — merge PR #13 to main
- Plan 007 closed 2026-05-25 with 9 commits on the branch. Phase A is the merge gate.

## Adaptations log

(none yet — will populate per-phase as we go)

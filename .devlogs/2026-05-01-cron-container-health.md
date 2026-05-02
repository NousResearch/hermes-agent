# Cron container-aware gateway health

**Date:** 2026-05-01
**Author:** Hermes Agent
**Branch:** cron-health-observability-20260501223311
**PR:** https://github.com/NousResearch/hermes-agent/pull/18634

## Goal

Make Hermes cron diagnostics accurate in container/shared-`HERMES_HOME` deployments where the Docker-supervised gateway owns the scheduler but its PID may not be visible from the inspecting CLI/container. Prevent operators from starting duplicate gateways because of a PID-namespace false negative.

## What Was Done

- Added plan/spec: `.hermes/plans/2026-05-01-cron-container-health.md`.
- Added ADR: `docs/adr/2026-05-01-container-aware-cron-health.md`.
- Updated `gateway/status.py` with `get_gateway_owner_status()` to report `local_pid_running`, `shared_lock_active`, or `not_running` from runtime lock/PID metadata.
- Updated `gateway/status.py` so `get_running_pid()` preserves PID/lock metadata when a shared runtime lock is active but the PID is not locally visible.
- Updated `hermes_cli/cron.py` so `hermes cron status` and `hermes cron list` use runtime-owner status instead of local PID-only detection.
- Added cron/status regression tests in `tests/hermes_cli/test_cron.py`.
- Added runtime-owner and metadata-preservation tests in `tests/hermes_cli/test_gateway_runtime_health.py`.
- Updated `website/docs/guides/cron-troubleshooting.md` with container-aware cron/gateway diagnosis and duplicate-gateway warning.

## Key Decisions

- Runtime lock ownership is now the cross-namespace scheduler-owner signal for cron diagnostics.
- `get_running_pid()` remains a local-PID helper for backward compatibility; broader cron diagnostics use the new owner-state helper.
- ADR required because this changes the operational source of truth for cron/gateway liveness in containerized deployments.
- No Docker Compose/service changes were made in this branch; this branch is limited to status logic, tests, docs, and ADR.

## Validation

- RED: `python -m pytest tests/hermes_cli/test_cron.py tests/hermes_cli/test_gateway_runtime_health.py -q` — failed as expected before production changes because `gateway.status.get_gateway_owner_status` did not exist; output captured at `/work/.hermes-data/tdd-evidence/2026-05-01-cron-health-red.txt`.
- GREEN targeted: `python -m pytest tests/hermes_cli/test_cron.py tests/hermes_cli/test_gateway_runtime_health.py -q` — passed after implementation; earlier output captured at `/work/.hermes-data/tdd-evidence/2026-05-01-cron-health-green-targeted.txt`.
- Final relevant regression: `python -m pytest tests/hermes_cli/test_cron.py tests/hermes_cli/test_gateway_runtime_health.py tests/hermes_cli/test_gateway.py -q` — 34 passed; output captured at `/work/.hermes-data/tdd-evidence/2026-05-01-cron-health-green-regression-final2.txt`.
- Compile check: `python -m py_compile gateway/status.py hermes_cli/cron.py` — passed.
- Static scan: `python /work/.hermes-data/skills/software-development/requesting-code-review/scripts/static_scan_diff.py --unstaged` — `STATIC_SCAN_PASS True`.
- Broader adjacent check: `python -m pytest tests/hermes_cli/test_cron.py tests/hermes_cli/test_gateway_runtime_health.py tests/hermes_cli/test_gateway.py tests/hermes_cli/test_gateway_service.py -q` — 130 passed, 7 failed in gateway service/systemd-root environment cases; output captured at `/work/.hermes-data/tdd-evidence/2026-05-01-cron-health-regression.txt`. These failures were outside touched files and not resolved in this branch.
- Independent review: delegate reviewer passed the diff with no security concerns or logic errors. Reviewer suggestions for metadata-preservation and branch coverage were addressed with additional tests.

## What Skills and Tools Were Used

- Skills: `hermes-agent`, `test-driven-development`, `requesting-code-review`, `devlog`.
- Tools: git worktree, pytest, py_compile, static diff scanner, independent `delegate_task` reviewer.

## Artifacts Updated

- `.hermes/plans/2026-05-01-cron-container-health.md`
- `docs/adr/2026-05-01-container-aware-cron-health.md`
- `.devlogs/2026-05-01-cron-container-health.md`
- `website/docs/guides/cron-troubleshooting.md`

## Related Repos

- `hermes-agent` worktree: `/work/.hermes-data/worktrees/hermes-agent-cron-health-20260501223311`

## Issues & Blockers

- Full gateway-service-adjacent regression includes existing environment-sensitive systemd/root failures in `tests/hermes_cli/test_gateway_service.py`; this branch did not change those paths. Follow-up: https://github.com/NousResearch/hermes-agent/issues/18630
- CLI-level shared-lock coverage is useful but intentionally out of scope for this branch's minimal unit/doc/ADR change. Follow-up: https://github.com/NousResearch/hermes-agent/issues/18631

## Key Learnings

- Cron status in shared-volume/container deployments should not be derived solely from locally visible PIDs.
- Active gateway runtime locks are a safer scheduler-owner signal than PID visibility across process namespaces, but they do not prove platform adapter health.
- Tests should explicitly cover both operator UI behavior (`cron status`/`list`) and metadata preservation behavior (`get_running_pid`).

## Next Steps

- Follow up on gateway service test stabilization: https://github.com/NousResearch/hermes-agent/issues/18630.
- Follow up on CLI-level shared-lock cron status coverage: https://github.com/NousResearch/hermes-agent/issues/18631.

## Prompting Notes

- **Initial ask:** Continue preserved task list after context compression and finish validation/review/devlog/final guard.
- **Clarifications needed:** None.
- **Corrections made:** Added direct metadata-preservation and branch coverage tests after independent review suggestions.
- **Scope drift:** Kept scope to cron/gateway status logic, tests, docs, and ADR; did not modify deployment scripts.

## Session Quality

- **Faithfulness:** Stayed on track — completed the preserved validation/review and devlog path without committing.
- **Prompt patterns:** The preserved task list and previous handoff gave clear success criteria; context compression required re-verifying files and evidence instead of relying on the handoff.

---
*Generated by Hermes Agent — devlog skill*

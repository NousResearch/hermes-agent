# Cron container-aware health and gateway ownership

Created: 2026-05-01
Branch: cron-health-observability-20260501223311
Worktree: /work/.hermes-data/worktrees/hermes-agent-cron-health-20260501223311

## Problem

`hermes cron status` and `hermes cron list` currently decide whether cron will fire by looking for locally visible gateway PIDs via `hermes_cli.gateway.find_gateway_pids()`. That is misleading in containerized deployments where the scheduler/gateway runs in another container but shares the same `HERMES_HOME` volume. In that case the gateway's process namespace may be invisible, while the shared runtime lock and status files are still authoritative enough to show that one gateway owner exists.

The current gateway status helper also risks treating a lock-held PID file as stale when the PID cannot be verified from the current namespace. That is unsafe for shared-volume/container deployments.

## Goals

1. Make cron status/list distinguish:
   - local gateway PID visible;
   - shared gateway runtime lock active but PID not locally visible;
   - no gateway owner detected.
2. Avoid deleting gateway PID/lock metadata when a runtime lock is actively held but the PID is not visible from this namespace.
3. Document the one-supervised-gateway-owner model for container/shared-HERMES_HOME cron deployments.
4. Preserve existing non-container behavior.

## Non-goals

- Add a new scheduler process or cron-only mode.
- Change cron job execution semantics.
- Change Docker Compose or deployment scripts in this PR.
- Make cross-host/network-filesystem locks a supported guarantee.

## Proposed implementation

### Production changes

1. `gateway/status.py`
   - Add a small public helper that reports gateway runtime owner state from the shared runtime lock/PID metadata without requiring local PID visibility.
   - Keep `get_running_pid()` backwards compatible for callers that need a local PID.
   - Change stale cleanup behavior so an active runtime lock with unverifiable PID does not delete PID/lock files.

2. `hermes_cli/cron.py`
   - Replace direct `find_gateway_pids()` cron health checks with a cron-local formatter that uses the gateway runtime helper.
   - Output should be explicit:
     - local PID visible: green, jobs fire automatically;
     - shared runtime lock active: green/notice, jobs likely fire from another namespace/container;
     - no owner: red/yellow, jobs will not fire automatically.

3. Docs
   - Update `website/docs/guides/cron-troubleshooting.md` with container/shared-volume diagnosis.
   - Add ADR `docs/adr/2026-05-01-container-aware-cron-health.md` documenting runtime lock as the cross-namespace scheduler-owner signal.

### Tests

1. Add/extend `tests/hermes_cli/test_cron.py`
   - RED: with no locally visible PID but an active gateway runtime lock, `cron status` should not say jobs will NOT fire.
   - RED: `cron list` should not warn that gateway is not running when a shared runtime lock is active.

2. Add/extend gateway runtime status tests
   - RED: when runtime lock is active but PID is not visible from current namespace, status detection reports `shared_lock_active` or equivalent and does not delete metadata.

## Acceptance criteria

- `pytest tests/hermes_cli/test_cron.py tests/hermes_cli/test_gateway_runtime_health.py` passes.
- Targeted tests show RED before production changes and GREEN after.
- `python -m py_compile gateway/status.py hermes_cli/cron.py` passes.
- Docs and ADR are present in the same branch.
- Repo-local devlog captures goal, files, decisions, RED/GREEN evidence, and learning triage.
- Final guard passes with `HERMES_TDD_EVIDENCE` populated.

## ADR rationale

ADR required. This changes the operational source of truth for cron/gateway liveness in containerized deployments and should be durable architecture knowledge, not just inline comments.

## Risks and mitigations

- Risk: Runtime lock held by a wedged process could make status optimistic.
  - Mitigation: message says shared runtime owner detected, not that every platform is healthy; recent runtime health lines still surface fatal platform states.
- Risk: Network filesystems may not preserve lock semantics.
  - Mitigation: docs explicitly scope this to verified local shared volumes/bind mounts.
- Risk: Existing callers expect PID-based detection.
  - Mitigation: keep `get_running_pid()` semantics and add separate owner-state helper for status/UI use.

# Context Health Update Survival Aftercare

Phase 9: Update survival smoke/regression tests.

Objective: Ensure Hermes update cannot silently bypass context governance.

This document is the post-update aftercare checklist for Context Health governance. It is a smoke/regression detection artifact, not gateway runtime implementation, not command activation, and not runtime config/profile activation.

## Smoke command

Run the smoke command after a Hermes update from the repository root:

```bash
python3 scripts/context_health_smoke.py --json
```

Expected output is a machine-readable summary or a clear PASS/FAIL report. The expected PASS/FAIL output must be reviewed before any deploy/restart or activation decision. A missing hook, missing file, or regression must produce FAIL / nonzero and HOLD.

## Post-update aftercare checklist

1. Run the smoke command immediately after a Hermes update.
2. Confirm policy loading is still detectable.
3. Confirm pre-turn intake hook presence is still detectable before raw append/persist/provider flow.
4. Confirm WCP provider payload enforcement is still detectable.
5. Confirm Task Boundary Firewall default-new behavior is still detectable.
6. Confirm closed task exclusion is still detectable.
7. Confirm retrieval scope enforcement is still detectable.
8. Confirm compact failure fallback still routes to safe HOLD behavior.
9. Confirm threshold does not revert to 85%-only path.
10. Confirm same-window rehydrate path or HOLD if not enabled is still detectable.
11. Confirm this update-aftercare checklist remains present and referenced.

If any check is missing after update, HOLD. Do not silently continue as if Context Health governance is intact.

## Safety contract

The smoke command is dry-run/default safe mode. It is temp HERMES_HOME/tmp_path based by contract and must use synthetic sentinel only inputs. It exits nonzero on missing hooks/regressions. Its report must contain no raw/private/secret/token/password material. It must provide a machine-readable summary or clear PASS/FAIL report.

The smoke/checklist must not touch live runtime surfaces:

```text
real ~/.hermes/state.db
live provider
network
secrets
tmux/session
profile/systemd/cron/wrapper/env/credential
gateway restart/deploy/activation
CLI slash command activation
runtime config/profile activation
```

Additional explicit boundaries:

```text
no real state DB read or mutation
no live provider call
no network dependency
no secret or credential inspection
no tmux/session control
no profile/systemd/cron/wrapper/env/credential mutation
no gateway restart/deploy/activation
no CLI slash command activation
no runtime config/profile activation
```

## Phase 9 scope boundary

Gateway/update survival is Phase 9 smoke/regression detection scope. It is not gateway runtime implementation. It is not a gateway restart, deploy, activation, or runtime behavior change.

Command activation is out of scope. Do not edit `hermes_cli/commands.py`, do not edit `cli.py`, do not add a CLI slash command, and do not execute `/rehydrate` or `/clear` as part of update-survival smoke.

Runtime config/profile activation is out of scope. Do not change config defaults, profiles, env, cron, systemd, wrappers, credentials, symlinks, or live session state from this smoke/checklist.

## HOLD criteria

HOLD if:

- the smoke command is missing;
- the smoke command cannot detect core hook loss;
- policy loading is not detectable;
- pre-turn intake hook presence is not detectable;
- WCP provider payload enforcement is not detectable;
- Task Boundary Firewall default-new behavior is not detectable;
- closed task exclusion is not detectable;
- retrieval scope enforcement is not detectable;
- compact failure fallback is not detectable;
- threshold has reverted to 85%-only path;
- same-window rehydrate path or HOLD if not enabled is not detectable;
- the smoke command requires live provider, network, secrets, real state DB, tmux/session, profile/systemd/cron/wrapper/env/credential, gateway restart/deploy/activation, CLI slash command activation, or runtime config/profile activation.

## Non-claims

This checklist does not prove full A/B contamination fix completion. It only defines the aftercare contract for update survival smoke/regression detection.

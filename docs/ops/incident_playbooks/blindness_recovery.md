# A6 Blindness Recovery Incident Playbook

Date: 2026-03-21
Status: active (Wave 8 A6 Ops Reliability Lane)

## 0) Trigger
This playbook is triggered when the observability layer (`layered_health_snapshot.sh` or `slo_evaluator_snapshot.sh`) fails consistently, resulting in a state where OpenClaw mutators are `failing` or blocked by `a6_observability_failed`.

## 1) Symptoms
- `continuity_now.sh --strict` shows `mutation_gate_projection: status=forbidden; posture=blocker`.
- `verify_then_resume.sh` throws `BLOCKER: verify_then_resume status=BLOCKER ... reason=a6_observability_failed`.

## 2) Mitigation Steps
1. **Identify the core failure:** Check the last layered health output directly.
   `jq . state/continuity/latest/layered_health_snapshot.json`
2. **Determine Process Death:** If the `alive` check failed, the OpenClaw service may be down. 
   - Ensure the process is restarted: `openclaw gateway restart`
3. **Determine Path Drift:** If the `ready` check failed, the state directory may have vanished or changed permissions.
   - Run `mkdir -p state/continuity/latest` and verify write access.
4. **Determine Staleness:** If the `truthful` check failed, the `continuity_now` script may be stuck.
   - Run `bash ops/openclaw/snapshot_ground_truth.sh` manually.
5. **Resume:** After fixing the underlying dependency, re-run `verify_then_resume.sh`. If observability scripts pass, the blocker clears.

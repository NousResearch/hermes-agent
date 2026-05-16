# Step 4: add-undersea-profile-health-inventory

## Objective
Add a read-only profile health inventory step so Undersea Friends can see profile readiness without touching gateways or secrets.

## Scope
- Modify or add scripts only if necessary; prefer extending harness status/reporting code.
- Read-only checks only:
  - profile directory exists
  - `config.yaml` exists
  - `SOUL.md` exists
  - `AGENTS.md` exists
  - required memories/handoff files exist when applicable
  - provider/model names are present without exposing API keys
- Do not read or print `.env` values.
- Do not run gateway restart/stop/start.

## Desired output
A profile inventory report that can be embedded into `status.json` or written as a separate generated status artifact, for example:

```json
{
  "profiles": [
    {"profile": "nemo", "alias": "니모", "role": "execution", "config": "present", "soul": "present", "gateway_check": "not-run"}
  ]
}
```

## Acceptance Criteria
- Inventory is read-only and never outputs secret values.
- Missing profile docs are reported as warnings, not auto-created.
- Tests or a dry-run command prove inventory works on a temporary fake profiles root.
- Documentation says live gateway/process checks remain explicit opt-in diagnostics.

# SecondBrain Memory Provider

Disabled-by-default Hermes memory provider for a controlled SecondBrain trial.

## Switch

The provider reads `$HERMES_HOME/secondbrain.json` on each call:

```json
{
  "enabled": false,
  "mode": "recall_only",
  "base_url": "http://127.0.0.1:3030",
  "project_scope": "secondbrain-phase4-placeholder",
  "timeout_ms": 1500
}
```

- `enabled: false` is the rollback/off switch.
- `mode: recall_only` is the only initially supported live mode.
- Once the provider is loaded in a Hermes session, changing `enabled` in this file is a runtime switch and does not require a gateway restart.
- `SECONDBRAIN_HERMES_TRIAL_ENABLED=false` is a hard-off environment override, but environment changes still require process restart to take effect.
- Secret values must be supplied via environment variables only, never committed.

Enable as active memory provider:

```bash
hermes config set memory.provider secondbrain
```

A new session/reset is required once after changing `memory.provider`, so the provider is loaded. After that, use the JSON switch for on/off.

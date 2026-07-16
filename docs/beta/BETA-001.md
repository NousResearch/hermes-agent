# BETA-001 — Configurable orchestrator identity

BETA-001 introduces a selectable agent identity while preserving Hermes as the default behavior.

## Activation

Edit `~/.hermes/config.yaml`:

```yaml
agent:
  mode: beta
```

Supported values:

- `hermes` — existing Hermes identity and behavior; default.
- `beta` — Beta Chief of Staff identity.

Unknown or malformed values fall back to `hermes` so a configuration error cannot prevent startup.

## Current scope

This milestone changes the top-level identity and operating policy only. It does not yet remove tools from the parent agent or implement the dynamic specialist registry. Beta uses the existing Hermes delegation and Kanban foundations.

The next milestone must move activation from the source/editable-install bootstrap into the packaged runtime initialization, then add the specialist registry and role-aware tool restrictions.

## Validation

Run:

```bash
pytest -q tests/test_beta_identity.py
```

Manual smoke test:

1. Start with no `agent.mode`; the assistant must identify as Hermes.
2. Set `agent.mode: beta` and restart the process.
3. Ask for its role; it must identify as Beta and describe itself as an orchestrator.
4. Set an invalid mode and restart; it must safely return to Hermes.

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

## Runtime integration

`AIAgent` resolves identity once from the active profile's `config.yaml` and `agent/system_prompt.py` consumes that session-stable result. This path is part of the packaged `agent` module, so editable installs, wheels, and official installers use the same behavior; `sitecustomize.py` is not required.

In Hermes mode, `SOUL.md` keeps its original behavior and replaces the built-in Hermes identity. In Beta mode, the Beta identity remains first and `SOUL.md` is appended as user customization. Identity is not re-resolved during a session, preserving prompt-cache stability.

This milestone changes the top-level identity and operating policy only. It does not remove tools from the parent agent or implement the specialist registry. Beta reuses the existing Hermes delegation and Kanban foundations.

## Validation

Run:

```bash
pytest -q tests/test_beta_identity.py tests/agent/test_system_prompt.py
```

Manual smoke test:

1. Start with no `agent.mode`; the assistant must identify as Hermes.
2. Set `agent.mode: beta` and restart the process.
3. Ask for its role; it must identify as Beta and describe itself as an orchestrator.
4. Set an invalid mode and restart; it must safely return to Hermes.

# Upstream Sync

This repo is a fork of:

- `https://github.com/NousResearch/hermes-agent.git`

The fork should stay as close to upstream as possible. MYNAH-specific behavior should live at the runtime edge, not deep inside Hermes core.

## Current remotes

- `origin` -> `git@github.com:ErniConcepts/hermes-agent-core.git`
- `upstream` -> `https://github.com/NousResearch/hermes-agent.git`

## Sync policy

- Do not merge upstream directly into `main` blindly.
- Always sync in a temporary branch first.
- Resolve conflicts conservatively.
- Prefer upstream behavior unless MYNAH would lose a required security or runtime property.

## Current MYNAH customization boundary

The goal is to keep MYNAH-specific behavior concentrated in these places:

- `mynah_runtime/service.py`
- `toolsets.py`
- `model_tools.py`
- runtime-facing tests such as `tests/test_mynah_runtime_service.py`

MYNAH should prefer:

- `SOUL.md` for runtime identity
- MYNAH-specific toolsets for capability restriction
- runtime env/config for deployment behavior

MYNAH should avoid growing deeper changes in:

- `run_agent.py`
- prompt assembly internals
- generic Hermes CLI/gateway behavior

## Known conflict hotspots

These are the files most likely to conflict during upstream syncs:

- `run_agent.py`
- `tests/conftest.py`
- `model_tools.py`
- `toolsets.py`
- `mynah_runtime/service.py`

## Current identity rule

- Upstream now supports `SOUL.md` as the primary identity file.
- MYNAH runtime identity should be seeded through `HERMES_HOME/SOUL.md`.
- MYNAH runtime should not depend on `MYNAH_AGENT_IDENTITY` prompt overrides anymore.

## Recommended sync procedure

1. Fetch upstream.
2. Create a temporary sync branch from local `main`.
3. Merge `upstream/main`.
4. Resolve conflicts with the smallest possible MYNAH-specific delta.
5. Run the focused MYNAH verification slice.
6. Only then merge or fast-forward back to `main`.

Example:

```powershell
git fetch upstream --prune
git checkout -b upstream-sync-YYYYMMDD
git merge upstream/main
```

## Post-merge verification

Run at minimum:

```powershell
python -m pytest tests/test_mynah_runtime_service.py tests/agent/test_prompt_builder_identity.py -o addopts=
python -m compileall mynah_runtime run_agent.py
```

If the MYNAH app changed runtime seeding or launch behavior, also run in `mynah`:

```powershell
python -m pytest tests/test_runtime_bootstrap.py tests/test_runtime_manager.py
python -m compileall src
```

## Resolution guidance

- If upstream improves a generic mechanism MYNAH already customizes, prefer adopting upstream and moving MYNAH back toward configuration.
- If MYNAH-specific behavior can be expressed through `SOUL.md`, env, or runtime wrapper code, do not patch deeper Hermes core.
- If a change would widen the runtime authority or weaken the lockdown model, keep the MYNAH behavior and document the reason.

## What success looks like

A good sync keeps:

- upstream core behavior largely intact
- MYNAH runtime identity via `SOUL.md`
- MYNAH runtime lockdown intact
- MYNAH-specific code concentrated in a small, easy-to-review surface

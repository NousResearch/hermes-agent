# KarinAI runtime notes

KarinAI uses Hermes Agent as the per-user agent runtime foundation.

This document captures KarinAI-specific runtime assumptions so product changes stay visible while the fork continues to track upstream Hermes.

## Intended boundaries

- `karinai-backend` owns product state, public APIs, auth, billing/usage metadata, schedules, run records, and workspace metadata.
- `karinai-backend/services/runtime-manager` owns privileged container lifecycle operations, routing, locks, restore/publish, resource enforcement, and heartbeats.
- `karinai-agent` owns the customized Hermes runtime distribution, runtime entrypoints, tool policy, profile/container templates, and agent image packaging.
- `karinai-infra` owns local/staging/prod deployment wiring and cross-service smoke tests.

## Runtime principles

- Keep one isolated Hermes profile/runtime per KarinAI user or workspace boundary.
- Treat upstream Hermes as the core agent engine; isolate KarinAI glue outside core files where possible.
- Document any required core patches in `docs/karinai-patches.md`.
- Avoid storing product secrets in committed config. Use runtime-provided environment/config injection.
- Keep backend control-plane responsibilities outside the agent fork unless the code is strictly runtime-local.

## Open implementation areas

- Runtime API/entrypoint shape.
- Container image build strategy.
- Tool policy defaults for managed KarinAI sessions.
- Profile volume layout and restore/publish behavior.
- Smoke tests validating backend-to-runtime-to-agent execution.

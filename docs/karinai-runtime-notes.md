# KarinAI runtime notes

KarinAI uses upstream Hermes Agent as the engine foundation for a productized per-user agent runtime.

This document keeps a short index of KarinAI-specific runtime assumptions so product work stays visible while the fork continues to track upstream Hermes. Runtime-local contracts are documented in this repo; product/control-plane contracts remain canonical in the backend repo.

## Canonical architecture

The canonical KarinAI per-user container runtime spec lives in the backend repo:
`karinai-backend/docs/architecture/per-user-container-runtime.md`.

That backend-owned document defines the control-plane contract, Open WebUI fork boundary, runtime-manager responsibilities, private `/v1/runs` routing, active-turn locking, snapshot/restore model, backend-owned cron wakeups, file/code execution policy, security prompt filtering, and security guardrails.

## Agent-side docs

- `docs/karinai-runtime-contract.md` defines the agent-runtime contract inside each user container.
- `docs/karinai-prompt-branding.md` defines how product-facing assistant identity should be template/config driven instead of hardcoded.
- `docs/karinai-patches.md` records unavoidable upstream Hermes core patches.

## Intended boundaries

- `karinai-backend` owns product state, public APIs, auth, billing/usage metadata, schedules, run records, conversations, and workspace metadata.
- `karinai-backend/services/runtime-manager` owns privileged container lifecycle operations, routing, locks, restore/publish, resource enforcement, internal API keys, and heartbeats.
- `karinai-agent` owns the customized upstream Hermes runtime distribution, managed runtime entrypoints, prompt templates, tool policy, profile/container templates, and agent image packaging.
- `karinai-infra` owns local/staging/prod deployment wiring and cross-service smoke tests.

## Runtime principles

- Product-facing runtime name is KarinAI agent; upstream Hermes remains the engine/base.
- The public UI and users must not call user containers directly.
- The canonical backend-to-agent execution API is private `/v1/runs` through runtime-manager.
- Backend product run ids are durable product state; agent runtime run ids are private execution handles.
- Tool policy, prompt identity, and managed-runtime config are rendered by backend/runtime-manager, not trusted from user-editable workspace state.
- The KarinAI agent may help users create scheduled work, but backend owns schedules and due-job firing.
- Raw provider/platform credentials stay outside user containers and are exposed only through scoped model/tool gateways.
- Keep KarinAI glue outside upstream core files where possible; document unavoidable core patches in `docs/karinai-patches.md`.

## Open implementation areas

- Managed runtime entrypoint and config rendering.
- KarinAI prompt template directory and renderer.
- Beta tool policy config and enforcement.
- Backend schedule-intent tool bridge.
- Container image build strategy for the KarinAI agent distribution.
- Profile/workspace volume layout and snapshot manifest.
- Product tests under `tests/karinai/` for managed mode, prompt branding, tool policy, cron bridging, and `/v1/runs` execution.
- Upstream sync and targeted test pass before substantial runtime code is added.

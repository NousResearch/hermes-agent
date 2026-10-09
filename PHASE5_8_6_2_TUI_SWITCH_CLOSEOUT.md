# Phase 5.8.6.2 — TUI model-switch coordinator closeout

Base: `e3192d4b85` (5.8.6.1). Branch:
`refactor/phase5-8-runtime-consumers`.

## Ownership

- TUI switch transaction remains `tui_gateway/model_switch.py`: live agent
  mutation, confirmation, deferred switching, rollback, session-only overrides,
  one-turn restore, profile scoping, worker restart, session events and global
  persistence decisions.
- `tui_gateway/model_switch_resolution.py` feeds caller-owned config and alias
  facts into canonical `models.selection` and `providers.routing`. The
  acquisition-only call to `hermes_cli.runtime_provider.resolve_runtime_provider`
  remains explicit Phase 6 debt; acquired route fields are inputs, not authority.
- The existing gateway-owned command parser, guard aggregator, result enrichment
  and persistence helpers were moved to single shared `application_model_*`
  owners. Gateway and TUI both consume these owners; no gateway dependency or
  forwarding compatibility module was introduced into TUI.
- Preflight compression warning was likewise moved from CLI into the shared
  application preflight owner; CLI and TUI references were updated.
- The shared selection guard again honours the active profile's
  `model.switch_context_confirm_tokens` setting, including zero to disable.
- A URL-bearing alias retains its endpoint when the request explicitly selects
  another provider but must never reuse that alias's credential.
- No process-global model environment mutation on per-session switches.
  Configuration writes use targeted atomic YAML updates.
- A related 5.8.6.1 regression was repaired: legacy rows with neither a custom
  endpoint nor a stable custom identity may recover the profile's configured
  named custom provider only when the stored model matches its default.
  Unrelated stale models cannot inherit that endpoint.

## Verification

- Model resolver, confirmation, preflight, gateway and architecture regressions:
  **49 passed**.
- Legacy custom-provider persistence and recovery suite: **33 passed** after
  test stubs were updated to bind the new TUI config authority.
- Large TUI server model-switch selection: **6 passed**.
- Direct alias-host and configurable threshold tests are included in the 49.
- Final TUI startup/rehydration and gateway launch regressions: **13 passed**.
- Ruff and Git diff checks passed on the changed production and test paths.
  Compilation is a separate closeout gate.

## Deferred work (not scope creep)

- 5.8.6.3 removes remaining TUI catalogue/picker/config/profile imports from
  `hermes_cli.models*`, `hermes_cli.model_switch_providers` and legacy
  model validation.
- The shared application enrichment and warning pipelines still call existing
  CLI-hosted validation/cost/data-policy leaf implementations. The original
  CLI warning aggregator is also still used by the CLI, auth picker and
  dashboard; do not treat those as a second permanent authority. Unifying
  those remaining presentation consumers is explicit later-phase work.
- 5.8.6.6 owns the already-observed dashboard router import of the obsolete
  `agent.model_metadata.is_local_endpoint`.
- Actual credential-source ownership, token persistence and pools remain Phase 6.

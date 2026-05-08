# Changelog

All notable changes to the `vmlx` provider plugin are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — Initial release

### Added
- Two `ProviderProfile` registrations under `plugins/model-providers/vmlx/`:
  - `vmlx` — primary, `http://localhost:8000/v1`, aliases
    `mlx`, `mlx-server`, `apple-mlx`, `vmlx-primary`.
  - `vmlx-janitor` — auxiliary, `http://localhost:8001/v1`, aliases
    `vmlx-aux`, `mlx-janitor`.
- Both profiles use `env_vars=()` and `fallback_models=()` — no API key,
  no cloud fallback (airgap by construction).
- Apple Silicon platform gate via `ImportError` on non-Darwin so the plugin
  is invisible to Linux/Windows contributors.
- README with hardware sizing, dual-port serving instructions, Hermes config,
  and 5-row troubleshooting table.
- pytest suite covering profile registration, default base URLs, and the
  non-Darwin import guard.

### Compatibility
- Built against the v0.13.0 plugin contract introduced in
  [#20324](https://github.com/NousResearch/hermes-agent/pull/20324)
  (ProviderProfile ABC + `plugins/model-providers/`).
- Follows the canonical pattern documented in
  [#20749](https://github.com/NousResearch/hermes-agent/pull/20749)
  (`website/docs/developer-guide/model-provider-plugin.md`).
- Apple Silicon (M1 / M2 / M3 / M4 / M5 family) only.

### Stability
- Pre-1.0: provider names, aliases, and default base URLs may shift in
  response to PR-review feedback. Once the plugin ships in an upstream
  release, semver applies and graduation to 1.0 freezes the surface.

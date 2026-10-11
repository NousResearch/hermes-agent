# pm/ — dependencies and Hermes environments

Applies on top of the root `AGENTS.md` (which carries the short form of the pinning rule).

## Dependency pinning policy

All dependencies carry upper bounds (litellm compromise #2796/#2810; Mini Shai-Hulud worm,
May 2026). PyPI: `>=floor,<next_major` (`"httpx>=0.28.1,<1"`); pre-1.0: `<0.(minor+2)`
(`>=0.29,<0.32`). Git URLs: 40-char commit SHA. GitHub Actions: SHA + `# vN` comment. CI-only
Python requirements: `==exact`. A bare `>=X.Y.Z` is rejected by CI and reviewers.
After changing `pyproject.toml`, run `hermes pm lock`, re-source `./activate`, and commit
`pyproject.toml` with `uv.lock`. Reference: #2810 (bounds), #9801 (SHA pinning + audit CI).

PM owns Hermes Python dependency changes. Use `pm.sync_venv(['extra'], explicit=True)`
for declared runtime extras, `hermes pm install` for setup/sync, and `hermes pm repair`
for damaged dependencies. Do not mutate Hermes environments with raw pip or uv.
Use `pm.build_environment` for fresh build outputs and `pm.ensure_environment` for
isolated tool environments. Callers receive an interpreter or tool path, not uv.
Nix's declarative uv2nix builds and unrelated user projects remain independently owned.

The `[tool.uv] exclude-newer = "14 days"` quarantine covers **Hermes's own dependencies only**
(every registry package in core's `uv.lock`). Plugin `python_dependencies` follow the plugin's own
policy: when PM generates the plugin workspace (`pm/workspace.py::_core_release_quarantine`) the
global cutoff moves onto each core-locked package, so plugin-only packages are not filtered and a
plugin still cannot drag a core package past the window. Teknium's ruling: "plugins dont have to
abide by our 14 day rule … Only hermes' dependencies themselves have to." We recommend (not require)
plugin authors adopt their own quarantine — the developer guide and `plugin-catalog/README.md` carry
that guidance.

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

The `[tool.uv] exclude-newer = "14 days"` quarantine covers core **and plugin** dependencies, one
policy for both: exact-pin a fresh direct dependency and exempt it with
`[tool.uv] exclude-newer-package = { name = false }` (core does this for its own direct deps);
everything else, transitive deps included, waits out the window. uv reads that table only at the
workspace root, so `pm/workspace.py::_release_quarantine` lifts each plugin's exemptions into the
generated root, filtered by `pm/plugin_declarations.py::quarantine_exemptions`: `false` on an
exact-pinned direct dependency that core's lock does not hold. A plugin can let its own fresh
release through; it cannot loosen a core package or a transitive one. `hermes plugins validate`
fails exemptions PM would ignore (`dependency quarantine` check), and
`scripts/ci/catalog_resolve_all.py` locks every catalog entry together with core.

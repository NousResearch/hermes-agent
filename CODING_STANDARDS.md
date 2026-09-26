# Hermes Agent coding standards

These are durable repository-wide implementation rules. `AGENTS.md` owns contributor safety, scope, and routing. A nested `AGENTS.md` may add local constraints.

## Facades and topical siblings

Large public modules are facades. Keep public entry points and compatibility-owned names in the facade. Put behavior in `<stem>_<topic>.py` siblings in the same package.

- Find and edit the topical sibling instead of appending behavior to a facade.
- Split a growing file or function along a coherent responsibility before adding more behavior.
- Avoid module-level import cycles. A sibling may late-import the facade inside a function when it needs the facade's patch seam.
- Patch where production reads the name. Check the call site's binding before choosing a monkeypatch target.
- In-tree code imports the defining module. Compatibility pointers and `PLUGIN-COMPAT` blocks exist only for external consumers and must not become internal dependencies.
- Internal moves update all callers and documentation in the same change. Do not add per-move re-export shims.

## Code shape

- Prefer a table or registry over four or more branches keyed by a name, kind, route, backend, command, or platform.
- Extend an existing resolver, writer, lifecycle owner, or orchestration path. Do not create a second implementation of the same policy.
- Keep comments and docstrings for intent, boundaries, and surprising constraints. Let names and structure explain mechanics.
- Avoid speculative flags, hooks without a consumer, impossible-error wrappers, and `try/except: pass` around code that should fail visibly.
- Resolve mutable configuration and profile-owned state at call time. Do not freeze it in import-time constants.
- Process identity uses canonical full-command matchers and start-time fingerprints. Never infer identity from an argv substring or bare PID existence.
- Argparse aliases return the literal token the user entered. Dispatch accepts the canonical name and every declared alias.
- Configuration files have one canonical writer. Preserve round-trip formatting and fail closed on unreadable input instead of replacing it.

## TypeScript

- Shared or distant UI state lives in small feature-owned nanostores. Rendering code uses `useStore`. Non-rendering actions use `$atom.get()`. Component-local interaction state stays local.
- Keep route roots thin. Put one job in each hook and colocate actions with the feature that owns them.
- Use interfaces for public props and shared object shapes. Extend React primitives with `React.ComponentProps`, `Omit`, or `Pick` instead of restating them.
- Prefer tables for IDs, routes, and views. Keep `src/app` for routes and pages, `src/store` for shared atoms, and `src/lib` for pure helpers.
- Make ignored promises explicit with `void`. For pure callbacks use `value => void update(value)`. For event handlers use `() => void save()`.
- Preserve reference identity when data has not changed. Avoid subscribing expensive trees to high-frequency state.

## Dependencies

- Every registry dependency has an upper bound. Python runtime ranges use a floor and a next-major ceiling. For pre-1.0 packages, cap below `0.(minor+2)`; for example, `>=0.4.2,<0.6`. CI-only Python requirements use exact pins.
- Git dependencies use a full 40-character commit SHA. GitHub Actions use a commit SHA with a version comment.
- Change Hermes Python dependencies through the package manager, not raw `pip` or `uv`. After `pyproject.toml` changes, run `hermes pm lock`, re-source `./activate`, and commit `uv.lock` with the manifest.
- Use `pm.build_environment` for fresh build outputs and `pm.ensure_environment` for isolated tool environments. Callers receive an interpreter or executable path, not package-manager internals.
- Hermes's release quarantine governs core locked dependencies. Plugin-only dependencies keep the plugin's policy and may not move a core package outside the core lock constraints.

## Platform and process boundaries

- Read control-host facts through `hermes_platform.host`. Add executable and resource discovery to `hermes_platform/resolver/`. Lookup does not install, download, or start software.
- Keep the control host, terminal execution target, and desktop client distinct. Host facts describe only the Python process host.
- Build subprocess environments through `tools/environments/local.py`. A child acting for a served profile uses `served_profile_child_env`. `os.environ.copy()` cannot carry profile context safely.
- Carry context variables into threads with the repository's scoped thread helper. Resolve profile state before spawning a process because context variables do not cross process boundaries.
- Test host-specific behavior on that host with one `@pytest.mark.platforms(...)` marker. Do not patch `sys.platform` or stack platform markers. Pure functions may accept platform as data.
- Use `pathlib.Path`, temporary directories, and platform helpers. Do not hardcode POSIX temporary paths, shell behavior, or user-home layouts.

## Testing

Run Python tests with `scripts/run_tests.sh`, never bare `pytest`. Tests mirror source ownership under `tests/<source-area>/`. JavaScript and TypeScript assertions stay in the owning package's Vitest suite so JavaScript-only changes select them in CI.

- Test public behavior, durable invariants, and relationships between data. Do not read production source text from a test or freeze counts, catalogs, versions, generated lists, or incidental implementation shape.
- Patch the binding production uses. Prefer real imports and isolated temporary homes for configuration, discovery, security, and I/O boundaries.
- Profile-sensitive tests isolate both `Path.home()` and `HERMES_HOME` when the code reads both. They exercise A -> B -> A where cross-profile leakage is possible.
- Time-sensitive tests use event synchronization and bounds wide enough for a loaded runner. A flaky pass on retry is still a defect.
- A test that needs another host is marked for that host. Declaration and pure-data invariants remain host-independent.
- Run the smallest faithful focused check first, then the owning directory or package. Build or run the real local integration when unit tests cannot prove the boundary.

Use `CONTRIBUTING.md` for environment setup and submission checks. Use the area's `AGENTS.md` for local test targets and invariants.
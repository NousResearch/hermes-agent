# Hermes Test Guide

These instructions apply to changes under `tests/`. The repository-wide
engineering invariants in the root `AGENTS.md` still apply.

## Run tests through the project wrapper

Always use `scripts/run_tests.sh`; do not invoke `pytest` directly. The wrapper
matches CI by isolating `HERMES_HOME`, removing credentials, setting UTC and a
stable locale, enabling xdist, and running each test file in a fresh subprocess.

```bash
scripts/run_tests.sh
scripts/run_tests.sh tests/gateway/
scripts/run_tests.sh tests/agent/test_foo.py::test_x
scripts/run_tests.sh -v --tb=long
```

The wrapper retries a failing test file once. A pass-on-retry is reported as
flaky and must be treated as a bug, not ignored. Prefer event-based
synchronization and generous wall-clock bounds over negative timing assertions.

## Isolation rules

- Tests must never read from or write to the user's real `~/.hermes`.
- Use the `_isolate_hermes_home` autouse fixture from `tests/conftest.py`.
- Profile tests must mock `Path.home()` and set `HERMES_HOME` to the same
  temporary root.
- No live network calls in unit tests.
- Test subprocesses cannot rely on process-global state created by another test
  file.

Canonical profile fixture:

```python
@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home
```

## Test behavior, not source shape

Never read source files in a test to assert that a string, call expression,
variable name, or regex still exists. Such tests fail on harmless refactors and
pass when the code is present but wired incorrectly. Extract logic into a pure
or dependency-injected function and execute it.

Do not write change-detector tests for values expected to evolve:

- model names or catalog snapshots;
- configuration version literals;
- provider or tool enumeration counts;
- hardcoded current lists.

Assert contracts and relationships instead:

- a provider catalog is non-empty;
- migration reaches `DEFAULT_CONFIG["_config_version"]`;
- plan-only models do not leak into a legacy list;
- every catalog model has a context-length entry.

## Put tests in the suite that CI will run

The change classifier chooses jobs from changed paths. Tests that inspect or
exercise JavaScript/TypeScript artifacts belong in the Vitest suite, not in
Python tests. This includes `package.json`, lockfiles, TypeScript configuration,
and `.ts`/`.tsx`/`.js` behavior.

Use Python tests for Python runtime behavior and cross-component integration
that genuinely enters through Python.

## Validation depth

- Small pure functions: focused unit tests.
- Resolution chains, configuration propagation, security boundaries, remote
  backends, or file/network I/O: exercise the real import and execution path
  against a temporary `HERMES_HOME`.
- Avoid mocks that bypass the integration seam being changed.
- A green mock-only test is insufficient when the reported failure occurred in
  wiring, discovery, configuration, or serialization.

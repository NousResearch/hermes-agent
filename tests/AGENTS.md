# Tests Guide

This directory owns the pytest suite. Prefer behavior and invariants over
snapshots of changing catalogs or counts.

## Runner

Always prefer the wrapper:

```bash
scripts/run_tests.sh
scripts/run_tests.sh tests/gateway/
scripts/run_tests.sh tests/agent/test_foo.py::test_bar -v --tb=long
scripts/run_tests.sh --no-isolate tests/foo/
```

The wrapper gives CI-like behavior: credential env vars unset, temp HOME and
HERMES_HOME, UTC timezone, C.UTF-8 locale, xdist, and subprocess isolation.

Direct `pytest` is acceptable only when a tool or IDE cannot invoke the wrapper.
At minimum activate the venv first. The in-tree isolation plugin still loads
from `pyproject.toml`.

## Subprocess Isolation

Every test normally runs in a fresh Python subprocess via `tests/_isolate_plugin.py`.
This prevents module globals, ContextVars, and singleton state from leaking
across tests.

Use `--no-isolate` only for focused debugging or when intentionally testing
state leakage.

## HERMES_HOME

Tests must not write to the real `~/.hermes`. The autouse fixture redirects
`HERMES_HOME` to a temp directory.

Profile tests must also mock `Path.home()` so profile roots resolve under the
temp directory:

```python
@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home
```

## Avoid Change-detector Tests

Do not assert snapshots of data expected to change, such as exact provider model
lists, config version literals, enumeration counts, or catalog snapshots.

Do assert relationships and contracts:

- a catalog has at least one entry for a provider,
- every model has a context length,
- migrations bump user config to the current version,
- plan-only models do not leak into legacy lists,
- a command registry projection contains a newly added command.

If the test fails whenever routine data is refreshed, convert it into an
invariant or delete it.

# Testing in Hermes Agent

Always use `scripts/run_tests.sh` to run tests. Do not call `pytest` directly.

## Environment Parity

`scripts/run_tests.sh` enforces parity with CI by setting a hermetic environment:
- Unsets credential variables (`*_API_KEY`, `*_TOKEN`, etc.)
- Sets `TZ=UTC`
- Sets `LANG=C.UTF-8`
- Enforces `-n auto` xdist workers
- Uses the in-tree subprocess-isolation plugin (`tests/_isolate_plugin.py`)

## Subprocess Isolation

Every test runs in a freshly-spawned Python subprocess. This prevents state leaks (ContextVars, module-level dicts/sets) between tests.
- Uses `multiprocessing.get_context("spawn")`
- Overhead: ~0.5–1.0s per test, amortized by xdist parallelism.
- Timeout: capped at 30 seconds per test (`isolate_timeout` in `pyproject.toml`).
- To disable isolation for fast feedback during debugging, pass `--no-isolate`:
  ```bash
  scripts/run_tests.sh --no-isolate tests/agent/test_foo.py
  ```

## Writing Invariant Tests (Anti-Change-Detector)

Do not write change-detector tests that fail when expected-to-change metadata changes (like hardcoded model catalogs or version numbers). Assert invariants or behaviors instead.

**Bad (breaks on every release):**
```python
assert "gemini-2.5-pro" in _PROVIDER_MODELS["gemini"]
assert DEFAULT_CONFIG["_config_version"] == 21
```

**Good (asserts behaviors/contracts):**
```python
assert "gemini" in _PROVIDER_MODELS
assert len(_PROVIDER_MODELS["gemini"]) >= 1
assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]
```

## Isolation from `~/.hermes/`

The `_isolate_hermes_home` fixture in `tests/conftest.py` redirects `HERMES_HOME` to a temp directory. Never write to `~/.hermes/` in tests.
If testing profiles, mock `Path.home()` to keep profiles inside the temp directory:
```python
@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home
```

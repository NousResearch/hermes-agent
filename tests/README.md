# Tests

Hermes uses `pytest` for the Python test suite. Prefer the canonical test
runner when working from a POSIX-like checkout:

```bash
scripts/run_tests.sh
```

The runner mirrors CI-oriented defaults:

- activates a local `.venv`, `venv`, or installed Hermes source venv
- clears pytest `addopts` before setting its own worker count
- runs with 4 xdist workers by default (`HERMES_TEST_WORKERS=4`)
- excludes `tests/integration` and `tests/e2e`
- runs `-m "not integration"`
- unsets credential-shaped environment variables before pytest starts
- pins deterministic runtime settings such as `TZ=UTC`, `LANG=C.UTF-8`, and
  `PYTHONHASHSEED=0`

Run a subset by passing paths or pytest selectors:

```bash
scripts/run_tests.sh tests/agent/
scripts/run_tests.sh tests/tools/test_example.py::test_specific_case
scripts/run_tests.sh --tb=long -v
```

## Direct pytest usage

Direct pytest is useful for quick local iteration, but remember that
`pyproject.toml` currently adds `-m 'not integration' -n auto` by default.
To override those defaults explicitly:

```bash
python -m pytest tests/path/to/test_file.py -o "addopts=" -n 0 -v
```

## Hermetic test invariants

`tests/conftest.py` protects local and CI runs by:

- redirecting `HERMES_HOME` to a per-test temporary directory
- unsetting credential variables such as API keys, tokens, and secrets
- blocking inherited Hermes session, gateway, and kanban variables
- setting deterministic runtime defaults
- resetting shared singleton state between tests

Tests should use `get_hermes_home()` for Hermes state paths. Code that reads
`Path.home() / ".hermes"` directly can escape the test sandbox and should be
fixed at the callsite.

## Integration and e2e tests

Integration and e2e tests are intentionally excluded from the default local
runner. Run them only when you have the required services and credentials set
up, and prefer documenting the required environment in the test file.

## Platform notes

`scripts/run_tests.sh` assumes POSIX virtualenv layout (`bin/activate`). On
native Windows, use a Python environment with pytest installed and invoke
pytest directly. Some tests that rely on POSIX behavior, such as symlinks,
file mode bits, or `signal.SIGALRM`, should use explicit skip guards.

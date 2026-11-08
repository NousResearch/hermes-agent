# Local test baseline

A fast, reliable set of test targets for the devtonic-agent fork, how to run
them, and which areas are known to fail or flake on which host.

## Running it

```bash
uv sync --group dev --group test
uv run python scripts/devtonic/baseline.py                 # default fast set (~2 min)
uv run python scripts/devtonic/baseline.py tests/cron      # explicit targets
uv run python scripts/devtonic/baseline.py --json out.json # machine-readable results
uv run python scripts/devtonic/baseline.py tests/x -- -k foo   # args after -- go to pytest
```

- Each **test file** runs in its own `python -m pytest` subprocess (the same
  isolation unit as upstream's `scripts/run_tests.sh` /
  `scripts/run_tests_parallel.py`), up to `-j` files in parallel. Running a whole
  directory in one process leaks module state between files and produces false
  failures — e.g. `tests/providers` shows 19 failures in one process and 0 per
  file. `--per-dir` restores the single-process mode for debugging.
- `TZ=UTC` and `PYTHONHASHSEED=0` are pinned. `tests/conftest.py` sandboxes
  `HERMES_HOME` and fences the network, so the baseline never touches a real
  `~/.hermes` install.
- `scripts/run_tests.sh` remains the CI-parity runner; it needs the
  `pm.build_env` test interpreter (see `tests/AGENTS.md`). `baseline.py` only
  needs the repo's `uv` environment.

## Default targets (must stay green)

Measured on macOS 15.5 arm64, Python 3.14.6 (uv-managed), SQLite 3.53.4:

| Target | Status | Passed | Failed | Errors | Skipped | Time (s) |
|---|---|---|---|---|---|---|
| `tests/devtonic` | pass | 7 | 0 | 0 | 0 | 1 |
| `tests/hermes_platform` | pass | 122 | 0 | 0 | 5 | 2 |
| `tests/conformance` | pass | 15 | 0 | 0 | 6 | 24 |
| `tests/providers` | pass | 93 | 0 | 0 | 0 | 23 |
| `tests/skills` | pass | 1567 | 0 | 0 | 1 | 14 |
| `tests/cron` | pass | 1594 | 0 | 0 | 11 | 63 |

Linux: not yet measured locally. Devtonic CI (`ubuntu-latest`) currently runs
only the import smoke plus changed test files; the next roadmap item expands it
and will fill in the Linux column.

## Opt-in targets

| Target | How to run | macOS result |
|---|---|---|
| `tests/acp_adapter` | `uv run --extra acp python scripts/devtonic/baseline.py tests/acp_adapter` | pass (421 passed, 37 skipped, 28 s) |
| `tests/monitoring` | default env | pass (19 passed, 2 skipped) |

`tests/acp_adapter` needs the `acp` extra (`agent-client-protocol`); without it
every file fails at import with `ModuleNotFoundError: No module named 'acp'`.
Note `uv run --extra acp` re-syncs the venv; a later plain `uv run` removes the
extra again.

## Known failures

| Target | Host | Result | Symptom |
|---|---|---|---|
| `tests/hermes_state` | macOS 15.5, Py 3.14.6, SQLite 3.53.4 | 68 failed / 1761 passed across 22 files | `sqlite3.DatabaseError: malformed database schema (messages_fts) - table messages_fts already exists` in the state.db repair / FTS rebuild / auto-vacuum paths (`test_state_db_malformed_repair.py`, `test_state_db_repair_non_destructive.py`, `test_hermes_state.py::TestVacuum`/`TestAutoMaintenance`, ...). Deterministic, not flaky. Suspected interaction with the newer bundled SQLite; untriaged. |

## Known flaky

None observed yet in the default set (two consecutive per-file runs were
identical). Record flakes here with host, file and symptom as they show up.

## Not yet measured

`tests/agent`, `tests/tools`, `tests/gateway`, `tests/hermes_cli`,
`tests/tui_gateway`, `tests/e2e`, `tests/integration`, `tests/pm`,
`tests/plugins`. These are large; measure them one at a time with explicit
`baseline.py` runs and add rows above.

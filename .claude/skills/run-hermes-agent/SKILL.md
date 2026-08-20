---
name: run-hermes-agent
description: Build, run, and drive the Hermes Agent CLI (NousResearch/hermes-agent). Use when asked to run hermes, test the hermes CLI, verify it's installed correctly, or check `hermes doctor`/`hermes -z`/`hermes chat` behavior.
---

Hermes Agent is a Python CLI (`hermes`, entrypoint `hermes_cli.main:main`)
managed with `uv`. Drive it via `.claude/skills/run-hermes-agent/smoke.sh`
— it builds the venv, checks version/doctor, exercises the non-interactive
`-z`/`--oneshot` invocation path (the scriptable surface most PRs touch,
as opposed to the interactive TUI/REPL), and runs a pytest subset.

All paths below are relative to the repo root.

## Prerequisites

- `uv` (any recent version — tested with `uv 0.12.0`). Node.js present
  but not required for the CLI itself (only for the separate `apps/desktop`
  Electron app and the `--tui`/`ui-tui` frontend, which this skill does
  not cover).
- Python is installed *by* `uv`, not by you — no manual Python setup needed.

## Setup / Build

```bash
bash .claude/skills/run-hermes-agent/smoke.sh
```

This is the whole build+verify path: pins Python 3.12.13 (`requires-python
>=3.11,<3.14`), runs `uv sync --extra dev`, then drives the CLI (see below).
Takes ~1-2 min cold (dependency download), seconds on a warm cache.

If you want the steps individually instead of the script:

```bash
uv python pin 3.12.13
uv sync --extra dev        # base deps + pytest/ruff/ty (the `dev` extra)
uv run hermes --version    # → Hermes Agent v0.19.0 (...) · upstream <sha>
```

## Run (agent path)

The smoke script is the driver. It has no interactive prompts — safe to
run unattended. What it does, in order:

| step | command | what it proves |
|---|---|---|
| version | `uv run hermes --version` | entrypoint resolves, venv is sane |
| health | `uv run hermes doctor` | config/deps/dirs check; exits 0 even with warnings |
| oneshot, no key | `uv run hermes -z "say hi"` | fails fast with a clear message, **exit 1** |
| oneshot, dummy key | `OPENROUTER_API_KEY=<dummy> uv run hermes -z "say hi"` | request path reaches the provider; prints the HTTP error as if it were the reply, **exit 0** (see Gotchas) |
| tests | `scripts/run_tests.sh tests/test_account_usage.py -q` | test harness works, CI-parity |

The no-key and dummy-key rows above are verified against current `main` (`hermes_cli/oneshot.py`), run inside an isolated `HERMES_HOME` — see the next paragraph for why that isolation matters.

To actually get a real answer out of `-z`/`hermes chat -q`, set a real
key first — `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, etc. (env var, or
`~/.hermes/.env` via `hermes setup`) — then:

```bash
uv run hermes -z "say hi"                    # one-shot, prints only the final text
uv run hermes chat -q "say hi" -Q            # same idea, quiet mode
```

Both are the scriptable/programmatic path. `hermes` with no args, or
`hermes chat`/`hermes --tui`, launch an interactive REPL/TUI instead —
those need a real terminal (tmux) and aren't covered by this skill.

## Run (human path)

```bash
uv run hermes setup   # interactive wizard: pick provider, enter API key
uv run hermes         # interactive chat once configured
```

## Test

```bash
uv sync --extra dev
scripts/run_tests.sh tests/test_account_usage.py -q   # → 4 passed
```

Use `scripts/run_tests.sh`, not raw `pytest` — it enforces hermetic
CI-parity (unset credential vars, `TZ=UTC`, `LANG=C.UTF-8`, per-file
subprocess isolation) per `AGENTS.md`'s Testing section. A bare `pytest`
invocation on a dev machine with real API keys set can diverge from CI in
ways that pass locally and fail (or vice versa) upstream. It only looks
for an interpreter at `./.venv`, `./venv`, or `$HERMES_PYTHON` — if
you've relocated the venv per the OneDrive Gotcha below, export
`HERMES_PYTHON=$UV_PROJECT_ENVIRONMENT/Scripts/python.exe` (or
`.../bin/python` on POSIX) first. `smoke.sh` does this automatically.

The full `tests/` tree is large; run a targeted subset rather than the
whole suite unless you specifically need full coverage. Note: any test
under `tests/test_atomic_replace_symlinks.py` will fail on Windows
without Developer Mode / admin (see Gotchas) — not a code bug.

---

## Gotchas

- **`uv sync` fails with `error: failed to remove directory ... Access
  is denied. (os error 5)` when the checkout lives under a
  OneDrive-synced folder.** OneDrive's cloud-sync file locking races
  with uv's atomic package-directory replace. Fix: point the venv
  outside OneDrive before syncing:
  ```bash
  export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/hermes-agent"
  uv sync --extra dev
  ```
  `smoke.sh` already does this. Retrying `uv sync` in place sometimes
  "succeeds" but leaves a half-replaced package — don't trust a retry,
  relocate instead.
- **`hermes -z` exits 0 even when the LLM call itself fails upstream**
  (e.g. bad API key → `HTTP 401: ...` printed to stdout as if it were
  the reply). Only a *local* config problem (no provider configured at
  all) exits 1. Verified directly against current `main`
  (`hermes_cli/oneshot.py`) with an isolated `HERMES_HOME` — the 401
  response text makes `response` non-empty, so the failure never reaches
  the `result.get("failed")` / empty-response checks that would otherwise
  return 1 or 2. A script checking "did this succeed" by exit code alone
  will treat a 401 as success — check the printed text too.
- **`hermes` loads `~/.hermes/.env` (or `$HERMES_HOME/.env`)
  unconditionally on every invocation** (`hermes_cli/main.py`, before
  dispatch). On a machine that already has a real provider configured,
  the "no key" / "dummy key" smoke steps above would silently call the
  real provider instead of exercising the failure path — `smoke.sh`
  isolates this by pointing `HERMES_HOME` at a fresh `mktemp -d` for its
  own run. Do the same (`HERMES_HOME=$(mktemp -d) ...`) if you run the
  oneshot steps manually outside the script.
- **`scripts/run_tests.sh` prints a `UnicodeEncodeError` traceback from
  `run_tests_parallel.py`'s progress printer on stock Windows consoles**
  (cp1252 can't encode the `✓` progress character), but the run still
  completes and reports `N passed, 0 failed` with exit 0 — cosmetic, not
  a real failure. Pre-existing in the repo's test runner, not specific to
  this skill.
- **`tests/test_atomic_replace_symlinks.py` fails on stock Windows**
  with `OSError: [WinError 1314] A required privilege is not held by
  the client` — creating symlinks needs Developer Mode or admin.
  Unrelated to the app; skip that file when smoke-testing.
- **`uv sync` (no `--extra dev`) does not install pytest.** `pytest`
  and friends live in the `dev` optional-dependency group in
  `pyproject.toml` — use `uv sync --extra dev` for anything test-related.

## Troubleshooting

- **`Python was not found; run without arguments to install from the
  Microsoft Store...`** when invoking `python`/`python3` directly:
  Windows' App Execution Alias is shadowing the command. Ignore it —
  use `uv run hermes ...` / `uv python pin` instead of calling `python`
  directly; `uv` manages its own interpreter and doesn't go through
  that alias.
- **`hermes doctor` reports `⚠ .env file missing` / `config.yaml not
  found`**: expected on a fresh clone. Not an error — `hermes doctor`
  still exits 0. Run `hermes setup` to fix, or ignore if you only need
  the CLI to *launch*, not actually chat.

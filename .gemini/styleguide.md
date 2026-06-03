# hermes-agent Code Review Style Guide

## Project Context

hermes-agent is "the self-improving AI agent — creates skills from experience,
improves them during use, and runs anywhere" (Nous Research, MIT). It is a
multi-provider LLM agent runtime. Primary stack: **Python >= 3.11** (the bulk
of the code), with a TypeScript docs/site surface and a Nix/flake build. Core
subsystems: `agent/` (runtime + provider adapters), `providers/` (LLM provider
base), `gateway/` (channel delivery + hooks), `acp_adapter/` / `acp_registry/`
(agent-comms protocol), `hermes_cli/`, `skills/` + `optional-skills/` +
`plugins/` (user-authored extension surfaces).

Tooling (from `pyproject.toml`): **ruff 0.15.10** (lint), **ty 0.0.21** (type
check, `python-version = 3.13`), **pytest 9** with `pytest-asyncio` +
`pytest-timeout` (30s per-test signal cap, `-m 'not integration'`). Dependency
manager: **uv** (`uv.lock`).

## Critical Review Areas

### 1. Supply-chain / dependency pinning (the #1 hard rule here)

Every direct dependency in `pyproject.toml` is **exact-pinned** (`==X.Y.Z`),
never a range. This is a deliberate security control, not a style choice: it
was tightened on 2026-05-12 after the Mini Shai-Hulud worm shipped a malicious
`mistralai` release to PyPI — an exact pin meant the only way a new version
reaches a user is an intentional bump on our side.

- **Flag** any new dependency added with a range specifier (`>=`, `~=`, `^`,
  `*`, or an open upper bound) instead of `==X.Y.Z`.
- **Flag** a `pyproject.toml` dependency bump that does **not** also update
  `uv.lock` (the lockfile must be regenerated with `uv lock` so transitive
  resolution stays consistent). A drifted `uv.lock` is a review blocker.
- **Flag** any new third-party import that has no corresponding pinned entry in
  `pyproject.toml`.

### 2. File I/O encoding (ruff `PLW1514`, load-bearing)

Bare `open()`, `Path.read_text()`, `Path.write_text()`, `read_bytes`-to-text,
etc. in **text mode without an explicit `encoding=`** default to the system
locale encoding — `cp1252` on US-locale Windows — which silently corrupts any
non-ASCII content. This caused three Windows-sandbox regressions in a single
debug session.

- **Flag** any new text-mode file read/write that omits `encoding="utf-8"`
  (outside `tests/`, `skills/`, `optional-skills/`, `plugins/`, which are
  intentionally exempt via per-file-ignores).

### 3. Secrets & provider credentials

This is an agent that talks to many LLM providers (`agent/anthropic_adapter.py`,
`bedrock_adapter.py`, `azure_identity_adapter.py`, etc.), so credential
handling is a primary attack surface.

- **No hardcoded API keys, tokens, or secrets** in source — they must come from
  environment variables or the configured secret source. Flag any literal that
  looks like a key (`sk-...`, `AKIA...`, bearer tokens, PEM blocks).
- **No secret values in logs.** Flag `print()`/`logging` calls that interpolate
  an API key, auth header, or full request payload that may carry credentials.
- Fail-closed on missing credentials: a provider adapter that silently proceeds
  (or falls back to an unauthenticated path) when its key is absent is a bug —
  it should raise a clear error.

### 4. Async correctness

The runtime is async (`pytest-asyncio`, `async_utils.py`, `auxiliary_client.py`).

- **Flag** a blocking call (sync network I/O, `time.sleep`, sync file I/O on a
  hot path) inside an `async def` — it stalls the event loop.
- **Flag** a coroutine that is created but never `await`ed (a bare
  `some_async()` whose result is discarded).
- **Flag** missing timeouts on outbound provider/network calls — the test suite
  enforces a 30s cap precisely because hangs are a known failure mode; the
  product code should bound its own I/O too.

### 5. Test discipline

- A test that hits a real external service (API key, Modal, network) MUST carry
  the `@pytest.mark.integration` marker — the default run is `-m 'not
  integration'`, so an unmarked test that needs a key will fail in CI for
  everyone. Flag new external-dependency tests that lack the marker.
- Don't introduce a test that relies on >30s of wall time; the per-test signal
  timeout is 30s and per-file isolation gives each file a fresh interpreter.

## Code Style Rules

### Python

- Target **Python >= 3.11** syntax (3.13 for type-checking). Prefer modern
  typing (`X | None` over `Optional[X]`, builtin generics `list[...]`).
- Keep `ty` clean: `unknown-argument` is `warn`, so a new call passing an
  argument the callee doesn't declare is a smell worth flagging.
- Follow the immutability bias of the codebase — prefer returning new objects
  over mutating arguments in place, especially for shared agent/runtime state.

### Writing style (applies to code comments and any user-facing strings)

- NEVER use em-dashes in strings, comments, or CLI output. Use commas, colons,
  or parentheses.
- Keep comments about *why*, not *what* — the code already says what it does.

### Error handling

- Never silently swallow an exception. Log with module context, e.g.
  `logger.error("[provider] anthropic call failed: %s", err)`, and re-raise or
  return a typed error — don't `except Exception: pass`.
- Validate external/untrusted input (provider responses, user-authored skill
  manifests, CLI args) at the boundary before use.

## What NOT to Flag

- ruff lints other than `PLW1514` — they are intentionally disabled in
  `[tool.ruff.lint]` while typechecks are being wrangled; only `PLW1514` is
  enforced. Do not flag unrelated style lints.
- Conventions inside `skills/`, `optional-skills/`, and `plugins/` — these are
  partially user-authored and carry their own conventions (they are exempt from
  `PLW1514` via per-file-ignores). Don't impose core-runtime rules there.
- File length alone (some adapters and the gateway are legitimately long).
- Missing docstrings on internal helpers (self-documenting code is preferred).
- `uv.lock` content diffs (the lockfile is generated; review the
  `pyproject.toml` change that drove it, not the resolved tree).

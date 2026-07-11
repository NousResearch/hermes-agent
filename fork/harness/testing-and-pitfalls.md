# Hermes Testing and Pitfall Rules

This document contains the detailed rules referenced by the root `AGENTS.md`.
It is part of the fork harness, not application runtime code. Read it before
editing tests, filesystem paths, gateway message guards, or test isolation.

## Known pitfalls

### Profile-safe paths

Use `get_hermes_home()` for code paths and `display_hermes_home()` for
user-facing messages. Never hardcode `~/.hermes/`; each profile has its own
`HERMES_HOME`.

### Interactive terminal UI

Do not add new `simple_term_menu` call sites. New interactive menus use the
stdlib curses UI, following `hermes_cli/tools_config.py`. `simple_term_menu`
remains only for legacy fallback paths because it can duplicate arrow-key
rendering in tmux and iTerm2.

### Terminal rendering

Do not emit `\033[K` from spinner or display code. Under `prompt_toolkit`
`patch_stdout` it can appear as literal text. Use space padding instead.

### Process-global tool state

`_last_resolved_tool_names` in `model_tools.py` is process-global. Child-agent
execution saves and restores it. New readers must tolerate the value being
temporarily stale while a child runs.

### Tool schema references

Schema descriptions must not name tools from other toolsets. Those tools may be
disabled or unavailable. Add conditional cross-references in
`get_tool_definitions()` instead, following the existing browser and code
execution post-processing pattern.

### Gateway message guards

Messages pass through both the base adapter pending-message guard and the
gateway runner command guard. Commands that must reach a blocked agent, such
as approval controls, must bypass both guards and dispatch inline. Do not route
them through `_process_message_background()` where session teardown can race.

### Stale branch merges

Before a squash merge, update the branch from current `main` and inspect the
resulting diff. A stale branch can silently restore an old version of an
unrelated file. Unexpected deletions are a stop signal.

### Live-path validation

Do not wire unused code into a live path without an end-to-end check using real
imports and a temporary `HERMES_HOME`. Mocks alone cannot prove provider
resolution, profile propagation, or file/network boundaries.

## Test isolation

Tests must not write to the real `~/.hermes/`. The autouse fixture in
`tests/conftest.py` redirects `HERMES_HOME` to a temporary directory. Profile
tests must also patch `Path.home()` so profile roots stay inside that temporary
directory:

```python
@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home
```

## Test runner

Use `scripts/run_tests.sh` for CI-parity runs. It unsets credential variables,
sets UTC and C.UTF-8, uses the repository's xdist settings, and applies the
subprocess isolation plugin.

```bash
scripts/run_tests.sh
scripts/run_tests.sh tests/gateway/
scripts/run_tests.sh tests/agent/test_foo.py::test_x
scripts/run_tests.sh -v --tb=long
```

Each test file runs in a fresh Python subprocess. This prevents module-level
state and ContextVars from leaking between files and keeps local results close
to CI behaviour.

On Windows, follow the repository's Windows test instructions in the root
`AGENTS.md` and use the project-supported Python runner. POSIX-only tests must
use the existing skip patterns for symlinks, mode bits, signals, and live
Winsock behaviour.

## Behaviour tests, not change detectors

Do not freeze data that is expected to change, such as model names, config
version literals, provider lists, or enumeration counts. Such tests only turn
routine updates into failures.

Prefer invariant tests:

```python
assert "gemini" in _PROVIDER_MODELS
assert len(_PROVIDER_MODELS["gemini"]) >= 1
assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]
assert not (set(moonshot_models) & coding_plan_only_models)
for model in _PROVIDER_MODELS["huggingface"]:
    assert model.lower() in DEFAULT_CONTEXT_LENGTHS_LOWER
```

The question is whether the relationship or contract still holds, not whether
the current catalog happens to contain a particular entry. Reviewers should
request invariants when a change-detector test is proposed.

## Safe agent workflow

Before changing a trust boundary, file path, tool policy, or process lifecycle:

1. Identify the owning profile, backend, and filesystem root.
2. Reproduce the current behaviour with a focused test or a real temporary
   `HERMES_HOME`.
3. Make the smallest change that preserves prompt caching, role alternation,
   approval boundaries, and cleanup semantics.
4. Run the focused test through the supported runner, then run lint or type
   checks for changed files.
5. Record skipped checks and residual risk in `_docs/` without committing that
   local implementation log.

Do not expose credentials in test fixtures, logs, screenshots, or generated
reports. Do not weaken a guard merely to make a test or local workflow pass.

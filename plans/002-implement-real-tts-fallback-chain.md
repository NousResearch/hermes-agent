# Plan 002: Implement a real TTS fallback chain

> **RE-SCOPED by Plan 005**: The fallback chain is now implemented in
> `agent-meow` (`agent_meow/hermes_voice_backends.py`) behind the voice
> gateway, not in `tools/tts_tool.py`. This plan is kept for reference but
> should not be executed as written. The live implementation is in the
> `agent-meow` repo under Plan 005 Tasks 1A/1B.

> **Executor instructions**: Follow this plan step by step. Run every verification command and confirm the expected result before moving to the next step. If anything in the "STOP conditions" section occurs, stop and report — do not improvise. When done, update the status row for this plan in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat 7b162e2da..HEAD -- tools/tts_tool.py tests/tools/test_tts_command_providers.py tests/tools/test_tts_piper.py data/config.yaml`
> If any in-scope file changed since this plan was written, compare the "Current state" excerpts against the live code before proceeding; on a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: plans/001-make-voice-runtime-source-of-truth-explicit.md
- **Category**: bug
- **Planned at**: commit `7b162e2da`, 2026-08-07

## Why this matters

The config currently advertises three TTS providers, but the runtime only picks one provider and has a single hardcoded fallback from Edge to NeuTTS. That means `edge -> piper-zh -> qwen3-tts` is not a real behavior today, even though the current config suggests a richer setup. Users will get unexpected failures instead of graceful degradation when Edge is unavailable or the Qwen bridge is down.

## Current state

- Relevant files:
  - `tools/tts_tool.py` — provider dispatch and fallback behavior.
  - `data/config.yaml` — current triple-provider config.
  - `tests/tools/test_tts_command_providers.py` — command-provider precedence tests.
  - `tests/tools/test_tts_piper.py` — Piper resolution/dispatch tests.
- Current excerpts:
  - `tools/tts_tool.py:3285-3327` — `elif provider == "piper"` branch is explicit, but the default branch still says `# Default: Edge TTS (free), with NeuTTS as local fallback` and only falls through to NeuTTS.
  - `data/config.yaml:76-91` — `provider: edge`, plus `providers.piper-zh` and `providers.qwen3-tts`, but no fallback list.
  - `tests/tools/test_tts_command_providers.py` already exercises built-in-vs-command precedence, which is the right test style to copy for fallback selection tests.
- Repo conventions to follow:
  - Python tests use `scripts/run_tests.sh`, not direct `pytest` (see `AGENTS.md`).
  - TTS tests prefer monkeypatching/stubbing provider internals rather than real network calls; see `tests/tools/test_tts_piper.py`.

## Commands you will need

| Purpose            | Command                                                                                                                                                                                              | Expected on success |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| Syntax check       | `python -m py_compile c:\Users\1\github-pr\hermes-agent\tools\tts_tool.py`                                                                                                                           | exit 0              |
| Targeted tests     | `c:\Users\1\github-pr\hermes-agent\scripts\run_tests.sh c:\Users\1\github-pr\hermes-agent\tests\tools\test_tts_command_providers.py c:\Users\1\github-pr\hermes-agent\tests\tools\test_tts_piper.py` | all pass            |
| New fallback tests | `c:\Users\1\github-pr\hermes-agent\scripts\run_tests.sh c:\Users\1\github-pr\hermes-agent\tests\tools\test_tts_fallback_chain.py`                                                                    | all pass            |

## Scope

**In scope**:

- `tools/tts_tool.py`
- `tests/tools/test_tts_command_providers.py`
- `tests/tools/test_tts_piper.py`
- `tests/tools/test_tts_fallback_chain.py` (new)
- `data/config.yaml` only if the final config schema needs an added fallback list example

**Out of scope**:

- Docker compose files
- `scripts/qwen3-tts-server.py`
- Provider package installation behavior
- README/INSTALL docs (handled in plan 004)

## Git workflow

- Branch: `advisor/002-tts-fallback-chain`
- Commit per logical unit; conventional commit style is fine
- Do not push or open a PR unless explicitly instructed

## Steps

### Step 1: Introduce an explicit fallback-chain config shape

Add a config reader that supports a list like `tts.fallback_providers: ["piper-zh", "qwen3-tts"]` while preserving today’s single-provider behavior when the list is absent. Keep the configured `tts.provider` as the first choice; the fallback list should contain only later attempts.

Do not make `providers` entries themselves imply ordering; ordering must be explicit.

**Verify**: `python -m py_compile c:\Users\1\github-pr\hermes-agent\tools\tts_tool.py` → exit 0

### Step 2: Refactor dispatch to attempt providers in order

Wrap the provider dispatch so `text_to_speech_tool()` can try the selected provider, then each configured fallback provider, logging which provider failed and why. Preserve current behavior for explicit per-call `provider=` overrides unless the chosen override is also supposed to honor the configured fallback chain.

Make the failure policy explicit:

- retry on import errors / provider unavailable / bridge/network failure
- do **not** silently skip on programmer errors (e.g. malformed config templates) without logging

**Verify**: existing targeted tests still pass via `scripts/run_tests.sh ...test_tts_command_providers.py ...test_tts_piper.py`

### Step 3: Add focused fallback-chain tests

Create `tests/tools/test_tts_fallback_chain.py` modeled after the existing TTS tests. Cover at least:

- Edge unavailable → falls back to `piper-zh`
- Edge unavailable and Piper unavailable → falls back to `qwen3-tts` command provider
- Explicit provider override still respects the intended semantics (documented in the test)
- A hard failure in the final fallback returns a clear error naming all attempted providers

**Verify**: `scripts/run_tests.sh ...test_tts_fallback_chain.py` → all pass

### Step 4: Update the sample config only if the schema changes

If you added `tts.fallback_providers`, update the sample in `data/config.yaml` so the triple-provider setup actually describes the intended chain. Keep this limited to schema/example alignment; broader docs belong to plan 004.

**Verify**: targeted tests still pass and no unrelated config behavior regressed

## Test plan

- Reuse the monkeypatch-heavy style from `tests/tools/test_tts_piper.py` and `tests/tools/test_tts_command_providers.py`.
- New tests should never hit real network providers.
- Cover one success path and at least two degradation paths.

## Done criteria

- [ ] Runtime can attempt `edge -> piper-zh -> qwen3-tts` in a configured order
- [ ] Existing TTS command-provider and Piper tests pass
- [ ] New fallback-chain tests exist and pass
- [ ] `tools/tts_tool.py` compiles cleanly
- [ ] No files outside scope modified

## STOP conditions

- The user actually wants provider selection to remain manual and only wanted docs clarified
- Fallback semantics for explicit `provider=` overrides are ambiguous after reading the current call sites
- Implementing the chain would require changing unrelated messaging/audio delivery code

## Maintenance notes

- Reviewer should scrutinize whether the fallback chain masks real configuration errors too aggressively.
- Keep provider-attempt logging explicit; hidden fallbacks make production debugging much harder.

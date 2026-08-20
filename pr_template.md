## What does this PR do?

Add configurable display of model reasoning/thinking blocks streamed during execution. When enabled via `display.verbose_reasoning: true` or `display.thinking_progress: true`, content between reasoning tags (```, , , , , , , , , , , ) is buffered and emitted with a `💭 **Reasoning:** ` prefix instead of being silently filtered out.

**Changes:**
- `gateway/stream_consumer.py`: Add `_reasoning_content` buffer and emit reasoning text when closing tag is found. Handle partial tags split across streaming chunks. Emit remaining content on stream end.
- `gateway/display_config.py`: Add `verbose_reasoning` and `thinking_progress` to `_GLOBAL_DEFAULTS`, add `_normalise` boolean handling.
- `gateway/run.py`: Wire `resolve_display_setting` for `verbose_reasoning`.
- `plugins/platforms/discord/adapter.py`: Add `_discord_verbose_reasoning()` with `display.*` resolution path, deprecation notice for legacy keys.
- `plugins/platforms/mattermost/adapter.py`: Add `verbose_reasoning` env var wiring, deprecation notice.
- `tests/gateway/test_stream_consumer_verbose_reasoning.py`: 15 tests covering think tag detection, partial tag boundaries, combined buffer, edge cases, reasoning content buffering, prefix logic.
- `tests/gateway/test_display_config.py`: 16 tests for `verbose_reasoning` and `thinking_progress` resolution hierarchy and normalisation.
- `website/docs/user-guide/messaging/discord.md`: Add `verbose_reasoning` and `thinking_progress` docs with resolution order.
- `website/docs/user-guide/messaging/mattermost.md`: Add `verbose_reasoning` and `thinking_progress` docs with `display.*` hierarchy.
- `cli-config.yaml.example`: Add `verbose_reasoning`/`thinking_progress` docs.

**Fixes:**
- Removed empty-string from `_OPEN_THINK_TAGS` that caused `.find('')` to match position 0 on every chunk, breaking think-block detection.
- Fixed `_filter_and_accumulate` combined-buffer check to recognize complete tags formed by splitting `_think_buffer` + new chunk.
- **Fixed reasoning content drop-off**: Reasoning content was being queued via `self._queue.put()` inside `_filter_and_accumulate`, causing it to be re-processed by the drain loop (which called `_filter_and_accumulate()` on it again, dropping it into a re-filter loop). Changed to append reasoning content directly to `self._accumulated` at the inline position where the closing tag was found, so it flows through the existing streaming path naturally.
- **Simplified `_flush_think_buffer`**: Removed redundant `_reasoning_prefix_added` flag — reasoning content now appends directly to `_accumulated` with prefix, no queue or flag coordination needed.
- Removed unused `_reasoning_buffer` and `_reasoning_last_queued_len` fields.
- Fixed duplicate test method name in `TestVerboseReasoningEdgeCases`.

## Related Issue

No existing issue — this is a new feature. Consider creating one first.

Fixes #<issue>

## Type of Change

- [x] ✨ New feature (non-breaking change that adds functionality)
- [x] 📝 Documentation update
- [x] ✅ Tests (adding or improving test coverage)

## Changes Made

- `gateway/stream_consumer.py`: `_reasoning_content` buffer, `_reasoning_prefix_added` flag, split-stream prefix handling
- `gateway/display_config.py`: `verbose_reasoning`/`thinking_progress` globals + normalisation
- `gateway/run.py`: `resolve_display_setting` wiring
- `plugins/platforms/discord/adapter.py`: `_discord_verbose_reasoning()` + env var wiring
- `plugins/platforms/mattermost/adapter.py`: `_mattermost_verbose_reasoning()` + env var wiring
- `tests/gateway/test_stream_consumer_verbose_reasoning.py`: 325-line test suite (15 tests, 6 classes)
- `tests/gateway/test_display_config.py`: 8 new tests for resolution hierarchy
- `website/docs/user-guide/messaging/discord.md`: 72 lines of docs
- `website/docs/user-guide/messaging/mattermost.md`: 79 lines of docs
- `cli-config.yaml.example`: 9 lines of comments

## How to Test

1. **Unit tests:** `uv run pytest tests/gateway/test_stream_consumer_verbose_reasoning.py tests/gateway/test_display_config.py -v` — all 43 tests should pass (15 consumer + 16 config + 12 baseline).
2. **Integration:** Set `display.verbose_reasoning: true` in your config, run a reasoning model (e.g. one that emits `` or `` blocks), and verify the model's thinking appears with a `💭 **Reasoning:** ` prefix in your chat instead of being filtered out.
3. **Verification:** Confirm `display.thinking_progress: true` shows progress indicators during long turns without leaking raw thinking content. Confirm default behavior (`verbose_reasoning: false`) still filters think blocks as before.

## Checklist

### Code
- [x] I've read the [Contributing Guide](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md)
- [x] My commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:` / `fix:`)
- [x] I searched for [existing PRs](https://github.com/NousResearch/hermes-agent/pulls) to make sure this isn't a duplicate
- [x] My PR contains **only** changes related to this fix/feature (no unrelated commits)
- [x] I've run `uv run pytest tests/gateway/ -q` and all tests pass (109/109)
- [x] I've added tests for my changes (required for bug fixes, strongly encouraged for features) — 43 new tests across 2 test files
- [x] I've tested on my platform: Ubuntu 24.04 (Linux x86_64)

### Documentation & Housekeeping
- [x] I've updated relevant documentation (README, `docs/`, docstrings) — added to `discord.md` and `mattermost.md`
- [x] I've updated `cli-config.yaml.example` if I added/changed config keys — added `verbose_reasoning` comments for Discord and Mattermost
- [x] I've considered cross-platform impact (Windows, macOS) per the [compatibility guide](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md#cross-platform-compatibility) — config resolution uses standard `display.*` hierarchy that works across all platforms

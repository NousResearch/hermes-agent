## What does this PR do?
The hygiene turn-hold deferral notice ("ℹ️ Context compression deferred — summary still streaming. Continuing without compression this turn.") reached every chat unconditionally. It is routine progress, not a failure: the turn proceeds immediately and the watermark-fenced summary is adopted at the next safe boundary (`#97963`), nothing is lost. On a chat with `hygiene_hard_message_limit` lowered for latency it shows up every few dozen messages and confuses non-technical users.

This PR gates the notice behind the existing `compression.progress_notices` opt-in (#52995), like the other routine compression statuses. Timeout and abort **warnings are unchanged** and still always reach the user.

## Related Issue
Follows up on #52995 (progress notices opt-in) and #97963 (turn-hold adoption).

## Type of Change
- [x] 🐛 Bug fix (non-breaking change that fixes an issue)
- [x] ✅ Tests (adding or improving test coverage)

## Changes Made
- `gateway/run_turn.py`: send `gateway.compress.turnhold_deferred` only when `_gateway_compression_progress_notices_enabled()`.
- Tests: the two existing turn-hold tests that assert the notice enable the gate explicitly; new `test_session_hygiene_turn_hold_notice_is_silent_when_progress_notices_off` asserts silence at the default.

## How to Test
```
pytest -q tests/gateway/test_session_hygiene.py tests/gateway/test_session_hygiene_turnhold_adoption.py tests/gateway/test_compression_progress_notices.py
```

## Checklist
- [x] Searched open and merged PRs/issues for duplicates (none for `turnhold_deferred`)
- [x] Tests added/updated and passing locally
- [x] Default behaviour becomes quieter; opt-in restores the old behaviour

🤖 Generated with [Claude Code](https://claude.com/claude-code)

# Upstream PR: Return Latest Tool Results When The Model Goes Empty

Target: `NousResearch/hermes-agent`

Prepared branch: `codex/upstream-empty-tool-recovery`

## Summary

Extract empty-tool-result recovery into `agent/tool_recovery.py` and use it when
the model exhausts empty-response retries after a tool-call turn. Instead of
returning a terminal `(empty)` response, Hermes returns a bounded summary of the
latest tool outputs.

## Why

Some OpenAI-compatible providers can return repeated empty final messages after
successfully executing tools. The user-visible `(empty)` response hides the real
tool evidence and makes a successful tool turn look like a failure. Returning the
latest tool output is a safer terminal fallback.

## Behavior

- Only activates after normal empty-response retries/fallbacks are exhausted.
- Only activates when recent tool results exist.
- Includes at most five latest tool results.
- Preserves existing empty-response behavior when there are no tool results.

## Verification

- `tests/agent/test_tool_recovery.py`
- `tests/run_agent/test_run_agent.py::TestRunConversation::test_tool_calls_then_empty_returns_latest_tool_results`

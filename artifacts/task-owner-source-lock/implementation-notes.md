# implementation-notes — owner/source-lock recovery guard

## Scope
- Prevent generic recovery/context-compaction turns from resuming work that belongs to a different chat/thread/session/agent owner.
- Keep the change repo-local and prompt/guard focused; do not touch unrelated recovery logic or external artifacts.

## Evidence / current understanding
- Failure class: a recovery request in one Telegram thread treated a compacted `## Active Task` from another Telegram topic/agent lane as live work.
- Existing `SUMMARY_PREFIX` already says latest user message wins over stale active tasks, but it does not explicitly require source/owner comparison when the latest message is a generic recovery request.
- `build_session_context_prompt()` already exposes the current platform/chat/thread context, but does not tell the model to reject cross-source tasks.

## Planned increment
1. Add RED tests that pin two invariants:
   - compaction summaries must include a source/owner lock and explicit-handoff requirement;
   - gateway session context must inject the current source lock for threaded sessions.
2. Apply minimal prompt-level fix in `agent/context_compressor.py` and `gateway/session.py`.
3. Run targeted pytest/ruff gates.

## Implementation log
- RED gate run before the fix:
  - `uv run pytest -q tests/agent/test_resume_stale_active_task.py tests/gateway/test_session_source_lock_prompt.py`
  - Result: 2 failed, 6 passed. Missing source/owner lock in both compaction prefix and session context prompt.
- Fix applied:
  - `SUMMARY_PREFIX` now requires source/owner comparison against `## Current Session Context` and says generic recovery/continue/resume is not an explicit handoff.
  - Added the previous prefix to `_HISTORICAL_SUMMARY_PREFIXES` so old persisted handoffs are re-normalized to the new source-lock wording.
  - `build_session_context_prompt()` now injects a Source/owner lock with current platform/chat/thread/session scope.
- First GREEN gate:
  - same pytest command
  - Result: 8 passed in 0.13s.
- Added a regression for old persisted handoffs so pre-source-lock summaries are upgraded during re-compaction.
- Broader targeted GREEN gate after restoring unrelated `uv.lock` drift:
  - `uv run --frozen pytest -q tests/agent/test_resume_stale_active_task.py tests/gateway/test_session_source_lock_prompt.py tests/gateway/test_session.py tests/gateway/test_pii_redaction.py tests/gateway/test_telegram_topic_mode.py tests/gateway/test_compression_session_id_persistence.py tests/run_agent/test_compression_persistence.py`
  - Result: 150 passed in 3.50s.
- Lint gate:
  - `uv run --frozen ruff check agent/context_compressor.py gateway/session.py tests/agent/test_resume_stale_active_task.py tests/gateway/test_session_source_lock_prompt.py`
  - Result: All checks passed.

## Sensitive data handling
- Use synthetic chat/thread/session IDs only in tests and notes.
- Do not write real Telegram IDs, message IDs, tokens, URLs, or credentials into artifacts.

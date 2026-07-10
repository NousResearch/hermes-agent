# STAGE-2 gate reconciliation fixes — 2026-07-10 (Apollo)

31 gate failures after the codex resolution pass. Each adjudicated code-vs-test on merit.
All were genuine merge-reconciliation issues, NOT resolver sloppiness. Fixes:

## Real CODE regressions the merge introduced (fixed the code)
1. **agent/agent_runtime_helpers.py — `NameError: known_tool_ids`** (broke ALL 16
   test_message_sequence_repair tests). The merge reconciled the fork's set→Counter migration
   (`known_tool_ids` → `known_tool_budget`) but left a stray `known_tool_ids.discard(tc_id)` from the
   old set impl. The Counter's `-= 1` already consumes the slot; the stray line was undefined-name +
   redundant. Removed it. → 16 tests green.
2. **cron/scheduler.py — one-shot dispatch claim dropped on the tick path.** Upstream added
   `claim_dispatch` (at-most-once for finite one-shots) but only in `run_one_job`; the fork's `tick()`
   dispatches through `_process_one_job` (the shared body), which never got the claim → a tick that
   dies mid-run could re-fire forever, and the test saw run-before-claim / refused-claim-still-ran.
   Added the claim guard to the top of `_process_one_job` (the body tick + run_job_now share).
   `run_one_job` keeps its own claim for the external-provider path; the two bodies don't call each
   other, so no double-claim. → 2 tests green.
3. **tui_gateway/server.py — untrusted client source no longer sanitized.** The merge replaced the
   fork's `platform=_sanitize_client_source(source)` with upstream's
   `_resolve_agent_platform(platform_override or source)`, which trusts `source` verbatim → a hostile
   label became the persisted platform. Reconciled: trusted internal `platform_override` keeps
   `_resolve_agent_platform`; untrusted `source` goes through `_sanitize_client_source` (slug guard →
   "tui"). → 1 test green.
4. **gateway/slash_commands.py — /compress chat-only `msgs` filter clobbered.** The merge took
   upstream's `msgs = [role in {user,assistant,tool}]` (full transcript) over the fork's chat-only
   projection `[{role,content} for role in {user,assistant} and content]`. That broke the fork's
   both-axes compress-feedback accounting (double-counted tool rows → "7→7" instead of "7→4", and
   double-counted them again in non_chat_rows). Restored the fork's chat-only projection + fixed the
   now-stale "pass the FULL transcript" comment. → 2 tests green (the 3rd, below, was a test fix).

## Stale TESTS encoding a pre-merge contract (updated the test to the merged contract)
5. **tests/agent/test_error_classifier.py** — change-detector enum test missing upstream's new
   `FailoverReason.ssl_cert_verification` member. Added it to the expected set (verified it's the
   ONLY delta; the enum superset is upstream's real addition).
6. **tests/gateway/test_compress_command.py** — (a) 3 count assertions fixed by the CODE fix #4;
   (b) the granular-breakdown test patched the fork's old `_resolve_gateway_model`, but the merged
   `_handle_compress_command` resolves model via the reconciled `_resolve_session_agent_runtime`
   (a real shared helper on both fork+upstream). Stubbed `_make_runner` with empty
   `_session_model_overrides` + `_rehydrate_session_model_override` + `_session_key_for_source` so the
   merged runtime-resolution path resolves cleanly through the patched `_resolve_gateway_model`.
7. **tests/test_tui_gateway_server.py** — all signature/contract drift from real upstream features the
   merge correctly adopted:
   - `_notify_session_boundary` gained a 3rd `platform` arg (upstream: CLI-parity lifecycle hooks) →
     updated 2 stub lambdas to `(event, session_id, platform=None)`.
   - `_notification_event_belongs_elsewhere` gained a leading `sid` arg (upstream: desktop
     origin_ui_session_id routing) → updated 5 call sites to pass a sid.
   - `list_sessions_rich` gained `compact_rows` kwarg (upstream) → 3 fake `_DB` methods now accept it.
   - `_SlashWorker(key, model, profile_home=...)` gained `profile_home` → made `_FakeWorker.__init__`
     variadic (5 fakes) so the create/close-race `_attach_worker` path installs the worker.
   - config-sync switch-model call gained `persist_override=False` kwarg (upstream) → added to the
     expected-call dict.

Net: 31 → 0. No fork feature dropped; no test weakened to hide a regression. Every CODE fix preserves
a fork behavior the merge would otherwise have silently lost; every TEST fix tracks a real upstream
feature the merge correctly adopted.

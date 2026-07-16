# Post-merge full-suite triage brief (2026-07-15 parity merge)

You are triaging test failures in the worktree `/Users/alexgierczyk/.hermes/worktrees/parity-2026-07-15`
(branch `sync/upstream-2026-07-15`, an UNCOMMITTED staged parity merge of upstream
`2ea39daeb` into `fork/main`). Full-suite log: `/tmp/parity-fullsuite-real-0715.log`.
102 FAILED tests across ~32 files + 22 files killed at the 300s per-file cap.

**DO NOT commit. DO NOT push. DO NOT run git merge/abort. STOP when done.**

## The method (for EVERY red, no exceptions)
1. Baseline-classify: run the same test on the fork/main baseline worktree
   `/Users/alexgierczyk/.hermes/worktrees/parity-baseline` with:
   `HOME=/Users/alexgierczyk /Users/alexgierczyk/.hermes/hermes-agent/venv/bin/python -m pytest <test> -q -o addopts="" -p no:randomly`
   (Do NOT trust `hermes_parity bisect` — it mis-classified a known merge regression as INHERITED earlier today. Always run the baseline directly.)
2. PASSES on baseline + FAILS on merge = MERGE REGRESSION → fix the CODE, preserving fork behavior.
3. FAILS on baseline too = INHERITED → leave it, list it in the report.
4. Test asserts a contract the merge legitimately changed (upstream feature adopted) = STALE TEST →
   update the TEST to the merged contract with a dated comment.
5. Run the test in the merge worktree with the same venv command to prove GREEN after each fix.

## Pre-adjudicated clusters (orchestrator decisions — follow them)
- **`ultra` reasoning-effort cluster** (~15 tests: test_reasoning_command gateway+cli,
  test_config_validation rejects_ultra, test_batch_runner_checkpoint rejects_ultra,
  test_primary_runtime_restore TestFallbackReasoningEffort×5, test_cronjob_schema
  reasoning_effort contract, test_discord_slash_commands descriptions_mention_max,
  test_hermes_constants web_reasoning_contract, test_auto_continue t6 enum docs):
  Upstream `7550c594c` (#62650) ADDED `max` and `ultra` as valid levels. The fork's #359
  tests predate it and assert ultra is rejected. VERDICT: `ultra` is now VALID — the merged
  contract accepts it. Update the fork tests to the merged contract (accept ultra; every
  surface that enumerates levels includes it symmetrically: python constants, web
  reasoning-effort.ts, cron schema, batch runner, fallback config validation, slash-command
  descriptions). Where a fork surface deliberately caps at `max` (check #359's intent via
  `git log -p 4739500b8`), keep the CAP only if the fork code still enforces it — code and
  test must agree; prefer upstream's full set unless fork code actively rejects ultra.
- **300s-timeout files (22, e.g. test_web_server, test_matrix, test_slack, test_scheduler,
  test_feishu, test_voice_command, test_kanban_*, test_profiles, test_auxiliary_client,
  test_computer_use)**: these are giant suites that hit the local per-file cap. Spot-check
  THREE of them on the baseline worktree the same way — if baseline also times out, classify
  ALL as ENVIRONMENT (local cap, CI will decide) and move on. Do NOT burn time re-running
  every one. If a spot-check PASSES quickly on baseline but times out on merge, treat that
  file as a real regression (likely a hang) and investigate.

## Unadjudicated clusters — classify on merit via the baseline method
- test_dashboard_auth_401_reauth TestTransparentRefreshOnAccessTokenEviction (5) +
  test_dashboard_auth_middleware full_login_round_trip
- test_completion_delivery (3) — the merge adopted upstream "completion-delivery
  de-duplication" (see docs/sync/review/resolution-decisions-2026-07-15.md); fork
  restart-durability behaviors must survive
- test_channel_directory TestBuildFromSessions (4)
- pending-CLI-message / clean-staged-input cluster: test_turn_context (2),
  test_cli_interrupt_ack_race, test_cli_shutdown_memory_messages,
  test_persist_platform_message_id, test_restart_backfill (discord)
- test_context_compressor TestPreflightDeferral (5)
- singles: test_client_e2e lifecycle, test_api_server_active_work_drain concurrency,
  test_mirror case_insensitive, test_session gateway_peer_fields, test_webhook_session_close
  awaits_async_session_db (AsyncSessionDB literal-await contract — likely a missed await in
  merged gateway code = regression), test_413_compression cannot_compress_further,
  test_codex_app_server user_message_not_duplicated, test_run_agent_conversation
  request_scoped_api_hooks, test_run_agent_streaming AnthropicInterruptHandler (2),
  test_state_db_malformed_repair strategy_b, test_approved_command_clean_slate,
  test_file_tools_live, test_mcp_stdio_init_timeout, test_mcp_tool_issue_948,
  test_openviking_provider (timeout file — spot-check rule applies)

## Constraints
- Preserve fork features: relay-lane headers, ContextVar approval gate, launchd/systemd exit
  contract, hygiene announce, messaging/moa toolsets, telegram queue preservation,
  include_timestamp opt-in, undo/redo, AsyncSessionDB literal-await.
- Never fix a red by weakening a security/durability assertion; fix the code.
- After all fixes: re-run ONLY the previously-red non-timeout files as one batch with the
  venv command and paste the summary line into the report.
- Write the full classification table (test → baseline verdict → class → action taken) to
  `docs/sync/review/fullsuite-triage-2026-07-15.md`.
- DO NOT commit. DO NOT push. STOP.

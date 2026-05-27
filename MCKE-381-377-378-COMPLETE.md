# MCKE-381 / MCKE-377 / MCKE-378 Complete

## Branch / Base

- Branch: `codex/mcke-381-377-378-remaining-hermes-fixes`
- HEAD: `26fa1ba6ee3bb625ea6397362a7828483845e252`
- Base `origin/main`: `2d5dcfabc312d43f87a4f0f44c45f62cf24a09b2`
- Divergence: `git rev-list --left-right --count HEAD...origin/main` => `2 0`

## What Was Replayed / Fixed

1. Replayed the safe portions of local reliability commit `1198d3e7f0d295e4be68161f9b3b4d47b8161ae3`:
   - Telegram/interactive wall-clock guard now uses a clearer state-preserving message.
   - Local parallel test runner falls back when `pytest-timeout` is not installed, preserving the outer per-file watchdog.
   - Added focused tests for both behaviors.
   - Stale MCKE-379/384 report files from that old commit were not replayed.

2. Replayed local context/provider reliability commit `6d7c525a1ec6203e32827457c4d11d095854a536` cleanly:
   - Context compression fails closed by default on summary failure instead of dropping middle context behind a static marker.
   - Added `compression.allow_fallback_marker` as the explicit legacy escape hatch.
   - Added compression preflight helper for configuration/context-fit diagnostics.
   - Classified `usage_limit_reached` and monthly/org usage-limit text as terminal provider quota/billing exhaustion after fallback recovery is exhausted.
   - Gateway Telegram sanitization now distinguishes provider quota exhaustion from transient rate limiting.
   - Added focused tests for compression, classifier, auxiliary payment detection, gateway provider diagnostics, and preflight.

3. Not replayed:
   - Claude CLI inactivity/fail-fast commit `33c0a7c72933556f32ac017a852e90c84089989b`: current `origin/main` no longer contains `agent/claude_code_cli_client.py`, so applying the old file-level patch would be unsafe.
   - Telegram approval-card retention/fallback commits `500ded6556bd4ad40d5d7345749abf02951f103d` / `38827606a97be7a57b96723802cad92b648c94db`: the retention commit conflicts in the current Telegram adapter. It was not forced over upstream changes.
   - Approval-brief enforcement commit `5170b233f94671ad8b25d77e1f3d0bf1eb5c5806`: broad approval/terminal changes were not replayed because the narrower Telegram retention patch already conflicted and this would risk overwriting newer approval hardening.

## Tests Run

- `python3 -m pytest ...` with system Python:
  - Result: failed before collection because system Python has no `pytest` module.

- Repo venv targeted pytest:
  - Command: `.venv/bin/python -m pytest tests/agent/test_conversation_loop_interactive_guard.py tests/test_run_tests_parallel.py tests/agent/test_context_compressor.py tests/run_agent/test_compression_feasibility.py tests/agent/test_error_classifier.py::TestClassifyApiError tests/agent/test_auxiliary_client.py::TestIsPaymentError tests/gateway/test_provider_failure_diagnostics.py -q -o addopts=`
  - Result: `206 passed in 7.30s`

- Py compile:
  - Command: `.venv/bin/python -m py_compile agent/agent_init.py agent/auxiliary_client.py agent/context_compressor.py agent/conversation_compression.py agent/conversation_loop.py agent/error_classifier.py gateway/run.py hermes_cli/config.py run_agent.py scripts/run_tests_parallel.py tests/agent/test_auxiliary_client.py tests/agent/test_context_compressor.py tests/agent/test_conversation_loop_interactive_guard.py tests/agent/test_error_classifier.py tests/gateway/test_provider_failure_diagnostics.py tests/run_agent/test_compression_feasibility.py tests/test_run_tests_parallel.py`
  - Result: PASS

## Files Changed

- `agent/agent_init.py`
- `agent/auxiliary_client.py`
- `agent/context_compressor.py`
- `agent/conversation_compression.py`
- `agent/conversation_loop.py`
- `agent/error_classifier.py`
- `gateway/run.py`
- `hermes_cli/config.py`
- `run_agent.py`
- `scripts/run_tests_parallel.py`
- `tests/agent/test_auxiliary_client.py`
- `tests/agent/test_context_compressor.py`
- `tests/agent/test_conversation_loop_interactive_guard.py`
- `tests/agent/test_error_classifier.py`
- `tests/gateway/test_provider_failure_diagnostics.py`
- `tests/run_agent/test_compression_feasibility.py`
- `tests/test_run_tests_parallel.py`
- `MCKE-381-377-378-COMPLETE.md`

## Remaining Protected Gates

- No remote push was performed.
- No external services were mutated by this recovery pass.
- No secrets/env/provider billing/OAuth edits were made.
- Restorely, STRStack, and DeadSaaS product repos were not mutated.
- Telegram approval-card retention/fallback remains a replay gate requiring a fresh adapter-aware patch or explicit approval to redesign against current upstream.

## Restorely Handoff Recommendation

Read-only Hermes state/audit evidence shows:

- `restorely-r2-launch-enablement-chain` is `completed_audited`; MCKE-166 is verified with R2 bucket/env/smoke-test evidence and no non-R2 protected gates crossed.
- `restorely-safe-launch-next-chain` is completed with product readiness `PARTIAL`; its next action was protected approval before R2, now superseded by the completed R2 chain.
- Current active Restorely follow-on is `restorely-apple-quality-qa-gauntlet-chain`. State is `paused_verified_step_needs_next_dispatch`; completed tickets are MCKE-354 through MCKE-358, pending tickets are MCKE-359 through MCKE-361.
- The exact next safe Restorely dispatch gate is MCKE-359 under the existing Apple-quality QA scope, after manual review/reconciliation of the two recorded stale-resume blocks for MCKE-357 and MCKE-358. Do not resume blindly from a stale PID; resume only from verified chain state/evidence.
- Still protected: Linear OAuth Public review (MCKE-142), Stripe LIVE/first paying customer (MCKE-108), indexing/public launch/SEO visibility changes, env/secrets/provider changes, external/customer sends, and customer-data mutation/deletion.

## Provider / Quota Recommendation

- Treat Claude `usage_limit_reached`, monthly usage-limit, org usage-limit, and equivalent subscription quota text as terminal for the current turn once credential rotation/fallback recovery is exhausted.
- Do not edit provider billing, OAuth, or secrets from Hermes reliability recovery.
- Operational policy: switch to a non-exhausted provider/model or wait for quota reset; gateway/user-facing replies should say provider quota is exhausted rather than implying a transient rate limit.
- Evidence in this pass: local history contained Claude quota/fail-fast work, but the old Claude CLI client file is absent from current `origin/main`; the safe applied fix is provider/error-classifier and gateway diagnostic handling.

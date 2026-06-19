# hermes-lcm Adoption Smoke — Isolated Profile

**Generated:** 2026-06-16 10:58 UTC
**Probe:** `scripts/probe_hermes_lcm_isolated.py --profile-dir /private/var/folders/wv/l241lswn1l5fks6jbnn_twrw0000gp/T/tmp.MwcjEINCxe --out /Users/alexgierczyk/.hermes/projects/hermes-agent-fork/docs/reports/hermes-lcm-adoption-smoke.md`
**Verdict:** **GO (isolated smoke clean)** — 7/7 checks passed

## Plugin under test

- **vendored_path:** `/Users/alexgierczyk/.hermes/projects/hermes-agent-fork/plugins/context_engine/lcm`
- **plugin_name:** `hermes-lcm`
- **plugin_version:** `0.16.2`
- **provenance:** `Source: github.com/stephenschoettler/hermes-lcm Source commit: 03b74f84440be99164ce3e2cd929917bc9550bfe Source branch/date: main, 2026-06-13 Source version: v0.16.2 (plugin.yaml version 0.16.2) Source staging path: ~/.hermes/worktrees/prd2v2-native-slimmer/staging/lcm-profile/plugins/hermes-lcm/ Adopted surface: Kyzcreig/hermes-agent private fork, plugins/context_engine/lcm/ Adopted package/module name: lcm (loader-safe directory for load_context_engine("lcm")) Upstream package/plugin identity preserved: hermes-lcm License: cleared for private fork use per PRD-6 LCM Context Engine Activation l`

## Isolation guarantees

- Engine loaded only from the fork-vendored plugin at `/Users/alexgierczyk/.hermes/projects/hermes-agent-fork/plugins/context_engine/lcm`.
- Profile/storage root was `/private/var/folders/wv/l241lswn1l5fks6jbnn_twrw0000gp/T/tmp.MwcjEINCxe` and passed the live-path refusal guard.
- No install script was run; no writes are made to live `~/.hermes/plugins` or profile directories.
- Each check uses a throwaway SQLite DB under the supplied temp/staging profile directory.
- Summarization is deterministic/stubbed for offline reproducibility.

## Smoke results

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1 | load+identity | PASS | ContextEngine subclass=True, engine.name='lcm', plugin_yaml=True |
| 2 | normal-chat/tool-ingestion | PASS | lcm_status.session_id='identity-s', lcm_describe.store_message_count=2 |
| 3 | threshold-compaction | PASS | should_compress(threshold)=True/should_compress(1000)=False=True; status=compacted, count=1, active 6<orig 8; DAG-summary-in-active=True |
| 4 | grep/describe/expand-byte-exact-recall | PASS | fact_out_of_active=True; grep total_results=4; describe_ok=True keys=depths,session_id,store_message_count; selected store_id=2; expand.content recovers raw 'DEPLOY-CODE-7F3A' byte-exact=True |
| 5 | bad-id-loud-error | PASS | lcm_expand(bad id) -> {"error": "Message store_id 999999 not found"} |
| 6 | reset-semantics | PASS | compression_count 1->0 after on_session_reset; grep-before=2; lossless store still answers grep after reset (all-scope)=2 |
| 7 | fail-open | PASS | summarizer LLM unavailable -> no crash=True, status=compacted, active_len=6, raw still grep-recoverable=2 |

## Raw check log

```
[PASS] load+identity — ContextEngine subclass=True, engine.name='lcm', plugin_yaml=True
[PASS] normal-chat/tool-ingestion — lcm_status.session_id='identity-s', lcm_describe.store_message_count=2
[PASS] threshold-compaction — should_compress(threshold)=True/should_compress(1000)=False=True; status=compacted, count=1, active 6<orig 8; DAG-summary-in-active=True
[PASS] grep/describe/expand-byte-exact-recall — fact_out_of_active=True; grep total_results=4; describe_ok=True keys=depths,session_id,store_message_count; selected store_id=2; expand.content recovers raw 'DEPLOY-CODE-7F3A' byte-exact=True
[PASS] bad-id-loud-error — lcm_expand(bad id) -> {"error": "Message store_id 999999 not found"}
[PASS] reset-semantics — compression_count 1->0 after on_session_reset; grep-before=2; lossless store still answers grep after reset (all-scope)=2
[PASS] fail-open — summarizer LLM unavailable -> no crash=True, status=compacted, active_len=6, raw still grep-recoverable=2
```

## Notes for the reviewer

- This isolated smoke proves in-process load, compaction, byte-exact recall, reset behavior, and fail-open fallback.
- It does not enable LCM in any live Hermes profile and does not prove a live model chooses retrieval tools unaided.
- Public redistribution remains gated by the upstream licensing posture recorded in `plugins/context_engine/lcm/VENDORED_FROM.txt`.

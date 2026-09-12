# Architecture Correction Set C7 — Optional-Path Scope Containment

Stage: ARCHITECTURE_CORRECTION
Status: ACCEPTED
Audit-Date: 2026-09-11
Affected-Task: T003
Accepted-C6-Source-SHA256: `b1d306fd1f6252d9034d0a84c5ba733f218cb101772ada0842d077bc4f7796f9`
Current-Integrated-Plan-SHA256: `2db48fe6f01e14a7cf36e5361027d6134591aa321ac5102b8b13baabb48ed7c0`
Owner-Decision: `ACCEPTED`
Owner-Review-Reference/Text: `Owner explicitly accepted unchanged C7 whole artifact specs/002-execution-router-public-api/drafts/architecture-correction-c7.md by exact SHA-256 60a2d38d69c4d1f9d1651b07d5b1eec6ac47033776db38626f89e667df8f7f40.`
Reviewed-At: `2026-09-11`
Accepted-Source-SHA256: `60a2d38d69c4d1f9d1651b07d5b1eec6ac47033776db38626f89e667df8f7f40`

## Correction

The 2026-09-11 source audit disproved C6-05's claim that these callbacks already have attempt/request identity fencing. C7 authorizes only a minimal, surface-owned token/guard on the concrete transcript- or delivery-mutating callbacks proven able to produce late output after a routed attempt closes. The guard exists only on the positively activated routed-continuation path. No-router/native turns retain the same callback objects and exact pre-T003 branch, with no new wrapper, catch/remap block, lifecycle metadata, cleanup or callback replacement.

Route-signature reconstruction cleanup is likewise conditional at the routed mutation point. It must not alter shared native `/model`, cache-mismatch, credential-error or existing rebuild semantics.

Continuation symbols are private underscored internal names. They are absent from `__all__`, facades, compatibility exports and public/plugin surfaces.

C7 adds no universal callback inventory, generic fencing subsystem, speculative hardening or ordinary-turn change.

## C6-06 test ceiling and mapping

C6-06 remains exactly 16 tests; no callback-enumeration, per-edge-case or speculative-hardening tests are added. Each retained test maps to accepted C6 behavior:

1. `test_main_turn_continuation_uses_exact_transcript_without_user_replay` — C6-01 exact transcript and no replay.
2. `test_main_turn_continuation_does_not_repeat_completed_tool_side_effect` — C6-01 no repeated side effect.
3. `test_main_turn_continuation_carries_spent_cursor_across_rebuild_and_primary_restore` — C6-02 spent cursor and exhaustion.
4. `test_main_turn_continuation_fresh_identity_and_lifecycle_order` — C6-02 fresh identity and ordering.
5. `test_cli_routed_fallback_continues_same_turn_once` — C6-03 Classic CLI routed continuation.
6. `test_cli_routed_fallback_budget_exhaustion_is_not_reset` — C6-02 Classic CLI budget preservation.
7. `test_cli_no_router_main_turn_and_native_fallback_are_unchanged` — C6-02 no-router/native baseline.
8. `test_tui_routed_fallback_continues_inside_one_prompt_submit` — C6-03 TUI routed continuation.
9. `test_tui_routed_fallback_budget_exhaustion_is_not_a_queued_followup` — C6-02 TUI budget and no queued replay.
10. `test_tui_no_router_main_turn_and_native_fallback_are_unchanged` — C6-02 no-router/native baseline.
11. `test_gateway_routed_fallback_continues_one_inbound_turn` — C6-03 Gateway routed continuation.
12. `test_gateway_routed_fallback_budget_exhaustion_never_queues_or_replays` — C6-02 Gateway budget and no replay.
13. `test_gateway_no_router_main_turn_and_native_fallback_are_unchanged` — C6-02 no-router/native baseline.
14. `test_oneshot_routed_fallback_continues_before_single_close` — C6-03 one-shot routed continuation.
15. `test_oneshot_routed_fallback_budget_exhaustion_does_not_recurse` — C6-02 one-shot budget and no recursion.
16. `test_oneshot_no_router_main_turn_and_native_fallback_are_unchanged` — C6-02 no-router/native baseline.

## Exact supersession and preserved scope

On owner acceptance, C7 supersedes only:

1. C6-05 item 6 phrase `ignored by the existing attempt/request identity fencing`, replacing that false premise with the routed-only surface-owned guard above.
2. C6-04 phrases requiring callback rebinding and existing reconstruction/cleanup, only to require a positive routed activation guard at each mutation point and to preserve exact no-router/native object identity and control flow.

All other C6 and T003 scope remains unchanged. Public contract version remains `1.0`. C7 authorizes no T004, T005, commit, installation, runtime/profile/config change or LIVE action.

## Owner review

- Decision: `ACCEPTED`
- Review reference/text: `Owner explicitly accepted unchanged C7 whole artifact specs/002-execution-router-public-api/drafts/architecture-correction-c7.md by exact SHA-256 60a2d38d69c4d1f9d1651b07d5b1eec6ac47033776db38626f89e667df8f7f40.`
- Reviewed at: `2026-09-11`
- Accepted unchanged DRAFT SHA-256: `60a2d38d69c4d1f9d1651b07d5b1eec6ac47033776db38626f89e667df8f7f40`

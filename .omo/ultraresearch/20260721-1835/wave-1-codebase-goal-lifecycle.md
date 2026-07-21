# Wave 1 — goal lifecycle researcher digest

## Verified findings

- Goal state is a per-session `state_meta` JSON record with only `active`,
  `paused`, `done`, and `cleared` statuses. A blocked/needs-input conclusion
  is currently represented as a `done` judge outcome with reason text; it must
  not be silently reinterpreted as a new runtime state.
- A judge `done` saves `GoalState` first, then appends an unconfirmed receipt.
  Receipt confirmation is same-session/same-workspace only and idempotent.
- Wait resumption uses a CAS-backed, leased `pending -> claimed` transition
  and is intentionally at-least-once. General `save_goal()` is a blind write;
  adding new state-mutating transitions would therefore need explicit
  concurrency design rather than assuming wait semantics apply everywhere.
- Compression migrates a goal by writing the child then clearing the parent;
  it is a two-step operation. Kanban has a separate task lifecycle and must
  not be conflated with a session goal.
- The worker confirmed `list_reusable_outcome_receipts()` has no production
  caller and recommends a shared, read-only outcome surface as the safest
  first extension.

## Suggested focused verification

```text
scripts/run_tests.sh tests/hermes_cli/test_goals.py tests/agent/test_verification_evidence.py tests/gateway/test_goal_verdict_send.py tests/gateway/test_goal_status_notice.py tests/gateway/test_goal_max_turns_config.py tests/tui_gateway/test_goal_command.py tests/tools/test_kanban_tools.py -q
```

## Worker EXPAND markers (verbatim)

- LEAD: `list_reusable_outcome_receipts()`에 production caller가 없음 — WHY: 검증·사용자확인 완료 학습 후보가 자동 주입 없이도 활용되지 않음 — ANGLE: shared CLI/gateway/TUI read-only outcomes surface
- LEAD: 일반 `/goal`의 blocked가 별도 상태가 아니라 `done` 사유로 종결됨 — WHY: 차단 감사·재개 정책을 확장할 때 현재 호환 의미를 깨기 쉬움 — ANGLE: GoalManager terminal transition과 receipt terminal_kind 통합
- LEAD: wait 재개만 CAS이고 일반 `save_goal()`은 blind write — WHY: 다중 frontend의 완료·pause·clear 경합이 마지막 writer 승리일 수 있음 — ANGLE: terminal transition CAS/versioning과 경쟁 회귀 테스트
- LEAD: 세션 migration은 child save 뒤 parent clear의 비원자적 두 단계 — WHY: 중간 crash 시 “정확히 하나의 active goal” 불변식이 깨질 수 있음 — ANGLE: SessionDB multi-key transaction 또는 crash-recovery reconciliation
- LEAD: `blocked`·`cancelled` outcome receipt kind에 production writer가 없음 — WHY: 현재 audit ledger가 judge-done 후보만 남김 — ANGLE: user clear/pause 및 genuine blocker의 audit 요구사항 여부 확인

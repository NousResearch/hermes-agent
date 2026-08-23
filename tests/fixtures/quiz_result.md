# result — compute_class router oracle (quiz)

## 1. Summary

compute_class 구현 전 고정 RED 오라클을 작성했다. production 코드는 수정하지 않았고 tests/** 아래에만 정상 5종 precedence fixture, I01~I12 known-bad fixture, 단일 route_task 공개 계약 테스트, kanban_create reasoning_effort E2E parity 테스트를 추가했다.

현재 기준 결과는 의도대로 RED다. fixture 자체 검증 2건은 GREEN이고, 미구현 production 계약은 21 failed / 2 passed다. 실패는 hermes_cli.compute_routing.route_task 부재 18건과 kanban_create reasoning_effort schema/handler 부재 3건으로 분리된다.

## 2. Files

- tests/fixtures/compute_routing_contract_v1.json
  - 정상 class 5종 specialist > architect > deep > standard > quick precedence
  - 구조적으로 유효한 I01~I12 mutant
  - capability snapshot, role boundary, fallback 자격표/상한, execution observation fixture
- tests/test_compute_routing_contract.py
  - 모든 정상/mutant를 동일 hermes_cli.compute_routing.route_task 키워드 인터페이스로 실행
  - role/assignee/write_scope/workspace/approval/toolsets/master_forbidden 불변
  - class/decision id/policy version 실행 전 명시 저장
  - actual route 없는 verified 금지, quick 검증 필수, unattended architect 차단
- tests/tools/test_kanban_compute_routing_contract.py
  - kanban_create reasoning_effort 닫힌 enum
  - HIGH 입력 → DB/event/show high → spawn --reasoning high parity
  - hihg 입력 거부 및 DB 변화 0

## 3. Commands + actual output

Command:

    G:labuntimehermes-bappenvScriptspython.exe -m pytest --basetemp <workspace>/tests/.pytest_compute_routing tests/test_compute_routing_contract.py::test_fixture_is_closed_complete_and_structurally_valid tests/test_compute_routing_contract.py::test_i12_expected_contracts_are_not_aliases -q

Output:

    ..                                                                       [100%]
    2 passed in 2.48s

Command:

    G:labuntimehermes-bappenvScriptspython.exe -m pytest --basetemp <workspace>/tests/.pytest_compute_routing_result tests/test_compute_routing_contract.py tests/tools/test_kanban_compute_routing_contract.py -q --tb=no

Output:

    .FFFFFFFFFFFFFFFFFF.FFF                                                  [100%]
    =========================== short test summary info ===========================
    FAILED tests/test_compute_routing_contract.py::test_normal_routes_use_precedence_and_preserve_role_boundary[N01-specialist-precedence]
    FAILED tests/test_compute_routing_contract.py::test_normal_routes_use_precedence_and_preserve_role_boundary[N02-architect-precedence]
    FAILED tests/test_compute_routing_contract.py::test_normal_routes_use_precedence_and_preserve_role_boundary[N03-deep-precedence]
    FAILED tests/test_compute_routing_contract.py::test_normal_routes_use_precedence_and_preserve_role_boundary[N04-standard-precedence]
    FAILED tests/test_compute_routing_contract.py::test_normal_routes_use_precedence_and_preserve_role_boundary[N05-quick-precedence]
    FAILED tests/test_compute_routing_contract.py::test_known_bad_mutant_is_dropped_through_route_task[I01-role-boundary-mutation]
    FAILED tests/test_compute_routing_contract.py::test_known_bad_mutant_is_dropped_through_route_task[I02-posthoc-class-inference]
    FAILED tests/test_compute_routing_contract.py::test_known_bad_mutant_is_dropped_through_route_task[I03-provider-without-model]
    FAILED tests/test_compute_routing_contract.py::test_known_bad_mutant_is_dropped_through_route_task[I04-invalid-reasoning-default-fallback]
    FAILED tests/test_compute_routing_contract.py::test_known_bad_mutant_is_dropped_through_route_task[I05-catalog-only-capability]
    FAILED tests/test_compute_routing_contract.py::test_known_bad_mutant_is_dropped_through_route_task[I06-work-failure-model-fallback]
    FAILED tests/test_compute_routing_contract.py::test_known_bad_mutant_is_dropped_through_route_task[I07-third-spawn-chain]
    FAILED tests/test_compute_routing_contract.py::test_known_bad_mutant_is_dropped_through_route_task[I08-fallback-widens-safety-boundary]
    FAILED tests/test_compute_routing_contract.py::test_known_bad_mutant_is_dropped_through_route_task[I09-verified-without-actual-model]
    FAILED tests/test_compute_routing_contract.py::test_known_bad_mutant_is_dropped_through_route_task[I10-quick-skips-verification]
    FAILED tests/test_compute_routing_contract.py::test_known_bad_mutant_is_dropped_through_route_task[I11-unattended-architect-spawn]
    FAILED tests/test_compute_routing_contract.py::test_quota_and_work_failures_remain_distinct[I12-quota]
    FAILED tests/test_compute_routing_contract.py::test_quota_and_work_failures_remain_distinct[I12-work]
    FAILED tests/tools/test_kanban_compute_routing_contract.py::test_kanban_create_schema_exposes_closed_reasoning_effort_enum
    FAILED tests/tools/test_kanban_compute_routing_contract.py::test_kanban_create_reasoning_effort_parity_to_db_event_show_and_spawn
    FAILED tests/tools/test_kanban_compute_routing_contract.py::test_kanban_create_rejects_reasoning_typo_without_db_mutation
    21 failed, 2 passed in 2.72s

Command:

    G:labuntimehermes-bappenvScriptspython.exe -m py_compile tests/test_compute_routing_contract.py tests/tools/test_kanban_compute_routing_contract.py

Output:

    PYCOMPILE_RC=0

Command:

    git diff --check

Output:

    DIFF_CHECK_RC=0

## 4. Verification (known-bad)

- I01 role/assignee/scope/workspace/approval mutation → rejected, spawn false.
- I02 class/decision/policy 미저장 + actual model 사후 역추정 → rejected, verified 금지.
- I03 provider-only route → provider_requires_model.
- I04 hihg → invalid_reasoning_effort; 조용한 profile fallback 금지.
- I05 catalog-only + auth/runtime 미관측 → prepared_not_observed.
- I06 test_failed → work_failed + same_owner rework; model fallback 금지.
- I07 fallback_index=1 이후 추가 spawn → fallback_exhausted.
- I08 fallback의 write_scope/toolset/approval/master 확대 → rejected.
- I09 actual provider/model/effort 없음 → observation_incomplete, verified 금지.
- I10 quick도 verification_required.
- I11 unattended architect → approval_required, spawn false.
- I12 quota는 routing_unavailable/fallback_started, test failure는 work_failed/verification_failed.

정상 fixture 5종과 mutant 12종을 모두 같은 route_task 인터페이스에 넣는다. I12는 quota/work 두 variant다. fixture 완전성 테스트가 precedence, I1~I12 연속성, 중복 없는 12개 ID, 공통 인자 구조를 별도로 검증한다.

## 5. Issues / Caveats

- RED는 의도된 구현 전 기준선이다.
- 기존 DB/spawn builder에는 reasoning_effort 저장/argv 배선이 있으나 kanban_create schema/handler가 값을 받지 않는다. HIGH가 NULL로 저장되고 hihg도 task 생성에 성공한다. created event/show 표면도 parity가 없다.
- 공유 venv에는 ruff 모듈이 없어 ruff는 실행하지 못했다. JSON parse, Python py_compile, git diff --check는 통과했다.
- pytest 임시 디렉터리는 제거했다.
- AGENTS.md가 참조한 RTK.md는 worktree와 인접 worktree에서 발견되지 않았다.
- 지정 결과 경로 G:/lab/workspace/mas/tasks/compute-class-routing-arena/workers/quiz/result.md는 샌드박스에서 쓰기 거부되어 이 대체본을 tests/fixtures/quiz_result.md에 보존했다. 원래 경로로 복사가 필요하다.

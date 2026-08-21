# Success criteria — evidence map

Generated: 2026-05-15

Each row maps a goal-stated success criterion to its concrete proof in
this scaffold: the file(s) that implement it and the test(s) that pin it.

## Goal recap

> Hermes is successful when every agent action is:
> **typed → transition-validated → governance-approved or denied →
> audited → replayable → test-pinned**
> while keeping Hermes' existing provider integrations and tools intact.

## Evidence

| # | Success criterion | Implementation | Test |
|---|---|---|---|
| 1 | **515-LOC `MultiStepLoop` kernel** | `agent/runtime/loop.py` (515 LOC verified by `wc -l`) | — |
| 2 | **Frozen typed step records** | `agent/runtime/steps.py` — `MemoryStep`, `TaskStep`, `ActionStep`, `PlanningStep`, `FinalAnswerStep`, `ToolCall`, `ToolOutput`, all `@dataclass(frozen=True, slots=True)` | `test_steps.py::test_task_step_is_frozen` · `test_kernel_contract.py::test_every_step_is_typed_frozen_dataclass` |
| 3 | **Mandatory governance gate before every tool action** | `agent/runtime/loop.py::_action_step` calls `GovernanceGate.evaluate()` for every call; denied calls never reach `tool_handler.handle()` | `test_governance.py::test_default_loop_is_deny_all` · `test_governance.py::test_denied_call_appears_in_audit_trail` · `test_kernel_contract.py::test_every_tool_call_has_a_governance_decision` |
| 4 | **Default-deny / fail-closed execution** | `MultiStepLoop.__init__` uses `DenyAllGovernance` if no policy supplied; `continue_on_error=False` by default | `test_failures.py::test_governance_default_denies_all_tools_terminates` · `test_failures.py::test_model_error_terminates_failclosed_by_default` · `test_kernel_contract.py::test_default_loop_is_fail_closed_on_tool_use` |
| 5 | **Legal transition validation** | `agent/runtime/transitions.py::LEGAL_TRANSITIONS` + `TransitionGuard.check()`; `MultiStepLoop._append` rejects illegal appends | `test_transitions.py` (5 tests) · `test_kernel_contract.py::test_only_legal_transitions_landed_in_memory` |
| 6 | **Typed failures instead of string errors** | `agent/runtime/steps.py::StepFailure(kind: FailureKind, message, details)` with 8 `FailureKind` literals | `test_failures.py` (4 tests) · `test_kernel_contract.py::test_failure_carries_typed_kind` |
| 7 | **Audited state mutation with reasons** | `agent/runtime/state.py::RunState` — `set`/`append`/`increment`/`delete` require non-empty `reason`; every mutation recorded as `StateMutation` | `test_state.py::test_mutation_without_reason_is_rejected` · `test_state.py::test_set_records_mutation_with_reason` · `test_kernel_contract.py::test_state_mutations_all_have_reasons` |
| 8 | **State frozen after run** | `RunState.freeze()` called in `_finalize` | `test_state.py::test_frozen_state_rejects_further_mutation` · `test_kernel_contract.py::test_state_is_frozen_after_run` |
| 9 | **Deterministic replay from trace** | `agent/runtime/replay.py::run_from_trace(jsonl)` + `RecordedModel` + `RecordedToolHandler` + `ScriptedGovernance`; `Clock` + `IdSource` injection (`FrozenClock`, `SequentialIdSource`) | `test_replay.py` (3 tests) · `test_kernel_contract.py::test_trace_replays_byte_identically` |
| 10 | **Byte-identical JSONL replay** | `AgentMemory.to_jsonl()` / `from_jsonl()` preserve `governance_decisions`, `failure`, and `tool_outputs` shapes | `test_replay.py::test_recorded_trace_replays_to_identical_jsonl` · `test_kernel_contract.py::test_trace_replays_byte_identically` |
| 11 | **Provider/tool integration intact** | `integration/hermes_model_shim.py`, `integration/hermes_tool_handler_shim.py`, `integration/aiagent_delegation.py` — exemplars showing each hermes provider adapter / tool dispatcher needs only a thin shim | `tests/runtime/test_integration_exemplars.py` — 7 tests using fake hermes internals proving the wiring pattern works |
| 12 | **62 isolated loop/kernel tests passing** | `tests/runtime/*` | `python3 -m unittest discover -s tests/runtime -v` |

## Mechanical audit (grep proofs)

| Check | Command | Result |
|---|---|---|
| No bare `except:` in kernel | `grep -nE 'except\s*:' agent/runtime/*.py` | 0 hits |
| `time.time()` only in `SystemClock` | `grep -n 'time\.time' agent/runtime/*.py` | 1 hit in `SystemClock.now()` + 1 in a docstring |
| `uuid.uuid4()` only in `UuidIdSource` + `ToolCall.new` test helper | `grep -n 'uuid\.uuid4' agent/runtime/*.py` | 2 hits (both opt-in) |
| No direct `state[...]` mutation | `grep -nE 'self\.state\[' agent/runtime/*.py` | 0 hits |

## File inventory

```
agent/runtime/
├── __init__.py           129 LOC   public surface
├── steps.py              170 LOC   typed step records + StepFailure
├── result.py              70 LOC   RunResult + token/timing
├── interfaces.py         157 LOC   protocols + Clock/IdSource defaults
├── transitions.py         42 LOC   TransitionGuard + state machine
├── governance.py         107 LOC   DenyAll/AllowAll/AllowList + Gate
├── state.py              124 LOC   RunState + audited mutations
├── callbacks.py           59 LOC   CallbackRegistry
├── memory.py             258 LOC   AgentMemory + JSONL roundtrip
├── loop.py               515 LOC   MultiStepLoop (the kernel)
└── replay.py             185 LOC   RecordedModel/Handler + run_from_trace
                       --------
                        1,816 LOC   kernel total

integration/
├── __init__.py            14 LOC
├── hermes_model_shim.py  ~135 LOC  provider-adapter shape
├── hermes_tool_handler_shim.py ~110 LOC  tool dispatch shape
└── aiagent_delegation.py ~165 LOC  end-to-end AIAgent → kernel pattern

tests/runtime/
├── test_steps.py                51 LOC    7 tests
├── test_memory.py              101 LOC    8 tests
├── test_callbacks.py            78 LOC    5 tests
├── test_transitions.py          57 LOC    5 tests
├── test_state.py                81 LOC    8 tests
├── test_governance.py          134 LOC    6 tests
├── test_failures.py             98 LOC    4 tests
├── test_replay.py              112 LOC    3 tests
├── test_loop.py                237 LOC    7 tests
├── test_kernel_contract.py     150 LOC    9 tests
└── test_integration_exemplars.py            7 tests
                              --------  --------
                                       69 tests
```

## Reproduce

```bash
cd docs/proposals/scaffold
python3 -m unittest discover -s tests/runtime -t . -v
```

Expected: `Ran 69 tests in ~0.005s — OK`

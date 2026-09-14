from __future__ import annotations

import copy
import importlib
from typing import Any

BOARD = "11111111-1111-4111-8111-111111111111"
TASK = "t_0123456789abcdef"
RUN_GENERATION = 7
POLICY_GENERATION = 12


def _api() -> tuple[Any, Any]:
    module = importlib.import_module("hermes_cli.kanban_admission")
    return module.AdmissionPolicy, module.AdmissionRequest


def _parsed_policy(raw: object) -> Any:
    AdmissionPolicy, _ = _api()
    return AdmissionPolicy.from_mapping(raw)


def _request(action: object, **overrides: object) -> Any:
    _, AdmissionRequest = _api()
    values: dict[str, object] = {
        "action": action,
        "board_uuid": BOARD,
        "task_id": TASK,
        "run_generation": RUN_GENERATION,
        "policy_generation": POLICY_GENERATION,
        "force": False,
    }
    values.update(overrides)
    return AdmissionRequest(**values)


def _policy(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "schema_version": 1,
        "generation": POLICY_GENERATION,
        "mode": "manual_allowlist",
        "paused": False,
        "admissions": [
            {
                "board_uuid": BOARD,
                "task_id": TASK,
                "run_generation": RUN_GENERATION,
            }
        ],
    }
    values.update(overrides)
    return values


def test_missing_malformed_and_unsupported_policy_fail_closed() -> None:
    policies = [
        None,
        {},
        _policy(schema_version=2),
        _policy(schema_version=1.0),
        _policy(generation=True),
        _policy(mode="unexpected"),
        _policy(admissions=[{"board_uuid": BOARD}]),
    ]

    for raw in policies:
        policy = _parsed_policy(raw)
        decision = policy.decide(_request("claim"))
        assert decision.allowed is False
        assert decision.reason == "invalid_policy"


def test_equal_valued_non_integer_policy_generations_are_denied() -> None:
    policy = _parsed_policy(_policy())

    for generation in (float(POLICY_GENERATION), True):
        decision = policy.decide(
            _request("claim", policy_generation=generation)
        )
        assert decision.allowed is False
        assert decision.reason == "policy_generation_mismatch"


def test_directly_constructed_invalid_policy_cannot_bypass_parser() -> None:
    AdmissionPolicy, _ = _api()
    bypass = AdmissionPolicy(
        schema_version=999,
        generation=POLICY_GENERATION,
        mode="open",
        paused=False,
        admissions=frozenset(),
    )

    decision = bypass.decide(_request("claim"))

    assert decision.allowed is False
    assert decision.reason == "invalid_policy"


def test_malformed_unhashable_action_is_denied_without_raising() -> None:
    policy = _parsed_policy(_policy(mode="open", admissions=[]))

    decision = policy.decide(_request(["claim"]))

    assert decision.allowed is False
    assert decision.reason == "unknown_action"


def test_diagnostic_read_is_explicitly_non_execution_enabling() -> None:
    policy = _parsed_policy(_policy(mode="frozen", paused=True))

    decision = policy.decide(
        _request(
            "diagnostic_read",
            board_uuid=None,
            task_id=None,
            run_generation=None,
        )
    )

    assert decision.allowed is True
    assert decision.execution_enabling is False
    assert decision.reason == "diagnostic_read"


def test_pause_denies_execution_even_when_force_and_exactly_allowlisted() -> None:
    policy = _parsed_policy(_policy(paused=True))

    for action in ("claim", "spawn", "promote", "unblock", "retry"):
        decision = policy.decide(_request(action, force=True))
        assert decision.allowed is False
        assert decision.reason == "paused"
        assert decision.execution_enabling is True


def test_frozen_mode_denies_every_execution_enabling_transition() -> None:
    policy = _parsed_policy(_policy(mode="frozen"))

    for action in ("claim", "spawn", "promote", "unblock", "retry"):
        decision = policy.decide(_request(action))
        assert decision.allowed is False
        assert decision.reason == "frozen"


def test_manual_allowlist_requires_exact_complete_identity() -> None:
    policy = _parsed_policy(_policy())

    assert policy.decide(_request("claim")).allowed is True

    mismatches = [
        {"board_uuid": "22222222-2222-4222-8222-222222222222"},
        {"task_id": "t_fedcba9876543210"},
        {"run_generation": RUN_GENERATION + 1},
        {"policy_generation": POLICY_GENERATION + 1},
        {"board_uuid": None},
        {"task_id": None},
        {"run_generation": None},
    ]
    for mismatch in mismatches:
        decision = policy.decide(_request("claim", **mismatch))
        assert decision.allowed is False
        assert decision.reason in {"identity_not_admitted", "policy_generation_mismatch"}


def test_open_mode_still_requires_complete_identity_and_policy_generation() -> None:
    policy = _parsed_policy(_policy(mode="open", admissions=[]))

    assert policy.decide(_request("spawn")).allowed is True
    assert policy.decide(_request("spawn", task_id=None)).allowed is False
    assert policy.decide(
        _request("spawn", policy_generation=POLICY_GENERATION + 1)
    ).allowed is False


def test_default_assignment_and_route_fallback_are_never_admissible_actions() -> None:
    policy = _parsed_policy(_policy(mode="open", admissions=[]))

    for action in ("default_assignment", "route_fallback"):
        decision = policy.decide(_request(action))
        assert decision.allowed is False
        assert decision.reason == "forbidden_action"


def test_parsing_and_decisions_do_not_mutate_inputs() -> None:
    raw = _policy()
    original = copy.deepcopy(raw)
    request = _request("claim")

    policy = _parsed_policy(raw)
    first = policy.decide(request)
    second = policy.decide(request)

    assert raw == original
    assert first == second

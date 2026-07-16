from __future__ import annotations

import importlib


OLD_TASK_A_CONTRACT_NEEDLE = "OLD_TASK_A_CONTRACT_NEEDLE_PHASE3_RED"
CURRENT_TASK_B_CONTRACT_WCP = "CURRENT_TASK_B_CONTRACT_WCP_PHASE3_RED"
UNSAFE_CONTRACT_SECRET = "UNSAFE_CONTRACT_SECRET_PHASE3_RED token=abc123"


def _load_wcp_module():
    # Phase 3 RED contract: this module must not exist yet in the current
    # approved scope. These tests document the future adapter API and are
    # expected to fail with ModuleNotFoundError until implementation is approved.
    return importlib.import_module("agent.working_context_packet")


def test_wcp_adapter_disabled_returns_use_original_contract():
    module = _load_wcp_module()
    decision = module.enforce_working_context_packet(
        policy={"enabled": False},
        api_messages=[{"role": "user", "content": OLD_TASK_A_CONTRACT_NEEDLE}],
        messages=[{"role": "user", "content": OLD_TASK_A_CONTRACT_NEEDLE}],
        context_health_intake=None,
    )

    assert decision.action == "use_original"
    assert decision.api_messages[0]["content"] == OLD_TASK_A_CONTRACT_NEEDLE


def test_wcp_adapter_enabled_replaces_provider_messages_without_old_task_a_needle():
    module = _load_wcp_module()
    decision = module.enforce_working_context_packet(
        policy={"enabled": True},
        api_messages=[
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": OLD_TASK_A_CONTRACT_NEEDLE},
            {"role": "user", "content": CURRENT_TASK_B_CONTRACT_WCP},
        ],
        messages=[{"role": "user", "content": OLD_TASK_A_CONTRACT_NEEDLE}],
        context_health_intake={"summary_path": "/tmp/summary.md"},
    )

    assert decision.action == "replace_api_messages"
    payload = repr(decision.api_messages)
    assert CURRENT_TASK_B_CONTRACT_WCP in payload
    assert OLD_TASK_A_CONTRACT_NEEDLE not in payload


def test_wcp_adapter_enabled_unsafe_returns_hold_without_secret_body():
    module = _load_wcp_module()
    decision = module.enforce_working_context_packet(
        policy={"enabled": True},
        api_messages=[{"role": "user", "content": UNSAFE_CONTRACT_SECRET}],
        messages=[{"role": "user", "content": UNSAFE_CONTRACT_SECRET}],
        context_health_intake={"safe": False, "reason": "unsafe"},
    )

    assert decision.action == "hold"
    assert decision.api_messages is None
    assert UNSAFE_CONTRACT_SECRET not in repr(decision)
    assert "abc123" not in repr(decision)

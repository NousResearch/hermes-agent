"""Tests for typed plugin aggregation at the kanban pre-claim boundary."""

from __future__ import annotations

import threading

import pytest

import hermes_cli.plugins as plugins
from hermes_cli.plugins import (
    BLOCK_KINDS,
    PluginContext,
    PluginManager,
    PluginManifest,
    PreClaimDisposition,
    PreClaimHookError,
    aggregate_hooks,
)


PLUGIN_ID = "test/preclaim-policy"


@pytest.fixture
def manager(monkeypatch):
    instance = PluginManager()
    monkeypatch.setattr(plugins, "_plugin_manager", instance)
    return instance


def _register_pre_claim(manager, plugin_id, callback):
    PluginContext(PluginManifest(name=plugin_id, key=plugin_id), manager).register_hook(
        "kanban_task_pre_claim", callback
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"decision": "allow"},
        {"decision": "defer"},
        {"decision": "defer", "reason": "try later"},
        *[
            {
                "decision": "block",
                "reason": "policy denied the claim",
                "block_kind": block_kind,
            }
            for block_kind in sorted(BLOCK_KINDS)
        ],
        {
            "decision": "complete",
            "summary": "work already verified",
            "evidence": {"checks": ["unit", "lint"]},
        },
    ],
)
def test_pre_claim_disposition_accepts_every_valid_decision_path(kwargs):
    disposition = PreClaimDisposition(plugin_id=PLUGIN_ID, **kwargs)

    assert disposition.decision == kwargs["decision"]
    assert disposition.plugin_id == PLUGIN_ID


@pytest.mark.parametrize(
    "kwargs",
    [
        {"decision": "unknown"},
        {"decision": "defer", "evidence": []},
        {"decision": "defer", "reason": 7},
        {"decision": "defer", "summary": 7},
        {"decision": "allow", "reason": "extra"},
        {"decision": "allow", "block_kind": "unsupported"},
        {"decision": "allow", "summary": "extra"},
        {"decision": "allow", "evidence": {"extra": True}},
        {"decision": "defer", "block_kind": "unsupported"},
        {"decision": "block", "block_kind": "unsupported"},
        {"decision": "block", "reason": "", "block_kind": "unsupported"},
        {"decision": "block", "reason": "   ", "block_kind": "unsupported"},
        {"decision": "block", "reason": "denied"},
        {
            "decision": "block",
            "reason": "denied",
            "block_kind": "not-a-supported-kind",
        },
        {
            "decision": "complete",
            "block_kind": "unsupported",
            "summary": "done",
            "evidence": {"proof": True},
        },
        {"decision": "complete", "evidence": {"proof": True}},
        {"decision": "complete", "summary": "", "evidence": {"proof": True}},
        {"decision": "complete", "summary": "   ", "evidence": {"proof": True}},
        {"decision": "complete", "summary": "done", "evidence": {}},
    ],
)
def test_pre_claim_disposition_rejects_decision_specific_invalid_matrix(kwargs):
    with pytest.raises(PreClaimHookError) as exc_info:
        PreClaimDisposition(plugin_id=PLUGIN_ID, **kwargs)

    assert PLUGIN_ID in str(exc_info.value)


def test_pre_claim_disposition_rejects_non_json_serializable_evidence():
    with pytest.raises(PreClaimHookError) as exc_info:
        PreClaimDisposition(
            decision="complete",
            summary="done",
            evidence={"unsupported": {1, 2}},
            plugin_id=PLUGIN_ID,
        )

    assert PLUGIN_ID in str(exc_info.value)


def test_pre_claim_disposition_rejects_block_kind_outside_public_allowlist():
    outside_kind = "outside-the-allowlist"
    assert outside_kind not in BLOCK_KINDS

    with pytest.raises(PreClaimHookError) as exc_info:
        PreClaimDisposition(
            decision="block",
            reason="denied",
            block_kind=outside_kind,
            plugin_id=PLUGIN_ID,
        )

    assert PLUGIN_ID in str(exc_info.value)


def test_aggregate_pre_claim_without_callbacks_defaults_to_allow(manager):
    result = aggregate_hooks("kanban_task_pre_claim", task={"id": "t1"})

    assert result == PreClaimDisposition(decision="allow")


def test_aggregate_pre_claim_invokes_all_callbacks_synchronously_with_exact_kwargs(
    manager,
):
    caller_thread = threading.get_ident()
    snapshot = {
        "task_id": "t1",
        "board": "default",
        "assignee": "worker",
        "source_status": "ready",
        "task": {"id": "t1", "title": "safe snapshot"},
        "dry_run": False,
    }
    calls = []

    def observe_first(**kwargs):
        calls.append(("first", threading.get_ident(), kwargs))

    def observe_second(**kwargs):
        calls.append(("second", threading.get_ident(), kwargs))
        return PreClaimDisposition(decision="defer", reason="later")

    manager._hooks["kanban_task_pre_claim"] = [observe_first, observe_second]

    result = aggregate_hooks("kanban_task_pre_claim", **snapshot)

    assert result.decision == "defer"
    assert calls == [
        ("first", caller_thread, snapshot),
        ("second", caller_thread, snapshot),
    ]


def test_aggregate_pre_claim_treats_none_as_allow(manager):
    manager._hooks["kanban_task_pre_claim"] = [lambda **_: None]

    assert aggregate_hooks("kanban_task_pre_claim").decision == "allow"


@pytest.mark.parametrize(
    ("order", "expected_plugin_id"),
    [
        (("allow", "defer"), "defer-plugin"),
        (("defer", "allow"), "defer-plugin"),
        (("defer", "complete"), "complete-plugin"),
        (("complete", "defer"), "complete-plugin"),
        (("complete", "block"), "block-plugin"),
        (("block", "complete"), "block-plugin"),
    ],
)
def test_aggregate_pre_claim_uses_decision_priority_not_registration_order(
    manager, order, expected_plugin_id
):
    returned = {
        "allow": PreClaimDisposition(decision="allow", plugin_id="allow-plugin"),
        "defer": PreClaimDisposition(
            decision="defer", reason="later", plugin_id="defer-plugin"
        ),
        "complete": PreClaimDisposition(
            decision="complete",
            summary="already done",
            evidence={"proof": True},
            plugin_id="complete-plugin",
        ),
        "block": PreClaimDisposition(
            decision="block",
            reason="denied",
            block_kind="policy_violation",
            plugin_id="block-plugin",
        ),
    }
    calls = []
    for decision in order:
        value = returned[decision]
        _register_pre_claim(
            manager,
            value.plugin_id,
            lambda value=value: calls.append(value.decision) or value,
        )

    result = aggregate_hooks("kanban_task_pre_claim")

    assert calls == list(order)
    assert isinstance(result, PreClaimDisposition)
    assert result.plugin_id == expected_plugin_id


def test_aggregate_pre_claim_keeps_registration_order_for_equal_rank(manager):
    first = PreClaimDisposition(decision="defer", reason="first", plugin_id="first")
    second = PreClaimDisposition(decision="defer", reason="second", plugin_id="second")
    manager._hooks["kanban_task_pre_claim"] = [
        lambda: first,
        lambda: second,
    ]

    result = aggregate_hooks("kanban_task_pre_claim")

    assert result.reason == "first"
    assert result.plugin_id == "first"


@pytest.mark.parametrize("invalid_response", ["allow", {"decision": "allow"}, 1])
def test_aggregate_pre_claim_rejects_invalid_type_with_registered_plugin_id(
    manager, invalid_response
):
    manifest = PluginManifest(name="display-name", key=PLUGIN_ID)
    context = PluginContext(manifest, manager)
    context.register_hook("kanban_task_pre_claim", lambda **_: invalid_response)

    with pytest.raises(PreClaimHookError) as exc_info:
        aggregate_hooks("kanban_task_pre_claim", task={"id": "t1"})

    assert PLUGIN_ID in str(exc_info.value)


def test_aggregate_pre_claim_does_not_trust_plugin_id_on_invalid_response(manager):
    class InvalidResponse:
        plugin_id = "spoofed-plugin"

    manifest = PluginManifest(name="display-name", key=PLUGIN_ID)
    PluginContext(manifest, manager).register_hook(
        "kanban_task_pre_claim", lambda **_: InvalidResponse()
    )

    with pytest.raises(PreClaimHookError) as exc_info:
        aggregate_hooks("kanban_task_pre_claim")

    assert PLUGIN_ID in str(exc_info.value)
    assert "spoofed-plugin" not in str(exc_info.value)


def test_aggregate_pre_claim_revalidates_mutated_disposition_with_registered_id(
    manager,
):
    invalid = PreClaimDisposition(decision="allow", plugin_id=PLUGIN_ID)
    invalid.reason = "added after validation"
    PluginContext(
        PluginManifest(name="display-name", key=PLUGIN_ID), manager
    ).register_hook("kanban_task_pre_claim", lambda **_: invalid)

    with pytest.raises(PreClaimHookError) as exc_info:
        aggregate_hooks("kanban_task_pre_claim")

    assert PLUGIN_ID in str(exc_info.value)


def test_aggregate_hooks_preserves_observer_list_semantics(manager):
    calls = []

    def first(**kwargs):
        calls.append(kwargs)
        return "first-result"

    manager._hooks["post_tool_call"] = [first, lambda **_: None]

    result = aggregate_hooks("post_tool_call", tool_name="terminal")

    assert result == ["first-result"]
    assert calls == [
        {
            "tool_name": "terminal",
            "telemetry_schema_version": "hermes.observer.v1",
        }
    ]

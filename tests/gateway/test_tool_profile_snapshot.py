from __future__ import annotations

from gateway.tool_profile_snapshot import (
    build_tool_loader_contract,
    build_tool_profile_snapshot,
    build_tool_registry,
    build_tool_risk_summary,
    evaluate_tool_load_request,
)


def _schema(name: str, description: str = "test tool") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_registry_classifies_core_tool_metadata() -> None:
    registry = build_tool_registry(
        [_schema("brainstack_recall"), _schema("terminal"), _schema("read_file")],
        toolset_memberships={
            "brainstack_recall": ["memory_provider:brainstack"],
            "terminal": ["hermes-cli"],
            "read_file": ["file"],
        },
    )

    assert registry["brainstack_recall"].capability_class == "memory.recall"
    assert registry["brainstack_recall"].data_exposure_class == "user_private"
    assert registry["terminal"].requires_human_approval is True
    assert registry["terminal"].side_effect_class == "local_execute"
    assert registry["read_file"].secrets_possible is True


def test_profile_snapshot_is_deterministic_and_cache_breaks_on_prefix_change() -> None:
    schemas = [_schema("terminal"), _schema("brainstack_recall")]
    registry = build_tool_registry(schemas)

    first = build_tool_profile_snapshot(
        "heavy_work",
        list(reversed(schemas)),
        registry,
        system_prompt="stable prompt",
        static_prefix="stable prefix",
    )
    second = build_tool_profile_snapshot(
        "heavy_work",
        schemas,
        registry,
        system_prompt="stable prompt",
        static_prefix="stable prefix",
        previous_static_prefix_hash=first.static_prefix_hash,
    )
    changed = build_tool_profile_snapshot(
        "heavy_work",
        schemas,
        registry,
        system_prompt="stable prompt",
        static_prefix="changed prefix",
        previous_static_prefix_hash=first.static_prefix_hash,
    )

    assert first.tool_names == ("brainstack_recall", "terminal")
    assert first.tool_schema_hash == second.tool_schema_hash
    assert second.cache_break_reason is None
    assert changed.cache_break_reason == "STATIC_PREFIX_HASH_CHANGED_REQUIRES_PROFILE_VERSION_BUMP"


def test_conversation_direct_loader_unavailable() -> None:
    registry = build_tool_registry([_schema("brainstack_recall")])
    contract = build_tool_loader_contract("conversation_direct", ["brainstack_recall"], registry)

    assert contract.loader_available is False
    allowed, reason = evaluate_tool_load_request(contract, registry, "brainstack_recall")
    assert allowed is False
    assert reason == "LOADER_UNAVAILABLE_FOR_PROFILE"


def test_tool_loader_allows_only_profile_local_enum() -> None:
    registry = build_tool_registry([_schema("brainstack_recall"), _schema("terminal")])
    contract = build_tool_loader_contract("conversation_tools", ["brainstack_recall"], registry)

    allowed, reason = evaluate_tool_load_request(contract, registry, "brainstack_recall")
    assert allowed is True
    assert reason == "ALLOWED"

    allowed, reason = evaluate_tool_load_request(contract, registry, "terminal")
    assert allowed is False
    assert reason == "TOOL_NOT_IN_ALLOWED_ENUM"


def test_tool_loader_preserves_side_effect_approval() -> None:
    registry = build_tool_registry([_schema("terminal")])
    contract = build_tool_loader_contract("heavy_work", ["terminal"], registry)

    allowed, reason = evaluate_tool_load_request(contract, registry, "terminal")
    assert allowed is False
    assert reason == "TOOL_SIDE_EFFECT_APPROVAL_REQUIRED"

    allowed, reason = evaluate_tool_load_request(
        contract,
        registry,
        "terminal",
        approval_granted=True,
    )
    assert allowed is True
    assert reason == "ALLOWED"


def test_config_disabled_and_gated_tools_rejected() -> None:
    registry = build_tool_registry(
        [_schema("brainstack_recall"), _schema("read_file")],
        gated_tools=["brainstack_recall"],
        config_disabled_tools=["read_file"],
    )
    contract = build_tool_loader_contract(
        "conversation_tools",
        ["brainstack_recall", "read_file"],
        registry,
    )

    allowed, reason = evaluate_tool_load_request(contract, registry, "brainstack_recall")
    assert allowed is False
    assert reason == "TOOL_GATED_UNAVAILABLE"

    allowed, reason = evaluate_tool_load_request(contract, registry, "read_file")
    assert allowed is False
    assert reason == "TOOL_CONFIG_DISABLED"


def test_risk_summary_marks_overlap_and_actions() -> None:
    registry = build_tool_registry(
        [_schema("terminal"), _schema("write_file"), _schema("read_file")]
    )
    summary = build_tool_risk_summary("heavy_work", registry, ["terminal", "write_file", "read_file"])

    assert summary.wrong_tool_risk == "high"
    assert summary.approval_required_tools == ["terminal", "write_file"]
    assert summary.overlap_groups["file_access"] == ["read_file", "write_file"]

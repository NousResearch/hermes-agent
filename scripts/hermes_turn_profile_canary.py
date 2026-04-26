#!/usr/bin/env python3
"""Write Phase 154 structural turn-profile canary artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gateway.tool_profile_snapshot import (  # noqa: E402
    build_tool_loader_contract,
    build_tool_registry,
    evaluate_tool_load_request,
    normalize_tool_schema,
)
from gateway.turn_profiles import SCHEMA_VERSION, resolve_turn_profile  # noqa: E402
from tools.registry import discover_builtin_tools, registry as tool_registry  # noqa: E402


def _json_dump(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _schemas_by_name() -> dict[str, dict[str, Any]]:
    discover_builtin_tools()
    schemas: dict[str, dict[str, Any]] = {}
    for name in sorted(tool_registry.get_all_tool_names()):
        schema = tool_registry.get_schema(name)
        if schema:
            schemas[name] = normalize_tool_schema(schema)
    return schemas


def build_artifacts() -> dict[str, Any]:
    current = ["hermes-cli"]
    conversation = resolve_turn_profile(
        platform="discord",
        prompt="Mi volt a debug markerem?",
        current_enabled_toolsets=current,
        env={},
    )
    url_memory = resolve_turn_profile(
        platform="discord",
        prompt="Jegyezd meg ezt az URL-t: https://example.com/repo",
        current_enabled_toolsets=current,
        env={},
    )
    keyword_not_heavy = resolve_turn_profile(
        platform="discord",
        prompt="nézd meg emlékezetből, beszéltünk-e erről",
        current_enabled_toolsets=current,
        env={},
    )
    direct = resolve_turn_profile(
        platform="discord",
        prompt="Mi volt a debug markerem?",
        current_enabled_toolsets=current,
        env={"HERMES_DISCORD_CONVERSATION_DIRECT": "1"},
    )
    cli = resolve_turn_profile(
        platform="cli",
        prompt="run tests",
        current_enabled_toolsets=current,
        env={},
    )
    rollback = resolve_turn_profile(
        platform="discord",
        prompt="Mi volt a debug markerem?",
        current_enabled_toolsets=["web", "file"],
        env={"HERMES_DISCORD_TURN_PROFILE": "heavy"},
    )
    heavy_web = resolve_turn_profile(
        platform="discord",
        prompt="/heavy web https://example.com",
        current_enabled_toolsets=current,
        env={},
    )
    heavy_file = resolve_turn_profile(
        platform="discord",
        prompt="/heavy file README.md",
        current_enabled_toolsets=current,
        env={},
    )
    heavy_code = resolve_turn_profile(
        platform="discord",
        prompt="/heavy code run tests",
        current_enabled_toolsets=current,
        env={},
    )
    heavy_debug = resolve_turn_profile(
        platform="discord",
        prompt="/heavy full-debug inspect everything",
        current_enabled_toolsets=current,
        env={},
    )

    schemas = _schemas_by_name()
    tool_registry_meta = build_tool_registry(schemas.values())
    conversation_loader = build_tool_loader_contract(
        "conversation_tools",
        ["memory", "session_search", "clarify"],
        tool_registry_meta,
    )
    direct_loader = build_tool_loader_contract(
        "conversation_direct",
        [],
        tool_registry_meta,
    )
    allowed_load = evaluate_tool_load_request(
        conversation_loader,
        tool_registry_meta,
        "session_search",
    )
    denied_cross_profile = evaluate_tool_load_request(
        conversation_loader,
        tool_registry_meta,
        "terminal",
    )
    direct_denied = evaluate_tool_load_request(
        direct_loader,
        tool_registry_meta,
        "session_search",
    )
    side_effect_denied = evaluate_tool_load_request(
        build_tool_loader_contract("heavy_code", ["terminal"], tool_registry_meta),
        tool_registry_meta,
        "terminal",
        approval_granted=False,
    )
    side_effect_allowed_with_approval = evaluate_tool_load_request(
        build_tool_loader_contract("heavy_code", ["terminal"], tool_registry_meta),
        tool_registry_meta,
        "terminal",
        approval_granted=True,
    )

    config_parity = {
        "schema": SCHEMA_VERSION,
        "discord_default": conversation.to_dict(),
        "conversation_direct_override": direct.to_dict(),
        "cli_local": cli.to_dict(),
        "rollback": rollback.to_dict(),
        "verdict": {
            "passed": (
                conversation.turn_profile == "conversation_tools"
                and direct.enabled_toolsets == ()
                and cli.enabled_toolsets == ("hermes-cli",)
                and rollback.rollback_override_active
            )
        },
    }
    conversation_canary = {
        "schema": SCHEMA_VERSION,
        "memory_question": conversation.to_dict(),
        "url_memory_false_positive": url_memory.to_dict(),
        "keyword_not_heavy": keyword_not_heavy.to_dict(),
        "verdict": {
            "passed": (
                conversation.enabled_toolsets == ("conversation_tools",)
                and url_memory.turn_profile == "conversation_tools"
                and url_memory.url_attachment_candidate_only
                and not url_memory.explicit_heavy
                and keyword_not_heavy.turn_profile == "conversation_tools"
            )
        },
    }
    heavy_canary = {
        "schema": SCHEMA_VERSION,
        "heavy_web": heavy_web.to_dict(),
        "heavy_file": heavy_file.to_dict(),
        "heavy_code": heavy_code.to_dict(),
        "heavy_full_debug": heavy_debug.to_dict(),
        "side_effect_denied_without_approval": {
            "allowed": side_effect_denied[0],
            "reason": side_effect_denied[1],
        },
        "side_effect_allowed_with_approval": {
            "allowed": side_effect_allowed_with_approval[0],
            "reason": side_effect_allowed_with_approval[1],
        },
        "verdict": {
            "passed": (
                heavy_web.enabled_toolsets == ("heavy_web",)
                and heavy_file.enabled_toolsets == ("heavy_file",)
                and heavy_code.enabled_toolsets == ("heavy_code",)
                and heavy_debug.enabled_toolsets == ("heavy_full_debug",)
                and not side_effect_denied[0]
                and side_effect_denied[1] == "TOOL_SIDE_EFFECT_APPROVAL_REQUIRED"
                and side_effect_allowed_with_approval[0]
            )
        },
    }
    loader_canary = {
        "schema": SCHEMA_VERSION,
        "conversation_tools_loader": conversation_loader.to_dict(),
        "conversation_direct_loader": direct_loader.to_dict(),
        "allowed_read_only_load": {"allowed": allowed_load[0], "reason": allowed_load[1]},
        "denied_cross_profile_load": {
            "allowed": denied_cross_profile[0],
            "reason": denied_cross_profile[1],
        },
        "denied_direct_load": {"allowed": direct_denied[0], "reason": direct_denied[1]},
        "cleanup_contract": {
            "ephemeral_turn_end_cleanup": conversation_loader.turn_end_cleanup_required,
            "pinned_session_end_cleanup": conversation_loader.session_end_cleanup_required,
            "profile_change_cleanup": conversation_loader.profile_change_cleanup_required,
        },
        "verdict": {
            "passed": (
                allowed_load[0]
                and not denied_cross_profile[0]
                and denied_cross_profile[1] == "TOOL_NOT_IN_ALLOWED_ENUM"
                and not direct_denied[0]
                and direct_denied[1] == "LOADER_UNAVAILABLE_FOR_PROFILE"
                and conversation_loader.turn_end_cleanup_required
                and conversation_loader.session_end_cleanup_required
            )
        },
    }
    return {
        "config_parity": config_parity,
        "conversation_canary": conversation_canary,
        "heavy_canary": heavy_canary,
        "loader_canary": loader_canary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = build_artifacts()
    _json_dump(output_dir / "154-CONFIG-PARITY.json", artifacts["config_parity"])
    _json_dump(output_dir / "154-CONVERSATION-CANARY.json", artifacts["conversation_canary"])
    _json_dump(output_dir / "154-HEAVY-BUNDLE-CANARY.json", artifacts["heavy_canary"])
    _json_dump(output_dir / "154-TOOL-LOADER-CANARY.json", artifacts["loader_canary"])
    passed = all(artifact["verdict"]["passed"] for artifact in artifacts.values())
    print(json.dumps({"schema": SCHEMA_VERSION, "passed": passed}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

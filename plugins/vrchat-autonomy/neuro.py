"""Neuro API (VedalAI neuro-sdk) helpers for the vrchat-autonomy plugin."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.openclaw.neuro_bridge import (
    DEFAULT_GAME_NAME,
    build_action_force_message,
    build_neuro_bridge_bootstrap,
    build_vrchat_neuro_actions,
    handle_neuro_action_message,
    neuro_sdk_vendor_status,
)
from tools.openclaw.vrchat_autonomy import load_autonomy_profile

DEFAULT_NEURO_WS_URL = "ws://127.0.0.1:8000"


def resolve_game_name(config: dict[str, Any] | None = None) -> str:
    if not config:
        return DEFAULT_GAME_NAME
    raw = str(config.get("neuro_game") or "").strip()
    return raw or DEFAULT_GAME_NAME


def resolve_ws_url(config: dict[str, Any] | None = None) -> str:
    if not config:
        return DEFAULT_NEURO_WS_URL
    raw = str(config.get("neuro_ws_url") or "").strip()
    return raw or DEFAULT_NEURO_WS_URL


def neuro_status(*, profile: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Vendor clone, profile, and action catalog for Neuro API bridge."""
    vendor = neuro_sdk_vendor_status()
    game = resolve_game_name(config)
    loaded = load_autonomy_profile(profile)
    return {
        "ok": vendor.get("success", False),
        "vendor": vendor,
        "game": game,
        "ws_url": resolve_ws_url(config),
        "profile": loaded,
        "actions": build_vrchat_neuro_actions(profile_path=profile),
    }


def neuro_bootstrap(
    *,
    profile: Path,
    config: dict[str, Any] | None = None,
    context: str = "",
    silent_context: bool = True,
) -> dict[str, Any]:
    """Build startup/context/actions/register messages for a Neuro websocket client."""
    game = resolve_game_name(config)
    payload = build_neuro_bridge_bootstrap(
        game=game,
        profile_path=profile,
        context=context,
        silent_context=silent_context,
    )
    payload["game"] = game
    payload["ws_url"] = resolve_ws_url(config)
    return payload


def neuro_build_messages(
    *,
    profile: Path,
    config: dict[str, Any] | None = None,
    context: str = "",
    silent_context: bool = True,
    force_action_names: list[str] | None = None,
    force_query: str = "",
    force_state: str = "",
    force_priority: str = "low",
    ephemeral_context: bool = True,
) -> dict[str, Any]:
    """Bootstrap plus optional actions/force message."""
    payload = neuro_bootstrap(
        profile=profile,
        config=config,
        context=context,
        silent_context=silent_context,
    )
    game = payload["game"]
    names = list(force_action_names or [])
    if force_query and names:
        payload["messages"].append(
            build_action_force_message(
                action_names=names,
                query=force_query,
                state=force_state,
                game=game,
                priority=force_priority,
                ephemeral_context=ephemeral_context,
            )
        )
    return payload


def neuro_handle_action(
    message: dict[str, Any],
    *,
    profile: Path,
    config: dict[str, Any] | None = None,
    retry_on_failure: bool = False,
    force_dry_run: bool = False,
) -> dict[str, Any]:
    """Validate one incoming Neuro action and route through VRChat safety gates."""
    game = resolve_game_name(config)
    result = handle_neuro_action_message(
        message,
        profile_path=profile,
        game=game,
        retry_on_failure=retry_on_failure,
        force_dry_run=force_dry_run,
    )
    result["game"] = game
    return result


def neuro_readiness(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Lightweight readiness slice for doctor/status (vendor files only)."""
    vendor = neuro_sdk_vendor_status()
    return {
        "vendor_ok": bool(vendor.get("success")),
        "vendor_path": vendor.get("path"),
        "commit": vendor.get("commit") or "",
        "specification_exists": bool(vendor.get("specification_exists")),
        "game": resolve_game_name(config),
        "ws_url": resolve_ws_url(config),
        "hint": (
            "Clone neuro-sdk into vendor/neuro-sdk (API/SPECIFICATION.md + LICENSE.md)."
            if not vendor.get("success")
            else "Use hermes vrchat-autonomy neuro bootstrap or scripts/vrchat_neuro_bridge.py."
        ),
    }


def neuro_vendor_status(*, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Vendor clone status plus submodule init command when missing."""
    vendor = neuro_sdk_vendor_status()
    init_cmd = "git submodule update --init vendor/neuro-sdk"
    ready = bool(vendor.get("success"))
    return {
        "ok": ready,
        "vendor": vendor,
        "game": resolve_game_name(config),
        "ws_url": resolve_ws_url(config),
        "init_command": init_cmd,
        "hint": (
            f"Run: {init_cmd}"
            if not ready
            else "Vendor OK — use `hermes vrchat-autonomy neuro bootstrap` or `neuro bridge`."
        ),
    }

"""Truthful, resumable readiness probes for installed Bot Marketplace profiles."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from hermes_cli.bot_catalog import BotCatalogEntry, BotSetupRequirement


def _metadata_path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "profile.yaml"


def _read_metadata() -> dict[str, Any]:
    from hermes_cli.bot_metadata import read_bot_metadata

    return read_bot_metadata()


def _effective_config() -> dict[str, Any]:
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_constants import get_hermes_home

    path = get_hermes_home() / "config.yaml"
    return load_user_config_effective(path) if path.is_file() else {}


def _toolset_ready(identifier: str) -> tuple[bool, str | None]:
    from hermes_cli.tools_config import _get_platform_tools
    from toolsets import get_toolset, resolve_toolset

    if get_toolset(identifier, include_registry=True) is None:
        return False, "This toolset is not available in the installed Hermes build."
    cfg = _effective_config()
    enabled = _get_platform_tools(cfg, "desktop") | _get_platform_tools(cfg, "cli")
    if identifier not in enabled:
        return False, "Enable this toolset for the bot profile."
    try:
        from model_tools import get_tool_definitions

        definitions = get_tool_definitions(
            enabled_toolsets=[identifier], quiet_mode=True, skip_tool_search_assembly=True,
        )
        exposed = {
            str(item.get("function", {}).get("name") or "")
            for item in definitions
            if isinstance(item, dict)
        }
        if not exposed.intersection(resolve_toolset(identifier)):
            return False, "This toolset is enabled but has no runtime-available tools in this profile."
    except Exception:
        return False, "Runtime availability for this toolset could not be verified."
    return True, None


def _plugin_ready(identifier: str) -> tuple[bool, str | None]:
    try:
        from hermes_cli.plugins import discover_plugins, get_plugin_manager

        discover_plugins()
        rows = get_plugin_manager().list_plugins()
    except Exception:
        return False, "Plugin discovery failed for this profile."
    row = next((item for item in rows if item.get("key") == identifier), None)
    if row is None:
        return False, "Install this plugin in the bot profile."
    if not row.get("enabled") or row.get("error"):
        return False, "Enable this plugin and resolve its load error."
    return True, None


def _connector_ready(identifier: str) -> tuple[bool, str | None]:
    try:
        from tools.connectors import connectors_available
        from tools.connectors.gateway.client import ConnectorClient

        if not connectors_available():
            return False, "Connector service is unavailable for this profile."
        rows = ConnectorClient().list_connectors()
        row = next((item for item in rows if str(item.get("connector") or "").lower() == identifier.lower()), None)
    except Exception:
        return False, "Connector status could not be read."
    if row is None:
        return False, "This connector is not in the canonical connector catalog."
    if not row.get("connected"):
        return False, "Connect an account for this connector."
    return True, None


def _requirement_status(requirement: BotSetupRequirement) -> dict[str, Any]:
    if requirement.kind == "toolset":
        ready, detail = _toolset_ready(requirement.id)
        action = "tools"
    elif requirement.kind == "command":
        ready = shutil.which(requirement.id) is not None
        detail = None if ready else f"Install the {requirement.id} executable on the runtime host."
        action = "install_command"
    elif requirement.kind == "plugin":
        ready, detail = _plugin_ready(requirement.id)
        action = "plugins"
    else:
        ready, detail = _connector_ready(requirement.id)
        action = "connect"
    return {
        "kind": requirement.kind,
        "id": requirement.id,
        "required": requirement.required,
        "purpose": requirement.purpose,
        "status": "ready" if ready else "needs_setup",
        "action": None if ready else action,
        "detail": detail,
    }


def _first_task_proof(entry: BotCatalogEntry) -> dict[str, Any]:
    """Bind proof to the starter request and a normally completed assistant turn."""
    from hermes_constants import get_hermes_home

    pending = {
        "status": "pending", "session_id": None,
        "request_message_id": None, "completion_message_id": None,
    }
    db_path = get_hermes_home() / "state.db"
    if not db_path.is_file():
        return pending
    db = None
    try:
        from hermes_state import SessionDB

        db = SessionDB(db_path=db_path, read_only=True)
        row = db.get_session_by_title("Bot Chat")
        session_id = str((row or {}).get("id") or "")
        if not session_id:
            return pending
        tip = db.get_compression_tip(session_id) or session_id
        messages = db.get_messages(tip, include_compacted=True)
        metadata = _read_metadata()
        installed_at = str((metadata.get("provenance") or {}).get("installed_at") or "")
        try:
            installed_timestamp = datetime.fromisoformat(installed_at.replace("Z", "+00:00")).timestamp()
        except (TypeError, ValueError):
            installed_timestamp = 0.0

        request = next((
            message for message in messages
            if message.get("role") == "user"
            and str(message.get("content") or "").strip() == entry.profile.starter_prompt.strip()
            and float(message.get("timestamp") or 0) >= installed_timestamp
        ), None)
        proof = {**pending, "session_id": tip}
        if request is None:
            return proof
        request_id = int(request["id"])
        proof["request_message_id"] = request_id
        completion = next((
            message for message in messages
            if int(message.get("id") or 0) > request_id
            and message.get("role") == "assistant"
            and message.get("finish_reason") == "stop"
            and str(message.get("content") or "").strip()
            and not message.get("tool_calls")
        ), None)
        if completion is not None:
            proof.update(status="complete", completion_message_id=int(completion["id"]))
        return proof
    except Exception:
        return pending
    finally:
        if db is not None:
            db.close()


def bot_setup_status(entry: BotCatalogEntry, *, runtime: dict[str, Any]) -> dict[str, Any]:
    declared = list(entry.setup.requirements)
    declared_keys = {(item.kind, item.id) for item in declared}
    for toolset in entry.capabilities.toolsets:
        if ("toolset", toolset) not in declared_keys:
            declared.append(BotSetupRequirement(
                kind="toolset", id=toolset, required=True,
                purpose="Required by this bot's reviewed capabilities.",
            ))
    requirements = [_requirement_status(item) for item in declared]
    runtime_ok = bool(runtime.get("ok"))
    first_task = _first_task_proof(entry)
    missing_required = [item for item in requirements if item["required"] and item["status"] != "ready"]
    setup_ready = runtime_ok and not missing_required
    state = "ready" if setup_ready and first_task["status"] == "complete" else "needs_setup"

    from hermes_cli.bot_metadata import mutate_bot_metadata

    def update_metadata(metadata: dict[str, Any]) -> None:
        metadata["setup_state"] = state
        metadata["setup_requirement_status"] = requirements
        metadata["first_task"] = first_task
        if state == "ready" and not metadata.get("setup_completed_at"):
            metadata["setup_completed_at"] = datetime.now(timezone.utc).isoformat()

    mutate_bot_metadata(update_metadata)

    return {
        "profile": _metadata_path().parent.name,
        "catalog_name": entry.name,
        "setup_state": state,
        "requirements": requirements,
        "runtime": {
            "ok": runtime_ok,
            "provider": runtime.get("provider"),
            "model": runtime.get("model"),
            "source": runtime.get("source"),
            "error": runtime.get("error"),
            "action": None if runtime_ok else "model",
            "reused_sign_in": bool(
                runtime_ok
                and runtime.get("provider") in {"openai-codex", "nous", "anthropic", "qwen-oauth"}
                and runtime.get("source") not in {None, "", "no-key-required"}
            ),
        },
        "first_task": {
            "status": first_task["status"],
            "session_id": first_task["session_id"],
        },
        "starter_prompt": None if first_task["status"] == "complete" else entry.profile.starter_prompt,
        "can_start_first_task": setup_ready and first_task["status"] != "complete",
        "can_activate_routines": state == "ready",
    }


def installed_bot_inventory() -> list[dict[str, Any]]:
    """Read live installed bot profiles without touching the catalog, agent runtime, or setup probes."""
    from hermes_cli.profiles import _iter_named_profile_dirs

    bots: list[dict[str, Any]] = []
    for profile_dir in _iter_named_profile_dirs():
        blueprint_path = profile_dir / "bot-blueprint.yaml"
        metadata_path = profile_dir / "profile.yaml"
        if not blueprint_path.is_file() or not metadata_path.is_file():
            continue
        try:
            manifest = yaml.safe_load(blueprint_path.read_text(encoding="utf-8"))
            metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
                continue
            entry = manifest.get("catalog_entry")
            presentation = entry.get("presentation") if isinstance(entry, dict) else None
            if not isinstance(entry, dict) or not isinstance(presentation, dict):
                continue
            catalog_name = str(entry.get("name") or "").strip()
            title = str(entry.get("title") or "").strip()
            summary = str(entry.get("summary") or "").strip()
            emoji = str(presentation.get("emoji") or "").strip()
            color = str(presentation.get("color") or "").strip()
            if not all((catalog_name, title, summary, emoji, color)):
                continue
            setup_state = str((metadata if isinstance(metadata, dict) else {}).get("setup_state") or "needs_setup")
            if setup_state not in {"ready", "needs_setup"}:
                setup_state = "needs_setup"
            bots.append({
                "profile": profile_dir.name,
                "catalog_name": catalog_name,
                "title": title,
                "summary": summary,
                "setup_state": setup_state,
                "presentation": {"emoji": emoji, "color": color},
            })
        except (OSError, UnicodeError, yaml.YAMLError):
            continue
    return bots


def installed_catalog_entry() -> BotCatalogEntry:
    """Load the immutable reviewed blueprint copied into this profile at install time."""
    path = _metadata_path().with_name("bot-blueprint.yaml")
    if not path.is_file():
        raise ValueError("installed Bot Marketplace blueprint is missing; reinstall this bot profile")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    manifest = raw if isinstance(raw, dict) else {}
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("catalog_entry"), dict):
        raise ValueError("installed Bot Marketplace blueprint is invalid")
    return BotCatalogEntry.model_validate(manifest["catalog_entry"])

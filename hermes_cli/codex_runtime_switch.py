"""Shared logic for the /codex-runtime slash command.

Authoritative runtime state lives under ``codex_runtime.*``.
``model.openai_runtime`` is legacy compatibility only and is neutralized by
rollback so old values do not continue to enable Codex app-server.

Both CLI (cli.py) and gateway (gateway/run.py) call into this module so the
behavior stays identical across surfaces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

VALID_RUNTIMES = ("auto", "codex_app_server")
DEFAULT_CODEX_RUNTIME = {"enabled": False, "mode": "responses_only", "allow_runtime_tools": []}
DEFAULT_RUNTIME_TOOLS: list[str] = []
PROTECTED_RUNTIME_TOOLS = {
    "memory", "session_search", "send_message", "cronjob",
    "clarify", "todo", "delegate_task",
}


@dataclass
class CodexRuntimeStatus:
    """Result of a /codex-runtime invocation. Callers render this however
    suits their surface (CLI uses Rich panels, gateway sends a text message)."""

    success: bool
    new_value: Optional[str] = None
    old_value: Optional[str] = None
    message: str = ""
    requires_new_session: bool = False
    codex_binary_ok: bool = True
    codex_version: Optional[str] = None


def parse_args(arg_string: str) -> tuple[Optional[str], list[str]]:
    """Parse the slash-command argument string. Returns (value, errors).

    No args         → return current state (value=None)
    'auto' / 'codex_app_server' / 'on' / 'off' → return that value
    anything else   → error
    """
    raw = (arg_string or "").strip().lower()
    if not raw:
        return None, []
    if raw in {"on", "codex", "enable"}:
        return "codex_app_server", []
    if raw in {"off", "default", "disable", "hermes", "rollback"}:
        return "auto", []
    if raw in VALID_RUNTIMES:
        return raw, []
    return None, [
        f"Unknown runtime {raw!r}. Use one of: auto, codex_app_server, on, off"
    ]


def _sanitize_allowlist(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        name = item.strip()
        if name and name not in PROTECTED_RUNTIME_TOOLS and name not in out:
            out.append(name)
    return out


def _get_codex_runtime(config: dict) -> dict:
    runtime = config.get("codex_runtime") if isinstance(config, dict) else None
    if not isinstance(runtime, dict):
        return dict(DEFAULT_CODEX_RUNTIME)
    enabled = runtime.get("enabled") is True
    mode = str(runtime.get("mode") or "responses_only").strip().lower()
    if mode not in {"responses_only", "app_server"}:
        mode = "responses_only"
    return {
        "enabled": enabled,
        "mode": mode,
        "allow_runtime_tools": _sanitize_allowlist(runtime.get("allow_runtime_tools")),
    }


def get_current_runtime(config: dict) -> str:
    """Return active runtime from the authoritative codex_runtime gate.

    Legacy ``model.openai_runtime`` is intentionally not authoritative.
    """
    runtime = _get_codex_runtime(config)
    if runtime.get("enabled") is True and runtime.get("mode") == "app_server":
        return "codex_app_server"
    return "auto"


def _disable_codex_runtime(config: dict) -> None:
    config["codex_runtime"] = dict(DEFAULT_CODEX_RUNTIME)
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        model_cfg.pop("openai_runtime", None)


def _enable_codex_runtime(config: dict) -> None:
    config["codex_runtime"] = {
        "enabled": True,
        "mode": "app_server",
        "allow_runtime_tools": list(DEFAULT_RUNTIME_TOOLS),
    }
    if not isinstance(config.get("model"), dict):
        config["model"] = {}
    # Compatibility marker only. Runtime resolution still requires codex_runtime.*.
    config["model"]["openai_runtime"] = "codex_app_server"


def set_runtime(config: dict, new_value: str) -> str:
    """Mutate the config dict in place. Returns the previous active value."""
    if new_value not in VALID_RUNTIMES:
        raise ValueError(
            f"invalid runtime {new_value!r}; must be one of {VALID_RUNTIMES}"
        )
    old = get_current_runtime(config)
    if new_value == "codex_app_server":
        _enable_codex_runtime(config)
    else:
        _disable_codex_runtime(config)
    return old


def check_codex_binary_ok() -> tuple[bool, Optional[str]]:
    """Best-effort verification that codex CLI is installed at acceptable
    version. Returns (ok, version_or_message)."""
    try:
        from agent.transports.codex_app_server import check_codex_binary

        return check_codex_binary()
    except Exception as exc:  # pragma: no cover
        return False, f"codex check failed: {exc}"


def _runtime_status_lines(config: dict) -> list[str]:
    runtime = _get_codex_runtime(config)
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    provider = str(model_cfg.get("provider") or "unknown").strip() or "unknown"
    model = str(model_cfg.get("default") or model_cfg.get("model") or "unknown").strip() or "unknown"
    allowed = runtime.get("allow_runtime_tools") or []
    gate_open = runtime.get("enabled") is True and runtime.get("mode") == "app_server"
    active_mode = "codex_app_server" if gate_open else "normal_tool_loop"
    legacy = model_cfg.get("openai_runtime") if isinstance(model_cfg, dict) else None
    return [
        f"provider/model: {provider} / {model}",
        f"active execution mode: {active_mode}",
        f"gate state: codex_runtime.enabled={runtime.get('enabled') is True} mode={runtime.get('mode')}",
        f"allowed runtime tools: {', '.join(allowed) if allowed else '(none)'}",
        f"legacy openai_runtime: {legacy!r}",
    ]


def apply(
    config: dict,
    new_value: Optional[str],
    *,
    persist_callback=None,
) -> CodexRuntimeStatus:
    """Top-level entry point used by both CLI and gateway handlers."""
    current = get_current_runtime(config)
    _binary_check: Optional[tuple[bool, Optional[str]]] = None

    def _check_binary_cached() -> tuple[bool, Optional[str]]:
        nonlocal _binary_check
        if _binary_check is None:
            _binary_check = check_codex_binary_ok()
        return _binary_check

    if new_value is None:
        ok, ver = _check_binary_cached()
        msg = "\n".join(
            [
                *_runtime_status_lines(config),
                f"codex CLI: {'OK ' + ver if ok else 'not available — ' + (ver or 'install with `npm i -g @openai/codex`')}",
            ]
        )
        return CodexRuntimeStatus(
            success=True,
            new_value=current,
            old_value=current,
            message=msg,
            codex_binary_ok=ok,
            codex_version=ver if ok else None,
        )

    if new_value == current:
        return CodexRuntimeStatus(
            success=True,
            new_value=current,
            old_value=current,
            message=f"codex_runtime already set to {current}",
        )

    if new_value == "codex_app_server":
        ok, ver_or_msg = _check_binary_cached()
        if not ok:
            return CodexRuntimeStatus(
                success=False,
                new_value=None,
                old_value=current,
                message="\n".join([
                    "Cannot enable codex_app_server runtime:",
                    f"{ver_or_msg or 'codex CLI not available'}",
                    "Install with: npm i -g @openai/codex",
                ]),
                codex_binary_ok=False,
                codex_version=None,
            )

    set_runtime(config, new_value)
    if persist_callback is not None:
        try:
            persist_callback(config)
        except Exception as exc:
            logger.exception("failed to persist codex_runtime change")
            return CodexRuntimeStatus(
                success=False,
                new_value=new_value,
                old_value=current,
                message=f"updated config in memory but persist failed: {exc}",
            )

    msg_lines = [f"openai_runtime: {current} → {new_value}", *_runtime_status_lines(config)]
    if new_value == "codex_app_server":
        ok, ver = _check_binary_cached()
        if ok:
            msg_lines.append(f"codex CLI: {ver}")
        try:
            from hermes_cli.codex_runtime_plugin_migration import migrate
            runtime_cfg = _get_codex_runtime(config)
            mig_report = migrate(config)
            user_servers = [s for s in mig_report.migrated if s != "hermes-tools"]
            if user_servers:
                msg_lines.append(
                    f"Migrated {len(user_servers)} MCP server(s): {', '.join(user_servers)}"
                )
            if mig_report.migrated_plugins:
                msg_lines.append(
                    f"Migrated {len(mig_report.migrated_plugins)} native Codex plugin(s): "
                    f"{', '.join(mig_report.migrated_plugins)}"
                )
            elif mig_report.plugin_query_error:
                msg_lines.append(f"Codex plugin discovery skipped: {mig_report.plugin_query_error}")
            if mig_report.wrote_permissions_default:
                msg_lines.append(
                    f"Default sandbox: {mig_report.wrote_permissions_default} (no approval prompt on every write)"
                )
            if "hermes-tools" in mig_report.migrated:
                allowed = runtime_cfg.get("allow_runtime_tools") or []
                if allowed:
                    msg_lines.append(
                        "Hermes tool callback registered for allowed tools: " + ", ".join(allowed)
                    )
                else:
                    msg_lines.append("Hermes tool callback registered with no Hermes tools exposed by default.")
                msg_lines.append(
                    "  (delegate_task, memory, session_search, send_message, cronjob, clarify, and todo stay on the default Hermes runtime.)"
                )
            msg_lines.append(f"  (config: {mig_report.target_path})")
            for err in mig_report.errors:
                msg_lines.append(f"⚠ MCP migration: {err}")
        except Exception as exc:
            msg_lines.append(f"⚠ MCP migration skipped: {exc}")
        msg_lines.append(
            "Codex app-server is enabled only for codex_runtime.enabled=True and codex_runtime.mode=app_server."
        )
        msg_lines.append(
            "Effective on next session — current cached agent keeps the prior runtime to preserve prompt cache."
        )
    else:
        msg_lines.append("OpenAI/Codex turns will use the default Hermes runtime.")
        msg_lines.append("Effective on next session.")

    return CodexRuntimeStatus(
        success=True,
        new_value=new_value,
        old_value=current,
        message="\n".join(msg_lines),
        requires_new_session=True,
        codex_binary_ok=True,
        codex_version=_binary_check[1] if _binary_check and _binary_check[0] else None,
    )

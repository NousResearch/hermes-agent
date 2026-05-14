"""Shared logic for the /codex-runtime slash command.

Toggles `model.openai_runtime` between "auto" (= chat_completions, Hermes'
default) and "codex_app_server" (= hand turns to a codex subprocess).

Both CLI (cli.py) and gateway (gateway/run.py) call into this module so the
behavior stays identical across surfaces.

The actual runtime resolution happens in hermes_cli.runtime_provider's
_maybe_apply_codex_app_server_runtime() helper, which reads the persisted
config value. This module just persists the value and reports the change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


VALID_RUNTIMES = ("auto", "codex_app_server")


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
    # Accept human-friendly synonyms
    if raw in ("on", "codex", "enable"):
        return "codex_app_server", []
    if raw in ("off", "default", "disable", "hermes"):
        return "auto", []
    if raw in VALID_RUNTIMES:
        return raw, []
    return None, [
        f"Unknown runtime {raw!r}. Use one of: auto, codex_app_server, on, off"
    ]


def get_current_runtime(config: dict) -> str:
    """Read the current `model.openai_runtime` value from a config dict.
    Returns 'auto' for unset / empty / unrecognized values."""
    if not isinstance(config, dict):
        return "auto"
    model_cfg = config.get("model") or {}
    if not isinstance(model_cfg, dict):
        return "auto"
    value = str(model_cfg.get("openai_runtime") or "").strip().lower()
    if value in VALID_RUNTIMES:
        return value
    return "auto"


def set_runtime(config: dict, new_value: str) -> str:
    """Mutate the config dict in place to persist the new runtime value.
    Returns the previous value for callers that want to report a delta."""
    if new_value not in VALID_RUNTIMES:
        raise ValueError(
            f"invalid runtime {new_value!r}; must be one of {VALID_RUNTIMES}"
        )
    old = get_current_runtime(config)
    if not isinstance(config.get("model"), dict):
        config["model"] = {}
    config["model"]["openai_runtime"] = new_value
    return old


def check_codex_binary_ok() -> tuple[bool, Optional[str]]:
    """Best-effort verification that codex CLI is installed at acceptable
    version. Returns (ok, version_or_message)."""
    try:
        from agent.transports.codex_app_server import check_codex_binary

        return check_codex_binary()
    except Exception as exc:  # pragma: no cover
        return False, f"codex check failed: {exc}"


def apply(
    config: dict,
    new_value: Optional[str],
    *,
    persist_callback=None,
) -> CodexRuntimeStatus:
    """Top-level entry point used by both CLI and gateway handlers.

    Args:
        config: in-memory config dict (will be mutated when new_value is set)
        new_value: desired runtime; None means "show current state only"
        persist_callback: optional callable taking the mutated config dict
            and persisting it to disk. Skipped when None (used by tests).

    Returns: CodexRuntimeStatus describing the outcome.
    """
    current = get_current_runtime(config)

    # Cache the codex binary check for this apply() call. Subprocess spawn
    # is cheap (~50ms for `codex --version`), but we'd otherwise call it up
    # to 3 times in the enable path (read-only/state, gate, success message).
    # None = not yet checked; (bool, str) = result.
    _binary_check: Optional[tuple[bool, Optional[str]]] = None

    def _check_binary_cached() -> tuple[bool, Optional[str]]:
        nonlocal _binary_check
        if _binary_check is None:
            _binary_check = check_codex_binary_ok()
        return _binary_check

    # Read-only call: just report state
    if new_value is None:
        ok, ver = _check_binary_cached()
        msg = (
            f"openai_runtime: {current}\n"
            f"codex CLI: {'OK ' + ver if ok else 'not available — ' + (ver or 'install with `npm i -g @openai/codex`')}"
        )
        return CodexRuntimeStatus(
            success=True,
            new_value=current,
            old_value=current,
            message=msg,
            codex_binary_ok=ok,
            codex_version=ver if ok else None,
        )

    # No change requested
    if new_value == current:
        return CodexRuntimeStatus(
            success=True,
            new_value=current,
            old_value=current,
            message=f"openai_runtime already set to {current}",
        )

    # If switching ON, verify codex CLI is installed before persisting —
    # an opt-in toggle that silently fails on the first turn is the
    # worst possible UX. Block here with a clear install hint.
    if new_value == "codex_app_server":
        ok, ver_or_msg = _check_binary_cached()
        if not ok:
            return CodexRuntimeStatus(
                success=False,
                new_value=None,
                old_value=current,
                message=(
                    "Cannot enable codex_app_server runtime: "
                    f"{ver_or_msg or 'codex CLI not available'}\n"
                    "Install with: npm i -g @openai/codex"
                ),
                codex_binary_ok=False,
                codex_version=None,
            )

    set_runtime(config, new_value)
    if persist_callback is not None:
        try:
            persist_callback(config)
        except Exception as exc:
            logger.exception("failed to persist openai_runtime change")
            return CodexRuntimeStatus(
                success=False,
                new_value=new_value,
                old_value=current,
                message=f"updated config in memory but persist failed: {exc}",
            )

    msg_lines = [
        f"openai_runtime: {current} → {new_value}",
    ]
    if new_value == "codex_app_server":
        ok, ver = _check_binary_cached()
        if ok:
            msg_lines.append(f"codex CLI: {ver}")
        # Auto-migrate Hermes' MCP servers + Codex's installed curated
        # plugins into ~/.codex/config.toml so the spawned codex subprocess
        # sees the same tool surface AND can call back into Hermes for
        # browser/web/delegate_task/vision/memory tools (#7 fix).
        # Failures are non-fatal — the runtime change still proceeds.
        try:
            from hermes_cli.codex_runtime_plugin_migration import migrate
            mig_report = migrate(config)
            # Tools/MCP servers (excluding the hermes-tools callback,
            # which is internal plumbing — surface separately).
            user_servers = [
                s for s in mig_report.migrated if s != "hermes-tools"
            ]
            if user_servers:
                msg_lines.append(
                    f"Migrated {len(user_servers)} MCP server(s): "
                    f"{', '.join(user_servers)}"
                )
            # Native Codex plugin migration (Linear, GitHub, etc.)
            if mig_report.migrated_plugins:
                msg_lines.append(
                    f"Migrated {len(mig_report.migrated_plugins)} native "
                    f"Codex plugin(s): {', '.join(mig_report.migrated_plugins)}"
                )
            elif mig_report.plugin_query_error:
                msg_lines.append(
                    f"Codex plugin discovery skipped: "
                    f"{mig_report.plugin_query_error}"
                )
            # Permissions + Hermes tool callback are always-on production
            # bits the user benefits from knowing about.
            if mig_report.wrote_permissions_default:
                msg_lines.append(
                    f"Default sandbox: {mig_report.wrote_permissions_default} "
                    f"(no approval prompt on every write)"
                )
            if "hermes-tools" in mig_report.migrated:
                msg_lines.append(
                    "Hermes tool callback registered: codex can now use "
                    "web_search, web_extract, browser_*, vision_analyze, "
                    "image_generate, skill_view, skills_list, text_to_speech, "
                    "kanban_* (worker + orchestrator) via MCP."
                )
                msg_lines.append(
                    "  (delegate_task, memory, session_search, todo run "
                    "only on the default Hermes runtime — they need the "
                    "agent loop context.)"
                )
            msg_lines.append(f"  (config: {mig_report.target_path})")
            for err in mig_report.errors:
                msg_lines.append(f"⚠ MCP migration: {err}")
        except Exception as exc:
            msg_lines.append(f"⚠ MCP migration skipped: {exc}")
        msg_lines.append(
            "OpenAI/Codex turns now run through `codex app-server` "
            "(terminal/file ops/patching inside Codex; "
            "Hermes tools available via MCP callback)."
        )
        msg_lines.append(
            "Effective on next session — current cached agent keeps "
            "the prior runtime to preserve prompt cache."
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
    )

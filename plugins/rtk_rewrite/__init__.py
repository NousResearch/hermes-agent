"""Built-in RTK rewrite plugin for Hermes.

Rewrites terminal tool commands through `rtk rewrite` before execution so the
agent receives RTK-compressed output when available.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_RTK_SETTINGS = {
    "enabled": "auto",
    "binary": "rtk",
    "rewrite_timeout_seconds": 2,
    "log_rewrites": False,
}

_rtk_available_cache: dict[str, bool] = {}


def _normalize_enabled(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    normalized = str(value or "auto").strip().lower()
    return normalized if normalized in {"auto", "true", "false"} else "auto"


def _coerce_timeout_seconds(value: Any) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return float(DEFAULT_RTK_SETTINGS["rewrite_timeout_seconds"])
    return timeout if timeout > 0 else float(DEFAULT_RTK_SETTINGS["rewrite_timeout_seconds"])


def _load_rtk_settings() -> dict[str, Any]:
    settings = dict(DEFAULT_RTK_SETTINGS)
    try:
        from hermes_cli.config import load_config

        config = load_config()
        terminal = config.get("terminal", {}) if isinstance(config, dict) else {}
        rtk = terminal.get("rtk", {}) if isinstance(terminal, dict) else {}
        if isinstance(rtk, dict):
            settings.update(rtk)
    except Exception:
        pass

    settings["enabled"] = _normalize_enabled(settings.get("enabled"))
    settings["binary"] = str(settings.get("binary") or DEFAULT_RTK_SETTINGS["binary"]).strip() or "rtk"
    settings["rewrite_timeout_seconds"] = _coerce_timeout_seconds(settings.get("rewrite_timeout_seconds"))
    settings["log_rewrites"] = bool(settings.get("log_rewrites", False))
    return settings


def _check_rtk(binary: str) -> bool:
    cached = _rtk_available_cache.get(binary)
    if cached is not None:
        return cached

    if os.sep in binary or (os.altsep and os.altsep in binary):
        available = Path(binary).expanduser().exists()
    else:
        available = shutil.which(binary) is not None

    _rtk_available_cache[binary] = available
    return available


def _try_rewrite(command: str, *, binary: str, timeout_seconds: float) -> Optional[str]:
    try:
        result = subprocess.run(
            [binary, "rewrite", command],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None

    rewritten = result.stdout.strip()
    if result.returncode == 0 and rewritten and rewritten != command:
        return rewritten
    return None


def _pre_tool_call(*, tool_name: str, args: dict[str, Any], **_kwargs) -> None:
    if tool_name != "terminal":
        return

    command = args.get("command")
    if not isinstance(command, str) or not command:
        return

    settings = _load_rtk_settings()
    if settings["enabled"] == "false":
        return

    binary = settings["binary"]
    if not _check_rtk(binary):
        return

    rewritten = _try_rewrite(
        command,
        binary=binary,
        timeout_seconds=settings["rewrite_timeout_seconds"],
    )
    if not rewritten:
        return

    if settings["log_rewrites"]:
        logger.info("[rtk] %s -> %s", command, rewritten)
    else:
        logger.debug("[rtk] %s -> %s", command, rewritten)
    args["command"] = rewritten


def register(ctx) -> None:
    settings = _load_rtk_settings()
    enabled = settings["enabled"]
    binary = settings["binary"]

    if enabled == "false":
        logger.debug("[rtk] disabled via terminal.rtk.enabled=false")
        return

    if not _check_rtk(binary):
        if enabled == "true":
            logger.warning("[rtk] terminal.rtk.enabled=true but binary '%s' was not found in PATH", binary)
        return

    ctx.register_hook("pre_tool_call", _pre_tool_call)
    logger.info("[rtk] rewrite hook registered using binary '%s'", binary)

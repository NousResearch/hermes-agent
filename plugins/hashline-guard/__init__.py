"""hashline-guard plugin: strict-match pre-check on patch `old_string` anchors.

Registers a `pre_tool_call` hook that blocks stale or ambiguous patch anchors
before the patch is applied. Fail-open on any unexpected error.

Note: the plugin directory contains a hyphen and cannot be a valid Python
package name, so hashline_core is loaded via importlib.util rather than a
relative import.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_CORE_PATH = Path(__file__).parent / "src" / "hashline_core.py"
_core_spec = importlib.util.spec_from_file_location("hashline_core", _CORE_PATH)
_core_mod = importlib.util.module_from_spec(_core_spec)
sys.modules.setdefault("hashline_core", _core_mod)
_core_spec.loader.exec_module(_core_mod)

verify_anchor = _core_mod.verify_anchor


def on_pre_tool_call(*args: Any, **kwargs: Any) -> Optional[Dict[str, str]]:
    """Pre-tool hook: block stale/ambiguous patch anchors.

    Returns ``{'action': 'block', 'message': ...}`` to stop the tool call,
    or ``None`` to allow it through.

    Hermes invokes this callback with ``tool_name`` and ``args`` kwargs
    (see ``hermes_cli/plugins.py`` ``_get_pre_tool_call_directive_details``).
    """
    tool_name = kwargs.get("tool_name", "")
    if tool_name != "patch":
        return None

    args = kwargs.get("args") or {}
    mode = args.get("mode", "")
    if mode != "replace":
        # V4A multi-file / traversal patches have their own checks.
        return None

    target_rel = args.get("path")
    old_string = args.get("old_string")
    if not target_rel or old_string is None:
        return None

    # Resolve target path relative to cwd when not absolute.
    cwd = kwargs.get("cwd") or str(Path.cwd())
    target = Path(target_rel)
    if not target.is_absolute():
        target = Path(cwd) / target

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return {
            "action": "block",
            "message": f"hashline-guard: file not found: {target}. Re-read the file and re-issue the patch.",
        }
    except Exception as exc:
        logger.debug("hashline-guard read error (%s): %s", target, exc)
        return None  # fail open

    status, reason = verify_anchor(text, old_string)
    if status == "block":
        return {
            "action": "block",
            "message": f"hashline-guard: {reason} in {target}. Re-read the file and re-issue the patch.",
        }
    return None


def register(ctx: Any) -> None:
    """Register the pre_tool_call hook with the Hermes plugin context."""
    ctx.register_hook("pre_tool_call", on_pre_tool_call)

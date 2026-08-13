"""hashline-guard plugin: strict-match pre-check on patch `old_string` anchors,
plus content-addressed anchored patching via `anchored_patch`.

Registers:
- pre_tool_call hook that blocks stale/ambiguous patch anchors
- anchored_patch tool: atomic read -> verify -> apply -> write
- hashline_compute tool: discover expected hashline values

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
verify_anchor_by_hash = _core_mod.verify_anchor_by_hash
find_all = _core_mod.find_all
compute_hashline = _core_mod.compute_hashline
context_hash = _core_mod.context_hash

SCHEMA = {
    "path": {"type": "string"},
    "old_string": {"type": "string"},
    "new_string": {"type": "string"},
    "expected_hashline": {"type": "string"},
    "window": {"type": "integer", "default": 2},
}


def handle_anchored_patch(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        path = args["path"]
        old_string = args["old_string"]
        new_string = args["new_string"]
        expected_hashline = args["expected_hashline"]
        window = int(args.get("window", 2))
    except Exception as exc:
        logger.debug("anchored_patch arg error: %s", exc)
        return _tool_error(f"anchored_patch: invalid args: {exc}", code="500", details={})

    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.debug("anchored_patch read error (%s): %s", path, exc)
        return _tool_error(f"anchored_patch: read failed: {path}", code="500", details={"error": str(exc)})

    status, payload = verify_anchor_by_hash(text, old_string, expected_hashline, window=window)
    if status == "block":
        logger.debug("anchored_patch blocked: %s", payload.get("reason"))
        return _tool_error(
            f"anchored_patch: {payload.get('reason')} in {path}",
            code="422",
            details={"found": payload.get("found", []), "lines": payload.get("lines", [])},
        )

    occurrence = payload
    matches = find_all(text, old_string)
    if occurrence < 0 or occurrence >= len(matches):
        return _tool_error("anchored_patch: occurrence index out of range", code="500", details={})

    start = matches[occurrence][0]
    end = matches[occurrence][1]
    updated = text[:start] + new_string + text[end:]

    try:
        with Path(path).open("w", encoding="utf-8", newline="") as fh:
            fh.write(updated)
    except Exception as exc:
        logger.debug("anchored_patch write error (%s): %s", path, exc)
        return _tool_error(f"anchored_patch: write failed: {path}", code="500", details={"error": str(exc)})

    return {"applied": True, "occurrence": occurrence, "hashline": expected_hashline}


def _tool_error(message: str, code: str = "500", details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "applied": False,
        "error": {"code": code, "message": message, "details": details or {}},
    }


def _context_snippet(file_text: str, old_string: str, occurrence_index: int, window: int = 2) -> str:
    matches = find_all(file_text, old_string)
    if not matches or occurrence_index < 0 or occurrence_index >= len(matches):
        return ""
    text = _core_mod._canonicalize(file_text)
    lines = text.splitlines()
    anchor_line = matches[occurrence_index][2] - 1
    start = max(0, anchor_line - window)
    end = min(len(lines), anchor_line + old_string.count("\n") + 1 + window)
    return "\n".join(lines[start:end])


def hashline_compute(args: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    try:
        path = args.get("path")
        old_string = args.get("old_string")
        window = int(args.get("window", 2))
        if not path or old_string is None:
            return _tool_error("hashline_compute requires path and old_string", code="500", details={})

        text = Path(path).read_text(encoding="utf-8", errors="replace")
        matches = find_all(text, old_string)
        if not matches:
            return {"hashlines": [], "count": 0}

        out = []
        for idx in range(len(matches)):
            line = matches[idx][2]
            hl = compute_hashline(text, old_string, idx, window=window)
            out.append({
                "line": line,
                "hashline": hl,
                "context": _context_snippet(text, old_string, idx, window=window),
            })
        return {"hashlines": out, "count": len(out)}
    except Exception as exc:
        logger.debug("hashline_compute failed: %s", exc)
        return _tool_error(f"hashline_compute failed: {type(exc).__name__}: {exc}", code="500", details={})


def on_pre_tool_call(*args: Any, **kwargs: Any) -> Optional[Dict[str, str]]:
    """Pre-tool hook: block stale/ambiguous patch anchors, and drift when expected_hashline is pinned.

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
        return None

    target_rel = args.get("path")
    old_string = args.get("old_string")
    if not target_rel or old_string is None:
        return None

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
        return None

    expected_hashline = args.get("expected_hashline")
    if expected_hashline is not None:
        status, payload = verify_anchor_by_hash(text, old_string, expected_hashline)
        if status == "ok":
            return None
        if isinstance(payload, dict) and "found" in payload and payload["found"]:
            actual = payload["found"][0]
            return {
                "action": "block",
                "message": (
                    f"hashline-guard: expected_hashline drifted in {target}. "
                    f"Actual hashline for the only occurrence: {actual}. "
                    "Re-pin expected_hashline to this value and retry."
                ),
            }
        return {
            "action": "block",
            "message": f"hashline-guard: {payload.get('reason', 'expected_hashline check failed')} in {target}.",
        }

    status, reason = verify_anchor(text, old_string)
    if status == "block":
        return {
            "action": "block",
            "message": f"hashline-guard: {reason} in {target}. Re-read the file and re-issue the patch.",
        }
    return None


def register(ctx: Any) -> None:
    """Register hooks/tools with the Hermes plugin context."""
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_tool(
        name="anchored_patch",
        toolset="file",
        schema=SCHEMA,
        handler=handle_anchored_patch,
    )
    ctx.register_tool(
        name="hashline_compute",
        toolset="hashline-guard",
        schema={
            "name": "hashline_compute",
            "description": "Return expected hashline values for every occurrence of an anchor in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Target file path."},
                    "old_string": {"type": "string", "description": "Anchor text to hash."},
                    "window": {"type": "integer", "description": "Surrounding context lines.", "default": 2},
                },
                "required": ["path", "old_string"],
            },
        },
        handler=hashline_compute,
        check_fn=lambda: True,
        description="Discover expected_hashline values for patch anchors.",
    )

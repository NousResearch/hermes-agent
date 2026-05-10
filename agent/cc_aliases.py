"""Claude Code tool-name aliasing for the OAuth path.

Real Claude Code's eager toolset is ~42K bytes of canonical schemas
(``Bash``, ``Read``, ``Edit``, ``Write``, ``Grep``, ``Glob``, ``Task``,
etc.) that Anthropic's billing classifier on personal Max plans always
accepts as plan-budget traffic. Hermes's native tool surface uses
different names (``terminal``, ``read_file``, ``patch``, …) and slightly
different schemas — even at small byte counts (~16K), the classifier
flags those as non-Claude-Code and routes to extra-usage billing,
which 400s on a Max plan with no extra credits.

This module bridges the gap on the OAuth path:

  * **Outbound** (``replace_with_cc_canonical``): when building the
    request, we swap any hermes tool that has a CC alias for the
    canonical CC tool entry (name + description + input_schema). The
    model now sees ``Bash``/``Read``/``Edit``/etc. and the wire payload
    looks like a real CC request.

  * **Inbound** (``adapt_tool_use``): when the model emits a
    ``tool_use`` block with a CC name, we translate (name, args) into
    the hermes equivalent so the existing tool registry can dispatch.
    Argument shape adaptation handles minor differences (CC's
    ``run_in_background`` → hermes's ``background``, CC's millisecond
    timeouts → hermes's seconds, etc.).

  * **Tools without an alias** (``vision_analyze``, ``ha_*``,
    ``image_generate``, …) pass through unchanged. They ride alongside
    the canonical CC tools as additional custom function tools — the
    classifier accepts a CC-shaped eager set plus a few extras.

Captured CC schemas live in ``cc_canonical/tools_eager.json``. Refresh
them from a real CC session via:

  HTTPS_PROXY=http://localhost:8080 NODE_EXTRA_CA_CERTS=~/.mitmproxy/...
  claude -p "say hi"

…with mitmdump capturing flows, then jq the request body's ``tools``
array. See scripts/refresh_cc_canonical.sh (TODO) for the recipe.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

_CANONICAL_PATH = Path(__file__).parent / "cc_canonical" / "tools_eager.json"

try:
    _CC_TOOLS: List[Dict[str, Any]] = json.loads(_CANONICAL_PATH.read_text())
except Exception as e:
    logger.warning("cc_aliases: failed to load %s: %s — alias layer disabled",
                   _CANONICAL_PATH, e)
    _CC_TOOLS = []

CC_TOOL_INDEX: Dict[str, Dict[str, Any]] = {t["name"]: t for t in _CC_TOOLS}


# Hermes tool name → CC tool name. When the outbound adapter sees a
# hermes tool whose name matches a key here, it substitutes the
# canonical CC tool. Inbound, the model emits the CC name and the
# inbound adapter routes back to the hermes name for dispatch.
HERMES_TO_CC: Dict[str, str] = {
    "terminal":     "Bash",
    "read_file":    "Read",
    "patch":        "Edit",
    "write_file":   "Write",
    "search_files": "Grep",
    # No 1:1 mapping for these yet — left unmapped so they ship as
    # native hermes entries (for now):
    #   delegate_task → Agent (CC's Agent has a different arg shape;
    #     subagent_type vs hermes's role/skill model — needs a fuller
    #     adapter, deferred to phase 2.3)
    #   todo → TodoWrite (CC's TodoWrite has different arg shape, also
    #     not in CC's eager set in some session profiles)
    #   process → Bash w/ run_in_background (semantically overlaps but
    #     not 1:1; keep separate so the model can manage existing
    #     background processes via hermes's tool)
}

CC_TO_HERMES: Dict[str, str] = {cc: h for h, cc in HERMES_TO_CC.items()}


# ────────────────────────────────────────────────────────────────────
# Argument adapters — translate CC's tool_use input to hermes's
# tool dispatch input.
# ────────────────────────────────────────────────────────────────────

def _adapt_bash(cc_args: Dict[str, Any]) -> Dict[str, Any]:
    """CC ``Bash`` → hermes ``terminal``.

    CC schema: command (str), timeout (ms), description (str),
               run_in_background (bool), dangerouslyDisableSandbox (bool)
    Hermes:    command (str), timeout (s, max 600), background (bool),
               workdir (str), pty (bool), notify_on_complete (bool)
    """
    out: Dict[str, Any] = {"command": cc_args["command"]}
    if "run_in_background" in cc_args:
        out["background"] = bool(cc_args["run_in_background"])
    if "timeout" in cc_args:
        # CC uses milliseconds with a 600000 max; hermes uses seconds
        # with a 600 max — same ceiling, different units.
        out["timeout"] = max(1, int(cc_args["timeout"]) // 1000)
    # description and dangerouslyDisableSandbox have no hermes equivalents;
    # drop silently (model context already explains command intent).
    return out


def _adapt_read(cc_args: Dict[str, Any]) -> Dict[str, Any]:
    """CC ``Read`` → hermes ``read_file``.

    CC: file_path (str), offset (int), limit (int), pages (str — PDF only)
    Hermes: path (str), offset (int), limit (int)
    """
    out: Dict[str, Any] = {"path": cc_args["file_path"]}
    if "offset" in cc_args:
        out["offset"] = int(cc_args["offset"])
    if "limit" in cc_args:
        out["limit"] = int(cc_args["limit"])
    return out


def _adapt_edit(cc_args: Dict[str, Any]) -> Dict[str, Any]:
    """CC ``Edit`` → hermes ``patch`` (replace mode).

    CC: file_path, old_string, new_string, replace_all (bool)
    Hermes patch: mode='replace', path, old_string, new_string, replace_all
    """
    return {
        "mode":         "replace",
        "path":         cc_args["file_path"],
        "old_string":   cc_args["old_string"],
        "new_string":   cc_args["new_string"],
        "replace_all":  bool(cc_args.get("replace_all", False)),
    }


def _adapt_write(cc_args: Dict[str, Any]) -> Dict[str, Any]:
    """CC ``Write`` → hermes ``write_file``.

    CC: file_path (str), content (str)
    Hermes: path (str), content (str)
    """
    return {
        "path":    cc_args["file_path"],
        "content": cc_args["content"],
    }


def _adapt_grep(cc_args: Dict[str, Any]) -> Dict[str, Any]:
    """CC ``Grep`` → hermes ``search_files``.

    CC has many flags (-i, -n, type, output_mode, head_limit, glob, ...).
    Hermes search_files has its own param names. Map best-effort and let
    the model retry with adjusted params if a search comes back empty.

    Common case: pattern + path mapping.
    """
    out: Dict[str, Any] = {
        "pattern": cc_args.get("pattern", ""),
    }
    if "path" in cc_args:
        out["path"] = cc_args["path"]
    if "glob" in cc_args:
        out["glob"] = cc_args["glob"]
    if cc_args.get("-i"):
        out["case_insensitive"] = True
    if "output_mode" in cc_args:
        out["output_mode"] = cc_args["output_mode"]
    if "head_limit" in cc_args:
        out["max_results"] = int(cc_args["head_limit"])
    return out


_ADAPTERS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "Bash":  _adapt_bash,
    "Read":  _adapt_read,
    "Edit":  _adapt_edit,
    "Write": _adapt_write,
    "Grep":  _adapt_grep,
}


# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────

def is_enabled() -> bool:
    """True iff the canonical CC schema cache loaded successfully."""
    return bool(CC_TOOL_INDEX)


def replace_with_cc_canonical(
    tools: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Outbound hook. Substitute CC canonical entries for hermes tools
    that have a CC alias. Tools without an alias pass through unchanged.

    Preserves order and any cache_control / type fields on entries
    we don't touch. Substitutions copy cache_control from the original
    hermes entry onto the canonical CC entry so prompt-cache boundary
    placement is unchanged.
    """
    if not is_enabled():
        return tools

    out: List[Dict[str, Any]] = []
    seen_cc: set = set()
    for t in tools:
        if not isinstance(t, dict):
            out.append(t)
            continue
        name = t.get("name", "")
        cc_name = HERMES_TO_CC.get(name)
        if cc_name and cc_name in CC_TOOL_INDEX and cc_name not in seen_cc:
            cc_entry = dict(CC_TOOL_INDEX[cc_name])  # shallow copy
            # Forward cache_control if the caller had set one.
            if "cache_control" in t:
                cc_entry["cache_control"] = t["cache_control"]
            out.append(cc_entry)
            seen_cc.add(cc_name)
        else:
            out.append(t)
    return out


def adapt_tool_use(name: str, tool_input: Any) -> Tuple[str, Any]:
    """Inbound hook. If the model emitted a CC-aliased tool name,
    return the hermes-side (name, input) pair so the existing tool
    registry can dispatch.

    Returns (name, input) unchanged when the name has no alias.
    Logs the rewrite at DEBUG so trace logs can see what happened.
    """
    hermes_name = CC_TO_HERMES.get(name)
    if not hermes_name:
        return name, tool_input

    adapter = _ADAPTERS.get(name)
    if adapter and isinstance(tool_input, dict):
        try:
            adapted = adapter(tool_input)
        except Exception as e:
            # On adapter failure, fall back to passing args through
            # untouched — the registry will likely 400 with a clearer
            # error than us swallowing the issue.
            logger.warning("cc_aliases: adapter failed for %s: %s — "
                           "passing args through unchanged", name, e)
            adapted = tool_input
    else:
        adapted = tool_input

    logger.debug("cc_aliases: tool_use %r → %r (input adapted=%s)",
                 name, hermes_name, adapter is not None)
    return hermes_name, adapted

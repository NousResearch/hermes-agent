"""Agent-facing tools for the google_meet plugin.

Tools:
  meet_join        — join a Google Meet URL (spawns Playwright bot locally
                     OR on a remote node host via node=<name>)
  meet_status      — report bot liveness + transcript progress
  meet_transcript  — read the current transcript (optional last-N)
  meet_leave       — signal the bot to leave cleanly
  meet_say         — (v2) speak text through the realtime audio bridge.
                     Requires the active meeting to have been joined with
                     mode='realtime'.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from plugins.google_meet import process_manager as pm


# ---------------------------------------------------------------------------
# Runtime gate
# ---------------------------------------------------------------------------

def check_meet_requirements() -> bool:
    """Return True when the plugin can actually run LOCALLY.

    Gates on:
      * Python ``playwright`` package importable
      * the plugin being on a supported platform (Linux or macOS)

    Note: remote-node operation (``node=<name>``) only needs the
    ``websockets`` dep on the gateway side — Chromium lives on the node.
    But the plugin-level gate keeps the v1 semantics; individual tool
    handlers relax the requirement when a node is addressed.
    """
    import platform as _p
    if _p.system().lower() not in ("linux", "darwin"):
        return False
    try:
        import playwright  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Node client helper
# ---------------------------------------------------------------------------

def _resolve_node_client(node: Optional[str]):
    """Return (NodeClient, node_name) for *node*, or (None, None) to run local.

    Raises RuntimeError with a readable message if the node is named but
    unresolvable, so the handler can surface a clear error to the agent.
    """
    if node is None or node == "":
        return None, None
    from plugins.google_meet.node.registry import NodeRegistry
    from plugins.google_meet.node.client import NodeClient

    reg = NodeRegistry()
    entry = reg.resolve(node if node != "auto" else None)
    if entry is None:
        raise RuntimeError(
            f"no registered meet node matches {node!r} — "
            "run `hermes meet node approve <name> <url> <token>` first"
        )
    client = NodeClient(url=entry["url"], token=entry["token"])
    return client, entry.get("name")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

MEET_JOIN_SCHEMA: Dict[str, Any] = {
    "name": "meet_join",
    "description": (
        "Join a Google Meet call and start scraping live captions into a "
        "transcript file. Only meet.google.com URLs are accepted; no calendar "
        "scanning, no auto-dial. Spawns a headless Chromium subprocess that "
        "runs in parallel with the agent loop — returns immediately. Poll "
        "with meet_status and read captions with meet_transcript. Reminder "
        "to the agent: you should announce yourself in the meeting (there is "
        "no automatic consent announcement)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": (
                    "Full https://meet.google.com/... URL. Required."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["transcribe", "realtime"],
                "description": (
                    "transcribe (default): listen-only, scrape captions. "
                    "realtime: also enable agent speech via meet_say "
                    "(requires OpenAI Realtime key + platform audio bridge)."
                ),
            },
            "guest_name": {
                "type": "string",
                "description": (
                    "Display name to use when joining as guest. Defaults to "
                    "'Hermes Agent'."
                ),
            },
            "duration": {
                "type": "string",
                "description": (
                    "Optional max duration before auto-leave (e.g. '30m', "
                    "'2h', '90s'). Omit to stay until meet_leave is called."
                ),
            },
            "headed": {
                "type": "boolean",
                "description": (
                    "Run Chromium headed instead of headless (debug only). "
                    "Default false."
                ),
            },
            "node": {
                "type": "string",
                "description": (
                    "Name of a registered remote node to run the bot on "
                    "(useful when the gateway runs on a headless Linux box "
                    "but the user's Chrome with a signed-in Google profile "
                    "lives on their Mac). Pass 'auto' to use the single "
                    "registered node. Default: run locally. Nodes are "
                    "approved via `hermes meet node approve`."
                ),
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    },
}

MEET_STATUS_SCHEMA: Dict[str, Any] = {
    "name": "meet_status",
    "description": (
        "Report the current Meet session state — whether the bot is alive, "
        "has joined, is sitting in the lobby, number of transcript lines "
        "captured, and last-caption timestamp."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "node": {"type": "string"},
        },
        "additionalProperties": False,
    },
}

MEET_TRANSCRIPT_SCHEMA: Dict[str, Any] = {
    "name": "meet_transcript",
    "description": (
        "Read the scraped transcript for the active Meet session. Returns "
        "full transcript unless 'last' is set, in which case returns the last "
        "N lines only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "last": {
                "type": "integer",
                "description": (
                    "Optional: return only the last N caption lines. Useful "
                    "for polling during a meeting without re-reading the "
                    "whole transcript."
                ),
                "minimum": 1,
            },
            "node": {"type": "string"},
        },
        "additionalProperties": False,
    },
}

MEET_LEAVE_SCHEMA: Dict[str, Any] = {
    "name": "meet_leave",
    "description": (
        "Leave the active Meet call cleanly, stop caption scraping, and "
        "finalize the transcript file. Safe to call when no meeting is "
        "active — returns ok=false with a reason."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "node": {"type": "string"},
        },
        "additionalProperties": False,
    },
}

MEET_SAY_SCHEMA: Dict[str, Any] = {
    "name": "meet_say",
    "description": (
        "Speak text into the active Meet call. Requires the active meeting "
        "to have been joined with mode='realtime'. The text is queued to "
        "the bot's OpenAI Realtime session; the generated audio is streamed "
        "into Chrome's fake microphone via a virtual audio device "
        "(PulseAudio null-sink on Linux, BlackHole on macOS). Returns "
        "immediately — the actual speech lags by a couple of seconds."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to speak."},
            "node": {"type": "string"},
        },
        "required": ["text"],
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _err(msg: str, **extra) -> str:
    return _json({"success": False, "error": msg, **extra})


def handle_meet_join(args: Dict[str, Any], **_kw) -> str:
    url = (args.get("url") or "").strip()
    if not url:
        return _err("url is required")
    mode = (args.get("mode") or "transcribe").strip().lower()
    if mode not in ("transcribe", "realtime"):
        return _err(f"mode must be 'transcribe' or 'realtime' (got {mode!r})")

    node = args.get("node")
    try:
        client, node_name = _resolve_node_client(node)
    except RuntimeError as e:
        return _err(str(e))

    if client is not None:
        # Remote path — delegate to the node host.
        try:
            res = client.start_bot(
                url=url,
                guest_name=str(args.get("guest_name") or "Hermes Agent"),
                duration=str(args.get("duration")) if args.get("duration") else None,
                headed=bool(args.get("headed", False)),
                mode=mode,
            )
            return _json({"success": bool(res.get("ok")), "node": node_name, **res})
        except Exception as e:
            return _err(f"remote node start_bot failed: {e}", node=node_name)

    # Local path — same as v1, with v2 params.
    if not check_meet_requirements():
        return _err(
            "google_meet plugin prerequisites missing — install with "
            "`pip install playwright && python -m playwright install "
            "chromium`. Plugin is supported on Linux and macOS only."
        )
    res = pm.start(
        url=url,
        headed=bool(args.get("headed", False)),
        guest_name=str(args.get("guest_name") or "Hermes Agent"),
        duration=str(args.get("duration")) if args.get("duration") else None,
        mode=mode,
    )
    return _json({"success": bool(res.get("ok")), **res})


def handle_meet_status(args: Dict[str, Any], **_kw) -> str:
    try:
        client, node_name = _resolve_node_client(args.get("node"))
    except RuntimeError as e:
        return _err(str(e))
    if client is not None:
        try:
            res = client.status()
            return _json({"success": bool(res.get("ok")), "node": node_name, **res})
        except Exception as e:
            return _err(f"remote node status failed: {e}", node=node_name)
    res = pm.status()
    return _json({"success": bool(res.get("ok")), **res})


def handle_meet_transcript(args: Dict[str, Any], **_kw) -> str:
    last = args.get("last")
    try:
        last_i = int(last) if last is not None else None
        if last_i is not None and last_i < 1:
            last_i = None
    except (TypeError, ValueError):
        last_i = None
    try:
        client, node_name = _resolve_node_client(args.get("node"))
    except RuntimeError as e:
        return _err(str(e))
    if client is not None:
        try:
            res = client.transcript(last=last_i)
            return _json({"success": bool(res.get("ok")), "node": node_name, **res})
        except Exception as e:
            return _err(f"remote node transcript failed: {e}", node=node_name)
    res = pm.transcript(last=last_i)
    return _json({"success": bool(res.get("ok")), **res})


def handle_meet_leave(args: Dict[str, Any], **_kw) -> str:
    try:
        client, node_name = _resolve_node_client(args.get("node"))
    except RuntimeError as e:
        return _err(str(e))
    if client is not None:
        try:
            res = client.stop()
            return _json({"success": bool(res.get("ok")), "node": node_name, **res})
        except Exception as e:
            return _err(f"remote node stop failed: {e}", node=node_name)
    res = pm.stop(reason="agent called meet_leave")
    return _json({"success": bool(res.get("ok")), **res})


def handle_meet_say(args: Dict[str, Any], **_kw) -> str:
    text = (args.get("text") or "").strip()
    if not text:
        return _err("text is required")
    try:
        client, node_name = _resolve_node_client(args.get("node"))
    except RuntimeError as e:
        return _err(str(e))
    if client is not None:
        try:
            res = client.say(text)
            return _json({"success": bool(res.get("ok")), "node": node_name, **res})
        except Exception as e:
            return _err(f"remote node say failed: {e}", node=node_name)
    res = pm.enqueue_say(text)
    return _json({"success": bool(res.get("ok")), **res})

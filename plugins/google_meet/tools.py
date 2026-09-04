"""Agent-facing tools for the google_meet plugin.

  meet_join        — join a Meet URL (locally, or on a remote node via node=<name>)
  meet_status      — bot liveness + transcript progress
  meet_transcript  — read the transcript (optional last-N)
  meet_leave       — signal the bot to leave cleanly
  meet_say         — speak text through the realtime bridge (mode='realtime' only)
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from plugins.google_meet import process_manager as pm


def check_meet_requirements() -> bool:
    """True when the plugin can run LOCALLY: Linux/macOS + importable ``playwright``.
    Remote-node operation only needs ``websockets``; handlers relax this gate when a node is addressed."""
    import importlib.util
    import platform as _p
    return (_p.system().lower() in {"linux", "darwin"}
            and importlib.util.find_spec("playwright") is not None)


def resolve_node(node: str):
    """``(NodeClient, node_name)`` for *node* (``'auto'`` = the sole registered node), or ``(None, None)``."""
    from plugins.google_meet.node.registry import NodeRegistry
    from plugins.google_meet.node.client import NodeClient
    entry = NodeRegistry().resolve(node if node != "auto" else None)
    if entry is None:
        return None, None
    return NodeClient(url=entry["url"], token=entry["token"]), entry.get("name")


_NODE_PROP = {"type": "string"}


def _str(description: str) -> Dict[str, Any]:
    return {"type": "string", "description": description}


def _schema(name: str, description: str, properties: Dict[str, Any], required=None) -> Dict[str, Any]:
    params: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        params["required"] = required
    params["additionalProperties"] = False
    return {"name": name, "description": description, "parameters": params}


MEET_JOIN_SCHEMA = _schema(
    "meet_join",
    "Join a Google Meet call and start scraping live captions into a transcript file. Only "
    "meet.google.com URLs are accepted; no calendar scanning, no auto-dial. Spawns a headless "
    "Chromium subprocess that runs in parallel with the agent loop — returns immediately. Poll "
    "with meet_status and read captions with meet_transcript. Reminder to the agent: you "
    "should announce yourself in the meeting (there is no automatic consent announcement).",
    {"url": _str("Full https://meet.google.com/... URL. Required."),
     "mode": {"type": "string", "enum": ["transcribe", "realtime"],
              "description": ("transcribe (default): listen-only, scrape captions. "
                              "realtime: also enable agent speech via meet_say "
                              "(requires OpenAI Realtime key + platform audio bridge).")},
     "guest_name": _str("Display name to use when joining as guest. Defaults to 'Hermes Agent'."),
     "duration": _str("Optional max duration before auto-leave (e.g. '30m', "
                      "'2h', '90s'). Omit to stay until meet_leave is called."),
     "headed": {"type": "boolean",
                "description": "Run Chromium headed instead of headless (debug only). Default false."},
     "node": _str("Name of a registered remote node to run the bot on (useful when the gateway "
                  "runs on a headless Linux box but the user's Chrome with a signed-in Google "
                  "profile lives on their Mac). Pass 'auto' to use the single registered node. "
                  "Default: run locally. Nodes are approved via `hermes meet node approve`.")},
    required=["url"])

MEET_STATUS_SCHEMA = _schema(
    "meet_status",
    "Report the current Meet session state — whether the bot is alive, has joined, is sitting "
    "in the lobby, number of transcript lines captured, and last-caption timestamp.",
    {"node": _NODE_PROP})

MEET_TRANSCRIPT_SCHEMA = _schema(
    "meet_transcript",
    "Read the scraped transcript for the active Meet session. Returns "
    "full transcript unless 'last' is set, in which case returns the last N lines only.",
    {"last": {"type": "integer",
              "description": ("Optional: return only the last N caption lines. Useful "
                              "for polling during a meeting without re-reading the whole transcript."),
              "minimum": 1},
     "node": _NODE_PROP})

MEET_LEAVE_SCHEMA = _schema(
    "meet_leave",
    "Leave the active Meet call cleanly, stop caption scraping, and finalize the transcript "
    "file. Safe to call when no meeting is active — returns ok=false with a reason.",
    {"node": _NODE_PROP})

MEET_SAY_SCHEMA = _schema(
    "meet_say",
    "Speak text into the active Meet call. Requires the active meeting to have been joined "
    "with mode='realtime'. The text is queued to the bot's OpenAI Realtime session; the "
    "generated audio is streamed into Chrome's fake microphone via a virtual audio device "
    "(PulseAudio null-sink on Linux, BlackHole on macOS). Returns immediately — the actual "
    "speech lags by a couple of seconds.",
    {"text": _str("Text to speak."), "node": _NODE_PROP},
    required=["text"])


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _err(msg: str, **extra) -> str:
    return _json({"success": False, "error": msg, **extra})


def _dispatch(node: Optional[str], op: str, remote, local) -> str:
    """Run *remote(client)* on the addressed node, else *local()*; wrap as a tool result."""
    if not node:
        res = local()
        return _json({"success": bool(res.get("ok")), **res})
    client, node_name = resolve_node(node)
    if client is None:
        return _err(f"no registered meet node matches {node!r} — "
                    "run `hermes meet node approve <name> <url> <token>` first")
    try:
        res = remote(client)
    except Exception as e:
        return _err(f"remote node {op} failed: {e}", node=node_name)
    return _json({"success": bool(res.get("ok")), "node": node_name, **res})


def handle_meet_join(args: Dict[str, Any], **_kw) -> str:
    url = (args.get("url") or "").strip()
    if not url:
        return _err("url is required")
    mode = (args.get("mode") or "transcribe").strip().lower()
    if mode not in {"transcribe", "realtime"}:
        return _err(f"mode must be 'transcribe' or 'realtime' (got {mode!r})")
    common: Dict[str, Any] = dict(
        url=url, guest_name=str(args.get("guest_name") or "Hermes Agent"),
        duration=str(args.get("duration")) if args.get("duration") else None,
        headed=bool(args.get("headed", False)), mode=mode)

    def _local():
        if not check_meet_requirements():
            return {"ok": False, "error": (
                "google_meet plugin prerequisites missing — install with "
                "`pip install playwright && python -m playwright install "
                "chromium`. Plugin is supported on Linux and macOS only.")}
        return pm.start(**common)

    return _dispatch(args.get("node"), "start_bot", lambda c: c.start_bot(**common), _local)


def handle_meet_status(args: Dict[str, Any], **_kw) -> str:
    return _dispatch(args.get("node"), "status", lambda c: c.status(), pm.status)


def handle_meet_transcript(args: Dict[str, Any], **_kw) -> str:
    try:
        last = int(args["last"]) if args.get("last") is not None else None
    except (TypeError, ValueError):
        last = None
    if last is not None and last < 1:
        last = None
    return _dispatch(args.get("node"), "transcript", lambda c: c.transcript(last=last),
                     lambda: pm.transcript(last=last))


def handle_meet_leave(args: Dict[str, Any], **_kw) -> str:
    return _dispatch(args.get("node"), "stop", lambda c: c.stop(),
                     lambda: pm.stop(reason="agent called meet_leave"))


def handle_meet_say(args: Dict[str, Any], **_kw) -> str:
    text = (args.get("text") or "").strip()
    if not text:
        return _err("text is required")
    return _dispatch(args.get("node"), "say", lambda c: c.say(text), lambda: pm.enqueue_say(text))

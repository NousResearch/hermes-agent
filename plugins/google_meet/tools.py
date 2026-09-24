"""Agent-facing tools for the google_meet plugin.

meet_join        — join a Meet URL (locally, or on a remote node via node=<name>)
meet_status      — bot liveness + transcript progress
meet_transcript  — read the transcript (optional last-N)
meet_leave       — signal the bot to leave cleanly
meet_say         — speak text through the realtime bridge (mode='realtime' only)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home
from plugins.google_meet import process_manager as pm


def check_meet_requirements() -> bool:
    """True when the plugin can run locally on a supported Playwright host."""
    import platform as _platform

    if _platform.system().lower() not in {"linux", "darwin"}:
        return False
    try:
        import playwright  # noqa: F401
    except ImportError:
        return False
    return True


def _default_auth_state() -> Optional[str]:
    """Return the saved local Meet auth state path when one is available."""
    path = Path(get_hermes_home()) / "workspace" / "meetings" / "auth.json"
    return str(path) if path.is_file() else None


def _resolve_duration(raw: Any) -> Optional[str]:
    """Return an explicit auto-leave duration, or keep the bot running indefinitely."""
    return str(raw) if raw else None


def resolve_node(node: str):
    """Return ``(NodeClient, node_name)`` for *node*, or ``(None, None)`` when absent."""
    from plugins.google_meet.node.client import NodeClient
    from plugins.google_meet.node.registry import NodeRegistry

    entry = NodeRegistry().resolve(node if node != "auto" else None)
    if entry is None:
        return None, None
    return NodeClient(url=entry["url"], token=entry["token"]), entry.get("name")


_NODE_PROP = {"type": "string"}


def _str(description: str) -> Dict[str, Any]:
    return {"type": "string", "description": description}


def _schema(
    name: str,
    description: str,
    properties: Dict[str, Any],
    required: Optional[list[str]] = None,
) -> Dict[str, Any]:
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
    {
        "url": _str("Full https://meet.google.com/... URL. Required."),
        "mode": {
            "type": "string",
            "enum": ["transcribe", "realtime"],
            "description": (
                "transcribe (default): listen-only, scrape captions. realtime: also enable "
                "agent speech via meet_say (requires OpenAI Realtime key + platform audio bridge)."
            ),
        },
        "guest_name": _str(
            "Display name to use when joining as guest. Defaults to 'Hermes Agent'."
        ),
        "duration": _str(
            "Optional max duration before auto-leave (e.g. '30m', '2h', '90s'). Omit to stay "
            "until meet_leave is called."
        ),
        "persist_after_session": {
            "type": "boolean",
            "description": (
                "Default false. Set true only when the user explicitly wants the bot to remain "
                "after the current Hermes session ends."
            ),
        },
        "use_auth_state": {
            "type": "boolean",
            "description": (
                "Default false. Set true to explicitly reuse saved local Google Meet auth state "
                "instead of joining as the configured guest."
            ),
        },
        "headed": {
            "type": "boolean",
            "description": "Run Chromium headed instead of headless (debug only). Default false.",
        },
        "node": _str(
            "Name of a registered remote node to run the bot on. Pass 'auto' to use the single "
            "registered node. Default: run locally. Nodes are approved with hermes meet node approve."
        ),
    },
    required=["url"],
)

MEET_STATUS_SCHEMA = _schema(
    "meet_status",
    "Report the current Meet session state — whether the bot is alive, has joined, is sitting "
    "in the lobby, number of transcript lines captured, and last-caption timestamp.",
    {"node": _NODE_PROP},
)

MEET_TRANSCRIPT_SCHEMA = _schema(
    "meet_transcript",
    "Read the scraped transcript for the active Meet session. Returns full transcript unless "
    "'last' is set, in which case returns the last N lines only.",
    {
        "last": {
            "type": "integer",
            "description": "Optional: return only the last N caption lines.",
            "minimum": 1,
        },
        "include_finished": {
            "type": "boolean",
            "description": (
                "Default false. Set true to explicitly read the most recent finished transcript "
                "owned by the current Hermes session when no meeting is active."
            ),
        },
        "node": _NODE_PROP,
    },
)

MEET_LEAVE_SCHEMA = _schema(
    "meet_leave",
    "Leave the active Meet call cleanly, stop caption scraping, and finalize the transcript "
    "file. Safe to call when no meeting is active — returns ok=false with a reason.",
    {"node": _NODE_PROP},
)

MEET_SAY_SCHEMA = _schema(
    "meet_say",
    "Speak text into the active Meet call. Requires the active meeting to have been joined "
    "with mode='realtime'. The text is queued to the bot's OpenAI Realtime session; the "
    "generated audio is streamed into Chrome's fake microphone via a virtual audio device "
    "(PulseAudio null-sink on Linux, BlackHole on macOS). Returns immediately — the actual "
    "speech lags by a couple of seconds.",
    {"text": _str("Text to speak."), "node": _NODE_PROP},
    required=["text"],
)


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _err(msg: str, **extra: Any) -> str:
    return _json({"success": False, "error": msg, **extra})


def _context_session_id(kwargs: Dict[str, Any]) -> Optional[str]:
    session_id = str(kwargs.get("session_id") or "").strip()
    return session_id or None


def _dispatch(node: Optional[str], op: str, remote, local) -> str:
    """Run *remote(client)* on an addressed node, otherwise run *local()*."""
    if not node:
        result = local()
        return _json({"success": bool(result.get("ok")), **result})
    client, node_name = resolve_node(node)
    if client is None:
        return _err(
            f"no registered meet node matches {node!r} — "
            "run hermes meet node approve <name> <url> <token> first"
        )
    try:
        result = remote(client)
    except Exception as exc:
        return _err(f"remote node {op} failed: {exc}", node=node_name)
    return _json({"success": bool(result.get("ok")), "node": node_name, **result})


def handle_meet_join(args: Dict[str, Any], **kwargs: Any) -> str:
    url = str(args.get("url") or "").strip()
    if not url:
        return _err("url is required")
    mode = str(args.get("mode") or "transcribe").strip().lower()
    if mode not in {"transcribe", "realtime"}:
        return _err(f"mode must be 'transcribe' or 'realtime' (got {mode!r})")

    node = args.get("node")
    use_auth_state = bool(args.get("use_auth_state", False))
    if node and use_auth_state:
        return _err(
            "use_auth_state is local-only; remote nodes must manage Google auth state on the node host"
        )

    common: Dict[str, Any] = {
        "url": url,
        "guest_name": str(args.get("guest_name") or "Hermes Agent"),
        "duration": _resolve_duration(args.get("duration")),
        "persist_after_session": bool(args.get("persist_after_session", False)),
        "headed": bool(args.get("headed", False)),
        "mode": mode,
        "session_id": _context_session_id(kwargs),
    }

    def _local() -> Dict[str, Any]:
        if not check_meet_requirements():
            return {
                "ok": False,
                "error": (
                    "google_meet plugin prerequisites missing — install with pip install playwright && "
                    "python -m playwright install chromium. Plugin is supported on Linux and macOS only."
                ),
            }
        return pm.start(
            **common,
            auth_state=_default_auth_state() if use_auth_state else None,
        )

    return _dispatch(
        node, "start_bot", lambda client: client.start_bot(**common), _local
    )


def handle_meet_status(args: Dict[str, Any], **_kwargs: Any) -> str:
    return _dispatch(
        args.get("node"), "status", lambda client: client.status(), pm.status
    )


def handle_meet_transcript(args: Dict[str, Any], **kwargs: Any) -> str:
    try:
        last = int(args["last"]) if args.get("last") is not None else None
    except (TypeError, ValueError):
        last = None
    if last is not None and last < 1:
        last = None
    include_finished = bool(args.get("include_finished", False))
    session_id = _context_session_id(kwargs)
    return _dispatch(
        args.get("node"),
        "transcript",
        lambda client: client.transcript(
            last=last,
            include_finished=include_finished,
            session_id=session_id,
        ),
        lambda: pm.transcript(
            last=last,
            include_finished=include_finished,
            session_id=session_id,
        ),
    )


def handle_meet_leave(args: Dict[str, Any], **_kwargs: Any) -> str:
    reason = "agent called meet_leave"
    return _dispatch(
        args.get("node"),
        "stop",
        lambda client: client.stop(reason=reason),
        lambda: pm.stop(reason=reason),
    )


def handle_meet_say(args: Dict[str, Any], **_kwargs: Any) -> str:
    text = str(args.get("text") or "").strip()
    if not text:
        return _err("text is required")
    return _dispatch(
        args.get("node"),
        "say",
        lambda client: client.say(text),
        lambda: pm.enqueue_say(text),
    )

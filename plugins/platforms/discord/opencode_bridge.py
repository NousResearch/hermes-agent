"""OpenCode permission bridge for the Discord gateway adapter.

Surfaces OpenCode's own permission requests (the agent asking before it
runs a tool) in a configured Discord channel and routes Accept/Reject
clicks back to OpenCode through its official server API.

Wire contract (verified against the official ``@opencode-ai/sdk`` types,
v1.18.25, ``packages/sdk/js/src/gen/types.gen.ts``):

- Events: ``GET {base_url}/event`` (SSE) delivers
  ``{"type": "permission.updated", "properties": Permission}`` where
  ``Permission = {id, type, pattern?, sessionID, messageID, callID?,
  title, metadata: {..}, time: {created}}``.  A reply is broadcast as
  ``permission.replied`` so every connected client (TUI included)
  observes the decision.
- Reply: ``POST {base_url}/session/{sessionID}/permissions/{permissionID}``
  with body ``{"response": "once" | "always" | "reject"}`` → 200 bool,
  400 bad request, 404 not found (already answered elsewhere).

Security posture (deliberate, non-negotiable):

- Off by default; enabled only via the discord platform ``extra`` key
  ``opencode_bridge`` (see :func:`parse_bridge_config`).
- Loopback-only ``base_url`` — a non-loopback target disables the bridge
  (fail-closed) instead of shipping permission metadata off-host.
- Only Discord users in the explicit ``allowed_user_ids`` allowlist may
  click; an empty allowlist disables the bridge entirely.
- Only ``"once"`` and ``"reject"`` replies are ever sent.  The ``"always"``
  response is intentionally unreachable from Discord — no persistent
  grants, no privilege escalation, no YOLO path.
- Timeout resolves to an explicit ``"reject"`` (fail-closed) so the
  OpenCode session never hangs on a missing reply.
- No secrets are read, stored, or transmitted; only event metadata the
  OpenCode server itself emits is displayed.

Out-of-repo counterpart: a small OpenCode plugin is only needed when the
TUI is not attached; any OpenCode client (TUI, IDE, this bridge) answers
through the same documented API above.  The bridge never writes OpenCode
configuration and never bypasses OpenCode's own permission engine.

OpenCode >= 1.18 additionally publishes the newer ``permission.asked``
event (``{id, sessionID, permission, patterns, metadata, always}``) and
answers on ``POST /permission/{requestID}/reply`` with ``{"reply": ..}``;
both shapes are accepted and each is answered on its own endpoint.

Command-guard spool (second request source)
-------------------------------------------

The out-of-repo OpenCode plugin ``befehlswaechter.js`` (and the matching
Claude Code PreToolUse hook) cannot use OpenCode's permission engine for
"path outside the project" decisions: a TUI running in auto mode answers
every native permission with ``once`` before Discord ever sees it.  The
guard therefore blocks the tool call itself and asks this bridge through a
local, user-private spool directory (default ``$HERMES_HOME/opencode_bridge``):

- ``requests/<id>.json`` — written atomically by the guard::

      {"version": 1, "id": "<[A-Za-z0-9_-]{8,64}>", "agent": "opencode",
       "created_at": <epoch s>, "expires_at": <epoch s>, "session_id": "..",
       "project": "/abs/project", "command": "cat ~/x", "path": "/Users/..",
       "access": "lesen" | "schreiben" | "ausführen" | "unklar"}

- ``decisions/<id>.json`` — written atomically by this bridge::

      {"version": 1, "id": "<id>", "decision": "once" | "reject",
       "source": "discord" | "timeout" | "drop", "decided_at": <epoch s>}

The guard polls for its decision file and denies on timeout, on a missing
or malformed decision, and when the gateway is not running at all.  The
bridge never answers a malformed or expired request (fail-closed: the
guard's own timeout then blocks the command), only ever writes ``once`` or
``reject``, and removes stale files on its scan cycle.  Only the request
metadata the guard chose to send (command, path, project, access kind) is
displayed; nothing is executed or read on the guard's behalf.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import guard mirrors the adapter module
    import discord

    DISCORD_AVAILABLE = True
except ImportError:  # pragma: no cover
    discord = None
    DISCORD_AVAILABLE = False


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

BRIDGE_CONFIG_KEY = "opencode_bridge"

DEFAULT_BASE_URL = "http://127.0.0.1:4096"
DEFAULT_TIMEOUT_SECONDS = 300
TIMEOUT_MIN_SECONDS = 30
TIMEOUT_MAX_SECONDS = 900
MAX_CONCURRENT_PROMPTS = 10
RESOLVED_MEMO_SIZE = 64

SSE_RECONNECT_MIN_SECONDS = 2.0
SSE_RECONNECT_MAX_SECONDS = 60.0

_TITLE_BUDGET = 400
_PATTERN_BUDGET = 200
_METADATA_BUDGET = 500
_COMMAND_BUDGET = 600
_PATH_BUDGET = 300

_TRUNCT_SUFFIX = "... [truncated]"

# Command-guard spool (see module docstring).
GUARD_SPOOL_DIRNAME = "opencode_bridge"
GUARD_REQUESTS_SUBDIR = "requests"
GUARD_DECISIONS_SUBDIR = "decisions"
GUARD_PROTOCOL_VERSION = 1
GUARD_SCAN_INTERVAL_SECONDS = 1.0
GUARD_MAX_LIFETIME_SECONDS = 3600  # a request may not ask to live longer
GUARD_DECISION_TTL_SECONDS = 3600  # orphaned decisions are swept after this
GUARD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
GUARD_AGENTS = {"opencode": "OpenCode", "claude-code": "Claude Code"}
GUARD_ACCESS_KINDS = frozenset({"lesen", "schreiben", "ausführen", "netz", "unklar"})
GUARD_SOURCE = "guard"
SSE_SOURCE = "sse"
# Request kinds: "permission" = one Accept/Reject decision (the original
# command-guard shape, also used for Claude Code's own permission prompts);
# "question" = a multiple-choice question set (Claude Code's AskUserQuestion)
# answered with labels or free text.
GUARD_KIND_PERMISSION = "permission"
GUARD_KIND_QUESTION = "question"
# "notice" = fire-and-forget session events (no decision file): the agent
# started a session (→ one Discord thread per session, prompts of that
# session are asked inside it), sent a further prompt, finished a turn, or
# spawned a child session whose prompts belong to the parent's thread.
GUARD_KIND_NOTICE = "notice"
GUARD_NOTICES = frozenset({"start", "prompt", "result", "child"})
GUARD_THREADS_FILE = "threads.json"
GUARD_THREAD_TTL_SECONDS = 7 * 24 * 3600
_NOTICE_TEXT_BUDGET = 1700
_THREAD_NAME_BUDGET = 100
GUARD_MAX_QUESTIONS = 4
GUARD_MAX_OPTIONS = 8
_DETAILS_BUDGET = 900
_QUESTION_BUDGET = 600
_OPTION_LABEL_BUDGET = 80  # Discord button label limit
_FREE_TEXT_BUDGET = 1500


def default_guard_dir() -> str:
    """Spool root: ``$HERMES_HOME/opencode_bridge`` (or ``~/.hermes/...``)."""
    home = os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"), ".hermes")
    return os.path.join(home, GUARD_SPOOL_DIRNAME)


def _truncate(text: str, budget: int) -> str:
    text = str(text or "")
    if len(text) <= budget:
        return text
    return text[: max(0, budget - len(_TRUNCT_SUFFIX))] + _TRUNCT_SUFFIX


def is_loopback_url(url: str) -> bool:
    """Return True when the URL's host is a loopback address.

    The bridge refuses non-loopback targets so permission metadata can
    never be shipped to a remote host by misconfiguration.
    """
    try:
        host = (urlsplit(str(url)).hostname or "").lower()
    except ValueError:
        return False
    return host in LOOPBACK_HOSTS


@dataclass(frozen=True)
class OpenCodeBridgeConfig:
    """Validated bridge configuration (from discord platform ``extra``).

    ``enabled`` is only True when every fail-closed precondition holds:
    explicit opt-in, a loopback base_url, a non-empty user allowlist, and
    a sane timeout.
    """

    enabled: bool = False
    base_url: str = DEFAULT_BASE_URL
    channel_id: str = ""
    allowed_user_ids: frozenset = frozenset()
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    disabled_reason: str = ""
    # Command-guard spool: on by default once the bridge itself is enabled;
    # ``guard_enabled: false`` keeps only the SSE path, ``guard_dir``
    # overrides the spool root.
    guard_enabled: bool = True
    guard_dir: str = ""
    # Optional per-agent channels, e.g. {"claude-code": "123"}: guard prompts
    # and session threads of that agent go there instead of ``channel_id``.
    agent_channels: Dict[str, str] = field(default_factory=dict)

    def channel_for(self, agent: str) -> str:
        return self.agent_channels.get(agent) or self.channel_id


def parse_bridge_config(extra: Any) -> OpenCodeBridgeConfig:
    """Parse and validate the bridge config from a platform ``extra`` dict.

    Fail-closed: any invalid or missing precondition yields a disabled
    config with ``disabled_reason`` filled in (logged once by the caller).
    """
    section = extra.get(BRIDGE_CONFIG_KEY) if isinstance(extra, dict) else None
    if not isinstance(section, dict):
        return OpenCodeBridgeConfig(disabled_reason="not configured")

    if not section.get("enabled"):
        return OpenCodeBridgeConfig(disabled_reason="disabled")

    base_url = str(section.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
    if not is_loopback_url(base_url):
        return OpenCodeBridgeConfig(
            base_url=base_url,
            disabled_reason=f"base_url is not loopback ({base_url})",
        )

    raw_users = section.get("allowed_user_ids") or []
    if isinstance(raw_users, str):
        raw_users = [p.strip() for p in raw_users.split(",")]
    allowed_user_ids = frozenset(
        str(u).strip() for u in raw_users if str(u).strip()
    )
    if not allowed_user_ids:
        return OpenCodeBridgeConfig(
            base_url=base_url,
            disabled_reason="allowed_user_ids is empty",
        )

    channel_id = str(section.get("channel_id") or "").strip()
    if not channel_id:
        return OpenCodeBridgeConfig(
            base_url=base_url,
            disabled_reason="channel_id is missing",
        )

    try:
        timeout_seconds = int(section.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS)
    except (TypeError, ValueError):
        timeout_seconds = DEFAULT_TIMEOUT_SECONDS
    timeout_seconds = max(TIMEOUT_MIN_SECONDS, min(TIMEOUT_MAX_SECONDS, timeout_seconds))

    guard_enabled = section.get("guard_enabled", True)
    guard_enabled = guard_enabled if isinstance(guard_enabled, bool) else True
    guard_dir = str(section.get("guard_dir") or "").strip() or default_guard_dir()

    raw_channels = section.get("agent_channels")
    agent_channels: Dict[str, str] = {}
    if isinstance(raw_channels, dict):
        for agent, value in raw_channels.items():
            agent, value = str(agent).strip(), str(value).strip()
            # Only known agents and plain numeric ids; anything else is
            # dropped so a typo cannot silently redirect prompts.
            if agent in GUARD_AGENTS and value.isdigit():
                agent_channels[agent] = value

    return OpenCodeBridgeConfig(
        enabled=True,
        base_url=base_url,
        channel_id=channel_id,
        allowed_user_ids=allowed_user_ids,
        timeout_seconds=timeout_seconds,
        guard_enabled=guard_enabled,
        guard_dir=guard_dir,
        agent_channels=agent_channels,
    )


@dataclass(frozen=True)
class OpenCodePermissionRequest:
    """One permission request awaiting a Discord decision.

    ``source`` says where it came from and therefore how the decision is
    delivered: ``"sse"`` requests are answered on the OpenCode server API
    (``reply_api`` picks the legacy or the v2 endpoint), ``"guard"``
    requests are answered through the spool's decisions directory.
    """

    permission_id: str
    session_id: str
    kind: str = ""
    pattern: str = ""
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = SSE_SOURCE
    reply_api: str = "legacy"
    # Guard-only display fields.
    agent: str = "opencode"
    command: str = ""
    path: str = ""
    project: str = ""
    access: str = ""
    expires_at: float = 0.0
    guard_kind: str = GUARD_KIND_PERMISSION
    tool: str = ""
    details: str = ""
    questions: tuple = ()
    # Notice-only fields.
    notice: str = ""
    text: str = ""
    started_at: float = 0.0
    parent_session_id: str = ""

    @property
    def short_session_id(self) -> str:
        return self.session_id[:12]

    @property
    def is_guard(self) -> bool:
        return self.source == GUARD_SOURCE

    @property
    def is_question(self) -> bool:
        return self.is_guard and self.guard_kind == GUARD_KIND_QUESTION

    @property
    def is_notice(self) -> bool:
        return self.is_guard and self.guard_kind == GUARD_KIND_NOTICE


def parse_permission_event(payload: Any) -> Optional[OpenCodePermissionRequest]:
    """Parse a ``permission.updated`` / ``permission.asked`` SSE payload.

    Returns None for anything that is not a well-formed permission
    request — malformed events are dropped (fail-closed), never guessed
    into shape.
    """
    if not isinstance(payload, dict):
        return None
    event_type = payload.get("type")
    if event_type not in ("permission.updated", "permission.asked"):
        return None
    props = payload.get("properties")
    if not isinstance(props, dict):
        return None
    permission_id = props.get("id")
    session_id = props.get("sessionID")
    if not isinstance(permission_id, str) or not permission_id:
        return None
    if not isinstance(session_id, str) or not session_id:
        return None
    metadata = props.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}

    if event_type == "permission.asked":
        patterns = props.get("patterns")
        if not isinstance(patterns, list):
            patterns = []
        pattern = ", ".join(str(p) for p in patterns)
        command = metadata.get("command")
        title = command if isinstance(command, str) and command.strip() else pattern
        return OpenCodePermissionRequest(
            permission_id=permission_id,
            session_id=session_id,
            kind=str(props.get("permission") or ""),
            pattern=pattern,
            title=title,
            metadata=metadata,
            reply_api="v2",
        )

    pattern = props.get("pattern")
    if pattern is not None and not isinstance(pattern, str):
        if isinstance(pattern, list):
            pattern = ", ".join(str(p) for p in pattern)
        else:
            pattern = json.dumps(pattern)
    return OpenCodePermissionRequest(
        permission_id=permission_id,
        session_id=session_id,
        kind=str(props.get("type") or ""),
        pattern=str(pattern) if pattern else "",
        title=str(props.get("title") or ""),
        metadata=metadata,
    )


def parse_guard_request(
    payload: Any, *, now: Optional[float] = None
) -> Optional[OpenCodePermissionRequest]:
    """Validate one spool request file's JSON into a guard request.

    Strict on purpose: anything unclear (wrong version, bad id, unknown
    agent or access kind, missing command/path, implausible lifetime,
    already expired) yields None and is never shown or answered — the
    guard's own timeout then keeps the command blocked.
    """
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != GUARD_PROTOCOL_VERSION:
        return None
    request_id = payload.get("id")
    if not isinstance(request_id, str) or not GUARD_ID_RE.match(request_id):
        return None
    agent = payload.get("agent")
    if agent not in GUARD_AGENTS:
        return None
    kind = payload.get("kind", GUARD_KIND_PERMISSION)
    if kind not in (GUARD_KIND_PERMISSION, GUARD_KIND_QUESTION, GUARD_KIND_NOTICE):
        return None
    project = payload.get("project")
    project = project if isinstance(project, str) else ""
    tool = payload.get("tool", "")
    tool = tool if isinstance(tool, str) else ""
    details = payload.get("details", "")
    details = details if isinstance(details, str) else ""
    questions: tuple = ()
    notice, text, parent_session_id, started_at = "", "", "", 0.0
    if kind == GUARD_KIND_NOTICE:
        notice = payload.get("notice")
        if notice not in GUARD_NOTICES:
            return None
        session = payload.get("session_id")
        if not isinstance(session, str) or not session.strip():
            return None
        text = payload.get("text", "")
        text = text if isinstance(text, str) else ""
        parent = payload.get("parent_session_id", "")
        parent_session_id = parent if isinstance(parent, str) else ""
        if notice == "child" and not parent_session_id:
            return None
        started = payload.get("started_at", 0.0)
        started_at = float(started) if isinstance(started, (int, float)) and not isinstance(started, bool) else 0.0
        command, path, access = notice, "-", "unklar"
    elif kind == GUARD_KIND_QUESTION:
        questions = _parse_questions(payload.get("questions"))
        if not questions:
            return None
        command = "; ".join(q["question"] for q in questions)
        path, access = "-", "unklar"
    else:
        command = payload.get("command")
        path = payload.get("path")
        if not isinstance(command, str) or not command.strip():
            return None
        if not isinstance(path, str) or not path.strip():
            return None
        access = payload.get("access")
        if access not in GUARD_ACCESS_KINDS:
            return None
    created_at = payload.get("created_at")
    expires_at = payload.get("expires_at")
    if isinstance(created_at, bool) or isinstance(expires_at, bool):
        return None
    if not isinstance(created_at, (int, float)) or not isinstance(expires_at, (int, float)):
        return None
    if expires_at <= created_at or expires_at - created_at > GUARD_MAX_LIFETIME_SECONDS:
        return None
    current = time.time() if now is None else now
    if expires_at <= current:
        return None
    session_id = payload.get("session_id")
    session_id = session_id if isinstance(session_id, str) and session_id else "-"
    return OpenCodePermissionRequest(
        permission_id=request_id,
        session_id=session_id,
        kind="befehlswaechter",
        title=command,
        source=GUARD_SOURCE,
        agent=str(agent),
        command=command,
        path=path,
        project=project,
        access=str(access),
        expires_at=float(expires_at),
        guard_kind=str(kind),
        tool=tool,
        details=details,
        questions=questions,
        notice=str(notice),
        text=text,
        started_at=started_at,
        parent_session_id=parent_session_id,
    )


class ThreadRegistry:
    """session_id → Discord thread, persisted next to the spool.

    Child sessions (subagents) map to their parent's thread. Entries expire
    after a week so the file cannot grow without bound.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._parents: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        sessions = data.get("sessions")
        parents = data.get("parents")
        if isinstance(sessions, dict):
            self._sessions = {
                str(k): v for k, v in sessions.items()
                if isinstance(v, dict) and isinstance(v.get("thread_id"), str)
            }
        if isinstance(parents, dict):
            self._parents = {str(k): str(v) for k, v in parents.items()}

    def _save(self) -> None:
        payload = {"sessions": self._sessions, "parents": self._parents}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".tmp-threads-", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(tmp_name, self.path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def root_session(self, session_id: str) -> str:
        seen: Set[str] = set()
        while session_id in self._parents and session_id not in seen:
            seen.add(session_id)
            session_id = self._parents[session_id]
        return session_id

    def thread_for(self, session_id: str) -> Optional[str]:
        entry = self._sessions.get(self.root_session(session_id))
        return entry.get("thread_id") if entry else None

    def set_thread(self, session_id: str, thread_id: str, channel_id: str, *, now: Optional[float] = None) -> None:
        self._sessions[session_id] = {
            "thread_id": str(thread_id),
            "channel_id": str(channel_id),
            "created_at": time.time() if now is None else now,
        }
        self._save()

    def set_parent(self, child_id: str, parent_id: str) -> None:
        if child_id == parent_id:
            return
        self._parents[child_id] = parent_id
        self._save()

    def prune(self, *, now: Optional[float] = None) -> None:
        current = time.time() if now is None else now
        stale = [
            sid for sid, entry in self._sessions.items()
            if current - float(entry.get("created_at") or 0) > GUARD_THREAD_TTL_SECONDS
        ]
        if not stale:
            return
        for sid in stale:
            del self._sessions[sid]
        self._parents = {c: p for c, p in self._parents.items() if p in self._sessions}
        self._save()


def _parse_questions(raw: Any) -> tuple:
    """Validate an AskUserQuestion-style question list; () when unclear."""
    if not isinstance(raw, list) or not 1 <= len(raw) <= GUARD_MAX_QUESTIONS:
        return ()
    out = []
    seen: Set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            return ()
        question = item.get("question")
        if not isinstance(question, str) or not question.strip() or question in seen:
            return ()
        seen.add(question)
        header = item.get("header", "")
        header = header if isinstance(header, str) else ""
        options_raw = item.get("options")
        if not isinstance(options_raw, list) or not 1 <= len(options_raw) <= GUARD_MAX_OPTIONS:
            return ()
        options = []
        labels: Set[str] = set()
        for opt in options_raw:
            if not isinstance(opt, dict):
                return ()
            label = opt.get("label")
            if not isinstance(label, str) or not label.strip() or label in labels:
                return ()
            labels.add(label)
            description = opt.get("description", "")
            options.append({
                "label": label,
                "description": description if isinstance(description, str) else "",
            })
        multi = item.get("multiSelect", False)
        out.append({
            "question": question,
            "header": header,
            "options": tuple(options),
            "multiSelect": multi if isinstance(multi, bool) else False,
        })
    return tuple(out)


class GuardSpool:
    """Filesystem mailbox between the command guard and this bridge.

    ``scan`` returns valid, unexpired, not-yet-seen requests; ``write_decision``
    publishes exactly one decision file atomically (temp file + rename) so
    the guard never reads a half-written JSON.  Everything lives under a
    user-private (0700) directory; the bridge never follows symlinks out of
    it and never executes anything it finds there.
    """

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.requests_dir = self.root / GUARD_REQUESTS_SUBDIR
        self.decisions_dir = self.root / GUARD_DECISIONS_SUBDIR
        self._invalid: Set[str] = set()

    def ensure(self) -> None:
        for directory in (self.root, self.requests_dir, self.decisions_dir):
            directory.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(directory, 0o700)
            except OSError:  # pragma: no cover - best effort on odd filesystems
                pass

    def _request_path(self, request_id: str) -> Path:
        return self.requests_dir / f"{request_id}.json"

    def _decision_path(self, request_id: str) -> Path:
        return self.decisions_dir / f"{request_id}.json"

    def scan(self, *, now: Optional[float] = None) -> List[OpenCodePermissionRequest]:
        """Return valid pending requests; drop expired files, remember bad ones."""
        current = time.time() if now is None else now
        found: List[OpenCodePermissionRequest] = []
        try:
            names = sorted(os.listdir(self.requests_dir))
        except FileNotFoundError:
            return found
        present = set(names)
        self._invalid &= present
        for name in names:
            if not name.endswith(".json"):
                continue
            request_id = name[: -len(".json")]
            if not GUARD_ID_RE.match(request_id):
                continue
            file_path = self.requests_dir / name
            if name in self._invalid:
                continue
            if file_path.is_symlink() or not file_path.is_file():
                self._invalid.add(name)
                continue
            try:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                # Possibly mid-write; a stale unreadable file is dropped once
                # its own deadline (or the max lifetime) has passed.
                self._drop_if_stale(file_path, current)
                continue
            request = parse_guard_request(payload, now=current)
            if request is None or request.permission_id != request_id:
                expires = payload.get("expires_at") if isinstance(payload, dict) else None
                if isinstance(expires, (int, float)) and not isinstance(expires, bool) and expires <= current:
                    self._unlink(file_path)
                else:
                    if name not in self._invalid:
                        logger.warning("OpenCode bridge: ignoring malformed guard request %s", name)
                    self._invalid.add(name)
                continue
            if not request.is_notice and self._decision_path(request_id).exists():
                continue  # already answered, guard has not collected it yet
            found.append(request)
        return found

    def remove_request(self, request_id: str) -> None:
        """Drop a consumed notice file (notices have no decision file)."""
        if GUARD_ID_RE.match(request_id):
            self._unlink(self._request_path(request_id))

    @property
    def threads_path(self) -> Path:
        return self.root / GUARD_THREADS_FILE

    def _drop_if_stale(self, file_path: Path, now: float) -> None:
        try:
            age = now - file_path.stat().st_mtime
        except OSError:
            return
        if age > GUARD_MAX_LIFETIME_SECONDS:
            self._unlink(file_path)

    def write_decision(
        self,
        request_id: str,
        decision: str,
        source: str,
        *,
        answers: Optional[Dict[str, Any]] = None,
        now: Optional[float] = None,
    ) -> None:
        """Publish one decision: ``once``/``reject``, or ``answer`` with answers."""
        if decision == "answer" and not isinstance(answers, dict):
            decision = "reject"
        if decision not in ("once", "reject", "answer"):
            decision = "reject"
        if not GUARD_ID_RE.match(request_id):
            raise ValueError(f"invalid guard request id: {request_id!r}")
        self.ensure()
        payload: Dict[str, Any] = {
            "version": GUARD_PROTOCOL_VERSION,
            "id": request_id,
            "decision": decision,
            "source": source,
            "decided_at": time.time() if now is None else now,
        }
        if decision == "answer":
            payload["answers"] = answers
        fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=self.decisions_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, self._decision_path(request_id))
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def sweep(self, *, now: Optional[float] = None) -> None:
        """Remove decisions nobody collected and leftover temp files."""
        current = time.time() if now is None else now
        try:
            names = os.listdir(self.decisions_dir)
        except FileNotFoundError:
            return
        for name in names:
            file_path = self.decisions_dir / name
            try:
                age = current - file_path.stat().st_mtime
            except OSError:
                continue
            if name.startswith(".tmp-") and age > 60:
                self._unlink(file_path)
            elif age > GUARD_DECISION_TTL_SECONDS:
                self._unlink(file_path)

    @staticmethod
    def _unlink(file_path: Path) -> None:
        try:
            file_path.unlink()
        except OSError:
            pass


class SseAssembler:
    """Incremental SSE line parser yielding JSON ``data:`` payloads.

    Feed one line at a time (without trailing newline); a complete event
    (blank line) returns the concatenated data payload as parsed JSON,
    or None when the block was not JSON or carried no data.
    """

    def __init__(self) -> None:
        self._data_lines: List[str] = []

    def feed(self, line: str) -> Optional[dict]:
        line = line.rstrip("\r")
        if line == "":
            block = "\n".join(self._data_lines)
            self._data_lines = []
            if not block:
                return None
            try:
                payload = json.loads(block)
            except ValueError:
                logger.debug("OpenCode bridge: non-JSON SSE payload dropped")
                return None
            return payload if isinstance(payload, dict) else None
        if line.startswith("data:"):
            self._data_lines.append(line[5:].lstrip(" "))
        # event:/id:/retry:/comment lines are irrelevant: the JSON payload
        # itself carries the event type.
        return None


class BridgePendingRegistry:
    """Tracks in-flight bridge prompts; first resolution wins.

    Also memoizes recently resolved permission ids so a redelivered
    ``permission.updated`` (SSE reconnect replay) cannot post a second
    prompt for a request that was already answered.
    """

    def __init__(self, max_concurrent: int = MAX_CONCURRENT_PROMPTS) -> None:
        self._max_concurrent = max_concurrent
        self._pending: Dict[str, str] = {}
        self._resolved_memo: List[str] = []
        self._resolved_set: Set[str] = set()

    def register(self, permission_id: str) -> bool:
        """Reserve a prompt slot; False when deduped or at capacity."""
        if permission_id in self._pending or permission_id in self._resolved_set:
            return False
        if len(self._pending) >= self._max_concurrent:
            logger.warning(
                "OpenCode bridge: %d prompts already pending, dropping %s",
                len(self._pending), permission_id,
            )
            return False
        self._pending[permission_id] = "pending"
        return True

    def resolve(self, permission_id: str, response: str) -> bool:
        """Record the first resolution; later calls are no-ops (False)."""
        if permission_id not in self._pending:
            return False
        del self._pending[permission_id]
        self._resolved_set.add(permission_id)
        self._resolved_memo.append(permission_id)
        if len(self._resolved_memo) > RESOLVED_MEMO_SIZE:
            dropped = self._resolved_memo.pop(0)
            self._resolved_set.discard(dropped)
        return True

    def is_pending(self, permission_id: str) -> bool:
        return permission_id in self._pending

    @property
    def pending_count(self) -> int:
        return len(self._pending)


class OpenCodeBridgeClient:
    """HTTP client for the OpenCode server API (SSE events + replies).

    Loopback-only by construction: both endpoints refuse to run against
    a non-loopback base_url.
    """

    def __init__(self, base_url: str, *, transport: Optional[Any] = None) -> None:
        if not is_loopback_url(base_url):
            raise ValueError(f"OpenCode bridge refuses non-loopback base_url: {base_url}")
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(10.0, read=None),
            transport=transport,
        )

    async def stream_events(self) -> Any:
        """Yield parsed SSE payloads from ``GET /event``, reconnecting on drop.

        Async generator; exits on cancellation. Reconnects use a bounded
        linear backoff so a dead OpenCode server costs one attempt per
        minute, not a busy loop.
        """
        assembler = SseAssembler()
        backoff = SSE_RECONNECT_MIN_SECONDS
        while True:
            try:
                async with self._client.stream("GET", "/event") as response:
                    response.raise_for_status()
                    backoff = SSE_RECONNECT_MIN_SECONDS
                    async for line in response.aiter_lines():
                        payload = assembler.feed(line)
                        if payload is not None:
                            yield payload
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("OpenCode bridge: event stream dropped (%s)", exc)
            await asyncio.sleep(backoff)
            backoff = min(SSE_RECONNECT_MAX_SECONDS, backoff * 2)

    async def reply(
        self, session_id: str, permission_id: str, response: str, api: str = "legacy"
    ) -> tuple[bool, int]:
        """POST a permission reply. Returns (delivered, status_code).

        ``api`` selects the endpoint the request was observed on: the
        legacy session-scoped route (``permission.updated``) or the v2
        ``/permission/{id}/reply`` route (``permission.asked``).
        ``delivered`` is False when OpenCode answered 4xx (notably 404 —
        the request was already answered by another client, e.g. the TUI).
        """
        if api == "v2":
            url = f"/permission/{permission_id}/reply"
            resp = await self._client.post(url, json={"reply": response})
        else:
            url = f"/session/{session_id}/permissions/{permission_id}"
            resp = await self._client.post(url, json={"response": response})
        if resp.status_code in (200, 204):
            return True, resp.status_code
        logger.warning(
            "OpenCode bridge: reply %s/%s -> %s rejected (%s)",
            session_id, permission_id, response, resp.status_code,
        )
        return False, resp.status_code

    async def aclose(self) -> None:
        await self._client.aclose()


class OpenCodeBridge:
    """Orchestrates event consumption, Discord prompts, and replies.

    Owns one asyncio task (see :meth:`run`) living inside the Discord
    adapter's event loop; per-request resolution happens either from a
    button click or the view's timeout, both funneled through
    :meth:`resolve` so first-wins semantics and the reply POST are in
    exactly one place.
    """

    def __init__(
        self,
        adapter: Any,
        config: OpenCodeBridgeConfig,
        client: Optional[OpenCodeBridgeClient] = None,
        spool: Optional[GuardSpool] = None,
    ) -> None:
        self._adapter = adapter
        self._config = config
        self._client = client or OpenCodeBridgeClient(config.base_url)
        self._registry = BridgePendingRegistry()
        self._task: Optional[asyncio.Task] = None
        self._guard_task: Optional[asyncio.Task] = None
        if spool is not None:
            self._spool: Optional[GuardSpool] = spool
        elif config.guard_enabled and config.guard_dir:
            self._spool = GuardSpool(config.guard_dir)
        else:
            self._spool = None
        self._threads: Optional[ThreadRegistry] = (
            ThreadRegistry(self._spool.threads_path) if self._spool is not None else None
        )

    @property
    def config(self) -> OpenCodeBridgeConfig:
        return self._config

    @property
    def registry(self) -> BridgePendingRegistry:
        return self._registry

    @property
    def spool(self) -> Optional[GuardSpool]:
        return self._spool

    @property
    def threads(self) -> Optional[ThreadRegistry]:
        return self._threads

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run())
        if self._spool is not None and (self._guard_task is None or self._guard_task.done()):
            self._guard_task = asyncio.create_task(self.run_guard_spool())

    async def run_guard_spool(self) -> None:
        """Poll the guard spool until cancelled; one prompt per new request."""
        assert self._spool is not None
        try:
            self._spool.ensure()
        except OSError as exc:
            logger.warning("OpenCode bridge: guard spool %s unusable: %s", self._spool.root, exc)
            return
        logger.info(
            "OpenCode bridge: watching guard spool %s for Discord channel %s",
            self._spool.root, self._config.channel_id,
        )
        try:
            while True:
                await self.poll_guard_spool_once()
                await asyncio.sleep(GUARD_SCAN_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            pass

    async def poll_guard_spool_once(self) -> int:
        """One scan cycle; returns how many prompts were posted."""
        assert self._spool is not None
        posted = 0
        try:
            requests = self._spool.scan()
        except Exception as exc:  # pragma: no cover - defensive: keep polling
            logger.warning("OpenCode bridge: guard spool scan failed: %s", exc)
            return 0
        for request in requests:
            if request.is_notice:
                try:
                    if await self._handle_notice(request):
                        posted += 1
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "OpenCode bridge: failed to handle %s notice %s: %s",
                        request.notice, request.permission_id, exc,
                    )
                finally:
                    # Notices are consumed exactly once; a failed one is not
                    # retried (a second thread or duplicate post is worse).
                    self._spool.remove_request(request.permission_id)
                continue
            if self._registry.is_pending(request.permission_id):
                continue
            try:
                if await self._post_prompt(request):
                    posted += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "OpenCode bridge: failed to post guard prompt for %s: %s",
                    request.permission_id, exc,
                )
        try:
            self._spool.sweep()
            if self._threads is not None:
                self._threads.prune()
        except Exception:  # pragma: no cover - best effort housekeeping
            pass
        return posted

    # ------------------------------------------------------------------
    # Session threads
    # ------------------------------------------------------------------

    async def _channel_for_agent(self, agent: str) -> Any:
        """The Discord channel this agent's prompts belong in."""
        client = self._adapter._client
        if client is None:
            return None
        channel_id = self._config.channel_for(agent)
        channel = client.get_channel(int(channel_id))
        if channel is None:
            try:
                channel = await client.fetch_channel(int(channel_id))
            except Exception as exc:
                logger.warning("OpenCode bridge: channel %s unreachable (%s)", channel_id, exc)
                return None
        return channel

    async def _main_channel(self) -> Any:
        return await self._channel_for_agent("")

    async def _thread_channel(self, session_id: str) -> Any:
        """The session's thread, or None when unknown / unreachable."""
        if self._threads is None or not session_id or session_id == "-":
            return None
        thread_id = self._threads.thread_for(session_id)
        if not thread_id:
            return None
        client = self._adapter._client
        if client is None:
            return None
        thread = client.get_channel(int(thread_id))
        if thread is None:
            try:
                thread = await client.fetch_channel(int(thread_id))
            except Exception as exc:
                logger.debug("OpenCode bridge: thread %s unreachable (%s)", thread_id, exc)
                return None
        return thread

    async def _target_channel(self, request: OpenCodePermissionRequest) -> Any:
        """Prompts of a session go to its thread, else to the agent's channel."""
        thread = await self._thread_channel(request.session_id)
        return thread if thread is not None else await self._channel_for_agent(request.agent)

    @staticmethod
    def _thread_name(request: OpenCodePermissionRequest) -> str:
        agent_label = GUARD_AGENTS.get(request.agent, request.agent)
        project = os.path.basename((request.project or "").rstrip("/")) or "Projekt"
        when = time.strftime("%d.%m. %H:%M", time.localtime(request.started_at or time.time()))
        return _truncate(f"{agent_label} · {project} · {when}", _THREAD_NAME_BUDGET)

    def _notice_text(self, request: OpenCodePermissionRequest) -> str:
        agent_label = GUARD_AGENTS.get(request.agent, request.agent)
        body = _truncate(request.text.strip(), _NOTICE_TEXT_BUDGET).replace("```", "'''")
        when = time.strftime("%H:%M", time.localtime(request.started_at or time.time()))
        if request.notice == "start":
            return (
                f"🧵 **{agent_label}-Sitzung gestartet** um {when}\n"
                f"**Projektordner:** `{_truncate(request.project or '-', _PATH_BUDGET)}` · **Session:** `{request.short_session_id}`\n\n"
                f"**Prompt:**\n{body or '_(kein Text)_'}"
            )
        if request.notice == "prompt":
            return f"📝 **Neuer Prompt** ({when})\n{body or '_(kein Text)_'}"
        if request.notice == "result":
            return f"✅ **Antwort** ({when})\n{body or '_(keine Textantwort)_'}"
        return body

    async def _open_session_thread(self, request: OpenCodePermissionRequest) -> tuple[Any, Optional[str]]:
        """Create the session's thread wherever the bridge channel allows it.

        Returns (thread, pending_start_text). The configured channel may be a
        text channel (thread hangs off a starter message), a forum channel
        (each session is a new forum post), or itself a thread (Discord
        forbids nesting, so the session thread is created in the parent and a
        pointer is posted in the watched thread). pending_start_text is the
        start message the caller still needs to post into the thread, or None
        when it was already delivered (forum post content).
        """
        channel = await self._channel_for_agent(request.agent)
        if channel is None:
            return None, None
        name = self._thread_name(request)
        start_text = self._notice_text(request)
        agent_label = GUARD_AGENTS.get(request.agent, request.agent)
        project = os.path.basename((request.project or "").rstrip("/")) or "-"
        pointer = f"🧵 **{agent_label}** · `{project}` · Session `{request.short_session_id}` — Verlauf und Rückfragen im Thread."

        forum_cls = getattr(discord, "ForumChannel", ()) if DISCORD_AVAILABLE else ()
        thread_cls = getattr(discord, "Thread", ()) if DISCORD_AVAILABLE else ()

        if isinstance(channel, forum_cls):
            created = await channel.create_thread(name=name, content=start_text)
            return getattr(created, "thread", created), None

        if isinstance(channel, thread_cls):
            parent = getattr(channel, "parent", None)
            if isinstance(parent, forum_cls):
                created = await parent.create_thread(name=name, content=start_text)
                thread = getattr(created, "thread", created)
                pending: Optional[str] = None
            elif parent is not None:
                kind = getattr(getattr(discord, "ChannelType", None), "public_thread", None)
                thread = await parent.create_thread(name=name, type=kind, auto_archive_duration=1440)
                pending = start_text
            else:
                return None, None
            await channel.send(content=f"{pointer} → {thread.mention}", allowed_mentions=self._allowed_mentions())
            return thread, pending

        # Plain text channel: thread attached to a starter message.
        starter = await channel.send(content=pointer, allowed_mentions=self._allowed_mentions())
        thread = await starter.create_thread(name=name, auto_archive_duration=1440)
        return thread, start_text

    async def _handle_notice(self, request: OpenCodePermissionRequest) -> bool:
        """Create/feed the session thread for a notice; True when a message went out."""
        if self._threads is None:
            return False
        if request.notice == "child":
            self._threads.set_parent(request.session_id, request.parent_session_id)
            return False
        thread = await self._thread_channel(request.session_id)
        if request.notice == "start" and thread is None:
            thread, pending = await self._open_session_thread(request)
            if thread is None:
                return False
            self._threads.set_thread(request.session_id, str(thread.id), self._config.channel_for(request.agent))
            logger.info("OpenCode bridge: thread %s opened for session %s", thread.id, request.session_id)
            if pending is None:
                return True  # start text already posted (forum post content)
            await thread.send(content=pending, allowed_mentions=self._allowed_mentions())
            return True
        if thread is None:
            # No thread (notice arrived before its start, or start failed):
            # fall back to the agent's channel so nothing is lost.
            thread = await self._channel_for_agent(request.agent)
            if thread is None:
                return False
        await thread.send(content=self._notice_text(request), allowed_mentions=self._allowed_mentions())
        return True

    async def run(self) -> None:
        """Consume the event stream until cancelled."""
        logger.info(
            "OpenCode bridge: streaming %s into Discord channel %s",
            self._config.base_url, self._config.channel_id,
        )
        try:
            async for payload in self._client.stream_events():
                request = parse_permission_event(payload)
                if request is None:
                    continue
                try:
                    await self._post_prompt(request)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "OpenCode bridge: failed to post prompt for %s: %s",
                        request.permission_id, exc,
                    )
        except asyncio.CancelledError:
            pass

    async def aclose(self) -> None:
        for task in (self._task, self._guard_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._task = None
        self._guard_task = None
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Discord prompt
    # ------------------------------------------------------------------

    def _with_mentions(self, content: str) -> str:
        """Prefix the allowlisted users so Discord notifies them.

        A guard prompt that nobody notices simply times out and blocks the
        agent's command (this happened in practice while the user was busy
        in a thread); the ping is the cheapest fix.
        """
        mentions = " ".join(f"<@{uid}>" for uid in sorted(self._config.allowed_user_ids) if uid.isdigit())
        return f"{mentions}\n{content}" if mentions else content

    def _allowed_mentions(self) -> Any:
        if not DISCORD_AVAILABLE:
            return None
        return discord.AllowedMentions(everyone=False, roles=False, users=True, replied_user=False)

    @staticmethod
    def _remaining_minutes(request: OpenCodePermissionRequest) -> Optional[int]:
        if not request.expires_at:
            return None
        return max(1, int(round((request.expires_at - time.time()) / 60)))

    def _guard_prompt_text(self, request: OpenCodePermissionRequest) -> tuple[str, str]:
        """German prompt for a command-guard / permission request."""
        command = _truncate(request.command, _COMMAND_BUDGET).replace("```", "'''")
        agent_label = GUARD_AGENTS.get(request.agent, request.agent)
        remaining = self._remaining_minutes(request)
        if request.tool:
            heading = f"🛡️ **{agent_label} bittet um Erlaubnis: {_truncate(request.tool, 60)}**"
        else:
            heading = "🛡️ **Befehlswächter: Zugriff außerhalb des Projekts**"
        lines = [heading, "", f"**Agent:** {agent_label}"]
        if request.tool:
            lines.append(f"**Werkzeug:** `{_truncate(request.tool, 60)}`")
        lines += ["**Befehl:**", f"```bash\n{command}\n```"]
        if request.details:
            details = _truncate(request.details, _DETAILS_BUDGET).replace("```", "'''")
            lines += ["**Details:**", f"```\n{details}\n```"]
        if request.path and request.path != "-":
            lines.append(f"**Pfad:** `{_truncate(request.path, _PATH_BUDGET)}`")
        lines.append(f"**Projektordner:** `{_truncate(request.project or '-', _PATH_BUDGET)}`")
        if request.access and request.access != "-":
            lines.append(f"**Zugriff:** {request.access}")
        lines += [
            f"**Session:** `{request.short_session_id}`",
            "",
            "**Einmal erlauben** lässt genau diese Aktion einmal durch, **Ablehnen** blockiert sie. "
            + (f"Ohne Antwort innerhalb von etwa {remaining} min wird automatisch abgelehnt. " if remaining else "Ohne Antwort wird automatisch abgelehnt. ")
            + "Dauerhafte Freigaben gibt es nicht.",
        ]
        return "\n".join(lines), f"```bash\n{command}\n```"

    def _question_prompt_text(self, request: OpenCodePermissionRequest, index: int) -> str:
        """German prompt for question ``index`` of a question request."""
        q = request.questions[index]
        agent_label = GUARD_AGENTS.get(request.agent, request.agent)
        remaining = self._remaining_minutes(request)
        total = len(request.questions)
        counter = f" ({index + 1}/{total})" if total > 1 else ""
        header = f" · {_truncate(q['header'], 60)}" if q.get("header") else ""
        lines = [
            f"❓ **{agent_label} fragt{counter}{header}**",
            "",
            _truncate(q["question"], _QUESTION_BUDGET),
            "",
        ]
        for opt in q["options"]:
            desc = f" — {_truncate(opt['description'], 200)}" if opt.get("description") else ""
            lines.append(f"• **{_truncate(opt['label'], _OPTION_LABEL_BUDGET)}**{desc}")
        lines += [
            "",
            f"**Projektordner:** `{_truncate(request.project or '-', _PATH_BUDGET)}` · **Session:** `{request.short_session_id}`",
            "",
            ("Mehrere Antworten möglich: anklicken und mit **Fertig** abschließen. " if q.get("multiSelect") else "Eine Antwort anklicken. ")
            + "**Andere Antwort…** öffnet ein Textfeld. "
            + (f"Ohne Antwort innerhalb von etwa {remaining} min wird die Frage abgebrochen." if remaining else "Ohne Antwort wird die Frage abgebrochen."),
        ]
        return "\n".join(lines)

    def _prompt_text(self, request: OpenCodePermissionRequest) -> tuple[str, str]:
        """Return (plain content, embed description) for the request."""
        if request.is_guard:
            return self._guard_prompt_text(request)
        title = _truncate(request.title or "(no title)", _TITLE_BUDGET)
        kind = request.kind or "permission"
        lines = [
            "🔐 **OpenCode Permission Request**",
            "",
            f"**Type:** `{_truncate(kind, 60)}`",
            f"**Request:**",
            f"```bash\n{title}\n```",
        ]
        if request.pattern:
            lines.append(f"**Pattern:** `{_truncate(request.pattern, _PATTERN_BUDGET)}`")
        lines.append(f"**Session:** `{request.short_session_id}`")
        lines.append("")
        lines.append("Answer **Accept** to allow this operation once, or **Reject** to deny it. Nothing is ever allowed persistently from Discord.")
        content = "\n".join(lines)

        embed_desc = f"```bash\n{title}\n```"
        return content, embed_desc

    async def _post_prompt(self, request: OpenCodePermissionRequest) -> bool:
        """Post one Accept/Reject prompt; True when a message went out."""
        if not self._registry.register(request.permission_id):
            return False
        channel = await self._target_channel(request) if request.is_guard else await self._main_channel()
        if channel is None:
            self._registry.resolve(request.permission_id, "drop")
            logger.warning(
                "OpenCode bridge: channel %s not found, dropping %s",
                self._config.channel_for(request.agent), request.permission_id,
            )
            return False

        if request.is_question:
            return await self._post_question_prompts(channel, request)

        content, embed_desc = self._prompt_text(request)
        if request.is_guard:
            embed = discord.Embed(
                title="🛡️ Befehlswächter" if not request.tool else f"🛡️ Erlaubnis: {_truncate(request.tool, 60)}",
                description=embed_desc,
                color=discord.Color.gold(),
            )
            if request.path and request.path != "-":
                embed.add_field(name="Pfad", value=_truncate(request.path, _PATH_BUDGET), inline=False)
            embed.add_field(name="Projektordner", value=_truncate(request.project or "-", _PATH_BUDGET), inline=False)
            if request.access and request.access != "-":
                embed.add_field(name="Zugriff", value=request.access, inline=True)
            embed.add_field(name="Agent", value=GUARD_AGENTS.get(request.agent, request.agent), inline=True)
        else:
            metadata_line = ""
            if request.metadata:
                metadata_line = _truncate(json.dumps(request.metadata, default=str), _METADATA_BUDGET)
            embed = discord.Embed(
                title="🔐 OpenCode Permission",
                description=embed_desc,
                color=discord.Color.gold(),
            )
            if metadata_line:
                embed.add_field(name="Details", value=f"```json\n{metadata_line}\n```", inline=False)

        timeout = self._config.timeout_seconds
        if request.is_guard and request.expires_at:
            # Never keep buttons alive past the guard's own deadline: the
            # command is already denied by then, a late click must not
            # look like it did anything.
            timeout = max(1, min(timeout, int(request.expires_at - time.time())))
        view = _get_view_class()(
            bridge=self,
            request=request,
            allowed_user_ids=self._config.allowed_user_ids,
            timeout=timeout,
        )
        if request.is_guard:
            content = self._with_mentions(content)
        try:
            msg = await channel.send(
                content=content, embed=embed, view=view, allowed_mentions=self._allowed_mentions()
            )
        except BaseException:
            # Nothing reached Discord: free the slot so the request is not
            # stuck "pending" forever (the guard's own timeout denies it).
            self._registry.resolve(request.permission_id, "drop")
            raise
        view._message = msg
        return True

    async def _post_question_prompts(self, channel: Any, request: OpenCodePermissionRequest) -> bool:
        """One message per question; the set resolves once all are answered."""
        timeout = self._config.timeout_seconds
        if request.expires_at:
            timeout = max(1, min(timeout, int(request.expires_at - time.time())))
        session = QuestionSession(self, request)
        view_class, _ = _get_question_classes()
        views = []
        try:
            for index in range(len(request.questions)):
                view = view_class(
                    session=session,
                    index=index,
                    allowed_user_ids=self._config.allowed_user_ids,
                    timeout=timeout,
                )
                content = self._question_prompt_text(request, index)
                if index == 0:
                    content = self._with_mentions(content)
                msg = await channel.send(
                    content=content, view=view, allowed_mentions=self._allowed_mentions()
                )
                view._message = msg
                views.append(view)
        except BaseException:
            self._registry.resolve(request.permission_id, "drop")
            for view in views:
                view.stop()
            raise
        session.views = views
        return True

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    async def resolve(
        self,
        request: OpenCodePermissionRequest,
        response: str,
        source: str,
        answers: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Resolve a request and deliver the reply.

        ``response`` is ``"once"`` or ``"reject"`` (``"answer"`` with
        ``answers`` for question requests). Returns a short outcome label
        the caller renders in Discord. First-wins: a late click after
        another client (TUI) answered or the timeout fired never
        double-posts a reply.
        """
        if response == "answer" and not (request.is_question and isinstance(answers, dict)):
            response = "reject"
        if response not in ("once", "reject", "answer"):
            response = "reject"
        if not self._registry.resolve(request.permission_id, response):
            return "already-resolved"
        if request.is_guard:
            if self._spool is None:
                return "reply-failed"
            try:
                self._spool.write_decision(request.permission_id, response, source, answers=answers)
            except Exception as exc:
                logger.error(
                    "OpenCode bridge: writing guard decision for %s failed: %s",
                    request.permission_id, exc,
                )
                return "reply-failed"
            logger.info(
                "OpenCode bridge: guard request %s answered %s (%s)",
                request.permission_id, response, source,
            )
            return "delivered"
        try:
            delivered, status = await self._client.reply(
                request.session_id, request.permission_id, response, request.reply_api
            )
        except Exception as exc:
            logger.error(
                "OpenCode bridge: reply POST failed for %s: %s",
                request.permission_id, exc,
            )
            return "reply-failed"
        if not delivered:
            if status == 404:
                return "resolved-elsewhere"
            return "reply-failed"
        logger.info(
            "OpenCode bridge: %s answered %s (%s)", request.permission_id, response, source
        )
        return "delivered"


class QuestionSession:
    """Collects the answers of one question request across its messages.

    First complete answer set wins; a timeout on any unanswered question
    rejects the whole request (the agent then gets no answers at all
    rather than a partial, misleading set).
    """

    def __init__(self, bridge: OpenCodeBridge, request: OpenCodePermissionRequest) -> None:
        self.bridge = bridge
        self.request = request
        self.answers: Dict[str, Any] = {}
        self.views: List[Any] = []
        self.finished = False

    @property
    def complete(self) -> bool:
        return len(self.answers) == len(self.request.questions)

    async def answer(self, index: int, value: Any) -> str:
        """Record one answer; returns the outcome once the set is complete."""
        if self.finished:
            return "already-resolved"
        question = self.request.questions[index]["question"]
        self.answers[question] = value
        if not self.complete:
            return "pending"
        self.finished = True
        outcome = await self.bridge.resolve(self.request, "answer", "discord", answers=dict(self.answers))
        for view in self.views:
            view.stop()
        return outcome

    async def abort(self, source: str) -> str:
        if self.finished:
            return "already-resolved"
        self.finished = True
        outcome = await self.bridge.resolve(self.request, "reject", source)
        for view in self.views:
            view.stop()
        return outcome


_VIEW_CLASS = None
_QUESTION_CLASSES = None


def _get_question_classes():
    """Build (once) the question view and free-text modal; requires discord.py."""
    global _QUESTION_CLASSES
    if _QUESTION_CLASSES is not None:
        return _QUESTION_CLASSES
    if not DISCORD_AVAILABLE:
        raise RuntimeError("discord.py is not installed")

    class FreeTextModal(discord.ui.Modal):
        """Free-text answer for one question ("Andere Antwort…")."""

        def __init__(self, view: "QuestionView") -> None:
            super().__init__(title=_truncate(view.question["header"] or "Andere Antwort", 45))
            self._view = view
            self.text = discord.ui.TextInput(
                label="Antwort",
                style=discord.TextStyle.paragraph,
                max_length=_FREE_TEXT_BUDGET,
                required=True,
            )
            self.add_item(self.text)

        async def on_submit(self, interaction: discord.Interaction) -> None:
            await self._view.submit(interaction, str(self.text.value).strip())

    class QuestionView(discord.ui.View):
        """Buttons for one question: one per option, plus free text / done."""

        def __init__(
            self,
            session: QuestionSession,
            index: int,
            allowed_user_ids: frozenset,
            timeout: int,
        ) -> None:
            super().__init__(timeout=timeout)
            self._session = session
            self._index = index
            self._allowed_user_ids = allowed_user_ids
            self._message = None
            self._selected: List[str] = []
            self._done = False
            self.question = session.request.questions[index]
            for opt in self.question["options"]:
                button = discord.ui.Button(
                    label=_truncate(opt["label"], _OPTION_LABEL_BUDGET),
                    style=discord.ButtonStyle.primary,
                )
                button.callback = self._option_callback(button, opt["label"])
                self.add_item(button)
            other = discord.ui.Button(label="Andere Antwort…", style=discord.ButtonStyle.secondary)
            other.callback = self._other_callback
            self.add_item(other)
            if self.question["multiSelect"]:
                done = discord.ui.Button(label="Fertig", style=discord.ButtonStyle.success)
                done.callback = self._done_callback
                self.add_item(done)

        def _authorized(self, interaction: discord.Interaction) -> bool:
            user = getattr(interaction, "user", None)
            uid = str(getattr(user, "id", "") or "")
            return bool(uid) and uid in self._allowed_user_ids

        async def _guard(self, interaction: discord.Interaction) -> bool:
            if not self._authorized(interaction):
                await interaction.response.send_message("Du darfst diese Frage nicht beantworten.", ephemeral=True)
                return False
            if self._done or self._session.finished:
                await interaction.response.send_message("Diese Frage wurde bereits beantwortet.", ephemeral=True)
                return False
            return True

        def _option_callback(self, button: Any, label: str) -> Callable[[Any], Awaitable[None]]:
            async def callback(interaction: discord.Interaction) -> None:
                if not await self._guard(interaction):
                    return
                if self.question["multiSelect"]:
                    if label in self._selected:
                        self._selected.remove(label)
                        button.style = discord.ButtonStyle.primary
                    else:
                        self._selected.append(label)
                        button.style = discord.ButtonStyle.success
                    await self._edit(interaction)
                    return
                await self.submit(interaction, label)
            return callback

        async def _other_callback(self, interaction: discord.Interaction) -> None:
            if not await self._guard(interaction):
                return
            await interaction.response.send_modal(FreeTextModal(self))

        async def _done_callback(self, interaction: discord.Interaction) -> None:
            if not await self._guard(interaction):
                return
            if not self._selected:
                await interaction.response.send_message("Bitte erst mindestens eine Antwort anklicken.", ephemeral=True)
                return
            await self.submit(interaction, list(self._selected))

        async def submit(self, interaction: discord.Interaction, value: Any) -> None:
            if self._done or self._session.finished:
                await interaction.response.send_message("Diese Frage wurde bereits beantwortet.", ephemeral=True)
                return
            self._done = True
            for child in self.children:
                child.disabled = True
            shown = ", ".join(value) if isinstance(value, list) else str(value)
            outcome = await self._session.answer(self._index, value)
            suffix = {
                "pending": "✅ Antwort gespeichert — bitte auch die anderen Fragen beantworten.",
                "delivered": "✅ Antwort übermittelt.",
                "reply-failed": "⚠️ Antwort konnte nicht abgelegt werden — die Frage bleibt unbeantwortet.",
            }.get(outcome, outcome)
            await self._edit(interaction, footer=f"**Antwort:** {_truncate(shown, 300)}\n{suffix}")

        async def _edit(self, interaction: discord.Interaction, footer: Optional[str] = None) -> None:
            content = None
            if footer is not None:
                base = getattr(interaction.message, "content", "") or ""
                content = f"{base}\n\n{footer}"
            try:
                if content is None:
                    await interaction.response.edit_message(view=self)
                else:
                    await interaction.response.edit_message(content=content, view=self)
            except Exception:
                logger.debug("OpenCode bridge: could not edit question message")

        async def on_timeout(self) -> None:
            if self._done or self._session.finished:
                return
            self._done = True
            for child in self.children:
                child.disabled = True
            await self._session.abort("timeout")
            msg = self._message
            if msg is None:
                return
            try:
                base = getattr(msg, "content", "") or ""
                await msg.edit(content=f"{base}\n\n⏱ Zeit abgelaufen — Frage abgebrochen.", view=self)
            except Exception:
                pass

    _QUESTION_CLASSES = (QuestionView, FreeTextModal)
    return _QUESTION_CLASSES


def _get_view_class():
    """Build (once) the Accept/Reject view; requires discord.py."""
    global _VIEW_CLASS
    if _VIEW_CLASS is not None:
        return _VIEW_CLASS
    if not DISCORD_AVAILABLE:
        raise RuntimeError("discord.py is not installed")

    class OpenCodePermissionView(discord.ui.View):
        """Two-button Accept/Reject prompt for one OpenCode permission.

        Authorization is the bridge's explicit allowlist (fail-closed:
        nobody outside it can click). The timeout resolves to an explicit
        reject so the OpenCode session never hangs. There is deliberately
        no persistent-allow button.
        """

        def __init__(
            self,
            bridge: OpenCodeBridge,
            request: OpenCodePermissionRequest,
            allowed_user_ids: frozenset,
            timeout: int,
        ) -> None:
            super().__init__(timeout=timeout)
            self._bridge = bridge
            self._request = request
            self._allowed_user_ids = allowed_user_ids
            self._message = None
            self._finished = False
            if request.is_guard:
                labels = {"Accept (once)": "Einmal erlauben", "Reject": "Ablehnen"}
                for child in self.children:
                    label = getattr(child, "label", None)
                    if label in labels:
                        child.label = labels[label]

        def _authorized(self, interaction: discord.Interaction) -> bool:
            user = getattr(interaction, "user", None)
            uid = str(getattr(user, "id", "") or "")
            return bool(uid) and uid in self._allowed_user_ids

        async def _answer(self, interaction: discord.Interaction, response: str) -> None:
            if not self._authorized(interaction):
                await interaction.response.send_message(
                    "Du darfst diese Anfrage nicht beantworten."
                    if self._request.is_guard
                    else "You're not allowed to answer OpenCode permission requests~",
                    ephemeral=True,
                )
                return
            outcome = await self._bridge.resolve(self._request, response, "discord")
            if outcome == "already-resolved":
                await interaction.response.send_message(
                    "Diese Anfrage wurde bereits beantwortet."
                    if self._request.is_guard
                    else "This request was already resolved~",
                    ephemeral=True,
                )
                return
            await self._finalize(interaction, response, outcome)

        async def _finalize(
            self,
            interaction: discord.Interaction,
            response: str,
            outcome: str,
        ) -> None:
            self._finished = True
            for child in self.children:
                child.disabled = True
            embed = interaction.message.embeds[0] if interaction.message.embeds else None
            if embed:
                if response == "once":
                    embed.color = (
                        discord.Color.green()
                        if outcome == "delivered"
                        else discord.Color.dark_grey()
                    )
                else:
                    embed.color = (
                        discord.Color.red()
                        if outcome == "delivered"
                        else discord.Color.dark_grey()
                    )
                if self._request.is_guard:
                    footer = {
                        "delivered": "Einmal erlaubt" if response == "once" else "Abgelehnt",
                        "reply-failed": "Antwort konnte nicht abgelegt werden — der Befehl bleibt blockiert",
                    }.get(outcome, outcome)
                else:
                    footer = {
                        "delivered": f"{'Accepted (once)' if response == 'once' else 'Rejected'}",
                        "resolved-elsewhere": "Already resolved by another OpenCode client",
                        "reply-failed": "Reply to OpenCode failed — answer in OpenCode directly",
                    }.get(outcome, outcome)
                embed.set_footer(text=footer)
            try:
                await interaction.response.edit_message(embed=embed, view=self)
            except Exception:
                logger.debug("OpenCode bridge: could not edit prompt message")

        async def _annotate_only(self, response: str, outcome: str) -> None:
            """Edit the message without an interaction (timeout path)."""
            self._finished = True
            for child in self.children:
                child.disabled = True
            msg = self._message
            if msg is None:
                return
            embed = msg.embeds[0] if msg.embeds else None
            if embed:
                embed.color = discord.Color.dark_grey()
                embed.set_footer(
                    text="⏱ Zeit abgelaufen — abgelehnt"
                    if self._request.is_guard
                    else "⏱ Timed out — rejected (fail-closed)"
                )
            try:
                await msg.edit(embed=embed, view=self)
            except Exception:
                pass  # message deleted or too old to edit

        @discord.ui.button(label="Accept (once)", style=discord.ButtonStyle.green)
        async def accept(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ) -> None:
            await self._answer(interaction, "once")

        @discord.ui.button(label="Reject", style=discord.ButtonStyle.red)
        async def reject(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ) -> None:
            await self._answer(interaction, "reject")

        async def on_timeout(self) -> None:
            if self._finished:
                return
            await self._bridge.resolve(self._request, "reject", "timeout")
            await self._annotate_only("reject", "timeout")

    _VIEW_CLASS = OpenCodePermissionView
    return _VIEW_CLASS

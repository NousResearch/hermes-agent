"""Persistent `claude -p` live-session manager (prompt-cache warm reuse).

Mirrors OpenClaw's ``claude-live-session.ts``: one long-lived
``claude -p --input-format stream-json`` child per Hermes conversation, held
warm across interleaved MCP tool cycles so the model's prompt cache stays hot
(~99.7% cache-read hit in the bench harness). The heavy machinery lives here;
``agent/claude_live_client.py`` drives turns and owns the MCP tool socket.

Responsibilities of this module (kept free of any Hermes-tool coupling so it is
independently testable with a fake process):

  * ``LiveSessionConfig`` — immutable spawn recipe + content fingerprint.
  * ``LiveSession`` — a single warm child: spawn / send-one-turn / teardown,
    with a dual watchdog (extend the quiet budget while a tool_use is
    outstanding) and clean detached-process-group teardown.
  * ``LiveSessionRegistry`` — ``Map[session_key -> LiveSession]`` with an LRU
    cap, idle timeout, fingerprint-mismatch respawn, and crash ``--resume``
    with orphaned-tool-use reseed.

Nothing here executes tools or knows about Hermes internals; the client injects
behaviour through plain values and callbacks.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Tunables (all env-overridable so ops can adjust without a redeploy).
# ---------------------------------------------------------------------------

_DEFAULT_IDLE_TIMEOUT_S = 600.0   # 10 min — evict a session idle this long.
_DEFAULT_LRU_CAP = 16             # Max concurrent warm children.
_DEFAULT_FRESH_QUIET_S = 90.0     # Cold-start: budget until first output.
_DEFAULT_RESUME_QUIET_S = 45.0    # Warm/resume: shorter first-output budget.
_DEFAULT_TOOL_QUIET_S = 300.0     # Quiet budget while a tool_use is outstanding.
_DEFAULT_TURN_HARD_DEADLINE_S = 1800.0  # Absolute backstop per turn.
_TEARDOWN_GRACE_S = 5.0
_MAX_REASSEMBLY_BYTES = 8 * 1024 * 1024  # cap the split-object reassembly buffer


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Config / fingerprint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveSessionConfig:
    """Immutable spawn recipe. Its ``fingerprint`` gates session reuse: any
    change (model, effort, system prompt, mcp config, argv, auth identity)
    yields a new fingerprint → the registry tears down and respawns."""

    command: str
    argv: tuple[str, ...]
    cwd: str
    env: dict[str, str]
    model: str
    effort: str
    system_prompt_hash: str
    mcp_config_hash: str
    auth_identity: str

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "effort": self.effort,
                "system_prompt_hash": self.system_prompt_hash,
                "mcp_config_hash": self.mcp_config_hash,
                "argv": list(self.argv),
                "auth_identity": self.auth_identity,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return _sha256_short(payload)


# ---------------------------------------------------------------------------
# Turn result
# ---------------------------------------------------------------------------


@dataclass
class TurnResult:
    """Everything the client needs from one warm turn."""

    text: str = ""
    reasoning: str = ""
    tool_uses: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    rate_limit_events: list[dict[str, Any]] = field(default_factory=list)
    session_id: str = ""
    result_event: Optional[dict[str, Any]] = None
    orphaned_tool_use: bool = False
    timed_out: bool = False


def usage_from_message(usage: Optional[dict[str, Any]]) -> dict[str, int]:
    """Extract the four token counters from an assistant/result usage block."""
    if not isinstance(usage, dict):
        return {}
    keys = (
        "input_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "output_tokens",
    )
    return {k: int(usage.get(k) or 0) for k in keys}


def _accumulate_usage(dst: dict[str, int], src: dict[str, int]) -> dict[str, int]:
    """Return a new dict summing counters (immutable — no in-place mutation)."""
    merged = dict(dst)
    for key, value in src.items():
        merged[key] = merged.get(key, 0) + int(value or 0)
    return merged


# ---------------------------------------------------------------------------
# Stream reader threads
# ---------------------------------------------------------------------------


class _StdoutReader:
    """Background thread: parse stream-json lines into an event queue."""

    def __init__(self, stream: Any):
        self._stream = stream
        self.events: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._buf = ""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            for line in iter(self._stream.readline, ""):
                if line == "":
                    break
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                try:
                    self.events.put(json.loads(line))
                    self._buf = ""
                except json.JSONDecodeError:
                    # stream-json occasionally splits a large object; buffer and
                    # retry. Cap the buffer so a run of unparseable output cannot
                    # grow it without bound or wedge the reader on garbage.
                    self._buf += line
                    try:
                        self.events.put(json.loads(self._buf))
                        self._buf = ""
                    except json.JSONDecodeError:
                        if len(self._buf) > _MAX_REASSEMBLY_BYTES:
                            self._buf = ""  # give up on this fragment, resync
                        continue
        except Exception:
            pass

    def get(self, timeout: float) -> dict[str, Any]:
        return self.events.get(timeout=timeout)


class _StderrReader:
    def __init__(self, stream: Any, maxlines: int = 60):
        from collections import deque

        self._stream = stream
        self.tail: "deque[str]" = deque(maxlen=maxlines)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            for line in iter(self._stream.readline, ""):
                if line == "":
                    break
                self.tail.append(line.rstrip("\n"))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# LiveSession
# ---------------------------------------------------------------------------


class LiveSession:
    """One warm ``claude -p`` child. Turns are serialized (single-flight)."""

    def __init__(
        self,
        config: LiveSessionConfig,
        *,
        popen: Callable[..., Any] = subprocess.Popen,
    ):
        self.config = config
        self._popen = popen
        self.proc: Any = None
        self._out: Optional[_StdoutReader] = None
        self._err: Optional[_StderrReader] = None
        self.session_id: str = ""
        self.last_activity: float = time.monotonic()
        self._turn_lock = threading.Lock()
        self._needs_fresh = False  # set when a prior turn left an orphaned tool_use
        self._spawned_at: float = 0.0
        # True once this live process's transcript already holds the conversation
        # prefix: after ≥1 sent turn, or when spawned via --resume (claude reloads
        # the transcript). A fresh process is False → the client must seed it with
        # the full prior history instead of only the new delta turn. Without this,
        # a fingerprint-drift respawn (/model, /effort switch) or an eviction would
        # silently drop all prior context.
        self.has_prior_context: bool = False

    # -- lifecycle ----------------------------------------------------------

    def spawn(self, *, resume_session_id: Optional[str] = None) -> None:
        argv = list(self.config.argv)
        resuming = bool(resume_session_id) and not self._needs_fresh
        if resuming:
            argv = argv + ["--resume", resume_session_id]
            # A resumed process reloads the full prior transcript, so it already
            # holds the conversation prefix — send only the delta turn.
            self.has_prior_context = True
        # Detached process group so teardown can group-kill and never orphan
        # the MCP bridge child claude itself spawned.
        kwargs: dict[str, Any] = dict(
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=self.config.cwd,
            env=self.config.env,
        )
        if hasattr(os, "setsid"):
            kwargs["start_new_session"] = True
        try:
            self.proc = self._popen(argv, **kwargs)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"claude live session: cannot launch `{self.config.command}` "
                "(the Claude Code CLI is not installed or not on PATH). Install "
                "it with `npm install -g @anthropic-ai/claude-code` and set "
                "HERMES_CLAUDE_CLI_COMMAND if it lives elsewhere."
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"claude live session: failed to launch `{self.config.command}`: {exc}"
            ) from exc
        if self.proc.stdin is None or self.proc.stdout is None:
            self.teardown()
            raise RuntimeError("claude live session: no stdin/stdout pipes")
        self._out = _StdoutReader(self.proc.stdout)
        self._err = _StderrReader(self.proc.stderr)
        self._spawned_at = time.monotonic()
        self.last_activity = self._spawned_at

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def fingerprint(self) -> str:
        return self.config.fingerprint

    def stderr_tail(self) -> str:
        if self._err is None:
            return ""
        return "\n".join(self._err.tail).strip()

    # -- turn driving -------------------------------------------------------

    def send_turn(
        self,
        user_text: str,
        *,
        fresh: bool,
        quiet_budget: Optional[float] = None,
        tool_quiet_budget: Optional[float] = None,
        hard_deadline: Optional[float] = None,
    ) -> TurnResult:
        """Write one user envelope, read events to the ``result`` event.

        Dual watchdog: the quiet budget is the max gap between events; while a
        tool_use is outstanding (seen tool_use, not yet its tool_result) the
        larger ``tool_quiet_budget`` applies, because a Hermes tool may be slow.
        """
        if not self.is_alive():
            raise RuntimeError("claude live session: process is not alive")

        quiet_budget = quiet_budget or (
            _env_float("HERMES_CLAUDE_LIVE_FRESH_QUIET_S", _DEFAULT_FRESH_QUIET_S)
            if fresh
            else _env_float("HERMES_CLAUDE_LIVE_RESUME_QUIET_S", _DEFAULT_RESUME_QUIET_S)
        )
        tool_quiet_budget = tool_quiet_budget or _env_float(
            "HERMES_CLAUDE_LIVE_TOOL_QUIET_S", _DEFAULT_TOOL_QUIET_S
        )
        hard_deadline = hard_deadline or _env_float(
            "HERMES_CLAUDE_LIVE_TURN_DEADLINE_S", _DEFAULT_TURN_HARD_DEADLINE_S
        )

        with self._turn_lock:  # single-flight: one turn per session at a time
            return self._drive_turn(
                user_text,
                quiet_budget=quiet_budget,
                tool_quiet_budget=tool_quiet_budget,
                hard_deadline=hard_deadline,
            )

    def _write_envelope(self, user_text: str) -> None:
        envelope = {
            "type": "user",
            "session_id": self.session_id or "",
            "parent_tool_use_id": None,
            "message": {"role": "user", "content": user_text},
        }
        self.proc.stdin.write(json.dumps(envelope) + "\n")
        self.proc.stdin.flush()

    def _drive_turn(
        self,
        user_text: str,
        *,
        quiet_budget: float,
        tool_quiet_budget: float,
        hard_deadline: float,
    ) -> TurnResult:
        assert self._out is not None
        self._write_envelope(user_text)
        # From here on this live process's transcript holds this turn, so the
        # next turn to the SAME process should be sent as a delta, not a reseed.
        self.has_prior_context = True

        result = TurnResult(session_id=self.session_id)
        outstanding: set[str] = set()
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        hard_stop = time.monotonic() + hard_deadline
        last_event = time.monotonic()

        while True:
            now = time.monotonic()
            if now >= hard_stop:
                result.timed_out = True
                break
            budget = tool_quiet_budget if outstanding else quiet_budget
            if now - last_event > budget:
                result.timed_out = True
                break
            if not self.is_alive():
                break
            try:
                evt = self._out.get(timeout=min(1.0, hard_stop - now))
            except queue.Empty:
                continue
            last_event = time.monotonic()
            self.last_activity = last_event
            done = self._consume_event(
                evt, result, outstanding, text_parts, reasoning_parts
            )
            if done:
                break

        result.text = "\n".join(p.strip() for p in text_parts if p.strip()).strip()
        result.reasoning = "\n".join(
            p.strip() for p in reasoning_parts if p.strip()
        ).strip()
        # Trailing tool_use without its tool_result → transcript is orphaned;
        # a later --resume would be rejected, so flag for a fresh reseed.
        result.orphaned_tool_use = bool(outstanding)
        self._needs_fresh = result.orphaned_tool_use or result.timed_out
        return result

    def _consume_event(
        self,
        evt: dict[str, Any],
        result: TurnResult,
        outstanding: set[str],
        text_parts: list[str],
        reasoning_parts: list[str],
    ) -> bool:
        """Fold one event into ``result``. Returns True on the turn's end."""
        etype = evt.get("type")
        if etype == "system" and evt.get("subtype") == "init":
            sid = evt.get("session_id")
            if isinstance(sid, str) and sid:
                self.session_id = sid
                result.session_id = sid
            return False
        if etype == "rate_limit_event":
            result.rate_limit_events.append(evt)
            return False
        if etype == "assistant":
            self._consume_assistant(evt, result, outstanding, text_parts, reasoning_parts)
            return False
        if etype == "user":
            _consume_tool_results(evt, outstanding)
            return False
        if etype == "result":
            result.result_event = evt
            return True
        return False

    def _consume_assistant(
        self,
        evt: dict[str, Any],
        result: TurnResult,
        outstanding: set[str],
        text_parts: list[str],
        reasoning_parts: list[str],
    ) -> None:
        message = evt.get("message") or {}
        result.usage = _accumulate_usage(
            result.usage, usage_from_message(message.get("usage"))
        )
        for block in message.get("content", []) or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                t = block.get("text")
                if isinstance(t, str) and t.strip():
                    text_parts.append(t)
            elif btype == "thinking":
                t = block.get("thinking") or block.get("text")
                if isinstance(t, str) and t.strip():
                    reasoning_parts.append(t)
            elif btype == "tool_use":
                tid = block.get("id")
                if isinstance(tid, str) and tid:
                    outstanding.add(tid)
                result.tool_uses.append(
                    {
                        "id": tid,
                        "name": block.get("name"),
                        "input": block.get("input"),
                    }
                )

    # -- teardown -----------------------------------------------------------

    def teardown(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        _kill_process_tree(proc)


def _consume_tool_results(evt: dict[str, Any], outstanding: set[str]) -> None:
    message = evt.get("message") or {}
    content = message.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            tid = block.get("tool_use_id")
            if isinstance(tid, str):
                outstanding.discard(tid)


def _kill_process_tree(proc: Any) -> None:
    """SIGTERM the group, grace, then SIGKILL the group. Guards the no-setsid
    fallback so we never leave a zombie ``claude`` (or its MCP bridge child)."""
    pgid: Optional[int] = None
    if hasattr(os, "getpgid") and hasattr(os, "killpg"):
        try:
            pgid = os.getpgid(proc.pid)
        except Exception:
            pgid = None

    def _signal(sig: int) -> None:
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
                return
            except Exception:
                pass
        try:
            proc.send_signal(sig)
        except Exception:
            pass

    try:
        if proc.poll() is None:
            _signal(signal.SIGTERM)
            try:
                proc.wait(timeout=_TEARDOWN_GRACE_S)
            except Exception:
                _signal(signal.SIGKILL)
                try:
                    proc.wait(timeout=_TEARDOWN_GRACE_S)
                except Exception:
                    pass
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class LiveSessionRegistry:
    """LRU + idle-timeout registry of warm sessions keyed by ``session_key``.

    Eviction is lazy (swept on every access) so behaviour is deterministic and
    unit-testable without a background reaper thread.
    """

    def __init__(
        self,
        *,
        idle_timeout_s: Optional[float] = None,
        lru_cap: Optional[int] = None,
        clock: Callable[[], float] = time.monotonic,
        popen: Callable[..., Any] = subprocess.Popen,
    ):
        self._sessions: "dict[str, LiveSession]" = {}
        self._lock = threading.RLock()
        self._idle_timeout = idle_timeout_s or _env_float(
            "HERMES_CLAUDE_LIVE_IDLE_TIMEOUT_S", _DEFAULT_IDLE_TIMEOUT_S
        )
        self._lru_cap = lru_cap or _env_int("HERMES_CLAUDE_LIVE_LRU_CAP", _DEFAULT_LRU_CAP)
        self._clock = clock
        self._popen = popen

    def get_or_create(
        self, session_key: str, config: LiveSessionConfig
    ) -> LiveSession:
        # ``teardown()`` blocks (SIGTERM grace → SIGKILL) so it is done OUTSIDE
        # the lock; holding the lock across a kill would stall every other
        # conversation's turn behind one slow child.
        doomed: list[LiveSession] = []
        with self._lock:
            doomed.extend(self._detach_idle())
            existing = self._sessions.get(session_key)
            if existing is not None:
                if existing.is_alive() and existing.fingerprint == config.fingerprint:
                    existing.last_activity = self._clock()
                    self._teardown_all(doomed)
                    return existing
                # Fingerprint drift (/model, /effort, tool change) or dead
                # child → detach for teardown and respawn fresh.
                doomed.append(existing)
                del self._sessions[session_key]
            session = LiveSession(config, popen=self._popen)
            session.spawn()
            session.last_activity = self._clock()
            self._sessions[session_key] = session
            doomed.extend(self._detach_over_cap())
        self._teardown_all(doomed)
        return session

    def recover(self, session_key: str) -> Optional[LiveSession]:
        """Respawn a crashed session with ``--resume`` (or fresh if orphaned)."""
        doomed: list[LiveSession] = []
        with self._lock:
            session = self._sessions.get(session_key)
            if session is None:
                return None
            captured = session.session_id
            orphaned = session._needs_fresh
            doomed.append(session)
            fresh = LiveSession(session.config, popen=self._popen)
            fresh._needs_fresh = orphaned
            fresh.spawn(resume_session_id=None if orphaned else captured)
            self._sessions[session_key] = fresh
        self._teardown_all(doomed)
        return fresh

    def drop(self, session_key: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_key, None)
        if session is not None:
            session.teardown()

    def shutdown(self) -> None:
        with self._lock:
            doomed = list(self._sessions.values())
            self._sessions.clear()
        self._teardown_all(doomed)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    # -- internal -----------------------------------------------------------

    @staticmethod
    def _teardown_all(sessions: list[LiveSession]) -> None:
        """Tear down detached sessions. Called with NO lock held so the blocking
        SIGTERM/SIGKILL grace never stalls other conversations."""
        for session in sessions:
            try:
                session.teardown()
            except Exception:
                pass

    def _detach_idle(self) -> list[LiveSession]:
        """Remove dead/idle sessions from the map and RETURN them for teardown
        by the caller outside the lock. Must be called under ``self._lock``."""
        now = self._clock()
        stale = [
            key
            for key, s in self._sessions.items()
            if not s.is_alive() or (now - s.last_activity) > self._idle_timeout
        ]
        detached: list[LiveSession] = []
        for key in stale:
            detached.append(self._sessions.pop(key))
        return detached

    def _detach_over_cap(self) -> list[LiveSession]:
        """Evict least-recently-active sessions above the cap; return them for
        teardown outside the lock. Must be called under ``self._lock``."""
        detached: list[LiveSession] = []
        while len(self._sessions) > self._lru_cap:
            victim = min(self._sessions.items(), key=lambda kv: kv[1].last_activity)[0]
            detached.append(self._sessions.pop(victim))
        return detached


# Process-wide singleton registry (one per Hermes process).
_REGISTRY: Optional[LiveSessionRegistry] = None
_REGISTRY_LOCK = threading.Lock()


def get_registry() -> LiveSessionRegistry:
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            _REGISTRY = LiveSessionRegistry()
        return _REGISTRY


def _shutdown_registry() -> None:
    """Kill every warm ``claude`` child at interpreter exit. Their process groups
    are detached (start_new_session=True) so they survive parent death otherwise
    — a leaked child per active conversation on every gateway restart."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        registry = _REGISTRY
    if registry is not None:
        try:
            registry.shutdown()
        except Exception:
            pass


atexit.register(_shutdown_registry)

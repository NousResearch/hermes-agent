"""Shared oracle helpers for the chaos (agent-turn liveness) E2E lane.

The chaos suites drive a REAL Hermes surface (AIAgent child process, GatewayRunner,
tui_gateway JSON-RPC subprocess) against ``tests/fakes/fake_llm_provider`` in a fault
mode and then check the same liveness invariants everywhere:

* bounded termination — the turn ends (answer or surfaced error) within the
  configured deadline plus a generous epsilon, never "whenever the fault clears";
* bounded provider calls — no retry storm;
* reusability — the same session accepts and answers a new message afterwards;
* history integrity — every assistant ``tool_call`` has a matching ``tool`` row,
  both in state.db and in what is sent back to the model;
* process hygiene — no process carrying the scenario's tag survives the session.

Everything here is surface-agnostic; surface drivers live next to their test file.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[4]

# One scenario's liveness budget. The fake faults last an HOUR; the configured
# deadlines below are seconds, so a turn that is still running at DEADLINE is
# waiting on the fault, not merely slow — margins are >10x the healthy runtime.
STALE_TIMEOUT_S = 3
REQUEST_TIMEOUT_S = 5
API_MAX_RETRIES = 2
MAX_TURNS = 8
TOOL_TIMEOUT_S = 3
TURN_DEADLINE_S = 90.0
# A retry storm is dozens-to-thousands of calls (#92450: ~64/s). Healthy fault
# handling with API_MAX_RETRIES=2 stays in single digits (stream retries x API retries).
MAX_PROVIDER_CALLS_PER_FAILED_TURN = 16
INTERRUPT_DEADLINE_S = 15.0
# Interrupt scenarios run with deadlines far beyond INTERRUPT_DEADLINE_S so only a
# working interrupt can end the turn in time.
LONG_TIMEOUT_S = 600
PROCESS_REAP_S = 15.0

PLAIN_MODEL = "fake-model"
# A slug with a reasoning stale-timeout floor: an explicit
# providers.<id>.stale_timeout_seconds must still win over the floor (#115024).
REASONING_MODEL = "deepseek-r1"


def new_tag() -> str:
    return f"chaos{uuid.uuid4().hex[:12]}"


def chaos_config(
    base_url: str,
    *,
    model: str = PLAIN_MODEL,
    stale_timeout: float = STALE_TIMEOUT_S,
    request_timeout: float = REQUEST_TIMEOUT_S,
    api_max_retries: int = API_MAX_RETRIES,
    max_turns: int = MAX_TURNS,
    extra: str = "",
) -> str:
    """config.yaml text with every liveness knob short and explicit (real keys:
    ``agent.api_max_retries``, ``agent.auto_recovery_cycles``, ``agent.max_turns``,
    ``providers.<id>.request_timeout_seconds`` / ``stale_timeout_seconds``)."""
    return (
        "model:\n"
        "  provider: custom\n"
        f"  base_url: {base_url}\n"
        f"  default: {model}\n"
        "  context_length: 128000\n"
        "agent:\n"
        f"  api_max_retries: {api_max_retries}\n"
        "  auto_recovery_cycles: 0\n"
        f"  max_turns: {max_turns}\n"
        "providers:\n"
        "  custom:\n"
        f"    request_timeout_seconds: {request_timeout}\n"
        f"    stale_timeout_seconds: {stale_timeout}\n"
        "compression:\n"
        "  enabled: false\n"
        "memory:\n"
        "  memory_enabled: false\n"
        "  user_profile_enabled: false\n"
        # Offline: the passive update check does a GitHub round-trip and, on a partial clone
        # whose objects lag upstream, spawns a git lazy fetch that outlives the gateway.
        "updates:\n"
        "  check: false\n"
        + extra
    )


def write_chaos_home(root: Path, base_url: str, **cfg: Any) -> tuple[Path, Path]:
    """Create ``root/home`` (fake $HOME) and ``root/home/.hermes`` (HERMES_HOME)."""
    home = root / "home"
    hermes_home = home / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text(chaos_config(base_url, **cfg), encoding="utf-8")
    (hermes_home / ".env").write_text("OPENAI_API_KEY=sk-fake-chaos\n", encoding="utf-8")
    return home, hermes_home


def hermetic_env(home: Path, hermes_home: Path, tag: str) -> dict[str, str]:
    """Child env: fake HOME/HERMES_HOME, no real credentials, repo importable, and a
    tag every descendant inherits so the orphan scan can find it after reparenting."""
    env = {
        k: v for k, v in os.environ.items()
        if not (
            k.endswith(("_API_KEY", "_TOKEN", "_SECRET"))
            or k.startswith(("HERMES_", "OPENROUTER", "ANTHROPIC", "OPENAI", "NOUS_"))
            or k in {"PYTEST_CURRENT_TEST"}
        )
    }
    env.update({
        "HOME": str(home),
        "HERMES_HOME": str(hermes_home),
        "OPENAI_API_KEY": "sk-fake-chaos",
        "PYTHONPATH": str(REPO_ROOT),
        "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
        "CHAOS_TAG": tag,
        # HOME *is* the tmp root, so the live-DB guard (pytest ancestry) would read the
        # tmp state.db as "production"; the documented child opt-out is safe here.
        "HERMES_STATE_DB_GUARD_BYPASS": "1",
        "TZ": "UTC",
        "NO_COLOR": "1",
    })
    return env


# ── process hygiene ─────────────────────────────────────────────────────────


def tagged_pids(tag: str, *, exclude: Iterable[int] = ()) -> list[int]:
    """PIDs whose environment carries ``CHAOS_TAG=<tag>`` (survives reparenting to init)."""
    needle = f"CHAOS_TAG={tag}".encode()
    skip = {os.getpid(), *exclude}
    found: list[int] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) in skip:
            continue
        try:
            with open(f"/proc/{entry}/environ", "rb") as fh:
                data = fh.read()
            with open(f"/proc/{entry}/stat", "rb") as fh:
                state = fh.read().rsplit(b")", 1)[1].split()[0]
        except OSError:
            continue
        if state == b"Z":
            continue  # zombie: already dead, only waiting on its parent's reap
        if needle in data.split(b"\0"):
            found.append(int(entry))
    return found


def describe_pids(pids: Iterable[int]) -> list[str]:
    out = []
    for pid in pids:
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            cmd = "?"
        out.append(f"{pid}: {cmd[:160]}")
    return out


def wait_no_tagged(tag: str, timeout: float = PROCESS_REAP_S) -> list[int]:
    """Poll until no tagged process remains; return survivors at the deadline."""
    deadline = time.monotonic() + timeout
    while True:
        left = tagged_pids(tag)
        if not left or time.monotonic() >= deadline:
            return left
        time.sleep(0.1)


def kill_tagged(tag: str) -> None:
    """Test cleanup only: SIGKILL anything still carrying our tag (never pkill -f).

    The conftest live-system guard refuses os.kill on PIDs reparented out of the test's
    subtree — exactly the orphans a red run leaves behind. The tag proves they are ours,
    so fall back to a pidfd signal (also immune to PID reuse) instead of leaking them."""
    for pid in tagged_pids(tag):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        except RuntimeError:
            pidfd_open = getattr(os, "pidfd_open", None)  # absent on some Python builds
            if pidfd_open is None:
                continue
            try:
                fd = pidfd_open(pid)
            except OSError:
                continue
            try:
                signal.pidfd_send_signal(fd, signal.SIGKILL)
            except OSError:
                pass
            finally:
                os.close(fd)


# ── history integrity ───────────────────────────────────────────────────────


def _call_ids(msg: dict[str, Any]) -> list[str]:
    calls = msg.get("tool_calls") or []
    if isinstance(calls, str):
        try:
            calls = json.loads(calls)
        except json.JSONDecodeError:
            return ["<unparseable tool_calls>"]
    return [c.get("id") or c.get("call_id") or "<missing id>" for c in calls if isinstance(c, dict)]


def unanswered_tool_calls(messages: list[dict[str, Any]]) -> list[str]:
    """tool_call ids with no ``tool`` message in the block right after their assistant.

    Wire rule (every provider): an assistant ``tool_calls`` message must be followed by
    one ``tool`` result per id before the next assistant/user message.
    """
    missing: list[str] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        ids = _call_ids(msg)
        if not ids:
            continue
        answered = set()
        for follow in messages[i + 1:]:
            if follow.get("role") != "tool":
                break
            answered.add(follow.get("tool_call_id"))
        missing.extend(tid for tid in ids if tid not in answered)
    return missing


def tool_results_by_id(messages: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "tool":
            content = msg.get("content")
            if not isinstance(content, str):
                content = json.dumps(content)
            out[msg.get("tool_call_id") or ""] = content
    return out


def persisted_messages(state_db: Path, session_id: str | None = None) -> list[dict[str, Any]]:
    """Active message rows from a real state.db, oldest first, OpenAI-shaped.

    With ``session_id=None`` the newest session that has messages is used."""
    conn = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True, timeout=10)
    try:
        conn.row_factory = sqlite3.Row
        if session_id is None:
            row = conn.execute(
                "SELECT session_id FROM messages ORDER BY id DESC LIMIT 1").fetchone()
            if row is None:
                return []
            session_id = row["session_id"]
        rows = conn.execute(
            "SELECT role, content, tool_call_id, tool_calls FROM messages "
            "WHERE session_id = ? AND active = 1 ORDER BY id", (session_id,)).fetchall()
    finally:
        conn.close()
    msgs = []
    for r in rows:
        m: dict[str, Any] = {"role": r["role"], "content": r["content"]}
        if r["tool_call_id"]:
            m["tool_call_id"] = r["tool_call_id"]
        if r["tool_calls"]:
            m["tool_calls"] = r["tool_calls"]
        msgs.append(m)
    return msgs


def integrity_ok(state_db: Path) -> str:
    conn = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True, timeout=10)
    try:
        return conn.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        conn.close()


@dataclass
class Timing:
    started: float
    ended: float | None = None

    @property
    def elapsed(self) -> float:
        return (self.ended or time.monotonic()) - self.started


def python_exe() -> str:
    return sys.executable

"""Transcript-ledger harness for the conversation-history (C3) and prompt-cache (C17) suites.

Everything real except the model: a real ``AIAgent`` + ``SessionDB`` on disk in-process, the
real ``tui_gateway`` stdio JSON-RPC server and the real ``hermes chat -q`` oneshot CLI as
subprocesses, all pointed at one recording ``FakeLLMServer``. The invariants below read the
three durable/observable projections of a conversation and cross-check them:

* what the model was sent (the fake's recorded request bodies),
* what the model will be sent on resume (``SessionDB.get_resume_conversations`` model view),
* what a user sees on resume (the display view, compaction archive included),
* what was billed (``sessions`` / ``session_model_usage`` vs the fake's reported usage).
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[4]

# Children get an ALLOWLISTED env: a developer or agent shell exports TERMINAL_CWD,
# HERMES_SESSION_*, AUXILIARY_*, provider keys ... any of which changes the child's prompt or
# routing and would make a surface-switch comparison diverge for reasons outside the product.
_ENV_ALLOW = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "USER", "LOGNAME", "SHELL", "SSL_CERT_FILE", "TZ")

# No network from probes: no update check, models.dev fetch fails fast on a closed port.
OFFLINE_CONFIG = "updates:\n  check: false\nmodels_dev:\n  url: http://127.0.0.1:9/api.json\n"
# The periodic memory/skill background review forks a second agent against the same provider on
# its own thread; its requests would race the scripted turns. Scenarios that want it opt in.
NO_BACKGROUND_REVIEW = "skills:\n  creation_nudge_interval: 0\nmemory:\n  nudge_interval: 0\n"

SUMMARY_TEXT = "Fake summary of the earlier conversation."


# ---------------------------------------------------------------------------------------------
# canonical forms


def canon(obj: Any) -> str:
    """Canonical JSON: key order is not semantic on the wire, every value byte is."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _text(content: Any) -> str:
    return content if isinstance(content, str) else canon(content)


def _args(raw: Any) -> str:
    """Tool-call arguments by meaning: the wire re-serializes the model's JSON compactly while the
    row keeps the model's bytes; the request stream's own byte-stability is ``prefix_breaks``' job."""
    try:
        return canon(json.loads(raw)) if isinstance(raw, str) else canon(raw)
    except ValueError:
        return str(raw)


def _calls(tool_calls: Any) -> tuple:
    out = []
    for tc in tool_calls or ():
        fn = tc.get("function") or {}
        out.append((tc.get("id"), fn.get("name"), _args(fn.get("arguments"))))
    return tuple(out)


def identity(msg: dict[str, Any]) -> tuple:
    """Logical identity of one conversation message as the MODEL sees it.

    User rows replay their ``api_content`` sidecar when present (surface notes, attachments),
    so that is the text the identity uses.
    """
    content = msg.get("api_content") if msg.get("role") == "user" and msg.get("api_content") else msg.get("content")
    return (msg.get("role"), _text(content or ""), msg.get("tool_call_id") or None, _calls(msg.get("tool_calls")))


def short(ident: tuple) -> str:
    role, text, tcid, calls = ident
    extra = f" tool_call_id={tcid}" if tcid else ""
    extra += f" calls={list(calls)}" if calls else ""
    return f"{role}:{text[:70]!r}{extra} #{hashlib.sha1(text.encode()).hexdigest()[:8]}"


def first_divergence(a: str, b: str) -> str:
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return f"first diverging byte at {n}: {a[max(0, n - 60):n + 60]!r}  vs  {b[max(0, n - 60):n + 60]!r}"


# ---------------------------------------------------------------------------------------------
# state.db views


def db_connect(hermes_home: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{hermes_home / 'state.db'}?mode=ro", uri=True, timeout=30)
    con.row_factory = sqlite3.Row
    return con


def views(hermes_home: Path, sid: str) -> tuple[list[dict], list[dict]]:
    """(model_history, display_history) exactly as a resuming surface loads them."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=hermes_home / "state.db")
    try:
        return db.get_resume_conversations(sid)
    finally:
        db.close()


def model_only_identities(hermes_home: Path, model: list[dict]) -> set[tuple]:
    """Identities of model-view rows stored as model-only (never part of the display projection)."""
    ids = [m["_row_id"] for m in model if m.get("_row_id") is not None]
    if not ids:
        return set()
    with db_connect(hermes_home) as con:
        hidden = {r[0] for r in con.execute(
            f"SELECT id FROM messages WHERE id IN ({','.join('?' * len(ids))}) "
            "AND COALESCE(json_extract(display_metadata, '$.model_only'), 0) != 0", ids)}
    return {identity(m) for m in model if m.get("_row_id") in hidden}


def row_counts(hermes_home: Path, sid: str) -> dict[str, int]:
    with db_connect(hermes_home) as con:
        total, active = con.execute(
            "SELECT count(*), coalesce(sum(active), 0) FROM messages WHERE session_id=?", (sid,)).fetchone()
        return {"total": total, "active": active}


def integrity_ok(hermes_home: Path) -> None:
    with db_connect(hermes_home) as con:
        assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def lineage(hermes_home: Path, sid: str) -> list[str]:
    """Root→tip chain of session ids ending at ``sid`` (compaction rotation may create children)."""
    chain = [sid]
    with db_connect(hermes_home) as con:
        while True:
            row = con.execute("SELECT parent_session_id FROM sessions WHERE id=?", (chain[0],)).fetchone()
            if not row or not row[0]:
                return chain
            chain.insert(0, row[0])


# ---------------------------------------------------------------------------------------------
# invariants


def is_summary(msg: dict[str, Any]) -> bool:
    return SUMMARY_TEXT in _text(msg.get("content") or "") or bool(msg.get("_compressed_summary"))


def assert_rows_exactly_once(hermes_home: Path, sid: str, where: str) -> None:
    """(a) at the storage layer: no two ACTIVE rows of a session are the same logical message.
    Readers dedupe defensively, so a double write can hide behind them; the rows cannot."""
    con = db_connect(hermes_home)
    try:
        rows = con.execute(
            "SELECT id, role, content, tool_call_id, tool_calls FROM messages "
            "WHERE session_id = ? AND active = 1 ORDER BY id", (sid,)).fetchall()
    finally:
        con.close()
    seen: dict[tuple, int] = {}
    dups = []
    for row_id, *ident in rows:
        key = tuple(ident)
        if key in seen:
            dups.append((seen[key], row_id, ident[0], str(ident[1])[:60]))
        seen.setdefault(key, row_id)
    assert not dups, f"{where}: the same message is persisted as more than one active row: {dups}"


def assert_exactly_once(msgs: Iterable[dict[str, Any]], where: str) -> None:
    """(a) No logical message appears twice in a projection (every scripted text is unique)."""
    seen: dict[tuple, int] = {}
    for i, m in enumerate(msgs):
        if m.get("role") == "system":
            continue
        k = identity(m)
        assert k not in seen, f"{where}: {short(k)} persisted twice (positions {seen[k]} and {i})"
        seen[k] = i


def model_payload(msgs: Iterable[dict[str, Any]]) -> list[tuple]:
    return [identity(m) for m in msgs if m.get("role") != "system"]


def assert_replay_equals_persisted(request: dict[str, Any], persisted: list[dict], where: str) -> None:
    """(b) The first request of turn N+1 replays exactly the persisted model history + the new user msg."""
    sent = model_payload(request["messages"])
    want = model_payload(persisted)
    assert sent[:-1] == want, (
        f"{where}: history sent to the model != persisted history\n"
        + _list_diff(want, sent[:-1], "persisted", "sent"))
    assert sent[-1][0] == "user", f"{where}: the turn's request does not end with the new user message"


def _list_diff(a: list[tuple], b: list[tuple], an: str, bn: str) -> str:
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    lines = [f"  lengths {an}={len(a)} {bn}={len(b)}; first difference at index {n}"]
    for label, seq in ((an, a), (bn, b)):
        lines.append(f"  {label}[{n}:{n + 3}] = " + "; ".join(short(x) for x in seq[n:n + 3]))
    return "\n".join(lines)


def prefix_breaks(requests: list[dict[str, Any]], *, tools: bool = True) -> list[tuple[int, str]]:
    """(C17) Indices i where request i is NOT a byte-identical extension of request i-1.

    Byte-stability covers the tools array (unless ``tools=False``), the system prompt and every
    earlier message.
    """
    breaks = []
    for i in range(1, len(requests)):
        prev, cur = requests[i - 1], requests[i]
        if tools and canon(prev.get("tools")) != canon(cur.get("tools")):
            breaks.append((i, "tools array changed: " + tools_diff(prev.get("tools"), cur.get("tools"))))
            continue
        pm, cm = prev["messages"], cur["messages"]
        if len(cm) < len(pm):
            breaks.append((i, f"message list shrank {len(pm)} -> {len(cm)}"))
            continue
        for j, (a, b) in enumerate(zip(pm, cm)):
            ca, cb = canon(a), canon(b)
            if ca != cb:
                breaks.append((i, f"messages[{j}] ({a.get('role')}) changed; {first_divergence(ca, cb)}"))
                break
    return breaks


def tools_breaks(requests: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """Indices i where request i sends a different tools array than request i-1."""
    return [(i, tools_diff(requests[i - 1].get("tools"), requests[i].get("tools")))
            for i in range(1, len(requests))
            if canon(requests[i - 1].get("tools")) != canon(requests[i].get("tools"))]


def tools_diff(a: Any, b: Any) -> str:
    an = {t["function"]["name"]: canon(t) for t in a or ()}
    bn = {t["function"]["name"]: canon(t) for t in b or ()}
    if list(an) != list(bn):
        return f"names/order differ: -{sorted(set(an) - set(bn))} +{sorted(set(bn) - set(an))}"
    for name in an:
        if an[name] != bn[name]:
            return f"tool {name!r}: {first_divergence(an[name], bn[name])}"
    return "?"


def assert_no_loss(ever_sent: dict[tuple, str], display: list[dict], model: list[dict], where: str) -> None:
    """Loss-free persistence: every message the model was ever sent is still on disk exactly once in
    the display history, and every message still in the model's working context is there too.
    Compaction may move a message out of the model view (summarized) but never out of display."""
    shown = [identity(m) for m in display if m.get("role") != "system" and not is_summary(m)]
    counts: dict[tuple, int] = {}
    for k in shown:
        counts[k] = counts.get(k, 0) + 1
    dup = [short(k) for k, c in counts.items() if c > 1]
    assert not dup, f"{where}: display history shows messages twice: {dup}"
    missing = [label for k, label in ever_sent.items() if k not in counts]
    assert not missing, (f"{where}: messages the model saw are gone from the durable transcript: {missing}\n"
                         f"  display now: {[short(k) for k in shown]}")
    model_missing = [short(identity(m)) for m in model
                     if m.get("role") != "system" and not is_summary(m) and identity(m) not in counts]
    assert not model_missing, f"{where}: model-view rows absent from display: {model_missing}"


def assert_inputs_shown_once(inputs: list[str], display: list[dict], model: list[dict], where: str) -> None:
    """Every text the user typed is on screen exactly once after a resume, and at most once in
    what the model is fed (compaction may summarize it away, never duplicate or merge-copy it)."""
    def hits(msgs: list[dict], text: str) -> int:
        return sum(_text(m.get("content") or "").count(text) for m in msgs if m.get("role") == "user")
    for text in inputs:
        shown, fed = hits(display, text), hits(model, text)
        assert shown == 1, f"{where}: user input {text[:60]!r} shown {shown}x in the resumed display history"
        assert fed <= 1, f"{where}: user input {text[:60]!r} replayed {fed}x to the model"


def billed_usage(records: list[dict[str, Any]], kind: str = "main") -> dict[str, int]:
    """What the fake provider reported for completed requests of ``kind``."""
    tot = {"calls": 0, "prompt": 0, "completion": 0, "cached": 0}
    for r in records:
        u = r.get("usage")
        if r["kind"] != kind or not u:
            continue
        tot["calls"] += 1
        tot["prompt"] += u.get("prompt_tokens", 0)
        tot["completion"] += u.get("completion_tokens", 0)
        tot["cached"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return tot


def assert_usage_matches(hermes_home: Path, sids: list[str], records: list[dict[str, Any]], where: str) -> None:
    """(f) state.db usage equals what the provider billed: main-loop totals on ``sessions`` and in
    the per-model ledger, no double count, no drop; cache reads are split out of input tokens."""
    want = billed_usage(records, "main")
    marks = ",".join("?" * len(sids))
    with db_connect(hermes_home) as con:
        s = con.execute(
            f"SELECT sum(input_tokens), sum(output_tokens), sum(cache_read_tokens), sum(api_call_count) "
            f"FROM sessions WHERE id IN ({marks})", sids).fetchone()
        u = con.execute(
            f"SELECT sum(input_tokens), sum(output_tokens), sum(cache_read_tokens), sum(api_call_count) "
            f"FROM session_model_usage WHERE session_id IN ({marks}) AND task=''", sids).fetchone()
    expect = (want["prompt"] - want["cached"], want["completion"], want["cached"], want["calls"])
    got_s = tuple(int(x or 0) for x in s)
    got_u = tuple(int(x or 0) for x in u)
    cols = "(input, output, cache_read, api_calls)"
    assert got_s == expect, f"{where}: sessions usage {cols} {got_s} != provider-billed {expect}"
    assert got_u == expect, f"{where}: session_model_usage main rows {cols} {got_u} != provider-billed {expect}"


# ---------------------------------------------------------------------------------------------
# subprocess surfaces


def child_env(home: Path, hermes_home: Path, workdir: Path) -> dict[str, str]:
    env = {k: os.environ[k] for k in _ENV_ALLOW if k in os.environ}
    env.update(
        HOME=str(home), HERMES_HOME=str(hermes_home), PWD=str(workdir),
        HERMES_TEST_ISOLATION=str(hermes_home),
        # The child's state.db IS this test's sandbox db; the live-DB guard would refuse it
        # because a pytest ancestor is present.
        HERMES_STATE_DB_GUARD_BYPASS="1",
        PYTHONPATH=str(REPO_ROOT), PYTHONUNBUFFERED="1", NO_COLOR="1", TMPDIR=str(home),
        TZ="UTC", LANG="C.UTF-8", PYTHONHASHSEED="0",
    )
    return env


class Spawned:
    """PIDs this test started; only these are ever signalled."""

    def __init__(self) -> None:
        self.procs: list[subprocess.Popen] = []

    def add(self, p: subprocess.Popen) -> subprocess.Popen:
        self.procs.append(p)
        return p

    def reap(self) -> list[int]:
        leaked = []
        for p in self.procs:
            if p.poll() is None:
                leaked.append(p.pid)
                p.kill()
                p.wait(timeout=10)
        return leaked


_SID_RE = re.compile(r"^session_id:\s*(\S+)\s*$", re.M)


def run_oneshot(env: dict[str, str], cwd: Path, prompt: str, spawned: Spawned, *,
                resume: str | None = None, timeout: float = 180.0) -> tuple[str, str]:
    """``hermes chat -q PROMPT -Q [--resume SID]`` in a fresh process -> (stdout, durable sid)."""
    cmd = [sys.executable, "-m", "hermes_cli.main", "chat", "-q", prompt, "-Q"]
    if resume:
        cmd += ["--resume", resume]
    p = spawned.add(subprocess.Popen(cmd, cwd=str(cwd), env={**env, "PWD": str(cwd)}, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8"))
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate(timeout=10)
        raise AssertionError(f"oneshot did not finish in {timeout}s; stderr tail: {err[-2000:]}")
    assert p.returncode == 0, f"oneshot rc={p.returncode}; stderr tail: {err[-3000:]}"
    sids = _SID_RE.findall(err)
    assert sids, f"oneshot printed no session_id line; stderr tail: {err[-2000:]}"
    return out, sids[-1]


class GatewayError(RuntimeError):
    pass


class TuiGateway:
    """The real ``python -m tui_gateway.entry`` stdio JSON-RPC server (what the TUI spawns)."""

    def __init__(self, env: dict[str, str], cwd: Path, spawned: Spawned, *, ready_timeout: float = 120.0):
        self.env, self.cwd, self.spawned, self.ready_timeout = env, str(cwd), spawned, ready_timeout
        self.proc: subprocess.Popen | None = None
        self.events: list[dict] = []
        self.stderr_lines: list[str] = []
        self._pending: dict[int, queue.Queue] = {}
        self._cond = threading.Condition()
        self._next_id = 0
        self._wlock = threading.Lock()
        self.stored: dict[str, str] = {}

    def __enter__(self) -> "TuiGateway":
        self.proc = self.spawned.add(subprocess.Popen(
            [sys.executable, "-m", "tui_gateway.entry"], cwd=self.cwd, env={**self.env, "PWD": self.cwd},
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1))
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self.wait_event("gateway.ready", timeout=self.ready_timeout)
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self, timeout: float = 30.0) -> None:
        p = self.proc
        if p is None or p.poll() is not None:
            return
        try:
            p.stdin.close()  # EOF -> orderly shutdown
        except OSError:
            pass
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=10)

    def _read_stdout(self) -> None:
        for raw in self.proc.stdout:
            line = raw.strip()
            if not line:
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            if frame.get("method") == "event":
                with self._cond:
                    self.events.append(frame.get("params") or {})
                    self._cond.notify_all()
            elif "method" in frame and "id" in frame:  # server->client request: decline
                self._write({"jsonrpc": "2.0", "id": frame["id"],
                             "error": {"code": -32601, "message": "test client declines server requests"}})
            elif "id" in frame and frame["id"] in self._pending:
                self._pending[frame["id"]].put(frame)
        with self._cond:
            self._cond.notify_all()

    def _read_stderr(self) -> None:
        for raw in self.proc.stderr:
            self.stderr_lines.append(raw.rstrip("\n"))

    def _write(self, frame: dict) -> None:
        with self._wlock:
            self.proc.stdin.write(json.dumps(frame) + "\n")
            self.proc.stdin.flush()

    def call(self, method: str, params: dict | None = None, timeout: float = 120.0) -> Any:
        self._next_id += 1
        rid = self._next_id
        q: queue.Queue = queue.Queue()
        self._pending[rid] = q
        try:
            self._write({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
            try:
                frame = q.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError(f"{method}: no reply in {timeout}s; stderr tail={self.stderr_lines[-8:]}")
        finally:
            self._pending.pop(rid, None)
        if "error" in frame:
            raise GatewayError(f"{method}: {frame['error']}")
        return frame.get("result")

    def mark(self) -> int:
        with self._cond:
            return len(self.events)

    def wait_event(self, etype: str, sid: str | None = None, *, start: int = 0, timeout: float = 120.0) -> dict:
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                for ev in self.events[start:]:
                    if ev.get("type") == etype and (sid is None or ev.get("session_id") == sid):
                        return ev
                if self.proc.poll() is not None:
                    raise RuntimeError(f"gateway exited rc={self.proc.returncode} waiting for {etype}; "
                                       f"stderr tail={self.stderr_lines[-10:]}")
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"no {etype} in {timeout}s; recent={[e.get('type') for e in self.events[-15:]]}")
                self._cond.wait(min(left, 0.5))

    def create(self) -> str:
        res = self.call("session.create", {"cwd": self.cwd})
        self.stored[res["session_id"]] = res["stored_session_id"]
        return res["session_id"]

    def resume(self, stored_sid: str) -> str:
        res = self.call("session.resume", {"session_id": stored_sid})
        self.stored[res["session_id"]] = res.get("stored_session_id") or stored_sid
        return res["session_id"]

    def call_when_idle(self, method: str, params: dict, timeout: float = 180.0) -> Any:
        """Idle-gated methods answer 4009 while the previous turn is still finalizing (after
        ``message.complete``); retry until the deadline instead of sleeping a fixed time."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                return self.call(method, params, timeout=timeout)
            except GatewayError as exc:
                if "4009" not in str(exc) or time.monotonic() > deadline:
                    raise
            time.sleep(0.2)

    def submit(self, sid: str, text: str, timeout: float = 120.0) -> dict:
        start = self.mark()
        self.call("prompt.submit", {"session_id": sid, "text": text}, timeout=timeout)
        return self.wait_event("message.complete", sid, start=start, timeout=timeout).get("payload") or {}


# ---------------------------------------------------------------------------------------------
# scripted model


class Script:
    """Main-turn responder: pops scripted actions; an action may be a callable run AT request
    arrival (so /steer and interrupt land while the request is genuinely in flight).

    Callables run on the fake provider's HTTP handler thread, where a raised assertion only drops
    the connection (the agent retries and the test moves on). ``errors`` keeps every such failure
    so the test re-raises it on its own thread (``raise_errors``)."""

    def __init__(self) -> None:
        self.actions: list[Any] = []
        self.n = 0
        self.session: Any = None
        self.steered: list[str] = []
        self.errors: list[str] = []

    def __call__(self, record: dict[str, Any]) -> Any:
        self.n += 1
        if self.actions:
            act = self.actions.pop(0)
            if not callable(act):
                return act
            try:
                return act(record)
            except BaseException:
                import traceback

                self.errors.append(traceback.format_exc())
                raise
        # Unique text + varying usage per answer, so a duplicated row or a double-counted
        # request is always distinguishable.
        from tests.fakes.fake_llm_provider import Text

        return Text(f"answer #{self.n}", prompt_tokens=900 + 13 * self.n, completion_tokens=5 + self.n,
                    cached_tokens=400 + self.n)

    def raise_errors(self, where: str) -> None:
        assert not self.errors, f"{where}: a scripted action failed on the provider thread:\n" + "\n".join(
            self.errors)


def big(label: str, n: int = 12000) -> str:
    """A user message large enough that summarizing a few of them genuinely shrinks the context."""
    body = " ".join(f"{label}-{i}" for i in range(n // (len(label) + 4)))
    return f"{label}: {body}"[:n]


# ---------------------------------------------------------------------------------------------
# in-process surface + per-step ledger


class InProcessSession:
    """A real ``AIAgent`` + ``SessionDB`` driven the way the classic CLI drives it: the surface
    owns ``history`` and hands it back each turn; ``/compress`` goes through ``compress_now``."""

    def __init__(self, base_url: str, hermes_home: Path, sid: str, *, platform: str = "cli") -> None:
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools
        from hermes_state import SessionDB
        from run_agent import AIAgent

        self.db = SessionDB(db_path=hermes_home / "state.db")
        # The CLI entrypoints enable the platform's configured toolsets; a bare AIAgent would
        # advertise a different tools array than every real surface resuming this session.
        toolsets = sorted(_get_platform_tools(load_config(), platform))
        self.agent = AIAgent(provider="custom", base_url=base_url, api_key="sk-fake-e2e", model="fake-model",
                             session_db=self.db, session_id=sid, quiet_mode=True, platform=platform,
                             enabled_toolsets=toolsets)
        self.history: list[dict[str, Any]] = []

    @property
    def sid(self) -> str:
        return self.agent.session_id

    def turn(self, text: str) -> dict[str, Any]:
        result = self.agent.run_conversation(text, conversation_history=self.history, task_id=self.sid)
        self.history = result.get("messages") or self.history
        return result

    def compress(self, args: str = "") -> Any:
        """Mirror ``hermes_cli.cli_session_mixin`` ``/compress``: install ``after_messages``, follow a
        rotated session id, re-flush the handoff on rotation, finalize the engine notification."""
        from agent.conversation_compression import finalize_context_engine_compression_notification
        from agent.conversation_compression_manual import compress_now, parse_compress_args

        before_sid = self.sid
        result = compress_now(self.agent, self.history, parse_compress_args(args), task_id=before_sid)
        if result.status != "compressed":
            return result
        self.history = result.after_messages
        if self.sid != before_sid:
            self.agent._flush_messages_to_session_db(self.history, None)
        finalize_context_engine_compression_notification(self.agent, committed=True)
        return result

    def close(self) -> None:
        try:
            self.agent.close()
        finally:
            self.db.close()


class Ledger:
    """Cross-checks request stream, model view, display view, row counts and usage after each step."""

    def __init__(self, srv: Any, hermes_home: Path, *, check_inputs: bool = True) -> None:
        self.srv, self.home, self.check_inputs = srv, hermes_home, check_inputs
        self.ever: dict[tuple, str] = {}
        self.inputs: list[str] = []
        self.model_only: set[tuple] = set()
        self.model_view: list[dict] | None = None
        self.counts: dict[str, int] | None = None
        self.compaction_request_idx: list[int] = []
        self._seen_main = 0

    def main(self) -> list[dict[str, Any]]:
        return self.srv.main_requests()

    def step(self, sid: str, label: str, *, compaction: bool = False, turn: bool = True) -> None:
        reqs = self.main()
        new = reqs[self._seen_main:]
        where = f"after step {label!r}"
        model, display = views(self.home, sid)
        if turn:
            assert new, f"{where}: the turn issued no model request"
            first = new[0]
            if compaction:
                # Pre-turn compaction rewrites what this request carries; a post-turn pass (micro)
                # rewrites the durable view after it. Either way the break lands on this request or
                # the next turn's first request; both are declared.
                self.compaction_request_idx += [self._seen_main, len(reqs)]
            else:
                if self.model_view is not None:
                    assert_replay_equals_persisted(first, self.model_view, where)
                sent = model_payload(first["messages"])
                persisted = model_payload(model)
                assert persisted[:len(sent)] == sent, (
                    f"{where}: the durable model view does not start with what this turn sent\n"
                    + _list_diff(sent, persisted[:len(sent)], "sent", "persisted"))
        elif compaction:
            self.compaction_request_idx.append(len(reqs))
        for r in new:
            for m in r["messages"]:
                if m.get("role") != "system" and not is_summary(m):
                    self.ever.setdefault(identity(m), short(identity(m)))
        for m in model:
            if m.get("role") != "system" and not is_summary(m):
                self.ever.setdefault(identity(m), short(identity(m)))
        assert_exactly_once(model, f"{where} (model view)")
        assert_rows_exactly_once(self.home, sid, where)
        # A model-only row (micro-compaction's merged user turn) stands in for rows that ARE displayed,
        # so it is exempt from the display no-loss check; its originals were each sent (and tracked) earlier.
        self.model_only |= model_only_identities(self.home, model)
        merged = self.model_only
        for k in merged:
            self.ever.pop(k, None)
        assert_no_loss(self.ever, display, [m for m in model if identity(m) not in merged], where)
        if self.check_inputs:
            assert_inputs_shown_once(self.inputs, display, model, where)
        counts = row_counts(self.home, sid)
        if self.counts is not None and sid == self._sid:
            assert counts["total"] >= self.counts["total"], f"{where}: durable rows were deleted {self.counts} -> {counts}"
            if not compaction:
                assert counts["active"] >= self.counts["active"], (
                    f"{where}: active rows shrank outside compaction {self.counts} -> {counts}")
        self.counts, self._sid = counts, sid
        self.model_view = model
        self._seen_main = len(reqs)

    _sid: str | None = None

"""Lane-private harness for the C12 messaging exactly-once suite (test_messaging_exactly_once.py).

What is real and what is fake:

* REAL: ``GatewayRunner`` constructed and ``start()``-ed normally (adapters are created through the
  real ``_create_adapter`` -> ``platform_registry`` path), the base-adapter session guard / busy
  queue / ``_send_with_retry`` / delivery ledger + boot redelivery sweep, the stream consumer,
  ``AIAgent``, SessionDB on disk, and the OpenAI-compatible client talking to
  ``tests/fakes/fake_llm_provider.py`` (which lives in the TEST process).
* FAKE: only the platform wire. ``FakePlatformAdapter`` implements the ``BasePlatformAdapter``
  contract (send / edit_message / delete_message / threads / per-platform length caps) against
  ``FakePlatformServer`` - the ground truth of what a user would SEE. Every accepted effect is
  journaled (fsync) to ``platform.jsonl`` BEFORE the call returns, so the journal survives a
  SIGKILL and a restarted gateway sees the same chat. Faults (send timeout, sent-but-ack-lost,
  429 retry_after, connection reset, a crash hold) are scripted per operation over RPC.

The gateway runs in a CHILD process (``python -m tests.e2e.core.delivery._fake_platform serve
<spool>``), controlled over a JSON-lines TCP socket. That keeps it isolated from the per-test
autouse fixtures of this suite's conftest (which close every SessionDB and reset plugin singletons
after each test), lets one ~15 s gateway boot serve a whole scenario matrix, and makes a real
``kill -9`` + restart on the same HERMES_HOME possible.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

POLL = 0.02
DEADLINE = 120.0
REPO_ROOT = Path(__file__).resolve().parents[4]

# Per-platform send/edit semantics (what the real service enforces, not what Hermes believes).
PROFILES: Dict[str, Dict[str, Any]] = {
    "telegram": {"max_len": 4096, "edits": True, "threads": False},
    "discord": {"max_len": 2000, "edits": True, "threads": True},
    "slack": {"max_len": 3900, "edits": True, "threads": True},
    "noedit": {"max_len": 4096, "edits": False, "threads": False},
}


def wait_until(predicate: Callable[[], Any], what: str, timeout: float = DEADLINE,
               proc: Optional[subprocess.Popen] = None, log: Optional[Path] = None) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if proc is not None and proc.poll() is not None:
            tail = log.read_text(errors="replace")[-4000:] if log and log.exists() else ""
            raise AssertionError(f"gateway child exited (rc={proc.returncode}) waiting for {what}\n{tail}")
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out after {timeout:.0f}s waiting for {what}")
        time.sleep(POLL)


def _append_jsonl(path: Path, record: dict) -> None:
    line = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a torn last line from a SIGKILLed writer
    return out


# Platform ground truth ------------------------------------------------------------------------


@dataclass
class PlatformMessage:
    message_id: str
    platform: str
    chat_id: str
    thread_id: Optional[str]
    text: str
    reply_to: Optional[str] = None
    deleted: bool = False
    edits: int = 0


class FakePlatformServer:
    """What the chat service holds. Thread-safe; journaled so it survives the gateway process."""

    def __init__(self, journal: Path) -> None:
        self._lock = threading.Lock()
        self.messages: Dict[str, PlatformMessage] = {}
        self.order: List[str] = []
        self.journal = journal
        for rec in read_jsonl(journal):
            self._apply(rec)

    def _apply(self, rec: dict) -> None:
        kind = rec["kind"]
        if kind == "post":
            msg = PlatformMessage(rec["message_id"], rec["platform"], rec["chat_id"], rec.get("thread_id"),
                                  rec["text"], rec.get("reply_to"))
            self.messages[msg.message_id] = msg
            self.order.append(msg.message_id)
        elif kind == "edit":
            msg = self.messages.get(rec["message_id"])
            if msg is not None:
                msg.text = rec["text"]
                msg.edits += 1
        elif kind == "delete":
            msg = self.messages.get(rec["message_id"])
            if msg is not None:
                msg.deleted = True

    def _commit(self, rec: dict) -> None:
        rec["t"] = time.time()
        _append_jsonl(self.journal, rec)
        self._apply(rec)

    def post(self, platform: str, chat_id: str, thread_id: Optional[str], text: str,
             reply_to: Optional[str]) -> str:
        with self._lock:
            mid = f"pm-{os.getpid()}-{len(self.order) + 1}"
            self._commit({"kind": "post", "message_id": mid, "platform": platform, "chat_id": str(chat_id),
                          "thread_id": thread_id, "text": text, "reply_to": reply_to})
            return mid

    def edit(self, message_id: str, text: str) -> bool:
        with self._lock:
            msg = self.messages.get(message_id)
            if msg is None or msg.deleted:
                return False
            self._commit({"kind": "edit", "message_id": message_id, "text": text})
            return True

    def delete(self, message_id: str) -> bool:
        with self._lock:
            if message_id not in self.messages:
                return False
            self._commit({"kind": "delete", "message_id": message_id})
            return True

    def visible(self, platform: Optional[str] = None, chat_id: Optional[str] = None) -> List[PlatformMessage]:
        with self._lock:
            return [m for m in (self.messages[i] for i in self.order)
                    if not m.deleted and (platform is None or m.platform == platform)
                    and (chat_id is None or m.chat_id == str(chat_id))]


# Fault injection ------------------------------------------------------------------------------


@dataclass
class Fault:
    """One scripted platform fault, consumed by the first ``times`` matching calls.

    ``op``: "send" | "edit" | "any" (send or edit) | "pre_ledger". ``contains``: substring the content must include (None = any call).
    ``kind``:
      timeout      - the platform never saw it; the adapter reports a timeout.
      ack_lost     - the platform accepted it (visible) but the ack never came back (timeout).
      rate_limit   - 429: refused, SendResult carries ``retry_after``.
      conn_error   - refused with a retryable connection reset.
      hold         - park the call BEFORE the platform accepts it (until SIGKILL).
      hold_after   - the platform accepts it, then park (the ack dies with the process).
    """

    platform: str
    op: str
    kind: str
    contains: Optional[str] = None
    retry_after: float = 0.3
    times: int = 1
    chat_id: Optional[str] = None
    hits: int = 0

    def matches(self, platform: str, op: str, content: str, chat_id: Optional[str] = None) -> bool:
        if self.hits >= self.times or platform != self.platform:
            return False
        if self.chat_id is not None and chat_id != self.chat_id:
            return False
        if self.op != op and not (self.op == "any" and op in ("send", "edit")):
            return False
        return self.contains is None or self.contains in content


# Adapter (child process) ----------------------------------------------------------------------


def _adapter_class():
    from gateway.config import Platform
    from gateway.platforms.base import BasePlatformAdapter, SendResult
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.platforms.helpers import MessageDeduplicator

    class FakePlatformAdapter(BasePlatformAdapter):
        """A platform adapter honouring the base contract against ``FakePlatformServer``."""

        splits_long_messages = True

        def __init__(self, config, *, name: str, profile: str, world: "ChildWorld"):
            super().__init__(config, Platform(name))
            self.pname = name
            self.world = world
            self.profile = PROFILES[profile]
            self.MAX_MESSAGE_LENGTH = self.profile["max_len"]
            self.SUPPORTS_MESSAGE_EDITING = self.profile["edits"]
            # Inbound de-duplication is adapter-owned in the contract; like most real adapters this
            # uses the shared helper, whose state lives on the adapter instance.
            self._dedup = MessageDeduplicator()

        # lifecycle
        async def connect(self, *, is_reconnect: bool = False) -> bool:
            self._running = True
            self._mark_connected()
            return True

        async def disconnect(self) -> None:
            self._running = False
            self._mark_disconnected()

        async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
            return {"name": f"chat-{chat_id}", "type": "dm", "chat_id": chat_id}

        async def send_typing(self, chat_id: str, metadata=None) -> None:
            return None

        # faults
        async def _fault(self, op: str, content: str, accept: Callable[[], Any],
                         chat_id: Optional[str] = None) -> Optional[Any]:
            fault = self.world.take_fault(self.pname, op, content, chat_id)
            if fault is None:
                return None
            if fault.kind == "timeout":
                return SendResult(success=False, error="Timed out waiting for platform response",
                                  error_kind="transient")
            if fault.kind == "ack_lost":
                accept()
                return SendResult(success=False, error="Timed out waiting for platform response",
                                  error_kind="transient")
            if fault.kind == "rate_limit":
                return SendResult(success=False, error=f"Too Many Requests: retry after {fault.retry_after}",
                                  retry_after=fault.retry_after, error_kind="rate_limited")
            if fault.kind == "conn_error":
                return SendResult(success=False, error="Connection reset by peer", retryable=True,
                                  error_kind="transient")
            if fault.kind in ("hold", "hold_after"):
                if fault.kind == "hold_after":
                    accept()
                self.world.note_hold(self.pname, op, content)
                await asyncio.Event().wait()  # parked until the process is killed
            raise AssertionError(f"unknown fault kind {fault.kind}")

        def _thread(self, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
            if not self.profile["threads"] or not metadata:
                return None
            tid = metadata.get("thread_id")
            return str(tid) if tid else None

        # outbound
        async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> SendResult:
            content = content or ""
            thread_id = self._thread(metadata)
            self.world.record_op({"platform": self.pname, "op": "send", "chat_id": str(chat_id),
                                  "thread_id": thread_id, "content": content,
                                  "interim": bool((metadata or {}).get("_interim_send"))})
            chunks = self.truncate_message(content, self.MAX_MESSAGE_LENGTH) if content else [""]
            posted: List[str] = []

            def accept() -> None:
                for chunk in chunks:
                    posted.append(self.world.server.post(self.pname, chat_id, thread_id, chunk, reply_to))

            faulted = await self._fault("send", content, accept, str(chat_id))
            if faulted is not None:
                return faulted
            accept()
            return SendResult(success=True, message_id=posted[-1], continuation_message_ids=tuple(posted[:-1]))

        async def edit_message(self, chat_id: str, message_id: str, content: str, *,
                               finalize: bool = False) -> SendResult:
            self.world.record_op({"platform": self.pname, "op": "edit", "chat_id": str(chat_id),
                                  "message_id": message_id, "content": content, "finalize": finalize})
            if not self.profile["edits"]:
                return SendResult(success=False, error="Not supported")
            if len(content) > self.MAX_MESSAGE_LENGTH:
                return SendResult(success=False, error="Bad Request: message is too long", error_kind="too_long")
            faulted = await self._fault("edit", content, lambda: self.world.server.edit(message_id, content),
                                        str(chat_id))
            if faulted is not None:
                return faulted
            if not self.world.server.edit(message_id, content):
                return SendResult(success=False, error="Bad Request: message to edit not found",
                                  error_kind="not_found")
            return SendResult(success=True, message_id=message_id)

        async def delete_message(self, chat_id: str, message_id: str) -> bool:
            self.world.record_op({"platform": self.pname, "op": "delete", "chat_id": str(chat_id),
                                  "message_id": message_id})
            return self.world.server.delete(message_id)

        # a crash point between "handler returned (turn persisted)" and "final text ledgered"
        async def _extract_response_content(self, response, event, session_key, **kw):
            chat = str(getattr(getattr(event, "source", None), "chat_id", "") or "")
            if self.world.take_fault(self.pname, "pre_ledger", str(response or ""), chat) is not None:
                self.world.note_hold(self.pname, "pre_ledger", str(response or ""))
                await asyncio.Event().wait()
            return await super()._extract_response_content(response, event, session_key, **kw)

        # inbound
        async def inject(self, text: str, message_id: str, *, chat_id: str = "c1", user_id: str = "u1",
                         thread_id: Optional[str] = None, chat_type: str = "dm") -> None:
            if self._dedup.is_duplicate(message_id):
                self.world.record_op({"platform": self.pname, "op": "inbound_duplicate_dropped",
                                      "chat_id": str(chat_id), "message_id": message_id})
                return
            self.world.record_op({"platform": self.pname, "op": "inbound", "chat_id": str(chat_id),
                                  "message_id": message_id, "content": text})
            source = self.build_source(chat_id=chat_id, chat_name=f"chat-{chat_id}", chat_type=chat_type,
                                       user_id=user_id, user_name="tester", thread_id=thread_id)
            event = MessageEvent(text=text, message_type=MessageType.TEXT, source=source, message_id=message_id)
            await self.handle_message(event)

    return FakePlatformAdapter


class ChildWorld:
    """Child-process state shared by every fake adapter instance (survives adapter re-creation)."""

    def __init__(self, spool: Path) -> None:
        self.spool = spool
        self.server = FakePlatformServer(spool / "platform.jsonl")
        self._lock = threading.Lock()
        self.faults: List[Fault] = []

    def take_fault(self, platform: str, op: str, content: str, chat_id: Optional[str] = None) -> Optional[Fault]:
        with self._lock:
            for f in self.faults:
                if f.matches(platform, op, content, chat_id):
                    f.hits += 1
                    _append_jsonl(self.spool / "faults_fired.jsonl",
                                  {"pid": os.getpid(), "platform": platform, "op": op, "kind": f.kind,
                                   "chat_id": chat_id, "content": content[:200]})
                    return f
        return None

    def note_hold(self, platform: str, op: str, content: str) -> None:
        _append_jsonl(self.spool / "holds.jsonl", {"pid": os.getpid(), "platform": platform, "op": op,
                                                   "content": content})

    def record_op(self, op: dict) -> None:
        op["pid"] = os.getpid()
        _append_jsonl(self.spool / "ops.jsonl", op)


def _serve(spool: Path) -> None:  # pragma: no cover - runs in the child process
    cfg = json.loads((spool / "config.json").read_text())
    world = ChildWorld(spool)

    # Knob-only: shrink the ledger's in-process redelivery backoff (30 s / 120 s) so a rejected final
    # is retried within the test's patience; the policy (who retries, with which marker) is untouched.
    import gateway.delivery_ledger as ledger
    ledger._RETRY_BACKOFF_SECONDS = tuple(cfg.get("ledger_backoff", (0.5, 1.0)))

    from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig, StreamingConfig
    from gateway.platform_registry import PlatformEntry, platform_registry
    from gateway.run import GatewayRunner

    cls = _adapter_class()
    platforms = {}
    for name, profile in cfg["platforms"].items():
        platform_registry.register(PlatformEntry(
            name=name, label=name, check_fn=lambda: True, source="plugin",
            adapter_factory=lambda pc, n=name, p=profile: cls(pc, name=n, profile=p, world=world),
            allow_all_env=f"{name.upper()}_ALLOW_ALL_USERS"))
        os.environ[f"{name.upper()}_ALLOW_ALL_USERS"] = "true"
        plat = Platform(name)
        platforms[plat] = PlatformConfig(enabled=True, token="fake-token",
                                         home_channel=HomeChannel(platform=plat, chat_id="home", name="home"))

    config = GatewayConfig(
        platforms=platforms,
        streaming=StreamingConfig(enabled=True, transport="edit", edit_interval=0.05, buffer_threshold=24,
                                  cursor=""),
        loop_watchdog=False, write_sessions_json=False, multiplex_profiles=False,
    )

    async def handle_rpc(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while True:
            line = await reader.readline()
            if not line:
                break
            req = json.loads(line)
            try:
                resp = await dispatch(req)
            except Exception as exc:  # surfaced to the test as an RPC error
                resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            writer.write((json.dumps(resp) + "\n").encode())
            await writer.drain()

    runner: Any = None

    async def dispatch(req: dict) -> dict:
        cmd = req["cmd"]
        if cmd == "inject":
            adapter = runner.adapters[Platform(req["platform"])]
            await adapter.inject(req["text"], req["message_id"], chat_id=req.get("chat_id", "c1"),
                                 thread_id=req.get("thread_id"))
            return {"ok": True}
        if cmd == "adapter_id":
            adapter = runner.adapters.get(Platform(req["platform"]))
            return {"ok": True, "id": id(adapter) if adapter is not None else None}
        if cmd == "reconnect":
            # The adapter's socket dies with a retryable error; the REAL reconnect watcher replaces it.
            old = runner.adapters[Platform(req["platform"])]
            old._set_fatal_error("fake_socket_closed", "socket closed by peer", retryable=True)
            await old._notify_fatal_error()
            return {"ok": True, "id": id(old)}
        if cmd == "fault":
            with world._lock:
                world.faults.append(Fault(**req["fault"]))
            return {"ok": True}
        if cmd == "busy":
            chats = req.get("chats")

            def mine(key: str) -> bool:
                return chats is None or any(re.search(rf":{re.escape(c)}(:|$)", key) for c in chats)

            busy = []
            for adapter in runner.adapters.values():
                keys = set(adapter._active_sessions) | set(adapter._pending_messages) | set(adapter._session_tasks)
                busy.extend(k for k in keys if mine(k))
            running = [k for k in list(runner._running_agents) if mine(k)]
            return {"ok": True, "busy": sorted(busy), "running": sorted(running)}
        raise ValueError(f"unknown rpc {cmd}")

    async def orphan_watchdog(parent: int) -> None:
        # The child runs in its own session (so kill9 can take the whole group); if the test
        # process dies without teardown (CI per-file timeout), exit instead of lingering.
        while os.getppid() == parent:
            await asyncio.sleep(0.5)
        os._exit(0)

    async def main() -> None:
        nonlocal runner
        parent = int(os.environ.get("C12_PARENT_PID") or os.getppid())
        watchdog = asyncio.ensure_future(orphan_watchdog(parent))  # noqa: F841 - lives with the loop
        runner = GatewayRunner(config)
        ok = await runner.start()
        if not ok:
            raise SystemExit(3)
        server = await asyncio.start_server(handle_rpc, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        _append_jsonl(spool / "ready.jsonl", {"pid": os.getpid(), "port": port})
        async with server:
            await server.serve_forever()

    asyncio.run(main())


# Test-process side ----------------------------------------------------------------------------


_KEEP_ENV = ("PATH", "LANG", "LC_ALL", "TMPDIR", "PYTHONPATH", "VIRTUAL_ENV", "SYSTEMROOT")


class GatewayProcess:
    """A real gateway in a child process on ``home``; RPC + SIGKILL + restart on the same state."""

    def __init__(self, root: Path, *, platforms: Dict[str, str], llm_base_url: str,
                 extra_config: str = "", ledger_backoff=(0.5, 1.0)) -> None:
        from tests.fakes.fake_llm_provider import write_hermes_home

        self.root = root
        self.home = root / "home"
        self.hermes_home = self.home / ".hermes"
        self.spool = root / "spool"
        self.spool.mkdir(parents=True, exist_ok=True)
        write_hermes_home(self.hermes_home, llm_base_url, extra_config=extra_config)
        (self.spool / "config.json").write_text(json.dumps(
            {"platforms": platforms, "ledger_backoff": list(ledger_backoff)}))
        self.proc: Optional[subprocess.Popen] = None
        self.sock: Optional[socket.socket] = None
        self._rfile = None
        self.boots = 0
        self.pids: List[int] = []

    @property
    def db_path(self) -> Path:
        return self.hermes_home / "state.db"

    @property
    def log(self) -> Path:
        return self.spool / f"child-{self.boots}.log"

    def env(self) -> Dict[str, str]:
        env = {k: os.environ[k] for k in _KEEP_ENV if k in os.environ}
        env.update({
            "HOME": str(self.home), "HERMES_HOME": str(self.hermes_home),
            # The child's HOME is itself a throwaway tmp dir, so ~/.hermes IS the temp home here; the
            # live-system guard (armed by the inherited isolation marker) would refuse it.
            "HERMES_STATE_DB_GUARD_BYPASS": "1",
            "HERMES_GATEWAY_LOCK_DIR": str(self.root / "gateway-locks"),
            "TZ": "UTC", "PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1", "C12_PARENT_PID": str(os.getpid()),
            "HERMES_DISABLE_LAZY_INSTALLS": "1", "TIRITH_ENABLED": "false",
            "AWS_EC2_METADATA_DISABLED": "true", "HERMES_HONCHO_HOST": "hermes",
            "PYTHONPATH": f"{REPO_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}".rstrip(os.pathsep),
        })
        return env

    def start(self) -> "GatewayProcess":
        assert self.proc is None
        self.boots += 1
        ready_before = len(read_jsonl(self.spool / "ready.jsonl"))
        with open(self.log, "wb") as log:
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "tests.e2e.core.delivery._fake_platform", "serve", str(self.spool)],
                cwd=str(REPO_ROOT), env=self.env(), stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        self.pids.append(self.proc.pid)
        rec = wait_until(
            lambda: next((r for r in read_jsonl(self.spool / "ready.jsonl")[ready_before:]
                          if r["pid"] == self.proc.pid), None),
            "gateway child ready", timeout=240, proc=self.proc, log=self.log)
        self.sock = socket.create_connection(("127.0.0.1", rec["port"]), timeout=120)
        self._rfile = self.sock.makefile("rb")
        return self

    def rpc(self, cmd: str, **kw: Any) -> dict:
        assert self.sock is not None, "gateway not running"
        self.sock.sendall((json.dumps({"cmd": cmd, **kw}) + "\n").encode())
        line = self._rfile.readline()
        if not line:
            raise AssertionError(f"gateway child closed the RPC channel during {cmd}\n"
                                 + self.log.read_text(errors="replace")[-4000:])
        resp = json.loads(line)
        assert resp.get("ok"), f"rpc {cmd} failed: {resp}"
        return resp

    def inject(self, platform: str, text: str, message_id: str, chat_id: str, thread_id: Optional[str] = None):
        return self.rpc("inject", platform=platform, text=text, message_id=message_id, chat_id=chat_id,
                        thread_id=thread_id)

    def fault(self, **fault: Any) -> None:
        self.rpc("fault", fault=fault)

    def idle(self, chats: Optional[List[str]]) -> bool:
        r = self.rpc("busy", chats=chats)
        return not r["busy"] and not r["running"]

    def wait_idle(self, chats: Optional[List[str]], what: str, timeout: float = DEADLINE, settle: int = 3) -> None:
        """Idle for ``settle`` consecutive polls (a queued drain task can hop between checks)."""
        streak = [0]

        def check() -> bool:
            streak[0] = streak[0] + 1 if self.idle(chats) else 0
            return streak[0] >= settle

        wait_until(check, what, timeout=timeout, proc=self.proc, log=self.log)

    def holds(self) -> List[dict]:
        return read_jsonl(self.spool / "holds.jsonl")

    def kill9(self) -> None:
        assert self.proc is not None
        assert self.proc.poll() is None, "gateway exited before the kill (a clean exit is not a crash test)"
        os.killpg(self.proc.pid, signal.SIGKILL)
        self.proc.wait(timeout=60)
        self._close()

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=60)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=30)
        self._close()

    def _close(self) -> None:
        for closer in (self._rfile, self.sock):
            try:
                if closer is not None:
                    closer.close()
            except OSError:
                pass
        self.sock = self._rfile = None
        self.proc = None

    # ground truth
    def platform_view(self) -> FakePlatformServer:
        return FakePlatformServer(self.spool / "platform.jsonl")

    def ops(self) -> List[dict]:
        return read_jsonl(self.spool / "ops.jsonl")


# Oracle ---------------------------------------------------------------------------------------

RECOVERY_MARKER_PREFIXES: tuple = ()


def recovery_markers() -> tuple:
    from gateway.delivery_ledger import FLOOD_MARKER, RECONNECTED_MARKER, RECOVERED_MARKER
    return (RECOVERED_MARKER, RECONNECTED_MARKER, FLOOD_MARKER)


_CHUNK_INDICATOR = re.compile(r"\s\(\d+/\d+\)\s*$")


def norm(text: str) -> str:
    """Compare text modulo whitespace: platform splitters (base truncate_message, the stream
    consumer's overflow split) cut at or inside whitespace and strip it, sometimes mid-word."""
    return "".join((text or "").split())


def join_chunks(parts: List[str]) -> str:
    """Re-join a reply rendered across several messages.

    The stream consumer's fallback continuation deliberately re-sends the broken word when the last
    landed edit ended mid-word (``StreamFallbackMixin._continuation_text``): a trailing partial word
    that is a proper prefix of the next chunk's first word is that documented overlap, not content.
    """
    out = ""
    for part in parts:
        if out and part and not out[-1].isspace():
            tail = re.split(r"\s", out)[-1]
            head = (part.split() or [""])[0]
            if tail and head != tail and head.startswith(tail):
                out = out[: len(out) - len(tail)]
        out += part
    return out


def header(aid: str) -> str:
    return f"<<{aid}>>"


def footer(aid: str) -> str:
    return f"<</{aid}>>"


@dataclass
class Copy:
    text: str
    marked: bool
    complete: bool
    message_ids: List[str]


def visible_copies(msgs: List[PlatformMessage], aid: str) -> List[Copy]:
    """Every visible rendition of answer ``aid``: starts at a message carrying its header and spans
    split chunks until the footer. A copy without a footer is incomplete (truncated/stale preview)."""
    markers = recovery_markers()
    cleaned = []
    for m in msgs:
        text, marked = m.text, False
        for mk in markers:
            if text.startswith(mk):
                text, marked = text[len(mk):], True
        cleaned.append((m.message_id, _CHUNK_INDICATOR.sub("", text), marked))
    copies: List[Copy] = []
    h, f = header(aid), footer(aid)
    for i, (mid, text, marked) in enumerate(cleaned):
        start = text.index(h) if h in text else None
        if start is None:
            stub = text.strip()
            if not (len(stub) >= 3 and h.startswith(stub)):
                continue
            # A preview cut inside the header itself. When the final edit then fails, the fallback
            # continuation carries the rest from the original cut (a prefix with no word boundary is
            # not backed up), so the header spans two messages: one rendition, not a stale stub.
            ahead = [c[1] for c in cleaned[i + 1:i + 2] if h not in c[1]]
            if not (ahead and join_chunks([stub, ahead[0]]).startswith(h)):
                copies.append(Copy(norm(stub), marked, False, [mid]))
                continue
            text, start = stub, 0
        parts, ids = [text[start:]], [mid]
        j = i
        while f not in parts[-1] and j + 1 < len(cleaned) and header(aid) not in cleaned[j + 1][1]:
            j += 1
            parts.append(cleaned[j][1])
            ids.append(cleaned[j][0])
            if f in cleaned[j][1]:
                break
        joined = join_chunks(parts)
        complete = f in joined
        if complete:
            joined = joined[: joined.index(f) + len(f)]
        copies.append(Copy(norm(joined), marked, complete, ids))
    return copies


def persisted_answers(db_path: Path, aid: str) -> List[str]:
    import sqlite3

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        rows = conn.execute(
            "SELECT content FROM messages WHERE role='assistant' AND content LIKE ? ORDER BY id",
            (f"%{header(aid)}%",)).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def persisted_user_rows(db_path: Path, token: str) -> List[str]:
    import sqlite3

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        rows = conn.execute("SELECT content FROM messages WHERE role='user' AND content LIKE ? ORDER BY id",
                            (f"%{token}%",)).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


if __name__ == "__main__":  # pragma: no cover - child process entry
    if len(sys.argv) >= 3 and sys.argv[1] == "serve":
        _serve(Path(sys.argv[2]))

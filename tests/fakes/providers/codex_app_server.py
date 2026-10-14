"""Fake ``codex app-server``: newline-delimited JSON-RPC over stdio, validated against the protocol.

Run as a script (``python codex_app_server.py --state-dir DIR app-server``) it plays the codex side of
the wire that ``agent/transports/codex_app_server.py`` speaks. Imported, it offers the test-side
harness (:class:`FakeCodex`) that installs an executable wrapper and reads the recorded transcript.

Protocol truth is the schema bundle emitted by ``codex app-server generate-json-schema`` (codex-cli
0.147): request params are checked field by field with the real server's serde error strings
(``-32600 "Invalid request: missing field `threadId`"``). The real server IGNORES unknown fields, so
the fake answers them normally but records each one as ``ignored`` — a field codex silently drops is
Hermes intent that never reaches the model, and the suite asserts none are sent. Responses Hermes
gives to server-initiated requests (approvals, elicitation) are validated the same way.

State lives in ``DIR``: ``scenario.json`` (scripted turns, consumed one per ``turn/start`` across
processes), ``threads.json`` (the "rollout store" that ``thread/resume`` reads, so a NEW Hermes process
can resume a thread) and ``transcript.jsonl`` (every message in both directions, plus spawn/exit
events with PIDs). Only stdlib: the wrapper runs it with the test interpreter.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

CLI_VERSION = "0.147.0"
INVALID_REQUEST = -32600
GRANDCHILD_RELEASE = "release-grandchildren"


# ---------------------------------------------------------------------------------------------------
# Protocol schema (subset of the codex app-server v2 bundle Hermes can reach) + serde-style validator
# ---------------------------------------------------------------------------------------------------

class Invalid(Exception):
    """A params/response value the real server's deserializer would reject."""


def _kind(value: Any) -> str:
    table = ((bool, "boolean"), (int, "integer"), (float, "floating point"), (str, "string"),
             (list, "sequence"), (dict, "map"), (type(None), "null"))
    return next((name for typ, name in table if isinstance(value, typ)), "value")


def _unexpected(value: Any) -> str:
    return f"{_kind(value)} `{value}`" if isinstance(value, (bool, int, float, str)) else _kind(value)


@dataclass
class Spec:
    check: Callable[[Any, str, list], None]

    def __call__(self, value: Any, path: str, ignored: list) -> None:
        self.check(value, path, ignored)


def _scalar(py: tuple, expected: str) -> Spec:
    def check(v: Any, path: str, ignored: list) -> None:
        if not isinstance(v, py) or (bool not in py and isinstance(v, bool)):
            raise Invalid(f"invalid type: {_unexpected(v)}, expected {expected}")
    return Spec(check)


STR, BOOL, INT = _scalar((str,), "a string"), _scalar((bool,), "a boolean"), _scalar((int,), "i64")
ANY = Spec(lambda v, p, i: None)


def nullable(spec: Spec) -> Spec:
    return Spec(lambda v, p, i: None if v is None else spec(v, p, i))


def enum(*values: str) -> Spec:
    def check(v: Any, path: str, ignored: list) -> None:
        if not isinstance(v, str):
            raise Invalid(f"invalid type: {_unexpected(v)}, expected a string")
        if v not in values:
            raise Invalid(f"unknown variant `{v}`, expected one of {', '.join(f'`{x}`' for x in values)}")
    return Spec(check)


def array(item: Spec) -> Spec:
    def check(v: Any, path: str, ignored: list) -> None:
        if not isinstance(v, list):
            raise Invalid(f"invalid type: {_unexpected(v)}, expected a sequence")
        for n, element in enumerate(v):
            item(element, f"{path}[{n}]", ignored)
    return Spec(check)


def obj(required: Optional[dict] = None, optional: Optional[dict] = None, *, open_map: bool = False) -> Spec:
    required, optional = required or {}, optional or {}

    def check(v: Any, path: str, ignored: list) -> None:
        if not isinstance(v, dict):
            raise Invalid(f"invalid type: {_unexpected(v)}, expected struct")
        for name in required:
            if name not in v:
                raise Invalid(f"missing field `{name}`")
        for name, value in v.items():
            spec = required.get(name) or optional.get(name)
            if spec is not None:
                spec(value, f"{path}.{name}", ignored)
            elif not open_map:
                ignored.append(f"{path}.{name}")
    return Spec(check)


def tagged(tag: str, variants: dict[str, Spec]) -> Spec:
    """serde internally tagged enum (``{"type": "text", ...}``)."""
    def check(v: Any, path: str, ignored: list) -> None:
        if not isinstance(v, dict):
            raise Invalid(f"invalid type: {_unexpected(v)}, expected internally tagged enum")
        if tag not in v:
            raise Invalid(f"missing field `{tag}`")
        variant = variants.get(v[tag]) if isinstance(v[tag], str) else None
        if variant is None:
            raise Invalid(f"unknown variant `{v[tag]}`, expected one of {', '.join(f'`{x}`' for x in variants)}")
        variant({k: x for k, x in v.items() if k != tag}, path, ignored)
    return Spec(check)


def one_of(*specs: Spec) -> Spec:
    def check(v: Any, path: str, ignored: list) -> None:
        errors = []
        for spec in specs:
            try:
                spec(v, path, [])
                return
            except Invalid as exc:
                errors.append(str(exc))
        raise Invalid(f"data did not match any variant of untagged enum ({'; '.join(errors)})")
    return Spec(check)


_TEXT_ELEMENT = obj({"byteRange": obj({"start": INT, "end": INT})}, {"placeholder": nullable(STR)})
_IMAGE_DETAIL = nullable(enum("auto", "low", "high", "original"))
USER_INPUT = tagged("type", {
    "text": obj({"text": STR}, {"text_elements": array(_TEXT_ELEMENT)}),
    "image": obj({"url": STR}, {"detail": _IMAGE_DETAIL}),
    "localImage": obj({"path": STR}, {"detail": _IMAGE_DETAIL}),
    "audio": obj({"url": STR}), "localAudio": obj({"path": STR}),
    "skill": obj({"name": STR, "path": STR}), "mention": obj({"name": STR, "path": STR}),
})
_ASK_FOR_APPROVAL = nullable(one_of(enum("untrusted", "on-request", "never"), obj({"granular": ANY})))
_THREAD_SETTINGS = {
    "approvalPolicy": _ASK_FOR_APPROVAL,
    "approvalsReviewer": nullable(enum("user", "auto_review", "guardian_subagent")),
    "baseInstructions": nullable(STR), "config": nullable(obj(open_map=True)), "cwd": nullable(STR),
    "developerInstructions": nullable(STR), "model": nullable(STR), "modelProvider": nullable(STR),
    "personality": nullable(enum("none", "friendly", "pragmatic")),
    "sandbox": nullable(enum("read-only", "workspace-write", "danger-full-access")), "serviceTier": nullable(STR),
}
_SANDBOX_POLICY = tagged("type", {
    "dangerFullAccess": obj(), "readOnly": obj(optional={"networkAccess": BOOL}),
    "externalSandbox": obj(optional={"networkAccess": ANY}),
    "workspaceWrite": obj(optional={"excludeSlashTmp": BOOL, "excludeTmpdirEnvVar": BOOL, "networkAccess": BOOL,
                                    "writableRoots": array(STR)}),
})

REQUEST_PARAMS: dict[str, Spec] = {
    "initialize": obj({"clientInfo": obj({"name": STR, "version": STR}, {"title": nullable(STR)})}, {
        "capabilities": nullable(obj(optional={
            "experimentalApi": BOOL, "extensions": nullable(obj(open_map=True)), "mcpServerOpenaiFormElicitation": BOOL,
            "optOutNotificationMethods": nullable(array(STR)), "requestAttestation": BOOL})),
    }),
    "thread/start": obj(optional={**_THREAD_SETTINGS, "ephemeral": nullable(BOOL), "serviceName": nullable(STR),
                                  "sessionStartSource": nullable(enum("startup", "clear")),
                                  "threadSource": nullable(STR)}),
    "thread/resume": obj({"threadId": STR}, _THREAD_SETTINGS),
    "turn/start": obj({"threadId": STR, "input": array(USER_INPUT)}, {
        "approvalPolicy": _ASK_FOR_APPROVAL, "approvalsReviewer": _THREAD_SETTINGS["approvalsReviewer"],
        "clientUserMessageId": nullable(STR), "cwd": nullable(STR), "effort": nullable(STR), "model": nullable(STR),
        "outputSchema": ANY, "personality": _THREAD_SETTINGS["personality"], "sandboxPolicy": nullable(_SANDBOX_POLICY),
        "serviceTier": nullable(STR), "summary": nullable(enum("auto", "concise", "detailed", "none")),
    }),
    "turn/interrupt": obj({"threadId": STR, "turnId": STR}),
    "turn/steer": obj({"threadId": STR, "input": array(USER_INPUT), "expectedTurnId": STR},
                      {"clientUserMessageId": nullable(STR)}),
    "thread/compact/start": obj({"threadId": STR}),
}

_EXEC_DECISION = one_of(enum("accept", "acceptForSession", "decline", "cancel"),
                        obj({"acceptWithExecpolicyAmendment": obj({"execpolicy_amendment": array(STR)})}),
                        obj({"applyNetworkPolicyAmendment": obj({"network_policy_amendment": obj(
                            {"action": enum("allow", "deny"), "host": STR})})}))
SERVER_REQUEST_RESULTS: dict[str, Spec] = {
    "item/commandExecution/requestApproval": obj({"decision": _EXEC_DECISION}),
    "item/fileChange/requestApproval": obj({"decision": enum("accept", "acceptForSession", "decline", "cancel")}),
    "item/permissions/requestApproval": obj({"permissions": obj(optional={
        "fileSystem": nullable(obj(open_map=True)), "network": nullable(obj({}, {"enabled": nullable(BOOL)}))})},
        {"scope": enum("turn", "session"), "strictAutoReview": nullable(BOOL)}),
    "mcpServer/elicitation/request": obj({"action": enum("accept", "decline", "cancel")},
                                         {"content": ANY, "_meta": ANY}),
}


def validate(spec: Spec, value: Any) -> tuple[Optional[str], list[str]]:
    """``(serde error or None, ignored field paths)``."""
    ignored: list[str] = []
    try:
        spec(value, "$", ignored)
    except Invalid as exc:
        return str(exc), ignored
    return None, ignored


# ---------------------------------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)


class _Store:
    """``threads.json``: thread rollouts + the global scripted-turn cursor, shared across processes."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict:
        if not self.path.exists():
            return {"threads": {}, "turn_cursor": 0}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, data: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)


class _TurnEnded(Exception):
    """A step ended the turn (failure, crash handled, interrupt)."""


class FakeAppServer:
    def __init__(self, state_dir: Path, argv: list[str]) -> None:
        self.state_dir = state_dir
        self.scenario = json.loads((state_dir / "scenario.json").read_text(encoding="utf-8"))
        self.store = _Store(state_dir / "threads.json")
        self.transcript_path = state_dir / "transcript.jsonl"
        self._write_lock = threading.Lock()
        self._record_lock = threading.Lock()
        self._next_server_id = 9000
        self._pending: dict[Any, dict] = {}
        self._pending_cv = threading.Condition()
        self._initialized = False
        self._interrupted: set[str] = set()
        self._turn_thread: Optional[threading.Thread] = None
        self.record({"event": "spawn", "argv": argv, "ppid": os.getppid()})

    # --- io -----------------------------------------------------------------------------------------
    def record(self, entry: dict) -> None:
        entry = {"pid": os.getpid(), "t": time.time(), **entry}
        with self._record_lock, open(self.transcript_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def send(self, msg: dict) -> None:
        self.record({"dir": "out", "msg": msg})
        with self._write_lock:
            sys.stdout.write(json.dumps(msg) + "\n")
            sys.stdout.flush()

    def notify(self, method: str, params: dict) -> None:
        self.send({"method": method, "params": params, "emittedAtMs": _now_ms()})

    def error(self, rid: Any, message: str, code: int = INVALID_REQUEST) -> None:
        self.send({"error": {"code": code, "message": message}, "id": rid})

    # --- main loop -----------------------------------------------------------------------------------
    def serve(self) -> None:
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                self.record({"dir": "in", "raw": raw, "violation": "not JSON"})
                continue
            if "method" in msg:
                self._on_client_message(msg)
            else:
                self._on_response(msg)
        self.record({"event": "stdin_eof"})

    def _on_client_message(self, msg: dict) -> None:
        method, rid, params = msg.get("method"), msg.get("id"), msg.get("params")
        if rid is None:  # notification
            known = method in {"initialized"}
            self.record({"dir": "in", "msg": msg, **({} if known else {"violation": f"unknown notification {method}"})})
            return
        spec = REQUEST_PARAMS.get(method)
        if spec is None:
            self.record({"dir": "in", "msg": msg, "violation": f"unknown method {method}"})
            self.error(rid, f"Invalid request: unknown variant `{method}`, expected one of "
                            + ", ".join(f"`{m}`" for m in REQUEST_PARAMS))
            return
        err, ignored = validate(spec, params if params is not None else {})
        self.record({"dir": "in", "msg": msg, "ignored": ignored, **({"violation": err} if err else {})})
        if err:
            self.error(rid, f"Invalid request: {err}")
            return
        if method != "initialize" and not self._initialized:
            self.error(rid, "Not initialized")
            return
        self._HANDLERS[method](self, rid, params or {})

    def _on_response(self, msg: dict) -> None:
        with self._pending_cv:
            pending = self._pending.get(msg.get("id"))
        entry: dict = {"dir": "in", "msg": msg}
        if pending is None:
            entry["violation"] = "response to unknown server request id"
        elif "result" in msg:
            err, ignored = validate(SERVER_REQUEST_RESULTS[pending["method"]], msg["result"])
            entry.update({"reply_to": pending["method"], "ignored": ignored, **({"violation": err} if err else {})})
        else:
            entry["reply_to"] = pending["method"]
        self.record(entry)
        if pending is not None:
            with self._pending_cv:
                pending["reply"] = msg
                self._pending_cv.notify_all()

    def server_request(self, method: str, params: dict, timeout: float = 60.0) -> dict:
        """Issue a server-initiated request and block for Hermes' reply."""
        with self._pending_cv:
            self._next_server_id += 1
            rid = self._next_server_id
            self._pending[rid] = {"method": method}
        self.send({"id": rid, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        with self._pending_cv:
            while "reply" not in self._pending[rid]:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.record({"event": "server_request_unanswered", "method": method, "id": rid})
                    return {}
                self._pending_cv.wait(remaining)
            return self._pending.pop(rid)["reply"]

    # --- request handlers ------------------------------------------------------------------------------
    def _initialize(self, rid: Any, params: dict) -> None:
        if self._initialized:
            self.error(rid, "Already initialized")
            return
        self._initialized = True
        self.send({"id": rid, "result": {
            "userAgent": f"{params['clientInfo']['name']}/{CLI_VERSION} (fake)", "codexHome": str(self.state_dir),
            "platformFamily": "unix", "platformOs": sys.platform}})

    def _thread_payload(self, thread_id: str, data: dict) -> dict:
        info = data["threads"][thread_id]
        return {
            "id": thread_id, "sessionId": thread_id, "preview": info.get("preview", ""), "ephemeral": False,
            "modelProvider": "openai", "createdAt": info["createdAt"], "updatedAt": int(time.time()),
            "status": {"type": "idle"}, "cwd": info["cwd"], "cliVersion": CLI_VERSION, "source": "vscode",
            "turns": [{"id": t["id"], "items": t["items"], "status": t["status"]} for t in info["turns"]],
        }

    def _thread_response(self, thread_id: str, data: dict) -> dict:
        info = data["threads"][thread_id]
        return {"thread": self._thread_payload(thread_id, data), "model": "gpt-fake", "modelProvider": "openai",
                "cwd": info["cwd"], "approvalPolicy": "on-request", "approvalsReviewer": "user",
                "sandbox": {"type": "workspaceWrite"}, "instructionSources": []}

    def _thread_start(self, rid: Any, params: dict) -> None:
        data = self.store.load()
        thread_id = str(uuid.uuid4())
        data["threads"][thread_id] = {"createdAt": int(time.time()), "cwd": params.get("cwd") or os.getcwd(),
                                      "turns": [], "developerInstructions": params.get("developerInstructions")}
        self.store.save(data)
        self.thread_id = thread_id
        self.send({"id": rid, "result": self._thread_response(thread_id, data)})
        self.notify("thread/started", {"thread": self._thread_payload(thread_id, data)})

    def _thread_resume(self, rid: Any, params: dict) -> None:
        data = self.store.load()
        thread_id = params["threadId"]
        if self.scenario.get("forget_threads") or thread_id not in data["threads"]:
            self.error(rid, f"no rollout found for thread id {thread_id}")
            return
        self.thread_id = thread_id
        self.send({"id": rid, "result": self._thread_response(thread_id, data)})

    def _turn_start(self, rid: Any, params: dict) -> None:
        data = self.store.load()
        thread_id = params["threadId"]
        if thread_id not in data["threads"]:
            self.error(rid, f"thread not found: {thread_id}")
            return
        turns = self.scenario.get("turns") or []
        cursor = data.get("turn_cursor", 0)
        script = turns[cursor] if cursor < len(turns) else {"steps": [{"kind": "message", "text": "(unscripted)"}]}
        data["turn_cursor"] = cursor + 1
        self.store.save(data)
        if script.get("start_error"):
            self.error(rid, script["start_error"])
            return
        turn_id = str(uuid.uuid4())
        self.send({"id": rid, "result": {"turn": {"id": turn_id, "items": [], "status": "inProgress"}}})
        self._turn_thread = threading.Thread(target=self._play_turn, args=(thread_id, turn_id, params, script),
                                             daemon=True)
        self._turn_thread.start()

    def _turn_interrupt(self, rid: Any, params: dict) -> None:
        self._interrupted.add(params["turnId"])
        self.send({"id": rid, "result": {}})

    def _turn_steer(self, rid: Any, params: dict) -> None:
        self.send({"id": rid, "result": {"turnId": params["expectedTurnId"]}})

    def _compact_start(self, rid: Any, params: dict) -> None:
        self.send({"id": rid, "result": {}})
        turn_id = str(uuid.uuid4())
        self._turn_thread = threading.Thread(
            target=self._play_turn, args=(params["threadId"], turn_id, None, {"steps": [{"kind": "compaction"}]}),
            daemon=True)
        self._turn_thread.start()

    _HANDLERS: dict[str, Callable[..., None]] = {
        "initialize": _initialize, "thread/start": _thread_start, "thread/resume": _thread_resume,
        "turn/start": _turn_start, "turn/interrupt": _turn_interrupt, "turn/steer": _turn_steer,
        "thread/compact/start": _compact_start,
    }

    # --- turn playback -----------------------------------------------------------------------------------
    def _play_turn(self, thread_id: str, turn_id: str, params: Optional[dict], script: dict) -> None:
        ctx = _TurnCtx(self, thread_id, turn_id)
        self.notify("turn/started", {"threadId": thread_id, "turn": {"id": turn_id, "items": [],
                                                                      "status": "inProgress"}})
        if params is not None:  # codex echoes the submitted input as a userMessage item
            ctx.item({"type": "userMessage", "id": ctx.new_id(), "content": params["input"]})
        status, error = "completed", None
        try:
            for step in script.get("steps", []):
                ctx.check_interrupt()
                STEPS[step["kind"]](ctx, step)
        except _TurnEnded as ended:
            status, error = ended.args[0], (ended.args[1] if len(ended.args) > 1 else None)
        data = self.store.load()
        data["threads"][thread_id]["turns"].append({"id": turn_id, "items": ctx.items, "status": status})
        self.store.save(data)
        turn: dict = {"id": turn_id, "items": [], "status": status}
        if error:
            turn["error"] = {"message": error}
        self.notify("turn/completed", {"threadId": thread_id, "turn": turn})


class _TurnCtx:
    def __init__(self, server: FakeAppServer, thread_id: str, turn_id: str) -> None:
        self.server, self.thread_id, self.turn_id = server, thread_id, turn_id
        self.items: list[dict] = []
        self._n = 0

    def new_id(self) -> str:
        self._n += 1
        return f"item_{self.turn_id[:8]}_{self._n}"

    def scope(self, **extra: Any) -> dict:
        return {"threadId": self.thread_id, "turnId": self.turn_id, **extra}

    def started(self, item: dict) -> None:
        self.server.notify("item/started", self.scope(item=item, startedAtMs=_now_ms()))

    def completed(self, item: dict) -> None:
        self.items.append(item)
        self.server.notify("item/completed", self.scope(item=item, completedAtMs=_now_ms()))

    def item(self, item: dict) -> None:
        self.started(item)
        self.completed(item)

    def check_interrupt(self) -> None:
        if self.turn_id in self.server._interrupted:
            raise _TurnEnded("interrupted")


def _step_reasoning(ctx: _TurnCtx, step: dict) -> None:
    item_id = ctx.new_id()
    ctx.started({"type": "reasoning", "id": item_id, "summary": [], "content": []})
    for index, part in enumerate(step.get("summary", [])):
        for chunk in (part[: len(part) // 2], part[len(part) // 2:]):
            ctx.server.notify("item/reasoning/summaryTextDelta", ctx.scope(itemId=item_id, delta=chunk,
                                                                           summaryIndex=index))
    ctx.completed({"type": "reasoning", "id": item_id, "summary": step.get("summary", []),
                   "content": step.get("content", [])})


def _step_command(ctx: _TurnCtx, step: dict) -> None:
    item_id = ctx.new_id()
    cwd = step.get("cwd") or ctx.server.store.load()["threads"][ctx.thread_id]["cwd"]
    base = {"type": "commandExecution", "id": item_id, "command": step["command"], "cwd": cwd,
            "commandActions": [{"type": "unknown", "command": step["command"]}]}
    ctx.started({**base, "status": "inProgress"})
    decision = "accept"
    if step.get("approval", True):
        reply = ctx.server.server_request("item/commandExecution/requestApproval", ctx.scope(
            itemId=item_id, startedAtMs=_now_ms(), command=step["command"], cwd=cwd, reason=step.get("reason")))
        decision = (reply.get("result") or {}).get("decision", "decline")
    if decision not in ("accept", "acceptForSession"):
        ctx.completed({**base, "status": "declined", "aggregatedOutput": None, "exitCode": None})
        return
    ctx.server.notify("item/commandExecution/outputDelta", ctx.scope(itemId=item_id, delta=step["output"]))
    ctx.completed({**base, "status": "completed" if step.get("exit_code", 0) == 0 else "failed",
                   "aggregatedOutput": step["output"], "exitCode": step.get("exit_code", 0), "durationMs": 7})


def _step_message(ctx: _TurnCtx, step: dict) -> None:
    item_id, text = ctx.new_id(), step["text"]
    ctx.started({"type": "agentMessage", "id": item_id, "text": ""})
    size = max(1, len(text) // max(1, step.get("chunks", 3)))
    for start in range(0, len(text), size):
        ctx.server.notify("item/agentMessage/delta", ctx.scope(itemId=item_id, delta=text[start:start + size]))
    ctx.completed({"type": "agentMessage", "id": item_id, "text": text})


def _step_message_partial(ctx: _TurnCtx, step: dict) -> None:
    """An agentMessage that streams deltas but never reaches item/completed (dies mid-item)."""
    item_id = ctx.new_id()
    ctx.started({"type": "agentMessage", "id": item_id, "text": ""})
    ctx.server.notify("item/agentMessage/delta", ctx.scope(itemId=item_id, delta=step["text"]))


def _step_usage(ctx: _TurnCtx, step: dict) -> None:
    breakdown = {"inputTokens": step["input"], "cachedInputTokens": step.get("cached", 0),
                 "outputTokens": step.get("output", 10), "reasoningOutputTokens": step.get("reasoning", 0)}
    breakdown["totalTokens"] = breakdown["inputTokens"] + breakdown["outputTokens"]
    ctx.server.notify("thread/tokenUsage/updated", ctx.scope(tokenUsage={
        "last": breakdown, "total": breakdown, "modelContextWindow": step.get("window", 272000)}))


def _step_compaction(ctx: _TurnCtx, step: dict) -> None:
    ctx.item({"type": "contextCompaction", "id": ctx.new_id()})


def _step_error_note(ctx: _TurnCtx, step: dict) -> None:
    ctx.server.notify("error", ctx.scope(error={"message": step["message"]}, willRetry=step.get("will_retry", True)))


def _step_fail(ctx: _TurnCtx, step: dict) -> None:
    raise _TurnEnded("failed", step["message"])


def _step_crash(ctx: _TurnCtx, step: dict) -> None:
    ctx.server.record({"event": "crash", "code": step.get("code", 1)})
    sys.stderr.write(step.get("stderr", "fake codex: fatal error") + "\n")
    sys.stderr.flush()
    os._exit(step.get("code", 1))


def _step_progress(ctx: _TurnCtx, step: dict) -> None:
    """Keep the turn busy for ``seconds``, streaming a delta every ``interval`` (a model that is working)."""
    item_id = ctx.new_id()
    ctx.started({"type": "agentMessage", "id": item_id, "text": ""})
    deadline = time.monotonic() + step["seconds"]
    text = ""
    while time.monotonic() < deadline:
        ctx.check_interrupt()
        text += "."
        ctx.server.notify("item/agentMessage/delta", ctx.scope(itemId=item_id, delta="."))
        time.sleep(step.get("interval", 0.25))
    ctx.check_interrupt()
    ctx.completed({"type": "agentMessage", "id": item_id, "text": step.get("text", text)})


def _step_permissions(ctx: _TurnCtx, step: dict) -> None:
    ctx.server.server_request("item/permissions/requestApproval", ctx.scope(
        itemId=ctx.new_id(), startedAtMs=_now_ms(), cwd=os.getcwd(), reason=step.get("reason"),
        permissions={"network": {"enabled": True}}))


# The descendant lives until the harness drops the release file (or 600 s pass), so the test side can
# retire an orphan without signalling a process that is no longer in its own subtree.
_GRANDCHILD = ("import os, sys, time\n"
               "deadline = time.monotonic() + 600\n"
               "while time.monotonic() < deadline and not os.path.exists(sys.argv[1]):\n"
               "    time.sleep(0.2)\n")


def _step_grandchild(ctx: _TurnCtx, step: dict) -> None:
    """A descendant in its own session (like codex's stdio MCP servers)."""
    child = subprocess.Popen([sys.executable, "-c", _GRANDCHILD, str(ctx.server.state_dir / GRANDCHILD_RELEASE)],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
    ctx.server.record({"event": "grandchild", "child_pid": child.pid})


STEPS: dict[str, Callable[[_TurnCtx, dict], None]] = {
    "reasoning": _step_reasoning, "command": _step_command, "message": _step_message, "usage": _step_usage,
    "message_partial": _step_message_partial,
    "compaction": _step_compaction, "error_note": _step_error_note, "fail": _step_fail, "crash": _step_crash,
    "progress": _step_progress, "permissions": _step_permissions, "grandchild": _step_grandchild,
}


def main(argv: list[str]) -> int:
    if "--version" in argv:
        print(f"codex-cli {CLI_VERSION}")
        return 0
    state_dir = Path(argv[argv.index("--state-dir") + 1])
    if "app-server" not in argv:
        sys.stderr.write(f"fake codex: unsupported argv {argv}\n")
        return 2
    server = FakeAppServer(state_dir, argv)

    def on_sigterm(*_: Any) -> None:
        server.record({"event": "exit", "how": "SIGTERM"})
        os._exit(143)

    signal.signal(signal.SIGTERM, on_sigterm)
    server.serve()
    server.record({"event": "exit", "how": "stdin_eof"})
    return 0


# ---------------------------------------------------------------------------------------------------
# Test-side harness
# ---------------------------------------------------------------------------------------------------

class FakeCodex:
    """One fake codex install: wrapper executable + state dir, and readers over the transcript."""

    def __init__(self, root: Path, turns: list[dict], **scenario: Any) -> None:
        self.state_dir = root / "fake_codex"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "scenario.json").write_text(json.dumps({"turns": turns, **scenario}), encoding="utf-8")
        self.bin = root / "bin" / "codex"
        self.bin.parent.mkdir(parents=True, exist_ok=True)
        self.bin.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{Path(__file__).resolve()}" '
                            f'--state-dir "{self.state_dir}" "$@"\n', encoding="utf-8")
        self.bin.chmod(0o755)

    def set_scenario(self, **changes: Any) -> None:
        path = self.state_dir / "scenario.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update(changes)
        path.write_text(json.dumps(data), encoding="utf-8")

    def entries(self) -> list[dict]:
        path = self.state_dir / "transcript.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def requests(self, method: Optional[str] = None) -> list[dict]:
        """Client requests Hermes sent (full transcript entries), optionally one method."""
        return [e for e in self.entries() if e.get("dir") == "in" and "method" in e.get("msg", {})
                and "id" in e["msg"] and (method is None or e["msg"]["method"] == method)]

    def replies_to(self, server_method: str) -> list[dict]:
        return [e for e in self.entries() if e.get("reply_to") == server_method]

    def violations(self) -> list[dict]:
        return [e for e in self.entries() if e.get("violation")]

    def ignored_fields(self) -> list[tuple[str, str]]:
        return [(str(e["msg"].get("method") or e.get("reply_to")), path) for e in self.entries()
                for path in e.get("ignored") or []]

    def spawned_pids(self) -> list[int]:
        return [e["pid"] for e in self.entries() if e.get("event") == "spawn"]

    def grandchild_pids(self) -> list[int]:
        return [e["child_pid"] for e in self.entries() if e.get("event") == "grandchild"]

    def threads(self) -> dict:
        return _Store(self.state_dir / "threads.json").load()["threads"]

    def assert_wire_clean(self) -> None:
        """Every request/response Hermes sent is valid AND carries no field codex would silently drop."""
        assert not self.violations(), f"protocol violations: {self.violations()}"
        assert not self.ignored_fields(), f"fields codex ignores (intent silently lost): {self.ignored_fields()}"


@dataclass
class CodexRun:
    """Outcome of :func:`run_codex_scenario`: the fake, the Hermes home and one ChatResult per CLI run."""
    fake: FakeCodex
    home: Any
    results: list
    session_id: str

    @property
    def output(self) -> str:
        return "".join(r.stdout + r.stderr for r in self.results)

    def process_entries(self, index: int) -> list[dict]:
        """Transcript entries of the ``index``-th app-server process (one per CLI run)."""
        pid = self.fake.spawned_pids()[index]
        return [e for e in self.fake.entries() if e.get("pid") == pid]

    def process_requests(self, index: int, method: str) -> list[dict]:
        return [e["msg"] for e in self.process_entries(index)
                if e.get("dir") == "in" and e.get("msg", {}).get("method") == method and "id" in e["msg"]]

    def cleanup(self) -> None:
        """Retire anything the fake spawned that is still alive (orphan scenarios).

        Orphaned descendants are released cooperatively first: by teardown they are reparented to init,
        outside the test's process subtree, where a live-system guard (rightly) refuses to signal them."""
        (self.fake.state_dir / GRANDCHILD_RELEASE).touch()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and any(pid_alive(p) for p in self.fake.grandchild_pids()):
            time.sleep(0.05)
        for pid in self.fake.spawned_pids() + self.fake.grandchild_pids():
            if pid_alive(pid):
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)


def run_codex_scenario(root: Path, turns: list[dict], runs: list[dict], *, config: Optional[dict] = None,
                       **scenario: Any) -> CodexRun:
    """Real ``hermes chat -q`` runs (``--resume`` after the first) against a fresh fake codex install.

    ``runs``: ``{"prompt", "args": [...], "then": {scenario changes applied after this run}}``."""
    from tests.e2e.core.providers._native_helpers import latest_session, make_home, run_chat

    fake = FakeCodex(root, turns, **scenario)
    model = {"provider": "openai", "default": "gpt-5.5", "openai_runtime": "codex_app_server",
             "codex_bin": str(fake.bin)}
    # The app-server owns auth; the key only satisfies Hermes' provider resolution and never leaves.
    home = make_home(root, model, env_file={"OPENAI_API_KEY": "sk-fake-codex-e2e"}, extra_config=config)
    results, session_id = [], None
    for run in runs:
        results.append(run_chat(home, run["prompt"], args=tuple(run.get("args", ())), resume=session_id,
                                timeout=run.get("timeout", 120)))
        session_id = latest_session(home)
        if run.get("then"):
            fake.set_scenario(**run["then"])
    assert session_id is not None
    return CodexRun(fake, home, results, session_id)


def pid_alive(pid: int) -> bool:
    """True while ``pid`` exists and is not a zombie."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, IndexError, ProcessLookupError):
        return False


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

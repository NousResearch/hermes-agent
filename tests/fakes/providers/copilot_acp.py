#!/usr/bin/env python3
"""Fake ACP agent executable standing in for ``copilot --acp --stdio`` (provider ``copilot-acp``).

Hermes' ``copilot-acp`` provider spawns an external agent process per model call and speaks the
Agent Client Protocol to it: JSON-RPC 2.0, one JSON object per line over stdio
(https://agentclientprotocol.com/protocol/overview). This module is that process. It

* answers ``<cmd> --help`` with a usage text advertising ``--acp`` (Hermes probes it before spawning);
* validates every client request against the published ACP schema (the ``agent-client-protocol``
  package's pydantic models, ``acp.schema``) plus the spec rules the models do not encode
  (initialize-first, absolute ``cwd``, no custom root fields, known ``sessionId``) and REJECTS
  malformed ones with a JSON-RPC ``-32602 Invalid params`` / ``-32600`` error, as an agent would;
* builds every message it sends from the same schema models, so a shape drift fails here first;
* replays a scripted turn per main-turn ``session/prompt`` (thought / message chunks with delays,
  ``tool_call`` / ``tool_call_update``, ``session/request_permission``, ``fs/read_text_file``,
  the ``{stopReason}`` result, JSON-RPC errors, a hard crash, chunks AFTER the result);
* appends every inbound and outbound message, with wall-clock time and pid, to
  ``<state>/transcript.jsonl`` (shared by every process of one scenario).

Test-side API: :class:`AcpFake` (write the script, build the launcher, read the transcript) and the
action builders (:func:`thought`, :func:`message`, ...). Run as a script it is the agent itself:
``copilot_acp.py --acp --stdio --state <dir>``.
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

PROTOCOL_VERSION = 1
TOOLS_MARKER = "Available tools"
AUX_ANSWER = "fake-acp auxiliary answer"
USAGE = """usage: copilot [options]

Options:
  --acp            Run as an Agent Client Protocol server
  --stdio          Use stdio transport (with --acp)
  --state <dir>    (fake) scenario directory holding script.json / transcript.jsonl
  -h, --help       Show help
"""


# ── test-side API ────────────────────────────────────────────────────────────────────────────


def thought(text: str, delay: float = 0.0) -> dict[str, Any]:
    return {"type": "thought", "text": text, "delay": delay}


def message(text: str, delay: float = 0.0) -> dict[str, Any]:
    return {"type": "message", "text": text, "delay": delay}


def tool_call(call_id: str, title: str, kind: str = "other", status: str = "pending") -> dict[str, Any]:
    return {"type": "tool_call", "id": call_id, "title": title, "kind": kind, "status": status}


def tool_update(call_id: str, status: str = "completed", text: str = "") -> dict[str, Any]:
    return {"type": "tool_update", "id": call_id, "status": status, "text": text}


def permission(call_id: str, title: str = "run a command") -> dict[str, Any]:
    return {"type": "permission", "id": call_id, "title": title}


def fs_read(path: str) -> dict[str, Any]:
    return {"type": "fs_read", "path": path}


def result(stop_reason: str = "end_turn") -> dict[str, Any]:
    return {"type": "result", "stopReason": stop_reason}


def rpc_error(code: int, msg: str, data: Any = None) -> dict[str, Any]:
    return {"type": "error", "code": code, "message": msg, "data": data}


def crash(exit_code: int = 1, stderr: str = "fatal: agent crashed") -> dict[str, Any]:
    return {"type": "crash", "code": exit_code, "stderr": stderr}


def hermes_tool_call(call_id: str, name: str, args: dict[str, Any]) -> str:
    """Text a model behind ACP emits to call a Hermes tool (ACP has no OpenAI tools channel)."""
    body = {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
    return f"<tool_call>{json.dumps(body)}</tool_call>"


class AcpFake:
    """One scenario directory: ``script.json`` in, ``transcript.jsonl`` out, plus a launcher."""

    def __init__(self, state_dir: Path, turns: list[list[dict[str, Any]]], *, models: list[str] | None = None,
                 current_model: str | None = None, ignore_sigterm: bool = False, load_session: bool = True,
                 aux_text: str = AUX_ANSWER):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        script = {"turns": turns, "models": models or [], "current_model": current_model,
                  "ignore_sigterm": ignore_sigterm, "load_session": load_session, "aux_text": aux_text}
        (self.state_dir / "script.json").write_text(json.dumps(script), encoding="utf-8")
        self.launcher = self.state_dir / "copilot"
        self.launcher.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{Path(__file__).resolve()}" "$@"\n', encoding="utf-8")
        self.launcher.chmod(0o755)

    def env(self) -> dict[str, str]:
        """Env vars that point Hermes' copilot-acp client at this fake."""
        return {"HERMES_COPILOT_ACP_COMMAND": str(self.launcher),
                "HERMES_COPILOT_ACP_ARGS": f"--acp --stdio --state {self.state_dir}"}

    def records(self) -> list[dict[str, Any]]:
        path = self.state_dir / "transcript.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def inbound(self, method: str | None = None) -> list[dict[str, Any]]:
        """Client->agent messages (requests, notifications, responses), optionally one method."""
        return [r for r in self.records() if r["dir"] == "in" and (method is None or r["msg"].get("method") == method)]

    def main_prompts(self) -> list[dict[str, Any]]:
        """Main-turn ``session/prompt`` records (those carrying Hermes' tool bridge)."""
        return [r for r in self.inbound("session/prompt") if r.get("main")]

    def aux_prompts(self) -> list[dict[str, Any]]:
        """Auxiliary ``session/prompt`` records (no tool bridge: compaction summaries, titles)."""
        return [r for r in self.inbound("session/prompt") if not r.get("main")]

    def invalid(self) -> list[dict[str, Any]]:
        return [r for r in self.records() if r["dir"] == "in" and r.get("errors")]

    def pids(self) -> list[int]:
        return sorted({r["pid"] for r in self.records() if r["dir"] == "spawn"})

    def events(self, kind: str) -> list[dict[str, Any]]:
        return [r for r in self.records() if r["dir"] == kind]


def prompt_text(record: dict[str, Any]) -> str:
    return "".join(b.get("text", "") for b in record["msg"]["params"].get("prompt", []) if b.get("type") == "text")


# ── agent process ────────────────────────────────────────────────────────────────────────────


class Agent:
    """One spawned agent process (one Hermes model call)."""

    def __init__(self, state_dir: Path):
        import acp.schema as schema  # validation/building uses the published ACP models

        self.s = schema
        self.state_dir = state_dir
        self.script = json.loads((state_dir / "script.json").read_text(encoding="utf-8"))
        self.initialized = False
        self.client_caps: dict[str, Any] = {}
        self.sessions: set[str] = set()
        self.turn: int | None = None
        self.next_id = 1000
        self.request_schemas = {
            "initialize": schema.InitializeRequest,
            "session/new": schema.NewSessionRequest,
            "session/load": schema.LoadSessionRequest,
            "session/prompt": schema.PromptRequest,
            "session/set_model": schema.SetSessionModelRequest,
            "session/set_config_option": schema.SetSessionConfigOptionSelectRequest,
            "session/cancel": schema.CancelNotification,
        }
        self.handlers = {
            "initialize": self._initialize, "session/new": self._new_session, "session/load": self._load_session,
            "session/set_model": self._ack, "session/set_config_option": self._set_config_option,
            "session/prompt": self._prompt,
        }

    # transcript + wire ------------------------------------------------------------------------

    def log(self, direction: str, msg: Any = None, **extra: Any) -> None:
        rec = {"t": time.time(), "pid": os.getpid(), "turn": self.turn, "dir": direction, "msg": msg, **extra}
        with open(self.state_dir / "transcript.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def send(self, msg: dict[str, Any]) -> None:
        self.log("out", msg)
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()

    def dump(self, model: Any) -> dict[str, Any]:
        return model.model_dump(by_alias=True, exclude_none=True, mode="json")

    def reply(self, msg_id: Any, result_model: Any) -> None:
        body = result_model if isinstance(result_model, dict) or result_model is None else self.dump(result_model)
        self.send({"jsonrpc": "2.0", "id": msg_id, "result": body})

    def error(self, msg_id: Any, code: int, text: str, data: Any = None) -> None:
        err = {"code": code, "message": text, **({"data": data} if data is not None else {})}
        self.send({"jsonrpc": "2.0", "id": msg_id, "error": err})

    def update(self, session_id: str, update_model: Any) -> None:
        note = self.s.SessionNotification(session_id=session_id, update=update_model)
        self.send({"jsonrpc": "2.0", "method": "session/update", "params": self.dump(note)})

    def read(self) -> dict[str, Any] | None:
        line = sys.stdin.readline()
        if not line:
            return None
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            self.log("in", line.rstrip("\n"), errors=[f"parse error: {exc}"])
            self.error(None, -32700, "Parse error")
            return {}
        return msg

    # validation -------------------------------------------------------------------------------

    def validate(self, msg: dict[str, Any]) -> list[str]:
        """Spec violations of one client request/notification (empty list = valid)."""
        errors: list[str] = []
        if msg.get("jsonrpc") != "2.0":
            errors.append("jsonrpc must be '2.0'")
        method = msg.get("method")
        if "id" in msg and not isinstance(msg["id"], (int, str)):
            errors.append("id must be a string or integer")
        model = self.request_schemas.get(method)
        if model is None:
            return errors
        params = msg.get("params")
        try:
            parsed = model.model_validate(params)
        except ValidationError as exc:
            return errors + [f"schema: {e['loc']} {e['msg']}" for e in exc.errors()]
        allowed = {f.alias or name for name, f in model.model_fields.items()}
        errors += [f"custom root field {k!r} (spec: use _meta)" for k in sorted(set(params) - allowed)]
        if method != "initialize" and not self.initialized:
            errors.append("request before initialize")
        if method in ("session/new", "session/load") and not os.path.isabs(parsed.cwd):
            errors.append("cwd must be an absolute path")
        sid = getattr(parsed, "session_id", None)
        if method not in ("session/load",) and sid is not None and sid not in self.sessions:
            errors.append(f"unknown sessionId {sid!r}")
        return errors

    # request handlers -------------------------------------------------------------------------

    def _initialize(self, msg_id: Any, params: dict[str, Any]) -> None:
        self.initialized = True
        self.client_caps = params.get("clientCapabilities") or {}
        caps = self.s.AgentCapabilities(load_session=bool(self.script.get("load_session")))
        self.reply(msg_id, self.s.InitializeResponse(
            protocol_version=PROTOCOL_VERSION, agent_capabilities=caps, auth_methods=[],
            agent_info=self.s.Implementation(name="fake-copilot", title="Fake Copilot", version="1.0.0")))

    def _session_response(self, session_id: str) -> Any:
        models = self.script.get("models") or []
        options = None
        if models:
            current = self.script.get("current_model") or models[0]
            options = [self.s.SessionConfigOptionSelect(
                id="model", name="Model", category="model", type="select", current_value=current,
                options=[self.s.SessionConfigSelectOption(value=m, name=m) for m in models])]
        return self.s.NewSessionResponse(session_id=session_id, config_options=options)

    def _new_session(self, msg_id: Any, params: dict[str, Any]) -> None:
        session_id = f"fake-sess-{os.getpid()}-{len(self.sessions) + 1}"
        self.sessions.add(session_id)
        self.reply(msg_id, self._session_response(session_id))

    def _load_session(self, msg_id: Any, params: dict[str, Any]) -> None:
        self.sessions.add(params["sessionId"])
        self.reply(msg_id, self.s.LoadSessionResponse())

    def _ack(self, msg_id: Any, params: dict[str, Any]) -> None:
        self.reply(msg_id, {})

    def _set_config_option(self, msg_id: Any, params: dict[str, Any]) -> None:
        models = self.script.get("models") or []
        if params.get("configId") != "model" or params.get("value") not in models:
            self.error(msg_id, -32602, "Invalid params", {"reason": "unknown config option or value"})
            return
        self.script["current_model"] = params["value"]
        self.reply(msg_id, self.s.SetSessionConfigOptionResponse(config_options=self._session_response("x").config_options))

    def _claim_turn(self) -> int:
        """Next scripted turn index, shared across every process of the scenario (file lock)."""
        with open(self.state_dir / "turn.counter", "a+", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.seek(0)
            index = int(fh.read().strip() or 0)
            fh.seek(0)
            fh.truncate()
            fh.write(str(index + 1))
            return index

    def _prompt(self, msg_id: Any, params: dict[str, Any]) -> None:
        text = "".join(b.get("text", "") for b in params["prompt"] if b.get("type") == "text")
        session_id = params["sessionId"]
        if TOOLS_MARKER not in text:  # auxiliary call (title/summary): never consumes a scripted turn
            self.update(session_id, self._chunk("agent_message_chunk", self.script.get("aux_text") or AUX_ANSWER))
            self.reply(msg_id, self.s.PromptResponse(stop_reason="end_turn"))
            return
        turns = self.script.get("turns") or []
        index = self._claim_turn()
        self.turn = index
        self.log("turn", {"index": index})
        actions = turns[index] if index < len(turns) else [message("(fake-acp: script exhausted)")]
        replied = False
        for action in actions:
            replied = self._run_action(action, msg_id, session_id) or replied
        if not replied:
            self.reply(msg_id, self.s.PromptResponse(stop_reason="end_turn"))

    # scripted actions -------------------------------------------------------------------------

    def _chunk(self, kind: str, text: str) -> Any:
        cls = {"agent_message_chunk": self.s.AgentMessageChunk, "agent_thought_chunk": self.s.AgentThoughtChunk}[kind]
        return cls(session_update=kind, content=self.s.TextContentBlock(type="text", text=text))

    def _run_action(self, action: dict[str, Any], msg_id: Any, session_id: str) -> bool:
        """Perform one scripted action; True when it answered the pending ``session/prompt``."""
        if action.get("delay"):
            time.sleep(float(action["delay"]))
        return bool(self._actions[action["type"]](self, action, msg_id, session_id))

    def _a_thought(self, action, msg_id, session_id):
        self.update(session_id, self._chunk("agent_thought_chunk", action["text"]))

    def _a_message(self, action, msg_id, session_id):
        self.update(session_id, self._chunk("agent_message_chunk", action["text"]))

    def _a_tool_call(self, action, msg_id, session_id):
        self.update(session_id, self.s.ToolCallStart(
            session_update="tool_call", tool_call_id=action["id"], title=action["title"], kind=action["kind"],
            status=action["status"]))

    def _a_tool_update(self, action, msg_id, session_id):
        content = None
        if action.get("text"):
            content = [self.s.ContentToolCallContent(
                type="content", content=self.s.TextContentBlock(type="text", text=action["text"]))]
        self.update(session_id, self.s.ToolCallProgress(
            session_update="tool_call_update", tool_call_id=action["id"], status=action["status"], content=content))

    def _a_permission(self, action, msg_id, session_id):
        req = self.s.RequestPermissionRequest(
            session_id=session_id,
            tool_call=self.s.ToolCallUpdate(tool_call_id=action["id"], title=action["title"], kind="execute"),
            options=[self.s.PermissionOption(option_id="allow-once", name="Allow once", kind="allow_once"),
                     self.s.PermissionOption(option_id="reject-once", name="Reject", kind="reject_once")])
        response = self._client_request("session/request_permission", self.dump(req))
        self._check_response(response, self.s.RequestPermissionResponse, "permission_outcome")

    def _a_fs_read(self, action, msg_id, session_id):
        if not (self.client_caps.get("fs") or {}).get("readTextFile"):
            self.log("fs_skipped", {"reason": "client did not advertise fs.readTextFile"})
            return
        req = self.s.ReadTextFileRequest(session_id=session_id, path=action["path"])
        response = self._client_request("fs/read_text_file", self.dump(req))
        self._check_response(response, self.s.ReadTextFileResponse, "fs_read_result")

    def _a_result(self, action, msg_id, session_id):
        self.reply(msg_id, self.s.PromptResponse(stop_reason=action["stopReason"]))
        return True

    def _a_error(self, action, msg_id, session_id):
        self.error(msg_id, int(action["code"]), action["message"], action.get("data"))
        return True

    def _a_crash(self, action, msg_id, session_id):
        self.log("crash", {"code": action["code"]})
        sys.stderr.write(action["stderr"] + "\n")
        sys.stderr.flush()
        os._exit(int(action["code"]))

    _actions = {"thought": _a_thought, "message": _a_message, "tool_call": _a_tool_call, "tool_update": _a_tool_update,
                "permission": _a_permission, "fs_read": _a_fs_read, "result": _a_result, "error": _a_error,
                "crash": _a_crash}

    def _client_request(self, method: str, params: dict[str, Any]) -> dict[str, Any] | None:
        self.next_id += 1
        req_id = self.next_id
        self.send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        while (msg := self.read()) is not None:
            if msg.get("id") == req_id and "method" not in msg:
                return msg
            self.dispatch(msg)
        return None

    def _check_response(self, response: dict[str, Any] | None, model: Any, kind: str) -> None:
        errors: list[str] = []
        if response is None:
            errors.append("client closed stdin before answering")
        elif "result" in response:
            try:
                model.model_validate(response["result"])
            except ValidationError as exc:
                errors += [f"schema: {e['loc']} {e['msg']}" for e in exc.errors()]
        elif not isinstance((response or {}).get("error"), dict):
            errors.append("response carries neither result nor error")
        self.log("in", response, kind=kind, errors=errors)

    # loop -------------------------------------------------------------------------------------

    def dispatch(self, msg: dict[str, Any]) -> None:
        method = msg.get("method")
        errors = self.validate(msg) if method else ["unsolicited response"]
        main = method == "session/prompt" and TOOLS_MARKER in "".join(
            b.get("text", "") for b in ((msg.get("params") or {}).get("prompt") or []) if isinstance(b, dict))
        self.log("in", msg, errors=errors, main=main)
        if "id" not in msg or not method:
            return  # notification (session/cancel) or stray response
        if errors:
            code = -32600 if errors == ["request before initialize"] else -32602
            self.error(msg["id"], code, "Invalid params" if code == -32602 else "Invalid request", {"errors": errors})
            return
        handler = self.handlers.get(method)
        if handler is None:
            self.error(msg["id"], -32601, f"Method not found: {method}")
            return
        handler(msg["id"], msg.get("params") or {})

    def serve(self) -> int:
        self.log("spawn", {"argv": sys.argv[1:], "cwd": os.getcwd()})
        while (msg := self.read()) is not None:
            if msg:
                self.dispatch(msg)
        if self.script.get("ignore_sigterm"):
            # A wedged agent: ignores SIGTERM AND stdin EOF; only SIGKILL ends it.
            self.log("wedged", {"reason": "stdin closed; ignoring EOF"})
            while True:
                time.sleep(3600)
        self.log("exit", {"reason": "stdin closed"})
        return 0


def _install_signals(agent: Agent) -> None:
    def _on_term(signum, _frame):
        if agent.script.get("ignore_sigterm"):
            agent.log("signal", {"signal": signum, "ignored": True})
            return
        agent.log("exit", {"reason": f"signal {signum}"})
        os._exit(0)

    signal.signal(signal.SIGTERM, _on_term)


def main(argv: list[str]) -> int:
    if not argv or "-h" in argv or "--help" in argv:
        sys.stdout.write(USAGE)
        return 0
    if "--acp" not in argv or "--state" not in argv:
        sys.stderr.write("error: unknown option; this fake only runs with --acp --stdio --state <dir>\n")
        return 1
    agent = Agent(Path(argv[argv.index("--state") + 1]))
    _install_signals(agent)
    return agent.serve()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

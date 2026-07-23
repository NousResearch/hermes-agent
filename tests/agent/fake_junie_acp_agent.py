"""A minimal fake `junie --acp=true` agent, speaking ACP via the official SDK.

Spawned as a real subprocess by the Junie ACP client tests so the client's SDK
integration (handshake, session lifecycle, streaming, permission + fs bridges,
process reuse / respawn, retry) is exercised end-to-end instead of mocked.

Behaviour is driven entirely by ``FAKE_JUNIE_*`` environment variables so each
test can shape a scenario without editing this file. Every handled method is
appended as one JSON line to ``$FAKE_JUNIE_LOG`` for assertions.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import acp
from acp.exceptions import RequestError
from acp.schema import (
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    SetSessionConfigOptionResponse,
    Usage,
)

_LOG_PATH = os.environ.get("FAKE_JUNIE_LOG", "")


def _log(record: dict[str, Any]) -> None:
    if not _LOG_PATH:
        return
    with open(_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


class FakeJunieAgent:
    def __init__(self) -> None:
        self._conn: acp.AgentSideConnection | None = None
        self._session_count = 0
        self._prompt_count = 0

    def on_connect(self, conn: acp.AgentSideConnection) -> None:
        self._conn = conn

    async def initialize(self, protocol_version: int, **_: Any) -> InitializeResponse:
        _log({"method": "initialize", "pid": os.getpid()})
        return InitializeResponse(protocol_version=1)

    async def new_session(self, cwd: str, mcp_servers: Any = None, **_: Any) -> NewSessionResponse:
        # Cross-process one-shot failure: the first process to reach new_session
        # errors (leaving a sentinel); a freshly respawned process succeeds.
        if _flag("FAKE_JUNIE_FAIL_SESSION_NEW_ONCE"):
            sentinel = os.environ.get("FAKE_JUNIE_SESSION_SENTINEL", "")
            if sentinel and not os.path.exists(sentinel):
                with open(sentinel, "w", encoding="utf-8") as fh:
                    fh.write("used")
                _log({"method": "session/new", "result": "error"})
                raise RequestError.internal_error({"details": "stale session/new"})
        self._session_count += 1
        sid = f"s{self._session_count}"
        _log({"method": "session/new", "session": sid})
        return NewSessionResponse(session_id=sid)

    async def set_config_option(
        self, config_id: str, session_id: str, value: Any, **_: Any
    ) -> SetSessionConfigOptionResponse:
        if _flag("FAKE_JUNIE_FAIL_MODEL") and config_id == "model":
            _log({"method": "session/set_config_option", "configId": config_id, "result": "error"})
            raise RequestError.invalid_params({"details": "unknown model"})
        _log({"method": "session/set_config_option", "configId": config_id, "value": value})
        return SetSessionConfigOptionResponse(config_options=[])

    async def prompt(self, prompt: Any, session_id: str, message_id: str | None = None, **_: Any) -> PromptResponse:
        self._prompt_count += 1
        _log({"method": "session/prompt", "session": session_id, "count": self._prompt_count})
        assert self._conn is not None

        if _flag("FAKE_JUNIE_FAIL_PROMPT"):
            raise RequestError.internal_error({"details": "prompt failed mid-turn"})

        # Optional: ask the client for permission and record what it answered.
        if _flag("FAKE_JUNIE_ASK_PERMISSION"):
            resp = await self._conn.request_permission(
                options=[{"optionId": "yes", "name": "Allow once", "kind": "allow_once"}],
                session_id=session_id,
                tool_call={"toolCallId": "consent", "title": "Do the thing"},
            )
            outcome = resp.outcome.model_dump(by_alias=True) if hasattr(resp.outcome, "model_dump") else resp.outcome
            _log({"method": "permission_outcome", "outcome": outcome})

        # Optional: read a file through the client's sandboxed fs bridge.
        read_path = os.environ.get("FAKE_JUNIE_READ_PATH", "")
        if read_path:
            r = await self._conn.read_text_file(path=read_path, session_id=session_id)
            _log({"method": "fs_read_result", "content": r.content})

        # Optional: attempt a write through the client's fs bridge.
        write_path = os.environ.get("FAKE_JUNIE_WRITE_PATH", "")
        if write_path:
            try:
                await self._conn.write_text_file(
                    content=os.environ.get("FAKE_JUNIE_WRITE_CONTENT", "hi"),
                    path=write_path,
                    session_id=session_id,
                )
                _log({"method": "fs_write_result", "ok": True})
            except RequestError as exc:
                _log({"method": "fs_write_result", "ok": False, "error": str(exc)})

        # Optional: stream native tool activity (never becomes an OpenAI tool_call).
        if _flag("FAKE_JUNIE_EMIT_TOOLCALL"):
            await self._conn.session_update(
                session_id=session_id,
                update={"sessionUpdate": "tool_call", "toolCallId": "t1",
                        "title": 'Found "*"', "kind": "other", "status": "pending"},
            )
            await self._conn.session_update(
                session_id=session_id,
                update={"sessionUpdate": "tool_call_update", "toolCallId": "t1", "status": "completed",
                        "content": [{"type": "content", "content": {"type": "text", "text": "gamma.log\n"}}]},
            )

        # Optional: stream a thought chunk (surfaces as reasoning).
        if _flag("FAKE_JUNIE_EMIT_THOUGHT"):
            await self._conn.session_update(
                session_id=session_id,
                update={"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "thinking..."}},
            )

        prefix = os.environ.get("FAKE_JUNIE_ANSWER_PREFIX", "ANSWER")
        await self._conn.session_update(
            session_id=session_id,
            update={"sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": f"{prefix}{self._prompt_count}"}},
        )

        usage = None
        if _flag("FAKE_JUNIE_USAGE"):
            usage = Usage(input_tokens=123, output_tokens=45, total_tokens=168)

        response = PromptResponse(stop_reason="end_turn", usage=usage)

        if _flag("FAKE_JUNIE_EXIT_AFTER_PROMPT"):
            # Emulate the process dying between turns so the client must respawn.
            asyncio.get_running_loop().call_later(0.05, os._exit, 0)

        return response

    async def close_session(self, session_id: str, **_: Any) -> None:
        _log({"method": "session/close", "session": session_id})
        return None

    async def cancel(self, session_id: str, **_: Any) -> None:
        _log({"method": "session/cancel", "session": session_id})

    async def authenticate(self, method_id: str, **_: Any) -> None:
        return None

    async def load_session(self, *a: Any, **k: Any) -> None:
        return None

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        return None


def main() -> None:
    asyncio.run(acp.run_agent(FakeJunieAgent()))


if __name__ == "__main__":
    main()

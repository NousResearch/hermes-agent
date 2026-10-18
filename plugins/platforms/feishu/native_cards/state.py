"""按真实入站消息绑定卡片；工具观察者不猜测当前聊天。"""

from contextvars import ContextVar
import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import threading
import time
from typing import Any


@dataclass
class CardTurn:
    chat_id: str
    thread_id: str
    reply_to: str
    owner_ids: tuple[str, ...] = ()
    chat_type: str = "dm"
    session_key: str = ""
    text: str = ""
    model: str = ""
    agent_session_id: str = ""
    started: float = field(default_factory=time.monotonic)
    card_id: str = ""
    message_id: str = ""
    draft_id: int | None = None
    final: bool = False
    cancelled: bool = False
    uncertain: bool = False
    fallback: bool = False
    error: str = ""
    calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    round: int = 0
    api_rounds: dict[str, int] = field(default_factory=dict)
    approvals: dict[str, tuple[str, str]] = field(default_factory=dict)
    orchestration: dict = field(default_factory=dict)
    moa_requests: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    io_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @property
    def key(self):
        return self.chat_id, self.thread_id, self.reply_to

    def status(self):
        with self.lock:
            return {
                "calls": deepcopy(list(self.calls.values())[-100:]),
                "orchestration": deepcopy(self.orchestration),
            }


class RoutingHub:
    def __init__(self):
        self.current = ContextVar("hermes_feishu_cards_turn", default=None)

    def pre_auxiliary(self, **payload):
        from .orchestration import observe_auxiliary

        observe_auxiliary(self.current.get(), payload, completed=False)

    def post_auxiliary(self, **payload):
        from .orchestration import observe_auxiliary

        observe_auxiliary(self.current.get(), payload, completed=True)

    def pre_approval(self, session_key="", request_id="", **kwargs):
        if self.current.get() is None or not session_key:
            return
        if not request_id:
            from tools.approval import list_gateway_approvals

            matches = [
                item
                for item in list_gateway_approvals(session_key)
                if item.get("command") == kwargs.get("command")
            ]
            if len(matches) == 1:
                request_id = matches[0].get("request_id", "")
        if request_id:
            turn = self.current.get()[1]
            digest = hashlib.sha256(str(kwargs.get("command", "")).encode()).hexdigest()
            with turn.lock:
                turn.approvals[request_id] = (session_key, digest)

    def pre_api(self, model="", session_id="", **kwargs):
        bound = self.current.get()
        if bound is not None and model:
            with bound[1].lock:
                if bound[1].agent_session_id and session_id != bound[1].agent_session_id:
                    return
                bound[1].agent_session_id = session_id
                bound[1].model = model

    def pre_tool(self, tool_name, tool_call_id="", api_request_id="", **kwargs):
        bound = self.current.get()
        if bound is None or not tool_call_id:
            return
        adapter, turn = bound
        with turn.lock:
            if turn.final or turn.cancelled:
                return
            if tool_call_id in turn.calls:
                return
            if api_request_id and api_request_id not in turn.api_rounds:
                turn.round += 1
                turn.api_rounds[api_request_id] = turn.round
            elif not api_request_id and not any(
                c["status"] == "running" for c in turn.calls.values()
            ):
                turn.round += 1
            turn.calls[tool_call_id] = {
                "id": tool_call_id,
                "name": tool_name,
                "round": turn.api_rounds.get(api_request_id, turn.round),
                "status": "running",
            }
        adapter.schedule_refresh(turn)

    def post_tool(self, tool_name, tool_call_id="", duration_ms=0, status="ok", **kwargs):
        bound = self.current.get()
        if bound is None:
            return
        adapter, turn = bound
        with turn.lock:
            record = turn.calls.get(tool_call_id)
            if record is None or turn.final or turn.cancelled:
                return
            record.update(
                status="done" if status in {"ok", "success", "completed"} else "failed",
                duration=max(0.0, duration_ms / 1000),
            )
        adapter.schedule_refresh(turn)

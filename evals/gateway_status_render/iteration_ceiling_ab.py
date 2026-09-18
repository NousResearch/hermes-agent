"""A/B probe: what the three user-facing gateway status lines render for the iteration counter.

Drives the REAL render paths (busy-ack, long-running heartbeat, inactivity-timeout diagnostic)
with a REAL ``AIAgent`` constructed with its default (unlimited) ``max_iterations`` and a
stub adapter that records the outbound text. Run on origin/main and on the fix branch:

    HERMES_HOME=$(mktemp -d) python evals/gateway_status_render/iteration_ceiling_ab.py [--finite 250]

Prints one JSON object per render site with the exact text a user would see.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock


def _agent(max_iterations: int | None):
    from run_agent import AIAgent

    kwargs: dict[str, Any] = dict(base_url="http://127.0.0.1:9/v1", api_key="sk-dummy", model="dummy-model",
                  quiet_mode=True, enabled_toolsets=[], disabled_toolsets=["*"])
    if max_iterations is not None:
        kwargs["max_iterations"] = max_iterations
    agent = AIAgent(**kwargs)
    agent._api_call_count = 3
    agent._current_tool = "terminal"
    agent._last_activity_desc = "terminal"
    agent._last_activity_ts = time.time() - 42
    return agent


async def _busy_ack(agent) -> str:
    import gateway.run as gr
    from gateway.platforms.base import SessionSource, build_session_key
    from gateway.platforms.event import MessageEvent, MessageType

    gr._load_gateway_config = lambda: {"display": {"platforms": {"telegram": {"busy_ack_detail": True}}}}
    runner = object.__new__(gr.GatewayRunner)
    runner._running_agents, runner._running_agents_ts = {}, {}
    runner._pending_messages, runner._busy_ack_ts, runner._queued_events = {}, {}, {}
    runner._draining, runner._busy_text_mode, runner._busy_input_mode = False, "interrupt", "interrupt"
    runner.adapters, runner.config, runner.session_store = {}, MagicMock(), None
    runner.config.group_sessions_per_user, runner.config.thread_sessions_per_user = True, False
    runner.hooks = MagicMock(); runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock(); runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _s: True
    source = SessionSource(platform=MagicMock(value="telegram"), chat_id="123", chat_type="private", user_id="u1")
    event = MessageEvent(text="status?", message_type=MessageType.TEXT, source=source, message_id="m1")
    sk = build_session_key(source)
    adapter = MagicMock()
    adapter._pending_messages, adapter._text_debounce, adapter._busy_text_debounce_seconds = {}, {}, 0.6
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock(); adapter.config.extra = {}
    adapter.platform = MagicMock(value="telegram")
    runner._running_agents[sk] = agent
    runner._running_agents_ts[sk] = time.time() - 600
    runner.adapters[source.platform] = adapter
    await runner._handle_active_session_busy_message(event, sk)
    return adapter._send_with_retry.call_args.kwargs.get("content", "")


async def _heartbeat(agent) -> str:
    from gateway.run_turn import GatewayTurnMixin
    from gateway.turn_context import TurnContext

    os.environ["HERMES_AGENT_NOTIFY_INTERVAL"] = "0.01"
    mixin = GatewayTurnMixin()
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="hb1"))
    mixin._adapter_for_source = MagicMock(return_value=adapter)
    mixin._should_emit_long_running_notification = MagicMock(side_effect=[True, False])
    disp = MagicMock()
    disp._display_surface_mode.return_value = "on"
    disp.resolve_display_setting.return_value = True
    ctx = TurnContext(source=SimpleNamespace(chat_id="c1", platform="telegram"), session_key="s1", agent_holder=[agent])
    await mixin._run_agent_notify_long_running(disp, ctx, [None])
    return adapter.send.await_args.args[1]


def _timeout(agent) -> str:
    import gateway.run as gr
    from gateway.run_turn import GatewayTurnMixin
    from gateway.turn_context import TurnContext

    gr.request_hard_interrupt = MagicMock()
    ctx = TurnContext(session_key="s1", agent_holder=[agent])
    return GatewayTurnMixin()._run_agent_timeout_result(SimpleNamespace(agent_timeout=1800.0), ctx)["final_response"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--finite", type=int, default=None, help="use a finite max_iterations instead of the default")
    args = ap.parse_args()
    sys.path.insert(0, os.getcwd())
    agent = _agent(args.finite)
    out = {
        "head": os.popen("git rev-parse --short HEAD").read().strip(),
        "max_iterations": agent.max_iterations,
        "busy_ack": asyncio.run(_busy_ack(agent)),
        "heartbeat": asyncio.run(_heartbeat(agent)),
        "timeout_diag": _timeout(agent),
    }
    out["sentinel_leaks"] = sum(str(sys.maxsize) in v for k, v in out.items() if k in ("busy_ack", "heartbeat", "timeout_diag"))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

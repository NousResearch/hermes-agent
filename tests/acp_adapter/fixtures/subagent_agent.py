"""Executable NO-MODEL FIXTURE: real ACP server and relay, scripted child activity."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import acp

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from tools.delegate_tool_progress import _ChildProgressRelay


class FixtureAgent:
    model = "no-model-fixture"
    tools = []
    system_prompt = "NO-MODEL FIXTURE"
    session_id = None

    def __init__(self):
        self.children = []

    def run_conversation(self, user_message, conversation_history, **kwargs):
        if not self.children:
            for child_id in ("alpha", "beta"):
                relay = _ChildProgressRelay(0, "private fixture goal", None, self.tool_progress_callback,
                                            2, child_id, None, 1, None, None, {})
                relay("subagent.start")
                relay("tool.started", "terminal", "private preview", {"secret": "never send"})
                relay("subagent.text", preview=f"[NO-MODEL FIXTURE] {child_id} public output")
                self.children.append(relay)
            nested = _ChildProgressRelay(0, "private nested goal", None, self.children[0],
                                         1, "nested", "alpha", 2, None, None, {})
            nested("subagent.start")
            nested("tool.started", "read_file", "private filename", {})
            nested("subagent.text", preview="[NO-MODEL FIXTURE] nested public output")
            self.children.append(nested)
        else:
            # These relays still hold the first turn's callback after the server rewires it.
            for relay, status in zip(self.children, ("completed", "failed", "interrupted")):
                relay("subagent.complete", status=status)
        reply = "[NO-MODEL FIXTURE] Root turn finished."
        return {"final_response": reply, "messages": conversation_history + [
            {"role": "user", "content": user_message}, {"role": "assistant", "content": reply},
        ]}

    def interrupt(self, *args, **kwargs):
        for relay in self.children:
            relay("subagent.complete", status="interrupted")


if __name__ == "__main__":
    asyncio.run(acp.run_agent(HermesACPAgent(SessionManager(agent_factory=FixtureAgent)), use_unstable_protocol=True))

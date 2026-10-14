"""A shared prompt whose ACP card got no accepted answer is shown again on reattach, never auto-answered."""
import asyncio
from types import SimpleNamespace

import pytest
from acp.schema import AllowedOutcome, DeniedOutcome

from acp_adapter.gateway_server import GatewayACPAgent
from hermes_cli.gateway_client import GatewayClientError

APPROVAL = {"kind": "approval", "prompt_id": "p-1", "execution_generation": 3, "command": "rm -r x",
            "description": "remove", "choices": ["once", "deny"]}
CLARIFY = {"kind": "clarify", "prompt_id": "q-1", "execution_generation": 3, "question": "Pick a color",
           "choices": ["blue", "green"], "multi_select": False}


class Editor:
    """Card-only editor: answers each card from a script (an outcome or a raised transport loss)."""

    def __init__(self, script):
        self.script, self.cards = list(script), []

    async def request_permission(self, session_id, tool_call, options):
        self.cards.append([option.option_id for option in options])
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return SimpleNamespace(outcome=step)


class Gateway:
    """The owner: still lists the prompt on every resume; the first respond RPC may be lost."""

    def __init__(self, prompt, lose_first_respond):
        self.prompt, self.lose, self.responses = prompt, lose_first_respond, []

    async def rpc(self, method, **params):
        if method == "session.resume":
            return {"session_id": params["session_id"], "prompts": [self.prompt], "pending": []}
        self.responses.append((method, params.get("choice", params.get("answer"))))
        if self.lose:
            self.lose = False
            raise GatewayClientError("Gateway disconnected")
        return {"status": "resolved"}


async def _settle(agent):
    for _ in range(5):
        await asyncio.gather(*agent._permissions.values(), return_exceptions=True)
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt, first, lose_first_respond, accept", [
    (APPROVAL, DeniedOutcome(outcome="cancelled"), False, ("allow_once", "approval.respond", "once")),
    (APPROVAL, ConnectionError("editor transport lost"), False, ("allow_once", "approval.respond", "once")),
    (APPROVAL, AllowedOutcome(outcome="selected", option_id="allow_once"), True, ("deny", "approval.respond", "deny")),
    (CLARIFY, ConnectionError("editor transport lost"), False, ("choice-1", "clarify.respond", "green")),
    (CLARIFY, AllowedOutcome(outcome="selected", option_id="choice-0"), True, ("choice-1", "clarify.respond", "green")),
], ids=["approval-editor-cancel", "approval-transport-loss", "approval-lost-respond",
        "clarify-transport-loss", "clarify-lost-respond"])
async def test_unaccepted_card_is_reoffered_on_reattach_without_replaying_its_answer(
        prompt, first, lose_first_respond, accept):
    option, method, value = accept
    editor = Editor([first, AllowedOutcome(outcome="selected", option_id=option)])
    gateway = Gateway(prompt, lose_first_respond)
    agent = GatewayACPAgent()
    agent._conn, agent._gateway, agent._snapshots["s"] = editor, gateway, {}
    agent._permission("s", prompt)  # the live request
    await _settle(agent)
    sent_first = list(gateway.responses)
    await agent._reattach(gateway)  # replacement connection / editor session/load: still pending
    await _settle(agent)
    assert len(editor.cards) == 2, "the unanswered prompt must be shown again"
    # The earlier answer is never auto-sent again; only the user's fresh choice reaches the owner.
    assert gateway.responses == sent_first + [(method, value)]
    await agent._reattach(gateway)  # accepted now: the owner's settle retires it, never a third card
    await _settle(agent)
    assert len(editor.cards) == 2

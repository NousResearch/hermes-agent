"""The CLI's approval answer must carry the panel's deny reason across the wire.

``GatewayAgent._on_approval`` hands the panel's choice to the gateway via
``approval.respond``. A deny-with-reason is a ``Denied`` choice (``==
"deny"``, ``.reason`` set): the str subclass flattens to plain ``"deny"`` in
JSON, so the reason must ride as its own field — otherwise the gateway relay
cannot forward what the user typed. A plain deny keeps the payload unchanged
(an old server's closed param set must keep accepting it).
"""

from hermes_cli.gateway_agent import GatewayAgent
from tools.approval_prompt import Denied


class _RecordingConnection:
    def __init__(self):
        self.calls = []

    def call(self, method, **params):
        self.calls.append((method, params))


def _answer(monkeypatch, choice, payload=None):
    connection = _RecordingConnection()
    agent = GatewayAgent.__new__(GatewayAgent)
    agent.connection = connection
    agent.session_id = "sess-1"
    agent._generation = 7
    monkeypatch.setattr("tools.terminal_tool._get_approval_callback",
                        lambda: (lambda command, description, **kwargs: choice))
    GatewayAgent._on_approval(agent, payload or {
        "command": "rm -rf /tmp/example", "description": "recursive delete",
        "choices": ["once", "deny"], "prompt_id": "p1", "execution_generation": 3,
    }, None, None)
    assert len(connection.calls) == 1
    return connection.calls[0]


def test_on_approval_sends_deny_reason_as_its_own_field(monkeypatch):
    """A Denied panel choice serializes its reason as a separate param."""
    method, params = _answer(monkeypatch, Denied("nope"))
    assert method == "approval.respond"
    assert params["choice"] == "deny"
    assert params["prompt_id"] == "p1"
    assert params["session_id"] == "sess-1"
    assert params["execution_generation"] == 3
    assert params["reason"] == "nope"


def test_on_approval_plain_deny_keeps_payload_unchanged(monkeypatch):
    """Without a reason the wire payload keeps today's exact shape."""
    method, params = _answer(monkeypatch, "deny")
    assert method == "approval.respond"
    assert params["choice"] == "deny"
    assert "reason" not in params

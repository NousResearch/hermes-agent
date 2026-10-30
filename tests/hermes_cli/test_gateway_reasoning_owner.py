"""A rotated gateway session must own the reasoning level its turns run at.

`/new`, `/resume` and `/branch` repoint ``agent.session_id``. The session just entered inherits the
profile's effort, so ``info["reasoning_effort"]`` already matches and the mutation that gives the
session an effort of its own used to be skipped. The gateway then resolved no effort for that
session's turns and the provider was never asked to reason: the live reasoning stream went silent
(no ``reasoning.delta``, only a blank ``reasoning_content`` placeholder).

The display flag travels the same way but on its own axis: it is per gateway session, so the sink
has to be re-assigned for the id the viewer just entered (``test_a_rotated_session_resends_the_
display_flag``).
"""

from types import SimpleNamespace

from hermes_cli.gateway_agent import GatewayAgent

INHERITED_INFO = {"reasoning_effort": "xhigh"}
CONFIG = {"enabled": True, "effort": "xhigh"}


class _Conn:
    """Records the RPCs the agent sends. A resume reply carries the fences a mutate needs."""

    def __init__(self):
        self.calls = []
        # ``_sync_show_reasoning`` memoises (session, show, epoch): a fresh epoch is a redial.
        self.epoch = 1

    def call(self, method, **params):
        self.calls.append((method, params))
        if method == "session.resume":
            return {"revision": 7, "execution_generation": 3}
        return {}

    def state(self, session_id):
        return SimpleNamespace(info={}, usage=None, usage_listeners=[], request_listeners=[], pending=[])

    def params(self, method):
        return [params for name, params in self.calls if name == method]


def _agent(conn, session_id, info):
    agent = GatewayAgent.__new__(GatewayAgent)
    agent.connection = conn
    agent._session_id = session_id
    agent._session_db = None
    agent._reasoning_config = None
    agent._show_reasoning_sent = None
    agent._reasoning_owner = None
    agent._state = lambda: SimpleNamespace(
        info=info, usage=None, usage_listeners=[], request_listeners=[], pending=[])
    agent._apply_info = lambda snapshot: None
    return agent


def test_a_new_session_commits_the_effort_it_inherits():
    conn = _Conn()
    agent = _agent(conn, "local-1", dict(INHERITED_INFO))
    assert agent._set_reasoning(CONFIG, explicit=False) is True
    mutates = conn.params("session.mutate")
    assert [call["session_id"] for call in mutates] == ["local-1"]
    assert mutates[0]["payload"] == {"effort": "xhigh"}


def test_a_rotated_session_commits_the_effort_again():
    conn = _Conn()
    agent = _agent(conn, "local-1", dict(INHERITED_INFO))
    agent._set_reasoning(CONFIG, explicit=False)
    agent._session_id = "local-2"  # /new repoints the id
    assert agent._set_reasoning(CONFIG, explicit=False) is True
    assert [call["session_id"] for call in conn.params("session.mutate")] == ["local-1", "local-2"]


def test_the_session_that_already_committed_is_not_asked_again():
    conn = _Conn()
    agent = _agent(conn, "local-1", dict(INHERITED_INFO))
    agent._set_reasoning(CONFIG, explicit=False)
    agent._set_reasoning(CONFIG, explicit=False)
    assert len(conn.params("session.mutate")) == 1


def test_an_explicit_level_is_always_committed():
    conn = _Conn()
    agent = _agent(conn, "local-1", dict(INHERITED_INFO))
    agent.pin_reasoning(CONFIG)
    agent.pin_reasoning(CONFIG)
    assert conn.params("session.mutate") and len(conn.params("session.mutate")) == 2


def test_a_rotated_session_resends_the_display_flag():
    conn = _Conn()
    agent = _agent(conn, "local-1", {})
    agent._reasoning_callback = lambda text: None
    agent._show_reasoning_sent = ("local-1", True)  # sent for the session it is leaving
    agent.session_id = "local-2"  # the setter /new repoints the id through
    sent = conn.params("config.set")
    assert [(call["session_id"], call["key"], call["value"]) for call in sent] == [
        ("local-2", "show_reasoning", "1")
    ]

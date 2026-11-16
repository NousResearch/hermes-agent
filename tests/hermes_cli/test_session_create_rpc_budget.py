"""``session.create`` builds the agent before answering, so it gets its own RPC budget.

The default 30 s is right for admissions and reads; it is not enough for the create verb when the
model-metadata probe waterfall runs, and the CLI then reports "Gateway did not answer session.create"
while the owner is still building the agent.
"""

from types import SimpleNamespace

from hermes_cli.gateway_client import DEFAULT_RPC_TIMEOUT, SESSION_CREATE_TIMEOUT
from hermes_cli.gateway_session_store import GatewaySessionStore


class _Connection:
    """Records every RPC budget; answers a create with a minimal snapshot."""

    def __init__(self):
        self.budgets = []

    def call(self, method, *, timeout=None, **params):
        self.budgets.append((method, timeout))
        if method == "session.create":
            return {"stored_session_id": "sid", "info": {}}
        return {"session_create": {"parameters": []}}

    def state(self, session_id):
        return SimpleNamespace(adopt_snapshot=lambda snapshot: None)


def test_session_create_passes_its_own_budget():
    connection = _Connection()
    store = GatewaySessionStore(connection)

    assert store.create({}) == "sid"
    budgets = dict(connection.budgets)
    assert budgets["session.create"] == SESSION_CREATE_TIMEOUT
    assert SESSION_CREATE_TIMEOUT > DEFAULT_RPC_TIMEOUT


def test_other_verbs_keep_the_default_budget():
    connection = _Connection()
    store = GatewaySessionStore(connection)

    store.client_env()

    budgets = dict(connection.budgets)
    assert budgets["runtime.describe"] is None  # None = DEFAULT_RPC_TIMEOUT

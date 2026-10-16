"""``gateway.run.AGENT_PENDING_SENTINEL`` — the placeholder for a starting turn.

The gateway puts this object into its running-agents map before the agent
exists. A plugin that reports on running sessions (a queue-status command, a
dashboard) must skip it, and can only do so by identity. The private spelling
stays as an alias bound to the same object; the gateway's own checks read the
public name.
"""

from gateway import run


def test_public_name_is_the_private_sentinel():
    assert run.AGENT_PENDING_SENTINEL is run._AGENT_PENDING_SENTINEL


def test_gateway_checks_read_the_public_name(monkeypatch):
    from types import SimpleNamespace

    from gateway.run_watchers import GatewaySessionWatchersMixin

    asked = []

    class Placeholder:
        def get_activity_summary(self):
            asked.append(True)
            return {}

    placeholder = Placeholder()
    monkeypatch.setattr(run, "AGENT_PENDING_SENTINEL", placeholder)
    owner = SimpleNamespace(_running_agents={"s": placeholder})
    assert GatewaySessionWatchersMixin._session_activity_for_stall(owner, "s") is None
    assert asked == []

"""``gateway.run.AGENT_PENDING_SENTINEL`` — the placeholder for a starting turn.

The gateway puts this object into its running-agents map before the agent
exists. A plugin that reports on running sessions (a queue-status command, a
dashboard) must skip it, and can only do so by identity. The public name is the
SAME object as the private spelling, so the gateway is unchanged.
"""

from gateway import run


def test_public_name_is_the_private_sentinel():
    assert run.AGENT_PENDING_SENTINEL is run._AGENT_PENDING_SENTINEL

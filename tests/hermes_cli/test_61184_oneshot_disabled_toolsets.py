"""
Regression test for issue #61184 - agent.disabled_toolsets is silently
ignored in the oneshot (-z) path.

The bug: hermes_cli/oneshot.py's _run_agent builds an AIAgent with
only enabled_toolsets. The disabled-toolset subtraction step in
get_tool_definitions() never runs in the oneshot path, so disabled
MCP servers' tools remain registered. When the intended MCP server
is unreachable, the agent silently falls back to the disabled one
without surfacing the substitution.

The fix: read agent.disabled_toolsets from config in _run_agent and
forward it to AIAgent's disabled_toolsets parameter.

Static-source tripwire: the oneshot path must read
agent.disabled_toolsets and pass it to AIAgent.
"""

import re
from pathlib import Path


def test_static_oneshot_forwards_disabled_toolsets():
    """Static-source check: the oneshot _run_agent function must read
    agent.disabled_toolsets and forward it to AIAgent.

    Fails on unfixed code, passes on fixed.
    """
    worktree = Path("/tmp/hermes-pr-work-60859/hermes-agent")
    oneshot_py = (worktree / "hermes_cli" / "oneshot.py").read_text()

    # Find the _run_agent function
    m = re.search(r"def _run_agent\(.*?(?=^def |\Z)", oneshot_py, re.MULTILINE | re.DOTALL)
    assert m, "_run_agent function not found"
    body = m.group(0)

    # The function must read agent.disabled_toolsets from config
    assert "disabled_toolsets" in body, (
        "#61184 regression: _run_agent does not read "
        "agent.disabled_toolsets from config. The oneshot path "
        "constructs AIAgent without forwarding the disabled-toolsets "
        "list, so disabled MCP servers' tools remain registered."
    )

    # And it must pass it to AIAgent
    # The disabled_toolsets= kwarg must be passed to the AIAgent call
    assert re.search(r"disabled_toolsets\s*=\s*_disabled_toolsets", body), (
        "#61184: _run_agent reads disabled_toolsets but does not pass "
        "it to AIAgent. The fix should call AIAgent with "
        "disabled_toolsets=_disabled_toolsets."
    )

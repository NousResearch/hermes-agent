"""A per-job MCP allowlist stays narrow whichever spelling names the server (#141468).

``enabled_toolsets`` accepts a server as its bare name (``finnhub``) or its toolset name
(``mcp-finnhub``). Either spelling is a selection: the job gets that server and no other.
"""

from cron.scheduler import _merge_mcp_into_per_job_toolsets

CFG = {
    "mcp_servers": {
        "finnhub": {"enabled": True},
        "playwright": {"enabled": True},
        "disabled_one": {"enabled": False},
    }
}
ENABLED = {"finnhub", "playwright"}


def test_the_toolset_name_spelling_is_treated_as_allowlist():
    assert _merge_mcp_into_per_job_toolsets(["web", "mcp-finnhub"], CFG) == ["web", "mcp-finnhub"]
    assert _merge_mcp_into_per_job_toolsets(["mcp-finnhub", "terminal"], CFG) == ["mcp-finnhub", "terminal"]


def test_a_configured_but_disabled_server_does_not_widen_the_allowlist():
    # A job pinned to a server the operator turned off must not fall through to every ENABLED
    # server: the disabled one stays off and nothing else rides in, in either spelling.
    for pick in ("mcp-disabled_one", "disabled_one"):
        result = _merge_mcp_into_per_job_toolsets(["web", pick], CFG)
        assert result == ["web", pick]
        assert not set(result) & ENABLED

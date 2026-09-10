"""Create/update clamp for H3: a persisted job never stores more toolsets than origin."""

from cron.scheduler import clamp_cron_enabled_toolsets_to_origin, _resolve_cron_enabled_toolsets


CFG = {
    "platform_toolsets": {
        "whatsapp_cloud": ["web", "file", "memory"],
        "telegram": ["web", "file", "memory"],
    }
}


def test_clamp_strips_terminal_when_origin_lacks_it():
    out = clamp_cron_enabled_toolsets_to_origin(
        ["terminal", "file", "web"],
        {"platform": "whatsapp_cloud"},
        CFG,
    )
    assert out is not None
    assert "terminal" not in out
    assert "file" in out and "web" in out


def test_clamp_leaves_cli_list_when_origin_missing():
    out = clamp_cron_enabled_toolsets_to_origin(
        ["web", "terminal"],
        None,
        CFG,
    )
    assert out == ["web", "terminal"]


def test_resolver_no_origin_keeps_per_job_list():
    """CLI jobs (retro_chase) keep an explicit terminal pin."""
    result = _resolve_cron_enabled_toolsets(
        {"enabled_toolsets": ["terminal", "file"]},
        {"mcp_servers": {}},
    )
    assert "terminal" in result
    assert "file" in result

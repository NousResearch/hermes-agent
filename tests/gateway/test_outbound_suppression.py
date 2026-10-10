"""Operator-configured outbound suppression (``suppress_outbound``) across gateway send rails."""

import pytest

from gateway import outbound_suppression
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import (
    _prepare_gateway_status_message,
    _sanitize_gateway_final_response,
)


def _clear_suppress_caches():
    outbound_suppression._CONFIG_CACHE.clear()
    outbound_suppression._COMPILED.clear()


@pytest.fixture
def suppress_config(monkeypatch, tmp_path):
    """Write a real config.yaml under a temp HERMES_HOME for suppress_outbound.

    Returns a setter: call it with (global_patterns, per_platform) to write
    the YAML and point config resolution at it. Every resolution then runs
    the real ``load_gateway_config()`` YAML bridges (top-level
    ``suppress_outbound`` plus ``platforms.<name>.suppress_outbound``) — no
    stubbing of the loader. Clears the config and compiled-pattern caches
    around each write so tests cannot leak into each other.
    """
    import json

    import gateway.run as run

    home = tmp_path / "hermes-home"
    home.mkdir()
    # load_gateway_config() resolves via get_hermes_home() (env), while
    # gateway.run's path helpers use the module-level _hermes_home snapshot —
    # point both at the temp home so the cache key and the loaded file agree.
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(run, "_hermes_home", home)

    def _set(global_patterns=None, per_platform=None):
        cfg_doc = {}
        if global_patterns is not None:
            cfg_doc["suppress_outbound"] = list(global_patterns)
        if per_platform:
            cfg_doc["platforms"] = {
                platform.value: {"suppress_outbound": list(patterns)}
                for platform, patterns in per_platform.items()
            }
        # JSON is valid YAML; avoids depending on a particular YAML writer in tests.
        (home / "config.yaml").write_text(json.dumps(cfg_doc), encoding="utf-8")
        # Two writes can land within one mtime tick, so drop the resolved
        # config rather than trusting the stamp within a single test.
        _clear_suppress_caches()
        return home

    _clear_suppress_caches()
    yield _set
    _clear_suppress_caches()


def test_suppress_outbound_drops_matching_final_response(suppress_config):
    """A configured pattern drops the final reply on a chat surface."""
    suppress_config(global_patterns=[r"^Liked it\.$"])

    assert _sanitize_gateway_final_response(Platform.TELEGRAM, "Liked it.") == ""
    # re.search semantics: unanchored patterns match anywhere.
    suppress_config(global_patterns=[r"Interrupting current task"])
    assert (
        _sanitize_gateway_final_response(
            Platform.TELEGRAM, "Interrupting current task to handle your message..."
        )
        == ""
    )


def test_suppress_outbound_drops_matching_status_message(suppress_config):
    """The same patterns cover the status/notice send path."""
    suppress_config(global_patterns=[r"Interrupting current task"])

    assert (
        _prepare_gateway_status_message(
            Platform.TELEGRAM, "lifecycle", "Interrupting current task..."
        )
        is None
    )


def test_suppress_outbound_exempts_raw_platforms(suppress_config):
    """Programmatic surfaces must never be muted by operator patterns."""
    suppress_config(global_patterns=[r".*"])  # suppress everything

    text = "Liked it."
    for platform in ("local", "api_server", "webhook", "msgraph_webhook"):
        assert _sanitize_gateway_final_response(platform, text) == text
        assert _prepare_gateway_status_message(platform, "warn", text) == text


def test_suppress_outbound_per_platform_extends_global(suppress_config):
    """platforms.<name>.suppress_outbound extends (not replaces) the global list."""
    suppress_config(
        global_patterns=[r"^Liked it\.$"],
        per_platform={Platform.TELEGRAM: [r"^Gateway restarted"]},
    )

    # Telegram gets global + its own pattern.
    assert _sanitize_gateway_final_response(Platform.TELEGRAM, "Liked it.") == ""
    assert (
        _sanitize_gateway_final_response(Platform.TELEGRAM, "Gateway restarted (v2)")
        == ""
    )
    # Other chat platforms only get the global pattern.
    assert _sanitize_gateway_final_response(Platform.DISCORD, "Liked it.") == ""
    assert (
        _sanitize_gateway_final_response(Platform.DISCORD, "Gateway restarted (v2)")
        == "Gateway restarted (v2)"
    )


def test_suppress_outbound_invalid_regex_warns_and_skips(suppress_config, caplog):
    """An invalid regex warns once, is skipped, and never crashes or over-drops."""
    import logging

    suppress_config(global_patterns=[r"[unclosed", r"^Liked it\.$"])

    with caplog.at_level(logging.WARNING):
        # Valid pattern still enforced despite the broken sibling.
        assert _sanitize_gateway_final_response(Platform.TELEGRAM, "Liked it.") == ""
        # Non-matching text passes through untouched.
        answer = "Here is the clean summary you asked for."
        assert _sanitize_gateway_final_response(Platform.TELEGRAM, answer) == answer

    assert any(
        "invalid suppress_outbound pattern" in record.getMessage()
        and "[unclosed" in record.getMessage()
        for record in caplog.records
    )


def test_suppress_outbound_empty_config_is_passthrough(suppress_config):
    """No configured patterns = zero behavior change."""
    suppress_config(global_patterns=[])

    answer = "Liked it."
    assert _sanitize_gateway_final_response(Platform.TELEGRAM, answer) == answer
    assert (
        _prepare_gateway_status_message(Platform.TELEGRAM, "info", answer) == answer
    )


def test_suppress_outbound_non_matching_text_untouched(suppress_config):
    """Patterns only drop matches; everything else flows through unchanged."""
    suppress_config(global_patterns=[r"^Liked it\.$"])

    answer = "I liked it. Here is the longer review you asked for."
    assert _sanitize_gateway_final_response(Platform.TELEGRAM, answer) == answer


def test_suppress_outbound_case_sensitive_as_written(suppress_config):
    """Patterns compile as written; operators opt into (?i) themselves."""
    suppress_config(global_patterns=[r"^liked it\.$"])
    assert _sanitize_gateway_final_response(Platform.TELEGRAM, "Liked it.") == "Liked it."

    suppress_config(global_patterns=[r"(?i)^liked it\.$"])
    assert _sanitize_gateway_final_response(Platform.TELEGRAM, "Liked it.") == ""


def test_get_suppress_outbound_resolution_order():
    """GatewayConfig.get_suppress_outbound: global first, then platform, deduped."""
    cfg = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                extra={"suppress_outbound": [r"^B$", r"^A$"]},
            )
        },
        suppress_outbound=[r"^A$"],
    )

    assert cfg.get_suppress_outbound(Platform.TELEGRAM) == [r"^A$", r"^B$"]
    assert cfg.get_suppress_outbound(Platform.DISCORD) == [r"^A$"]
    assert cfg.get_suppress_outbound(None) == [r"^A$"]


def test_suppress_outbound_loaded_from_real_config_yaml(suppress_config):
    """load_gateway_config() bridges suppress_outbound from a real config.yaml.

    Exercises the actual YAML loader (temp HERMES_HOME) for both the global
    key and the platforms.<name>.suppress_outbound per-platform extension.
    """
    from gateway.config import load_gateway_config

    suppress_config(
        global_patterns=[r"^Liked it\.$"],
        per_platform={Platform.TELEGRAM: [r"^Gateway restarted"]},
    )

    cfg = load_gateway_config()
    assert cfg.suppress_outbound == [r"^Liked it\.$"]
    telegram_cfg = cfg.platforms.get(Platform.TELEGRAM)
    assert telegram_cfg is not None
    assert telegram_cfg.extra.get("suppress_outbound") == [r"^Gateway restarted"]
    assert cfg.get_suppress_outbound(Platform.TELEGRAM) == [
        r"^Liked it\.$",
        r"^Gateway restarted",
    ]
    assert cfg.get_suppress_outbound(Platform.DISCORD) == [r"^Liked it\.$"]


def test_suppress_outbound_routed_profiles_do_not_share_cache(tmp_path):
    """Context-local profile homes must never reuse each other's rules.

    Regression for the mtime-only cache key: two profile config files with
    identical mtimes are distinct cache entries because the resolved config
    path is part of the identity. The cache is deliberately NOT cleared
    between the profile switches below — that reuse is what's under test.
    """
    import os

    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()
    (home_a / "config.yaml").write_text(
        "suppress_outbound:\n  - '^From profile A$'\n", encoding="utf-8"
    )
    (home_b / "config.yaml").write_text(
        "suppress_outbound:\n  - '^From profile B$'\n", encoding="utf-8"
    )
    # Force identical mtimes so an mtime-only cache key would alias them.
    stat_a = (home_a / "config.yaml").stat()
    os.utime(home_b / "config.yaml", ns=(stat_a.st_atime_ns, stat_a.st_mtime_ns))

    _clear_suppress_caches()
    try:
        token = set_hermes_home_override(home_a)
        try:
            assert (
                _sanitize_gateway_final_response(Platform.TELEGRAM, "From profile A")
                == ""
            )
            assert (
                _sanitize_gateway_final_response(Platform.TELEGRAM, "From profile B")
                == "From profile B"
            )
        finally:
            reset_hermes_home_override(token)

        token = set_hermes_home_override(home_b)
        try:
            # With the old mtime-only key this resolved profile A's rules.
            assert (
                _sanitize_gateway_final_response(Platform.TELEGRAM, "From profile B")
                == ""
            )
            assert (
                _sanitize_gateway_final_response(Platform.TELEGRAM, "From profile A")
                == "From profile A"
            )
        finally:
            reset_hermes_home_override(token)
    finally:
        _clear_suppress_caches()


@pytest.mark.asyncio
async def test_suppress_outbound_covers_active_session_shutdown_notice(suppress_config):
    """The direct shutdown-notification send honors suppress_outbound."""
    from unittest.mock import MagicMock

    from gateway.session import build_session_key
    from tests.gateway.restart_test_helpers import (
        make_restart_runner,
        make_restart_source,
    )

    suppress_config(global_patterns=[r"is (restarting|shutting down)"])

    runner, adapter = make_restart_runner()
    source = make_restart_source()
    session_key = build_session_key(source)
    runner._running_agents = {session_key: MagicMock()}
    runner._cache_session_source(session_key, source)

    await runner._notify_active_sessions_of_shutdown()
    assert adapter.sent_calls == []

    # Control: with no matching pattern the same rail delivers the notice.
    suppress_config(global_patterns=[])
    await runner._notify_active_sessions_of_shutdown()
    assert len(adapter.sent_calls) == 1
    assert "shutting down" in adapter.sent_calls[0][1]


@pytest.mark.asyncio
async def test_suppress_outbound_covers_home_channel_shutdown_broadcast(suppress_config):
    """The home-channel shutdown broadcast honors suppress_outbound too."""
    from gateway.config import HomeChannel
    from tests.gateway.restart_test_helpers import make_restart_runner

    suppress_config(global_patterns=[r"is (restarting|shutting down)"])

    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-chat",
        name="Telegram Home",
    )

    await runner._notify_active_sessions_of_shutdown()
    assert adapter.sent_calls == []

    suppress_config(global_patterns=[])
    await runner._notify_active_sessions_of_shutdown()
    assert len(adapter.sent_calls) == 1
    assert adapter.sent_calls[0][0] == "home-chat"
    assert "shutting down" in adapter.sent_calls[0][1]


@pytest.mark.asyncio
async def test_suppress_outbound_covers_platform_notice_rail(suppress_config):
    """Setup/operational notices (``_deliver_platform_notice``) honor suppress_outbound."""
    from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source

    suppress_config(global_patterns=[r"^Setup nag"])
    runner, adapter = make_restart_runner()
    runner._delivery_adapter_for = lambda source: adapter
    source = make_restart_source()

    await runner._deliver_platform_notice(source, "Setup nag: configure a home channel")
    assert adapter.sent_calls == []

    await runner._deliver_platform_notice(source, "Something the user needs to know")
    assert [c[1] for c in adapter.sent_calls] == ["Something the user needs to know"]

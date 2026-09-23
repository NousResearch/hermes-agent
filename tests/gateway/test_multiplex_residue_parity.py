"""Multiplex parity: a served profile reads ITS ``sessions.*`` / proxy env, never the launch
profile's; a stale served route never re-scaffolds an archived profile; MCP discovery runs once
per served profile home."""

import logging
import threading

import pytest

from agent import secret_scope as ss
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    alpha = root / "profiles" / "alpha"
    for home in (root, alpha):
        (home / "sessions").mkdir(parents=True)
        (home / ".env").write_text("", encoding="utf-8")
    (root / "config.yaml").write_text(
        "sessions:\n  cjk_fts: true\n  search_slow_ms: 1000\n", encoding="utf-8")
    (alpha / "config.yaml").write_text(
        "sessions:\n  cjk_fts: false\n  search_slow_ms: 50\n", encoding="utf-8")
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    ss.set_multiplex_active(False)
    yield root, alpha
    ss.set_multiplex_active(False)


def test_served_profile_reads_its_own_sessions_settings(two_homes, monkeypatch):
    root, alpha = two_homes
    from hermes_state_fts import _cjk_fts_config_enabled
    from hermes_state_search import _search_slow_ms
    # The multiplexer bridged the LAUNCH (default) profile's sessions.* into env at import.
    monkeypatch.setenv("HERMES_CJK_FTS", "true")
    monkeypatch.setenv("HERMES_SEARCH_SLOW_MS", "1000")
    assert _cjk_fts_config_enabled() is True and _search_slow_ms() == 1000.0  # unscoped = env bridge
    token = set_hermes_home_override(str(alpha))
    try:
        assert _cjk_fts_config_enabled() is False
        assert _search_slow_ms() == 50.0
    finally:
        reset_hermes_home_override(token)


def test_per_turn_sessions_bridge_skips_secondary_scope(two_homes, monkeypatch):
    root, alpha = two_homes
    from gateway import run as gw_run
    monkeypatch.delenv("HERMES_CJK_FTS", raising=False)
    ss.set_multiplex_active(True)
    with gw_run._profile_runtime_scope(alpha):
        gw_run._bridge_max_turns_from_config(alpha)
    assert "HERMES_CJK_FTS" not in __import__("os").environ, "secondary scope must not write the process env"


def test_resolve_proxy_url_reads_routed_profile_scope(two_homes, monkeypatch):
    root, alpha = two_homes
    from gateway.platforms.base import resolve_proxy_url
    monkeypatch.setenv("TELEGRAM_PROXY", "socks5://default-proxy:1080")  # launch profile's .env
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope({"TELEGRAM_PROXY": "socks5://alpha-proxy:1080"})
    try:
        assert resolve_proxy_url("TELEGRAM_PROXY") == "socks5://alpha-proxy:1080"
    finally:
        ss.reset_secret_scope(token)
    token = ss.set_secret_scope({})  # a served profile with NO proxy must not borrow the default's
    try:
        monkeypatch.delenv("HTTPS_PROXY", raising=False)
        assert resolve_proxy_url("TELEGRAM_PROXY") is None
    finally:
        ss.reset_secret_scope(token)
    ss.set_multiplex_active(False)
    assert resolve_proxy_url("TELEGRAM_PROXY") == "socks5://default-proxy:1080"  # standalone: env


def test_stale_served_turn_never_recreates_archived_profile(two_homes):
    """#94590: a multiplexer still holding an archived profile's route must not re-scaffold it."""
    import shutil
    root, alpha = two_homes
    from gateway import run as gw_run
    shutil.rmtree(alpha)
    ss.set_multiplex_active(True)
    with gw_run._profile_runtime_scope(alpha):
        from hermes_state import SessionDB
        with pytest.raises(FileNotFoundError):
            SessionDB()
    assert not alpha.exists()


def test_mcp_discovery_slot_is_per_profile_home(two_homes, monkeypatch):
    """#67605: alpha building an agent after default must still get ITS discovery run."""
    root, alpha = two_homes
    import hermes_cli.mcp_startup as ms
    from hermes_constants import get_hermes_home
    for home in (root, alpha):
        (home / "config.yaml").write_text("mcp_servers:\n  demo:\n    command: /bin/true\n", encoding="utf-8")
    monkeypatch.setattr(ms, "_mcp_discovery_started", set())
    monkeypatch.setattr(ms, "_mcp_discovery_thread", {})
    seen, done = [], threading.Event()

    def fake_discover(**_kw):
        seen.append(get_hermes_home().name)
        done.set()

    monkeypatch.setattr("tools.mcp_tool_discovery.discover_mcp_tools", fake_discover)
    monkeypatch.setattr("tools.mcp_tool_discovery.get_mcp_status", lambda *a, **k: [{"connected": True}])
    for home in (root, alpha):
        token = set_hermes_home_override(str(home))
        try:
            done.clear()
            ms.start_background_mcp_discovery(logger=logging.getLogger("t"), thread_name=f"mcp-{home.name}")
            assert done.wait(5)
            ms.wait_for_mcp_discovery(timeout=5)
        finally:
            reset_hermes_home_override(token)
    assert seen == [root.name, "alpha"]

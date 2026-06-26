import pytest
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from gateway.config import Platform, GatewayConfig
from hermes_constants import get_hermes_home

@pytest.fixture
def clean_hermes_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    
    # Create profiles directory and files
    profiles_dir = hermes_home / "profiles"
    profiles_dir.mkdir()
    
    for name in ("profilea", "profileb"):
        pdir = profiles_dir / name
        pdir.mkdir()
        (pdir / "home").mkdir()
        (pdir / "SOUL.md").write_text(f"I am {name}", encoding="utf-8")
        (pdir / "memories").mkdir()
        (pdir / "sessions").mkdir()
        (pdir / "config.yaml").write_text("agent:\n  system_prompt: overridden", encoding="utf-8")
        
        # Write identity marker to satisfy profile safety check
        from hermes_cli.profiles import write_profile_identity_marker
        write_profile_identity_marker(name, pdir, profiles_dir, overwrite=True)

    # Set up global SOUL.md
    (hermes_home / "SOUL.md").write_text("I am main agent", encoding="utf-8")
    
    # Write topic profiles config
    import json
    topic_profiles = {
        "telegram:dm:111:222": "profilea",
        "telegram:dm:111:333": "profileb",
    }
    with open(hermes_home / "topic_profiles.json", "w", encoding="utf-8") as f:
        json.dump(topic_profiles, f)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    with patch("gateway.run._hermes_home", hermes_home):
        yield hermes_home

def test_session_key_routing_unconditional(clean_hermes_home):
    """Verify session key incorporates profile even when multiplex_profiles is False."""
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={})
    runner.config.multiplex_profiles = False
    
    # We must mock _normalize_source_for_session_key
    runner._normalize_source_for_session_key = lambda src: src

    source_a = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="111",
        chat_type="dm",
        thread_id="222",
    )
    source_b = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="111",
        chat_type="dm",
        thread_id="333",
    )
    source_main = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="111",
        chat_type="dm",
        thread_id="444",
    )

    key_a = runner._session_key_for_source(source_a)
    key_b = runner._session_key_for_source(source_b)
    key_main = runner._session_key_for_source(source_main)

    assert source_a.profile == "profilea"
    assert source_b.profile == "profileb"
    assert source_main.profile is None

    assert "profilea" in key_a
    assert "profileb" in key_b
    assert "profilea" not in key_main and "profileb" not in key_main

def test_dynamic_session_db_and_store_scoping(clean_hermes_home):
    """Verify that session_store and _session_db are dynamically resolved per-profile."""
    runner = GatewayRunner(config=GatewayConfig(platforms={}))
    runner._normalize_source_for_session_key = lambda src: src
    
    source_a = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="111",
        chat_type="dm",
        thread_id="222",
    )
    
    # Initially global db path
    global_db_path = clean_hermes_home / "state.db"
    assert Path(runner._session_db.db_path).resolve() == global_db_path.resolve()
    
    # Simulate a routed run using _routed_profile_for_source helper
    routed = runner._routed_profile_for_source(source_a)
    assert routed == "profilea"
    
    profile_home = clean_hermes_home / "profiles" / "profilea"
    from gateway.run import _profile_runtime_scope
    with _profile_runtime_scope(profile_home):
        # Inside the scope, get_hermes_home() points to profilea
        assert get_hermes_home().resolve() == profile_home.resolve()
        # session_store and _session_db should resolve to profile-specific DB paths
        profile_db_path = profile_home / "state.db"
        assert Path(runner._session_db.db_path).resolve() == profile_db_path.resolve()
        assert Path(runner.session_store.sessions_dir).resolve() == (profile_home / "sessions").resolve()

    # Outside the scope, back to global
    assert Path(runner._session_db.db_path).resolve() == global_db_path.resolve()

def test_soul_cache_busting(clean_hermes_home):
    """Verify that updating SOUL.md busts the agent cache signature."""
    runner = GatewayRunner(config=GatewayConfig(platforms={}))
    
    # Check signature initially
    sig1 = runner._agent_config_signature(
        model="gpt-4",
        runtime={},
        enabled_toolsets=[],
        ephemeral_prompt="",
    )
    
    # Modify SOUL.md
    soul_file = clean_hermes_home / "SOUL.md"
    soul_file.write_text("updated identity content", encoding="utf-8")
    
    # Force different mtime/stat
    import os
    stat = soul_file.stat()
    os.utime(soul_file, (stat.st_atime, stat.st_mtime + 5.0))
    
    sig2 = runner._agent_config_signature(
        model="gpt-4",
        runtime={},
        enabled_toolsets=[],
        ephemeral_prompt="",
    )
    
    assert sig1 != sig2

def test_tilde_expansion_isolated(clean_hermes_home):
    """Verify that tilde (~) expansion resolves to profile-specific home inside routing scope."""
    from gateway.run import _profile_runtime_scope
    from tools.file_tools import _expand_tilde
    import concurrent.futures
    from contextvars import copy_context

    profile_home = clean_hermes_home / "profiles" / "profilea"
    
    # Setup the config/home mode to enable profile home mode
    monkeypatch_env = os.environ.copy()
    monkeypatch_env["TERMINAL_HOME_MODE"] = "profile"
    
    with patch.dict(os.environ, monkeypatch_env):
        with _profile_runtime_scope(profile_home):
            # Directly inside scope:
            assert _expand_tilde("~/SOUL.md") == str(profile_home / "home" / "SOUL.md")
            
            # Inside ThreadPoolExecutor worker thread (simulates tool execution thread):
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            ctx = copy_context()
            res = pool.submit(ctx.run, lambda: _expand_tilde("~/SOUL.md")).result()
            assert res == str(profile_home / "home" / "SOUL.md")
            pool.shutdown()

@pytest.mark.asyncio
async def test_telegram_topic_new_command_isolated(clean_hermes_home):
    """Verify that /new executed in a Telegram topic lane is correctly routed to the profile scope and DB."""
    from gateway.platforms.base import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from gateway.config import Platform, GatewayConfig
    
    runner = GatewayRunner(config=GatewayConfig(platforms={}))
    runner._normalize_source_for_session_key = lambda src: src
    
    # We must mock adapters lookup so send does not crash or block
    mock_adapter = MagicMock()
    runner.adapters = {Platform.TELEGRAM: mock_adapter}
    
    profile_home = clean_hermes_home / "profiles" / "profilea"
    profile_db_path = profile_home / "state.db"
    from hermes_state import SessionDB
    db = SessionDB(db_path=profile_db_path)
    db.enable_telegram_topic_mode(chat_id="111", user_id="user_999")

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="111",
        chat_type="dm",
        thread_id="222",  # profilea is mapped to this
        user_id="user_999",
    )
    
    event = MessageEvent(
        text="/new",
        message_id="999",
        source=source,
    )
    
    # Bypass confirmation dialog so execute runs immediately
    async def mock_confirm(*args, **kwargs):
        return await kwargs["execute"]()
    runner._maybe_confirm_destructive_slash = mock_confirm

    # Let's run _handle_message.
    await runner._handle_message(event)

    # Check profilea's database for the topic binding!
    profile_home = clean_hermes_home / "profiles" / "profilea"
    profile_db_path = profile_home / "state.db"
    assert profile_db_path.exists()
    
    from hermes_state import SessionDB
    db = SessionDB(db_path=profile_db_path)
    binding = db.get_telegram_topic_binding(chat_id="111", thread_id="222")
    assert binding is not None
    assert binding["chat_id"] == "111"
    assert binding["thread_id"] == "222"

    # Verify that the GLOBAL database does NOT have this binding, proving isolation!
    global_db_path = clean_hermes_home / "state.db"
    if global_db_path.exists():
        global_db = SessionDB(db_path=global_db_path)
        global_binding = global_db.get_telegram_topic_binding(chat_id="111", thread_id="222")
        assert global_binding is None

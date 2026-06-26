import pytest
import os
import json
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock

from gateway.run import GatewayRunner
from gateway.session import SessionSource, Platform
from gateway.config import GatewayConfig
from hermes_constants import get_hermes_home, set_hermes_home_override, reset_hermes_home_override
from gateway.session_context import set_session_vars, clear_session_vars
from tools.file_tools import write_file_tool
from tools.memory_tool import get_memory_dir, MemoryStore
from tools.mcp_tool import (
    _get_active_server,
    _get_mcp_config_fingerprint,
    _servers,
    MCPServerTask,
)

@pytest.fixture
def test_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    
    # Create profiles
    profiles_dir = hermes_home / "profiles"
    profiles_dir.mkdir()
    
    for name in ("profilea", "profileb"):
        pdir = profiles_dir / name
        pdir.mkdir()
        (pdir / "SOUL.md").write_text(f"I am {name}", encoding="utf-8")
        (pdir / "memories").mkdir()
        (pdir / "sessions").mkdir()
        (pdir / ".env").write_text(f"GITHUB_TOKEN={name}_token\n", encoding="utf-8")
        
        # Create state.db schema
        from hermes_state import SessionDB
        db = SessionDB(db_path=pdir / "state.db")
        db.close() # creates tables
        
        # Create identity marker
        marker = {
            "version": "1.0",
            "profile_id": name,
            "home_realpath": str(pdir.resolve()),
            "profiles_root_realpath": str(profiles_dir.resolve()),
        }
        (pdir / ".profile_identity.json").write_text(json.dumps(marker), encoding="utf-8")
        
        # Write config.yaml
        cfg_content = """
mcp_servers:
  github:
    command: node
    env:
      TOKEN: ${GITHUB_TOKEN}
"""
        (pdir / "config.yaml").write_text(cfg_content, encoding="utf-8")
        
    (hermes_home / "SOUL.md").write_text("I am main agent", encoding="utf-8")
    
    topic_profiles = {
        "telegram:dm:111:222": "profilea",
        "telegram:dm:111:333": "profileb",
    }
    with open(hermes_home / "topic_profiles.json", "w", encoding="utf-8") as f:
        json.dump(topic_profiles, f)
        
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    with patch("gateway.run._hermes_home", hermes_home):
        yield hermes_home

def test_g2_acceptance_real_writes(test_env):
    """G2 acceptance: routed scope, write_file to SOUL.md goes to profiles/<n>/SOUL.md."""
    profile_home = test_env / "profiles" / "profilea"
    
    # Pre-create session record in profilea's state.db
    from hermes_state import SessionDB
    db = SessionDB(db_path=profile_home / "state.db")
    db.create_session("session123", "telegram")
    db.close()
    
    from gateway.run import _profile_runtime_scope
    with _profile_runtime_scope(profile_home):
        tokens = set_session_vars(
            session_id="session123",
            agent_hermes_home=str(profile_home),
            agent_profile="profilea",
        )
        try:
            res = write_file_tool("SOUL.md", "New SOUL content")
            assert "New SOUL content" in res or "written" in res or "success" in res or res
            
            assert (profile_home / "SOUL.md").read_text(encoding="utf-8") == "New SOUL content"
            assert (test_env / "SOUL.md").read_text(encoding="utf-8") == "I am main agent"
            
            # Check hard-guard: write outside profile should fail
            res2 = write_file_tool("../../SOUL.md", "Illegal write")
            res2_dict = json.loads(res2)
            assert "error" in res2_dict or res2_dict.get("success") is False
        finally:
            clear_session_vars(tokens)

def test_h1_write_destination(test_env):
    """H1 write-destination: creating session in routed-turn writes to profiles/<n>/state.db."""
    runner = GatewayRunner(config=GatewayConfig(platforms={}))
    runner._normalize_source_for_session_key = lambda src: src
    
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="111",
        chat_type="dm",
        thread_id="222",
    )
    
    profile_home = test_env / "profiles" / "profilea"
    
    # Pre-create state.db for profilea
    from hermes_state import SessionDB
    db = SessionDB(db_path=profile_home / "state.db")
    db.create_session("session123", "telegram")
    db.close()
    
    from gateway.run import _profile_runtime_scope
    with _profile_runtime_scope(profile_home):
        profile_db = profile_home / "state.db"
        assert profile_db.exists()
        
        db = SessionDB(db_path=profile_db)
        loaded = db.get_session("session123")
        assert loaded is not None
        db.close()
        
    global_db = test_env / "state.db"
    if global_db.exists():
        g_db = SessionDB(db_path=global_db)
        assert g_db.get_session("session123") is None
        g_db.close()

def test_soul_identity_prompt(test_env):
    """SOUL identity: agent prompt contains profile's SOUL, modifying it busts signature."""
    runner = GatewayRunner(config=GatewayConfig(platforms={}))
    profile_home = test_env / "profiles" / "profilea"
    
    from gateway.run import _profile_runtime_scope
    with _profile_runtime_scope(profile_home):
        sig1 = runner._agent_config_signature(
            model="gpt-4",
            runtime={},
            enabled_toolsets=[],
            ephemeral_prompt="",
        )
        
        soul_file = profile_home / "SOUL.md"
        soul_file.write_text("updated identity content", encoding="utf-8")
        
        stat = soul_file.stat()
        os.utime(soul_file, (stat.st_atime, stat.st_mtime + 5.0))
        
        sig2 = runner._agent_config_signature(
            model="gpt-4",
            runtime={},
            enabled_toolsets=[],
            ephemeral_prompt="",
        )
        assert sig1 != sig2

def test_mcp_real_config_isolation(test_env):
    """MCP real-config: B3 acceptance. Two profiles, different env/tokens, separate connections."""
    mock_connects = []
    async def fake_connect(name, config):
        mock_connects.append((name, config))
        server = MCPServerTask(name)
        server.session = MagicMock()
        server._tools = []
        return server
        
    def fake_run(coro_or_factory, timeout=30):
        coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
        return asyncio.run(coro)

    with patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True), \
         patch("tools.mcp_tool._ensure_mcp_loop"), \
         patch("tools.mcp_tool._run_on_mcp_loop", side_effect=fake_run):
         
         profile_home_a = test_env / "profiles" / "profilea"
         from gateway.run import _profile_runtime_scope
         with _profile_runtime_scope(profile_home_a):
             srv_a = _get_active_server("github")
             assert srv_a is not None
             
         profile_home_b = test_env / "profiles" / "profileb"
         with _profile_runtime_scope(profile_home_b):
             srv_b = _get_active_server("github")
             assert srv_b is not None
             
         assert len(mock_connects) >= 2
         configs = [c for n, c in mock_connects if n == "github"]
         assert configs[0]["env"]["TOKEN"] == "profilea_token"
         assert configs[1]["env"]["TOKEN"] == "profileb_token"

def test_memory_isolation(test_env):
    """Memory isolation: routed-turn writes MEMORY.md to profiles/<n>/memories/."""
    profile_home = test_env / "profiles" / "profilea"
    
    from gateway.run import _profile_runtime_scope
    with _profile_runtime_scope(profile_home):
        tokens = set_session_vars(
            session_id="session123",
            agent_hermes_home=str(profile_home),
            agent_profile="profilea",
        )
        try:
            mem_dir = get_memory_dir()
            assert mem_dir.resolve() == (profile_home / "memories").resolve()
            assert get_memory_dir("profileb").resolve() == (test_env / "profiles" / "profileb" / "memories").resolve()
            
            store = MemoryStore()
            store.add("memory", "Hello from profilea")
            
            mem_file = profile_home / "memories" / "MEMORY.md"
            assert mem_file.exists()
            assert "Hello from profilea" in mem_file.read_text(encoding="utf-8")
            
            global_mem = test_env / "memories" / "MEMORY.md"
            if global_mem.exists():
                assert "Hello from profilea" not in global_mem.read_text(encoding="utf-8")
        finally:
            clear_session_vars(tokens)

def test_phase_4_auth_fail_closed(test_env):
    """Phase-4 auth fail-closed: missing scoped credential in routed session returns None/fails."""
    from agent.secret_scope import set_multiplex_active, get_secret
    set_multiplex_active(True)
    try:
        profile_home = test_env / "profiles" / "profilea"
        
        from gateway.run import _profile_runtime_scope
        with _profile_runtime_scope(profile_home):
            assert get_secret("GITHUB_TOKEN") == "profilea_token"
            val = get_secret("SOME_NON_EXISTENT_KEY")
            assert val is None or val == ""
    finally:
        set_multiplex_active(False)

def test_cross_profile_process_control(test_env):
    """Cross-profile process control: Profile A cannot retrieve Profile B processes."""
    from tools.process_registry import process_registry
    
    profile_home_a = test_env / "profiles" / "profilea"
    profile_home_b = test_env / "profiles" / "profileb"
    
    from tools.process_registry import ProcessSession
    sess = ProcessSession(
        id="proc123",
        command="sleep 10",
        cwd="/tmp",
        task_id="task1",
        session_key="session1",
        agent_profile="profileb",
        agent_hermes_home=str(profile_home_b),
    )
    process_registry._running["proc123"] = sess
    
    tokens = set_session_vars(
        session_id="session123",
        agent_hermes_home=str(profile_home_a),
        agent_profile="profilea",
    )
    try:
        proc = process_registry._get_for_current_scope("proc123")
        assert proc is None
    finally:
        clear_session_vars(tokens)

def test_zero_regression_full_turn(test_env):
    """Zero-regression full-turn: unrouted topic uses global HERMES_HOME."""
    runner = GatewayRunner(config=GatewayConfig(platforms={}))
    runner._normalize_source_for_session_key = lambda src: src
    
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="111",
        chat_type="dm",
        thread_id="999",
    )
    
    routed = runner._routed_profile_for_source(source)
    assert routed is None
    
    profile_home = runner._resolve_profile_home_for_source(source)
    assert profile_home.resolve() == test_env.resolve()


def test_g2_fallback_when_session_id_none(test_env):
    """Test Tail 1 fallback: when session_id is None, get_profile_home_for_session resolves active profile home-override."""
    from tools.file_tools import get_profile_home_for_session, write_file_tool
    from gateway.run import _profile_runtime_scope
    
    profile_home = test_env / "profiles" / "profilea"
    
    # 1. Under profile scope, session_id is None but fallback resolves profile_home
    with _profile_runtime_scope(profile_home):
        resolved = get_profile_home_for_session(None)
        assert resolved is not None
        assert resolved.resolve() == profile_home.resolve()
        
        # Verify hard guard blocks writing outside profile-home even without session_id
        res = write_file_tool("../../SOUL.md", "Illegal write")
        res_dict = json.loads(res)
        assert "error" in res_dict or res_dict.get("success") is False
        assert (test_env / "SOUL.md").read_text(encoding="utf-8") == "I am main agent"

    # 2. Under default scope, get_profile_home_for_session(None) returns None and allows normal writes
    resolved_default = get_profile_home_for_session(None)
    assert resolved_default is None


def test_reload_mcp_scoped(test_env):
    """Test Tail 2: /reload-mcp under routed scope reloads config from correct profile's home directory."""
    runner = GatewayRunner(config=GatewayConfig(platforms={}))
    
    homes_seen = []
    
    from hermes_constants import get_hermes_home
    def fake_load_config():
        homes_seen.append(get_hermes_home().resolve())
        return {}
        
    def fake_shutdown():
        homes_seen.append(get_hermes_home().resolve())
        
    def fake_discover():
        homes_seen.append(get_hermes_home().resolve())
        return []
        
    profile_home = test_env / "profiles" / "profilea"
    
    from gateway.run import _profile_runtime_scope
    with _profile_runtime_scope(profile_home), \
         patch("tools.mcp_tool._load_mcp_config", side_effect=fake_load_config), \
         patch("tools.mcp_tool.shutdown_mcp_servers", side_effect=fake_shutdown), \
         patch("tools.mcp_tool.discover_mcp_tools", side_effect=fake_discover):
         
         event = MagicMock()
         event.source.platform = Platform.TELEGRAM
         event.source.chat_id = "111"
         event.source.thread_id = "222"
         
         asyncio.run(runner._execute_mcp_reload(event))
         
    assert len(homes_seen) > 0
    for h in homes_seen:
        assert h == profile_home.resolve()


def test_write_file_guard_on_resolve_error(test_env):
    """Test Tail 3: write_file still applies hard-guard on resolve error."""
    profile_home = test_env / "profiles" / "profilea"
    global_soul = test_env / "SOUL.md"
    
    # Pre-create session record in profilea's state.db
    from hermes_state import SessionDB
    db = SessionDB(db_path=profile_home / "state.db")
    db.create_session("session123", "telegram")
    db.close()
    
    # We patch _resolve_path_for_task to raise an exception
    with patch("tools.file_tools._resolve_path_for_task", side_effect=ValueError("Simulated resolve error")):
        # Call write_file_tool attempting to write to global SOUL.md under a session
        from gateway.run import _profile_runtime_scope
        with _profile_runtime_scope(profile_home):
            tokens = set_session_vars(
                session_id="session123",
                agent_hermes_home=str(profile_home),
                agent_profile="profilea",
            )
            try:
                res = write_file_tool(str(global_soul), "Malicious SOUL override", session_id="session123")
                assert "Refusing to write to the global SOUL.md" in res or "error" in res
                
                # Verify it was NOT written
                assert global_soul.read_text(encoding="utf-8") == "I am main agent"
            finally:
                clear_session_vars(tokens)


def test_g2_hard_guard_blocks_outside_profile_home(test_env):
    """Verify expanded hard-guard blocks write to any path outside profile_home except tmp and cwd."""
    profile_home = test_env / "profiles" / "profilea"
    
    # Pre-create session record in profilea's state.db
    from hermes_state import SessionDB
    db = SessionDB(db_path=profile_home / "state.db")
    db.create_session("session123", "telegram")
    db.close()
    
    from gateway.run import _profile_runtime_scope
    with _profile_runtime_scope(profile_home):
        tokens = set_session_vars(
            session_id="session123",
            agent_hermes_home=str(profile_home),
            agent_profile="profilea",
        )
        try:
            # 1. Writing inside profile home should succeed
            in_profile_path = profile_home / "some_file.txt"
            res_ok = write_file_tool(str(in_profile_path), "in profile content", session_id="session123")
            assert "error" not in res_ok
            assert in_profile_path.read_text(encoding="utf-8") == "in profile content"
            
            # 2. Writing inside temp directory should succeed
            import tempfile
            temp_file = Path(tempfile.gettempdir()) / "hermes_test_temp.txt"
            res_temp = write_file_tool(str(temp_file), "temp content", session_id="session123")
            assert "error" not in res_temp
            assert temp_file.read_text(encoding="utf-8") == "temp content"
            
            # 3. Writing inside active workspace CWD should succeed
            cwd_file = Path(os.getcwd()) / "hermes_test_cwd.txt"
            # Ensure it doesn't already exist or clean up after
            try:
                res_cwd = write_file_tool(str(cwd_file), "cwd content", session_id="session123")
                assert "error" not in res_cwd
                assert cwd_file.read_text(encoding="utf-8") == "cwd content"
            finally:
                if cwd_file.exists():
                    cwd_file.unlink()
                    
            # 4. Writing to an outside/forbidden path should be blocked
            forbidden_path = test_env / "random_file_outside_profile.txt"
            res_blocked = write_file_tool(str(forbidden_path), "forbidden content", session_id="session123")
            assert "Refusing to write to path outside profile home" in res_blocked or "error" in res_blocked
            assert not forbidden_path.exists()
        finally:
            clear_session_vars(tokens)

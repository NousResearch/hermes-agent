"""
Tests for hermes_cli.mcp_config — ``hermes mcp`` subcommands.

These tests mock the MCP server connection layer so they run without
any actual MCP servers or API keys.
"""

import argparse
import os
from pathlib import Path

import pytest


def _set_interactive_stdin(monkeypatch, *, is_tty: bool = True) -> None:
    from unittest.mock import MagicMock

    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = is_tty
    monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    """Redirect all config I/O to a temp directory."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "hermes_cli.config.get_hermes_home", lambda: tmp_path
    )
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    monkeypatch.setattr(
        "hermes_cli.config.get_config_path", lambda: config_path
    )
    monkeypatch.setattr(
        "hermes_cli.config.get_env_path", lambda: env_path
    )
    return tmp_path


def _make_args(**kwargs):
    """Build a minimal argparse.Namespace."""
    defaults = {
        "name": "test-server",
        "url": None,
        "mcp_command": None,
        "args": None,
        "auth": None,
        "preset": None,
        "env": None,
        "mcp_action": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _seed_config(tmp_path: Path, mcp_servers: dict):
    """Write a config.yaml with the given mcp_servers."""
    import yaml

    config = {"mcp_servers": mcp_servers, "_config_version": 9}
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)


class FakeTool:
    """Mimics an MCP tool object returned by the SDK."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description


# ---------------------------------------------------------------------------
# Tests: cmd_mcp_list
# ---------------------------------------------------------------------------

class TestMcpList:
    def test_list_empty_config(self, tmp_path, capsys):
        from hermes_cli.mcp_config import cmd_mcp_list

        cmd_mcp_list()
        out = capsys.readouterr().out
        assert "No MCP servers configured" in out

    def test_list_with_servers(self, tmp_path, capsys):
        _seed_config(tmp_path, {
            "ink": {
                "url": "https://mcp.ml.ink/mcp",
                "enabled": True,
                "tools": {"include": ["create_service", "get_service"]},
            },
            "github": {
                "command": "npx",
                "args": ["@mcp/github"],
                "enabled": False,
            },
        })
        from hermes_cli.mcp_config import cmd_mcp_list

        cmd_mcp_list()
        out = capsys.readouterr().out
        assert "ink" in out
        assert "github" in out
        assert "2 selected" in out  # ink has 2 in include
        assert "disabled" in out  # github is disabled

    def test_list_enabled_default_true(self, tmp_path, capsys):
        """Server without explicit enabled key defaults to enabled."""
        _seed_config(tmp_path, {
            "myserver": {"url": "https://example.com/mcp"},
        })
        from hermes_cli.mcp_config import cmd_mcp_list

        cmd_mcp_list()
        out = capsys.readouterr().out
        assert "myserver" in out
        assert "enabled" in out


# ---------------------------------------------------------------------------
# Tests: cmd_mcp_remove
# ---------------------------------------------------------------------------

class TestMcpRemove:
    def test_remove_existing_server(self, tmp_path, capsys, monkeypatch):
        _seed_config(tmp_path, {
            "myserver": {"url": "https://example.com/mcp"},
        })
        monkeypatch.setattr("builtins.input", lambda _: "y")
        from hermes_cli.mcp_config import cmd_mcp_remove

        cmd_mcp_remove(_make_args(name="myserver"))

        out = capsys.readouterr().out
        assert "Removed" in out

        # Verify config updated
        from hermes_cli.config import load_config

        config = load_config()
        assert "myserver" not in config.get("mcp_servers", {})


    def test_remove_cleans_oauth_tokens(self, tmp_path, capsys, monkeypatch):
        _seed_config(tmp_path, {
            "oauth-srv": {"url": "https://example.com/mcp", "auth": "oauth"},
        })
        monkeypatch.setattr("builtins.input", lambda _: "y")
        # Also patch get_hermes_home in the mcp_config module namespace
        monkeypatch.setattr(
            "hermes_cli.mcp_config.get_hermes_home", lambda: tmp_path
        )

        # Create a fake token file
        token_dir = tmp_path / "mcp-tokens"
        token_dir.mkdir()
        token_file = token_dir / "oauth-srv.json"
        token_file.write_text("{}")

        from hermes_cli.mcp_config import cmd_mcp_remove

        cmd_mcp_remove(_make_args(name="oauth-srv"))
        assert not token_file.exists()


# ---------------------------------------------------------------------------
# Tests: cmd_mcp_add
# ---------------------------------------------------------------------------

class TestMcpAdd:

    def test_add_http_server_all_tools(self, tmp_path, capsys, monkeypatch):
        """Add an HTTP server, accept all tools."""
        fake_tools = [
            FakeTool("create_service", "Deploy from repo"),
            FakeTool("list_services", "List all services"),
        ]

        def mock_probe(name, config, **kw):
            return [(t.name, t.description) for t in fake_tools]

        monkeypatch.setattr(
            "hermes_cli.mcp_config._probe_single_server", mock_probe
        )
        # No auth, accept all tools
        inputs = iter(["n", ""])  # no auth needed, enable all
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        from hermes_cli.mcp_config import cmd_mcp_add

        cmd_mcp_add(_make_args(name="ink", url="https://mcp.ml.ink/mcp"))
        out = capsys.readouterr().out
        assert "Saved" in out
        assert "2/2 tools" in out

        # Verify config written
        from hermes_cli.config import load_config

        config = load_config()
        assert "ink" in config.get("mcp_servers", {})
        assert config["mcp_servers"]["ink"]["url"] == "https://mcp.ml.ink/mcp"

    def test_add_yes_skips_every_prompt(self, tmp_path, capsys, monkeypatch):
        """--yes drives a full add without touching input() even once.

        input() is wired to raise, so any surviving prompt site (auth,
        save-anyway, tool selection) fails the test rather than hanging it.
        """
        fake_tools = [
            FakeTool("create_service", "Deploy from repo"),
            FakeTool("list_services", "List all services"),
        ]

        monkeypatch.setattr(
            "hermes_cli.mcp_config._probe_single_server",
            lambda name, config, **kw: [(t.name, t.description) for t in fake_tools],
        )

        def _no_input(prompt=""):
            raise AssertionError(f"--yes must not prompt; got {prompt!r}")

        monkeypatch.setattr("builtins.input", _no_input)

        from hermes_cli.mcp_config import cmd_mcp_add

        cmd_mcp_add(_make_args(name="auto", url="https://mcp.example.com/mcp", yes=True))
        out = capsys.readouterr().out
        assert "auto-enabling" in out
        assert "2/2 tools" in out

        from hermes_cli.config import load_config

        assert "auto" in load_config().get("mcp_servers", {})

    def test_add_yes_with_auth_but_no_key_errors(self, tmp_path, capsys, monkeypatch):
        """--yes --auth header without the env var fails loudly, never prompts.

        Cron/CI that asked for auth but has no credential should get an
        actionable error, not a password prompt nothing can answer.
        """
        def _probe_should_not_run(name, config, **kw):
            raise AssertionError("must error before probing")

        monkeypatch.setattr(
            "hermes_cli.mcp_config._probe_single_server", _probe_should_not_run
        )

        def _no_input(prompt=""):
            raise AssertionError(f"--yes must not prompt; got {prompt!r}")

        monkeypatch.setattr("builtins.input", _no_input)

        from hermes_cli.mcp_config import cmd_mcp_add

        cmd_mcp_add(_make_args(
            name="needs-auth",
            url="https://mcp.example.com/mcp",
            auth="header",
            yes=True,
        ))
        out = capsys.readouterr().out
        assert "MCP_NEEDS_AUTH_API_KEY" in out

        from hermes_cli.config import load_config

        assert "needs-auth" not in load_config().get("mcp_servers", {})

    def test_add_stdio_server_with_env(self, tmp_path, capsys, monkeypatch):
        """Stdio servers can persist explicit environment variables."""
        fake_tools = [FakeTool("search", "Search repos")]

        def mock_probe(name, config, **kw):
            assert config["env"] == {
                "MY_API_KEY": "secret123",
                "DEBUG": "true",
            }
            return [(t.name, t.description) for t in fake_tools]

        monkeypatch.setattr(
            "hermes_cli.mcp_config._probe_single_server", mock_probe
        )
        monkeypatch.setattr("builtins.input", lambda _: "")

        from hermes_cli.mcp_config import cmd_mcp_add

        cmd_mcp_add(_make_args(
            name="github",
            mcp_command="npx",
            args=["@mcp/github"],
            env=["MY_API_KEY=secret123", "DEBUG=true"],
        ))
        out = capsys.readouterr().out
        assert "Saved" in out

        from hermes_cli.config import load_config

        config = load_config()
        srv = config["mcp_servers"]["github"]
        assert srv["env"] == {
            "MY_API_KEY": "secret123",
            "DEBUG": "true",
        }


    def test_add_preset_fills_transport(self, tmp_path, capsys, monkeypatch):
        """A preset fills in command/args when no explicit transport given."""
        monkeypatch.setattr(
            "hermes_cli.mcp_config._MCP_PRESETS",
            {"testmcp": {"command": "npx", "args": ["-y", "test-mcp-server"], "display_name": "Test MCP"}},
        )
        fake_tools = [FakeTool("do_thing", "Does a thing")]

        def mock_probe(name, config, **kw):
            assert name == "myserver"
            assert config["command"] == "npx"
            assert config["args"] == ["-y", "test-mcp-server"]
            assert "env" not in config
            return [(t.name, t.description) for t in fake_tools]

        monkeypatch.setattr(
            "hermes_cli.mcp_config._probe_single_server", mock_probe
        )
        monkeypatch.setattr("builtins.input", lambda _: "")

        from hermes_cli.mcp_config import cmd_mcp_add
        from hermes_cli.config import read_raw_config

        cmd_mcp_add(_make_args(name="myserver", preset="testmcp"))
        out = capsys.readouterr().out
        assert "Saved" in out

        config = read_raw_config()
        srv = config["mcp_servers"]["myserver"]
        assert srv["command"] == "npx"
        assert srv["args"] == ["-y", "test-mcp-server"]
        assert "env" not in srv


# ---------------------------------------------------------------------------
# Tests: cmd_mcp_test
# ---------------------------------------------------------------------------

class TestMcpTest:

    def test_test_success(self, tmp_path, capsys, monkeypatch):
        _seed_config(tmp_path, {
            "ink": {"url": "https://mcp.ml.ink/mcp"},
        })

        def mock_probe(name, config, **kw):
            return [("create_service", "Deploy"), ("list_services", "List all")]

        monkeypatch.setattr(
            "hermes_cli.mcp_config._probe_single_server", mock_probe
        )
        from hermes_cli.mcp_config import cmd_mcp_test

        cmd_mcp_test(_make_args(name="ink"))
        out = capsys.readouterr().out
        assert "Connected" in out
        assert "Tools discovered: 2" in out

    def test_probe_uses_configured_connect_timeout(self, monkeypatch):
        """OAuth-capable probes must not hard-code a short 30s timeout."""
        import asyncio
        from hermes_cli import mcp_config
        import tools.mcp_tool as mcp_tool

        captured = {}

        class FakeServer:
            _tools = []

            async def shutdown(self):
                captured["shutdown"] = True

        async def fake_connect(name, config):
            return FakeServer()

        def fake_run_on_mcp_loop(coro, timeout):
            captured["outer_timeout"] = timeout
            return asyncio.run(coro)

        async def fake_wait_for(awaitable, timeout):
            captured["inner_timeout"] = timeout
            return await awaitable

        monkeypatch.setattr(mcp_tool, "_ensure_mcp_loop", lambda: None)
        monkeypatch.setattr(mcp_tool, "_stop_mcp_loop_if_idle", lambda: None)
        monkeypatch.setattr(mcp_tool, "_connect_server", fake_connect)
        monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", fake_run_on_mcp_loop)
        monkeypatch.setattr(mcp_config.asyncio, "wait_for", fake_wait_for)

        assert mcp_config._probe_single_server(
            "supabase", {"connect_timeout": 300}
        ) == []
        assert captured["inner_timeout"] == 300.0
        assert captured["outer_timeout"] == 310.0
        assert captured["shutdown"] is True


# ---------------------------------------------------------------------------
# Tests: env var interpolation
# ---------------------------------------------------------------------------

class TestEnvVarInterpolation:


    def test_interpolate_cursor_env_prefix(self, monkeypatch):
        """Cursor-style ${env:VAR} resolves the same secret as ${VAR}."""
        monkeypatch.setenv("MY_KEY", "secret123")
        from tools.mcp_tool import _interpolate_env_vars

        assert _interpolate_env_vars("Bearer ${env:MY_KEY}") == "Bearer secret123"


    def test_env_ref_name_strips_prefix(self):
        from tools.mcp_tool import _env_ref_name

        assert _env_ref_name("env:API_KEY") == "API_KEY"
        assert _env_ref_name("API_KEY") == "API_KEY"
        assert _env_ref_name(" env:API_KEY ") == "API_KEY"


class TestContextVarInterpolation:
    """Cursor-style context variables: ${userHome}, ${workspaceFolder},
    ${workspaceFolderBasename}, ${pathSeparator}, ${/}."""

    def test_user_home(self):
        import os

        from tools.mcp_tool import _interpolate_env_vars

        assert _interpolate_env_vars("${userHome}") == os.path.expanduser("~")

    def test_path_separator_and_slash_shorthand(self):
        import os

        from tools.mcp_tool import _interpolate_env_vars

        assert _interpolate_env_vars("${pathSeparator}") == os.sep
        assert _interpolate_env_vars("${/}") == os.sep

    def test_workspace_folder_and_basename(self, monkeypatch):
        import tools.mcp_tool as mcp_tool

        monkeypatch.setattr(
            mcp_tool, "_workspace_folder", lambda: "/srv/projects/myapp"
        )
        assert mcp_tool._interpolate_env_vars("${workspaceFolder}") == (
            "/srv/projects/myapp"
        )
        assert mcp_tool._interpolate_env_vars(
            "${workspaceFolderBasename}"
        ) == "myapp"

    def test_workspace_folder_falls_back_to_cwd(self, monkeypatch):
        import os

        import tools.file_tools as file_tools
        from tools.mcp_tool import _workspace_folder

        monkeypatch.setattr(
            file_tools, "_authoritative_workspace_root", lambda task_id="default": None
        )
        assert _workspace_folder() == os.getcwd()

    def test_mixed_string_with_env_and_context_vars(self, monkeypatch):
        import os

        import tools.mcp_tool as mcp_tool

        monkeypatch.setenv("MY_TOKEN", "tok-1")
        monkeypatch.setattr(mcp_tool, "_workspace_folder", lambda: "/ws/app")
        result = mcp_tool._interpolate_env_vars(
            "${userHome}${/}.cache${/}${workspaceFolderBasename}-${MY_TOKEN}"
        )
        home = os.path.expanduser("~")
        assert result == f"{home}{os.sep}.cache{os.sep}app-tok-1"

    def test_context_names_are_case_sensitive(self, monkeypatch):
        """${USERHOME} is NOT a context var — it keeps env-var semantics
        (literal placeholder when unset)."""
        monkeypatch.delenv("USERHOME", raising=False)
        from tools.mcp_tool import _interpolate_env_vars

        assert _interpolate_env_vars("${USERHOME}") == "${USERHOME}"

    def test_unknown_ref_keeps_literal_placeholder(self, monkeypatch):
        monkeypatch.delenv("NOT_A_REAL_VAR_XYZ", raising=False)
        from tools.mcp_tool import _interpolate_env_vars

        assert _interpolate_env_vars("${NOT_A_REAL_VAR_XYZ}") == (
            "${NOT_A_REAL_VAR_XYZ}"
        )

    def test_context_vars_in_nested_config(self, monkeypatch):
        import os

        import tools.mcp_tool as mcp_tool

        monkeypatch.setattr(mcp_tool, "_workspace_folder", lambda: "/ws/app")
        cfg = {
            "command": "npx",
            "args": ["-y", "server-fs", "${workspaceFolder}"],
            "cwd": "${workspaceFolder}",
            "env": {"CACHE": "${userHome}${/}.cache"},
            "headers": {"X-Ws": "${workspaceFolderBasename}"},
        }
        out = mcp_tool._interpolate_env_vars(cfg)
        home = os.path.expanduser("~")
        assert out["args"][2] == "/ws/app"
        assert out["cwd"] == "/ws/app"
        assert out["env"]["CACHE"] == f"{home}{os.sep}.cache"
        assert out["headers"]["X-Ws"] == "app"


# ---------------------------------------------------------------------------
# Tests: probe-path env resolution (#37792)
# ---------------------------------------------------------------------------

class TestProbeEnvResolution:
    """The probe path must resolve ``${ENV}`` before connecting, so the
    discovery probe behaves like runtime tool loading. Regression for #37792
    where `hermes mcp add --auth header` sent a literal
    ``Authorization: Bearer ${MCP_X_API_KEY}`` and got 401."""

    def test_resolve_interpolates_header(self, monkeypatch):
        from hermes_cli.mcp_config import _resolve_mcp_server_config

        monkeypatch.setenv("MCP_N8N_API_KEY", "jwt-token-xyz")
        resolved = _resolve_mcp_server_config({
            "url": "http://localhost:5678/mcp-server/http",
            "headers": {"Authorization": "Bearer ${MCP_N8N_API_KEY}"},
        })
        assert resolved["headers"]["Authorization"] == "Bearer jwt-token-xyz"

    def test_active_secret_scope_does_not_load_dotenv_into_process_env(
        self, tmp_path, monkeypatch
    ):
        from agent.secret_scope import reset_secret_scope, set_secret_scope
        from hermes_cli.mcp_config import _resolve_mcp_server_config

        monkeypatch.setenv("MCP_SHARED_API_KEY", "default-secret")
        token = set_secret_scope({"MCP_SHARED_API_KEY": "profile-secret"})
        try:
            resolved = _resolve_mcp_server_config({
                "headers": {"Authorization": "Bearer ${MCP_SHARED_API_KEY}"},
            })
        finally:
            reset_secret_scope(token)

        assert resolved["headers"]["Authorization"] == "Bearer profile-secret"
        assert os.environ["MCP_SHARED_API_KEY"] == "default-secret"


    def test_probe_resolves_before_connect(self, monkeypatch):
        """_probe_single_server must pass the RESOLVED config to _connect_server."""
        import hermes_cli.mcp_config as mc

        monkeypatch.setenv("MCP_N8N_API_KEY", "jwt-token-xyz")

        seen = {}

        class _FakeTool:
            name = "do_thing"
            description = "a tool"

        class _FakeServer:
            _tools = [_FakeTool()]

            async def shutdown(self):
                return None

        async def _fake_connect(name, config):
            seen["config"] = config
            return _FakeServer()

        monkeypatch.setattr("tools.mcp_tool._connect_server", _fake_connect)

        tools = mc._probe_single_server("n8n", {
            "url": "http://localhost:5678/mcp-server/http",
            "headers": {"Authorization": "Bearer ${MCP_N8N_API_KEY}"},
        })

        assert tools == [("do_thing", "a tool")]
        assert seen["config"]["headers"]["Authorization"] == "Bearer jwt-token-xyz"


class TestProbeCapabilityGating:
    """The ``details`` probe must not fire prompts/list or resources/list at
    servers that either disabled them in config or never advertised them.

    Regression for the Unreal MCP server case: it answers
    ``Call to unknown method "prompts/list"``, so an unconditional probe logged
    a hard error and ``tools.prompts: false`` (the documented workaround) had no
    effect because the probe never consulted config or capabilities.
    """

    class _FakeTool:
        name = "do_thing"
        description = "a tool"

    class _Caps:
        def __init__(self, prompts=None, resources=None):
            self.prompts = prompts
            self.resources = resources

    class _InitResult:
        def __init__(self, caps):
            self.capabilities = caps

    def _make_server(self, called, caps):
        outer = self

        class _Result(list):
            @property
            def prompts(self):
                return self

            @property
            def resources(self):
                return self

        class _Session:
            async def list_prompts(self_inner):
                called.append("prompts")
                return _Result()

            async def list_resources(self_inner):
                called.append("resources")
                return _Result()

        class _FakeServer:
            _tools = [outer._FakeTool()]
            session = _Session()
            initialize_result = outer._InitResult(caps)

            async def shutdown(self_inner):
                return None

        return _FakeServer()

    def _run_probe(self, monkeypatch, config, caps):
        import hermes_cli.mcp_config as mc

        called: list[str] = []

        async def _fake_connect(name, cfg):
            return self._make_server(called, caps)

        monkeypatch.setattr("tools.mcp_tool._connect_server", _fake_connect)
        details: dict = {}
        mc._probe_single_server("srv", config, details=details)
        return called, details

    def test_config_disables_prompts_probe(self, monkeypatch):
        # Server advertises both, but user turned prompts off.
        caps = self._Caps(prompts=object(), resources=object())
        called, details = self._run_probe(
            monkeypatch, {"url": "http://x/mcp", "tools": {"prompts": False}}, caps
        )
        assert "prompts" not in called
        assert "resources" in called


    def test_advertised_and_enabled_is_probed(self, monkeypatch):
        caps = self._Caps(prompts=object(), resources=object())
        called, details = self._run_probe(monkeypatch, {"url": "http://x/mcp"}, caps)
        assert set(called) == {"prompts", "resources"}


class TestStripBearerPrefix:
    """Pasted tokens that already include ``Bearer `` would otherwise produce
    ``Bearer Bearer <jwt>`` once the header template adds its own prefix."""

    def test_bare_token_unchanged(self):
        from hermes_cli.mcp_config import _strip_bearer_prefix

        assert _strip_bearer_prefix("eyJabc123") == "eyJabc123"


class TestBearerAuthPersistence:
    def test_secret_and_header_are_persisted_separately(self):
        from hermes_cli.config import get_env_value
        from hermes_cli.mcp_config import _save_bearer_auth_token

        headers = _save_bearer_auth_token("My Server", "Bearer secret-value")

        assert headers == {
            "Authorization": "Bearer ${MCP_MY_SERVER_API_KEY}",
        }
        assert get_env_value("MCP_MY_SERVER_API_KEY") == "secret-value"

    def test_empty_token_is_rejected(self):
        from hermes_cli.mcp_config import _save_bearer_auth_token

        with pytest.raises(ValueError, match="Bearer token is required"):
            _save_bearer_auth_token("empty", "Bearer   ")


# ---------------------------------------------------------------------------
# Tests: config helpers
# ---------------------------------------------------------------------------

class TestConfigHelpers:
    def test_save_and_load_mcp_server(self, tmp_path):
        from hermes_cli.mcp_config import _save_mcp_server, _get_mcp_servers

        _save_mcp_server("mysvr", {"url": "https://example.com/mcp"})
        servers = _get_mcp_servers()
        assert "mysvr" in servers
        assert servers["mysvr"]["url"] == "https://example.com/mcp"


    def test_env_key_for_server(self):
        from hermes_cli.mcp_config import _env_key_for_server

        assert _env_key_for_server("ink") == "MCP_INK_API_KEY"
        assert _env_key_for_server("my-server") == "MCP_MY_SERVER_API_KEY"
        assert _env_key_for_server("my.server") == "MCP_MY_SERVER_API_KEY"
        assert _env_key_for_server("github/mcp") == "MCP_GITHUB_MCP_API_KEY"


# ---------------------------------------------------------------------------
# Tests: dispatcher
# ---------------------------------------------------------------------------

class TestDispatcher:
    def test_no_action_shows_list(self, tmp_path, capsys):
        from hermes_cli.mcp_config import mcp_command

        _seed_config(tmp_path, {})
        mcp_command(_make_args(mcp_action=None))
        out = capsys.readouterr().out
        assert "Commands:" in out or "No MCP servers" in out


# ---------------------------------------------------------------------------
# Tests: Task 7 consolidation — cmd_mcp_remove evicts manager cache,
# cmd_mcp_login forces re-auth
# ---------------------------------------------------------------------------


class TestMcpRemoveEvictsManager:
    def test_remove_evicts_in_memory_provider(self, tmp_path, capsys, monkeypatch):
        """After cmd_mcp_remove, the MCPOAuthManager no longer caches the provider."""
        _seed_config(tmp_path, {
            "oauth-srv": {"url": "https://example.com/mcp", "auth": "oauth"},
        })
        monkeypatch.setattr("builtins.input", lambda _: "y")
        monkeypatch.setattr(
            "hermes_cli.mcp_config.get_hermes_home", lambda: tmp_path
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _set_interactive_stdin(monkeypatch)

        from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests
        reset_manager_for_tests()

        mgr = get_manager()
        mgr.get_or_build_provider(
            "oauth-srv", "https://example.com/mcp", None,
        )
        assert mgr._key("oauth-srv") in mgr._entries

        from hermes_cli.mcp_config import cmd_mcp_remove
        cmd_mcp_remove(_make_args(name="oauth-srv"))

        assert mgr._key("oauth-srv") not in mgr._entries


class TestMcpLogin:
    def test_login_rejects_unknown_server(self, tmp_path, capsys):
        _seed_config(tmp_path, {})
        from hermes_cli.mcp_config import cmd_mcp_login
        cmd_mcp_login(_make_args(name="ghost"))
        out = capsys.readouterr().out
        assert "not found" in out


    def test_login_false_success_no_token(self, tmp_path, capsys, monkeypatch):
        """Probe lists tools without auth (Google Drive), but no token landed.

        The server allows tools/list without auth (DCR 400'd), so the probe
        succeeds yet no OAuth token exists. Login must NOT claim success — it
        should warn and point the user at pre-registered client_id config.
        """
        _seed_config(tmp_path, {
            "googledrive": {
                "url": "https://drivemcp.googleapis.com/mcp/v1",
                "auth": "oauth",
            },
        })
        # Probe returns tools even though auth never completed.
        monkeypatch.setattr(
            "hermes_cli.mcp_config._probe_single_server",
            lambda name, cfg, connect_timeout=30: [
                ("search_files", "d"), ("read_file_content", "d"),
            ],
        )
        # No token file is created → _oauth_tokens_present() returns False.
        from hermes_cli.mcp_config import cmd_mcp_login

        cmd_mcp_login(_make_args(name="googledrive"))
        out = capsys.readouterr().out

        assert "no OAuth token was obtained" in out
        assert "Authenticated" not in out
        assert "client_id" in out

    def test_login_genuine_success_with_token(self, tmp_path, capsys, monkeypatch):
        """Probe lists tools AND a token exists → report real success."""
        _seed_config(tmp_path, {
            "realserver": {"url": "https://mcp.example.com/mcp", "auth": "oauth"},
        })
        token_dir = tmp_path / "mcp-tokens"

        # cmd_mcp_login wipes tokens before probing, then the real OAuth flow
        # writes a fresh token during the probe. Simulate that: the mocked
        # probe drops a token file, mirroring a successful authorization.
        seen = {}

        def mock_probe(name, cfg, connect_timeout=30):
            seen["connect_timeout"] = connect_timeout
            token_dir.mkdir(exist_ok=True)
            (token_dir / "realserver.json").write_text('{"access_token": "x"}')
            return [("a", "d"), ("b", "d"), ("c", "d")]

        monkeypatch.setattr(
            "hermes_cli.mcp_config._probe_single_server", mock_probe
        )

        from hermes_cli.mcp_config import cmd_mcp_login

        cmd_mcp_login(_make_args(name="realserver"))
        out = capsys.readouterr().out

        assert "Authenticated — 3 tool(s) available" in out
        assert "no OAuth token" not in out
        # The login path must grant a human enough time to finish the browser
        # OAuth round-trip — far longer than the 30s probe default.
        assert seen["connect_timeout"] >= 180


# ---------------------------------------------------------------------------
# Tests: cmd_mcp_reauth (GH#36767)
# ---------------------------------------------------------------------------

class TestMcpReauth:
    def test_reauth_all_visits_only_oauth_servers_in_order(
        self, tmp_path, capsys, monkeypatch
    ):
        """--all re-auths every oauth server (skipping non-oauth), serially."""
        _seed_config(tmp_path, {
            "gh": {"url": "https://gh.example.com/mcp", "auth": "oauth"},
            "jira": {"url": "https://jira.example.com/mcp", "auth": "oauth"},
            "localstdio": {"command": "foo"},  # no url / no oauth → skipped
            "apikey": {"url": "https://k.example.com/mcp", "headers": {"x": "y"}},
        })
        visited = []
        monkeypatch.setattr(
            "hermes_cli.mcp_config._reauth_oauth_server",
            lambda name, cfg: visited.append(name) or True,
        )
        from hermes_cli.mcp_config import cmd_mcp_reauth

        cmd_mcp_reauth(_make_args(name=None, all=True))
        out = capsys.readouterr().out

        assert visited == ["gh", "jira"]
        assert "Re-authenticated 2/2 server(s)" in out

    def test_reauth_all_reports_partial_failures(self, tmp_path, capsys, monkeypatch):
        """A server that fails to re-auth is counted but doesn't abort the rest."""
        _seed_config(tmp_path, {
            "a": {"url": "https://a.example.com/mcp", "auth": "oauth"},
            "b": {"url": "https://b.example.com/mcp", "auth": "oauth"},
        })
        monkeypatch.setattr(
            "hermes_cli.mcp_config._reauth_oauth_server",
            lambda name, cfg: name == "a",  # only 'a' succeeds
        )
        from hermes_cli.mcp_config import cmd_mcp_reauth

        cmd_mcp_reauth(_make_args(name=None, all=True))
        out = capsys.readouterr().out

        assert "Re-authenticated 1/2 server(s)" in out


    def test_reauth_unknown_server(self, tmp_path, capsys):
        _seed_config(tmp_path, {
            "gh": {"url": "https://gh.example.com/mcp", "auth": "oauth"},
        })
        from hermes_cli.mcp_config import cmd_mcp_reauth

        cmd_mcp_reauth(_make_args(name="ghost", all=False))
        out = capsys.readouterr().out
        assert "not found" in out

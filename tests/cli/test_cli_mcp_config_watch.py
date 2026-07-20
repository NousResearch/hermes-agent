"""Tests for automatic MCP reload when config.yaml mcp_servers section changes."""
import time
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_cli(tmp_path, mcp_servers=None, extra_config=None):
    """Create a minimal HermesCLI instance with mocked config."""
    import cli as cli_mod
    obj = object.__new__(cli_mod.HermesCLI)
    cfg = {"mcp_servers": mcp_servers or {}}
    if extra_config:
        cfg.update(extra_config)
    obj.config = cfg
    obj._agent_running = False
    obj._last_config_check = 0.0
    obj._config_mcp_servers = mcp_servers or {}

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("mcp_servers: {}\n")
    obj._config_mtime = cfg_file.stat().st_mtime

    obj._reload_mcp = MagicMock()
    obj._busy_command = MagicMock()
    obj._busy_command.return_value.__enter__ = MagicMock(return_value=None)
    obj._busy_command.return_value.__exit__ = MagicMock(return_value=False)
    obj._slow_command_status = MagicMock(return_value="reloading...")

    return obj, cfg_file


class TestMCPConfigWatch:

    def test_no_change_does_not_reload(self, tmp_path):
        """If mtime and mcp_servers unchanged, _reload_mcp is NOT called."""
        obj, cfg_file = _make_cli(tmp_path)

        with patch("hermes_cli.config.get_config_path", return_value=cfg_file):
            obj._check_config_mcp_changes()

        obj._reload_mcp.assert_not_called()

    def test_mtime_change_with_same_mcp_servers_does_not_reload(self, tmp_path):
        """If file mtime changes but mcp_servers is identical, no reload."""
        import yaml
        obj, cfg_file = _make_cli(tmp_path, mcp_servers={"fs": {"command": "npx"}})

        # Write same mcp_servers but touch the file
        cfg_file.write_text(yaml.dump({"mcp_servers": {"fs": {"command": "npx"}}}))
        # Force mtime to appear changed
        obj._config_mtime = 0.0

        with patch("hermes_cli.config.get_config_path", return_value=cfg_file):
            obj._check_config_mcp_changes()

        obj._reload_mcp.assert_not_called()

    def test_new_mcp_server_triggers_reload(self, tmp_path):
        """Adding a new MCP server to config triggers auto-reload."""
        import yaml
        obj, cfg_file = _make_cli(tmp_path, mcp_servers={})

        # Simulate user adding a new MCP server to config.yaml
        cfg_file.write_text(yaml.dump({"mcp_servers": {"github": {"url": "https://mcp.github.com"}}}))
        obj._config_mtime = 0.0  # force stale mtime

        with patch("hermes_cli.config.get_config_path", return_value=cfg_file):
            obj._check_config_mcp_changes()

        obj._reload_mcp.assert_called_once()

    def test_removed_mcp_server_triggers_reload(self, tmp_path):
        """Removing an MCP server from config triggers auto-reload."""
        import yaml
        obj, cfg_file = _make_cli(tmp_path, mcp_servers={"github": {"url": "https://mcp.github.com"}})

        # Simulate user removing the server
        cfg_file.write_text(yaml.dump({"mcp_servers": {}}))
        obj._config_mtime = 0.0

        with patch("hermes_cli.config.get_config_path", return_value=cfg_file):
            obj._check_config_mcp_changes()

        obj._reload_mcp.assert_called_once()

    def test_interval_throttle_skips_check(self, tmp_path):
        """If called within CONFIG_WATCH_INTERVAL, stat() is skipped."""
        obj, cfg_file = _make_cli(tmp_path)
        obj._last_config_check = time.monotonic()  # just checked

        with patch("hermes_cli.config.get_config_path", return_value=cfg_file), \
             patch.object(Path, "stat") as mock_stat:
            obj._check_config_mcp_changes()
            mock_stat.assert_not_called()

        obj._reload_mcp.assert_not_called()

    def test_missing_config_file_does_not_crash(self, tmp_path):
        """If config.yaml doesn't exist, _check_config_mcp_changes is a no-op."""
        obj, cfg_file = _make_cli(tmp_path)
        missing = tmp_path / "nonexistent.yaml"

        with patch("hermes_cli.config.get_config_path", return_value=missing):
            obj._check_config_mcp_changes()  # should not raise

        obj._reload_mcp.assert_not_called()

    def test_optout_disables_auto_reload(self, tmp_path, capsys):
        """When mcp.auto_reload_on_config_change is False, a changed
        mcp_servers section must NOT trigger an automatic reload — but the
        change is still detected and the user is told how to apply it.

        This protects the provider prompt cache: every automatic reload
        rebuilds the agent tool surface and invalidates cached prefixes.

        The toggle is the top-level ``mcp:`` section in config.yaml, and the
        watcher reads it from the same freshly-parsed file it diffs — so
        flipping the toggle and editing mcp_servers in one edit behaves
        correctly.
        """
        import yaml
        obj, cfg_file = _make_cli(
            tmp_path,
            mcp_servers={},
        )

        # Simulate a changed mcp_servers section with auto-reload opted out.
        cfg_file.write_text(yaml.dump({
            "mcp": {"auto_reload_on_config_change": False},
            "mcp_servers": {"github": {"url": "https://mcp.github.com"}},
        }))
        obj._config_mtime = 0.0  # force stale mtime

        with patch("hermes_cli.config.get_config_path", return_value=cfg_file):
            obj._check_config_mcp_changes()

        obj._reload_mcp.assert_not_called()

        out = capsys.readouterr().out
        assert "reload skipped" in out
        assert "/reload-mcp" in out
        assert "prompt cache" in out

    def test_optout_updates_snapshot_so_reload_mcp_applies_cleanly(self, tmp_path):
        """After an opted-out change, the watcher must not re-notify every
        tick: the snapshot is updated so the same content compares equal on
        the next pass."""
        import yaml
        obj, cfg_file = _make_cli(tmp_path, mcp_servers={})

        cfg_file.write_text(yaml.dump({
            "mcp": {"auto_reload_on_config_change": False},
            "mcp_servers": {"github": {"url": "https://mcp.github.com"}},
        }))
        obj._config_mtime = 0.0

        with patch("hermes_cli.config.get_config_path", return_value=cfg_file):
            obj._check_config_mcp_changes()
            # Second pass: same file content, new mtime — no reload, no change.
            obj._last_config_check = 0.0
            obj._config_mtime = 0.0
            obj._check_config_mcp_changes()

        obj._reload_mcp.assert_not_called()
        assert obj._config_mcp_servers == {"github": {"url": "https://mcp.github.com"}}

    def test_optout_path_is_top_level_mcp_not_auxiliary(self, tmp_path):
        """Regression guard: the opt-out toggle is the top-level
        ``mcp.auto_reload_on_config_change`` key, NOT ``auxiliary.mcp``
        (which holds side-LLM task provider settings).

        A config that sets ONLY ``auxiliary.mcp.auto_reload_on_config_change:
        false`` must NOT disable the reload."""
        import yaml
        obj, cfg_file = _make_cli(
            tmp_path,
            mcp_servers={},
        )

        cfg_file.write_text(yaml.dump({
            "auxiliary": {"mcp": {"auto_reload_on_config_change": False}},
            "mcp_servers": {"github": {"url": "https://mcp.github.com"}},
        }))
        obj._config_mtime = 0.0

        with patch("hermes_cli.config.get_config_path", return_value=cfg_file):
            obj._check_config_mcp_changes()

        # Reload happened because the aux-task path is not the toggle.
        obj._reload_mcp.assert_called()

    def test_env_var_templates_do_not_false_positive_on_unrelated_saves(
        self, tmp_path, monkeypatch, capsys
    ):
        """Regression for the '/reasoning triggers MCP reload' bug (#55701).

        Init snapshots mcp_servers from the loaded config, which has been
        through _expand_env_vars() — so ``${MCP_GH_API_KEY}`` is stored
        expanded.  The watcher re-parses the RAW yaml.  Without expanding the
        watcher side too, the comparison is always unequal whenever any
        template is in use, so EVERY config.yaml rewrite (e.g.
        save_config_value('agent.reasoning_effort', ...) from /reasoning)
        fired a full MCP reconnect.
        """
        import yaml
        monkeypatch.setenv("MCP_GH_API_KEY", "sekrit-token")

        raw_servers = {
            "github": {
                "url": "https://mcp.github.com",
                "headers": {"Authorization": "Bearer ${MCP_GH_API_KEY}"},
            }
        }
        expanded_servers = {
            "github": {
                "url": "https://mcp.github.com",
                "headers": {"Authorization": "Bearer sekrit-token"},
            }
        }
        # Init snapshot holds the EXPANDED form (as load_cli_config produces).
        obj, cfg_file = _make_cli(tmp_path, mcp_servers=expanded_servers)

        # Unrelated-key save: mcp_servers content identical (raw templates),
        # only reasoning_effort changed — mtime moves.
        cfg_file.write_text(yaml.dump({
            "agent": {"reasoning_effort": "high"},
            "mcp_servers": raw_servers,
        }))
        obj._config_mtime = 0.0

        with patch("hermes_cli.config.get_config_path", return_value=cfg_file):
            obj._check_config_mcp_changes()

        obj._reload_mcp.assert_not_called()
        assert "MCP server config changed" not in capsys.readouterr().out

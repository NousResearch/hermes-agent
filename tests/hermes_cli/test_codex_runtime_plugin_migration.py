"""Tests for the codex MCP plugin migration helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli.codex_runtime_plugin_migration import (
    MIGRATION_MARKER,
    MigrationReport,
    _format_toml_value,
    _strip_existing_managed_block,
    _translate_one_server,
    migrate,
    render_codex_toml_section,
)


# ---- per-server translation ----

class TestTranslateOneServer:
    def test_stdio_basic(self):
        cfg, skipped = _translate_one_server("filesystem", {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            "env": {"FOO": "bar"},
        })
        assert cfg == {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            "env": {"FOO": "bar"},
        }
        assert skipped == []

    def test_stdio_with_cwd(self):
        cfg, _ = _translate_one_server("custom", {
            "command": "/usr/bin/myserver",
            "cwd": "/var/lib/mcp",
        })
        assert cfg["cwd"] == "/var/lib/mcp"

    def test_http_basic(self):
        cfg, skipped = _translate_one_server("api", {
            "url": "https://x.example/mcp",
            "headers": {"Authorization": "Bearer abc"},
        })
        assert cfg == {
            "url": "https://x.example/mcp",
            "http_headers": {"Authorization": "Bearer abc"},
        }
        assert skipped == []

    def test_sse_falls_under_streamable_http_with_warning(self):
        cfg, skipped = _translate_one_server("sse_server", {
            "url": "http://localhost:8000/sse",
            "transport": "sse",
        })
        assert cfg["url"] == "http://localhost:8000/sse"
        assert any("sse" in s.lower() for s in skipped)

    def test_timeouts_translate(self):
        cfg, _ = _translate_one_server("x", {
            "command": "y",
            "timeout": 180,
            "connect_timeout": 30,
        })
        assert cfg["tool_timeout_sec"] == 180.0
        assert cfg["startup_timeout_sec"] == 30.0

    def test_non_numeric_timeout_skipped(self):
        cfg, skipped = _translate_one_server("x", {
            "command": "y",
            "timeout": "not-a-number",
        })
        assert "tool_timeout_sec" not in cfg
        assert any("timeout" in s and "numeric" in s for s in skipped)

    def test_disabled_server_emits_enabled_false(self):
        cfg, _ = _translate_one_server("x", {
            "command": "y",
            "enabled": False,
        })
        assert cfg["enabled"] is False

    def test_enabled_true_omitted(self):
        cfg, _ = _translate_one_server("x", {"command": "y", "enabled": True})
        assert "enabled" not in cfg  # codex defaults to true

    def test_command_and_url_prefers_stdio_warns(self):
        cfg, skipped = _translate_one_server("x", {
            "command": "y", "url": "http://z",
        })
        assert "command" in cfg
        assert "url" not in cfg
        assert any("url" in s for s in skipped)

    def test_no_transport_returns_none(self):
        cfg, skipped = _translate_one_server("broken", {"description": "x"})
        assert cfg is None
        assert "no command or url" in skipped[0]

    def test_sampling_dropped_with_warning(self):
        cfg, skipped = _translate_one_server("x", {
            "command": "y",
            "sampling": {"enabled": True, "model": "gemini-3-flash"},
        })
        assert "sampling" not in cfg
        assert any("sampling" in s for s in skipped)

    def test_unknown_keys_warned(self):
        cfg, skipped = _translate_one_server("x", {
            "command": "y",
            "totally_made_up_key": "value",
        })
        assert "totally_made_up_key" not in cfg
        assert any("totally_made_up_key" in s for s in skipped)

    def test_non_dict_input(self):
        cfg, skipped = _translate_one_server("x", "notadict")  # type: ignore[arg-type]
        assert cfg is None


# ---- TOML rendering ----

class TestTomlValueFormatter:
    def test_string_quoted(self):
        assert _format_toml_value("hello") == '"hello"'

    def test_string_with_quotes_escaped(self):
        assert _format_toml_value('a"b') == '"a\\"b"'

    def test_bool(self):
        assert _format_toml_value(True) == "true"
        assert _format_toml_value(False) == "false"

    def test_int(self):
        assert _format_toml_value(42) == "42"

    def test_float(self):
        assert _format_toml_value(180.0) == "180.0"

    def test_list_of_strings(self):
        assert _format_toml_value(["a", "b"]) == '["a", "b"]'

    def test_inline_table(self):
        out = _format_toml_value({"FOO": "bar"})
        assert out == '{ FOO = "bar" }'

    def test_empty_inline_table(self):
        assert _format_toml_value({}) == "{}"

    def test_string_with_newline_escaped(self):
        """TOML basic strings don't allow literal newlines — a path or
        env var containing a newline must use \\n. Otherwise codex would
        refuse to load the config."""
        out = _format_toml_value("line one\nline two")
        assert "\n" not in out  # no raw newline in output
        assert "\\n" in out

    def test_string_with_tab_escaped(self):
        out = _format_toml_value("col1\tcol2")
        assert "\t" not in out
        assert "\\t" in out

    def test_string_with_other_controls_escaped(self):
        for raw, expected in [
            ("\r", "\\r"),
            ("\f", "\\f"),
            ("\b", "\\b"),
        ]:
            out = _format_toml_value(f"x{raw}y")
            assert raw not in out, f"{raw!r} should be escaped"
            assert expected in out, f"{expected!r} should be in output"

    def test_windows_path_escaped_correctly(self):
        out = _format_toml_value(r"C:\Users\Alice\.codex")
        # Each backslash should be doubled
        assert out == r'"C:\\Users\\Alice\\.codex"'

    def test_atomic_write_no_temp_leak_on_success(self, tmp_path):
        """The atomic-write path uses tempfile.mkstemp + rename. On
        success the temp file should not be left behind."""
        migrate({"mcp_servers": {"x": {"command": "y"}}},
                codex_home=tmp_path,
                discover_plugins=False,
                expose_hermes_tools=False,
                default_permission_profile=None)
        # config.toml should exist
        assert (tmp_path / "config.toml").exists()
        # And no .config.toml.* temp files left behind
        leftover = [p.name for p in tmp_path.iterdir()
                    if p.name.startswith(".config.toml.")]
        assert leftover == [], f"temp file leaked after migration: {leftover}"

    def test_atomic_write_cleanup_on_rename_failure(self, tmp_path, monkeypatch):
        """If rename fails partway through (out of disk, permissions,
        crash), the temp file must be cleaned up. Otherwise repeated
        failed migrations would pile up .config.toml.* files."""
        from pathlib import Path as _Path
        original_replace = _Path.replace

        def failing_replace(self, target):
            raise OSError("simulated disk full")

        monkeypatch.setattr(_Path, "replace", failing_replace)
        report = migrate(
            {"mcp_servers": {"x": {"command": "y"}}},
            codex_home=tmp_path,
            discover_plugins=False,
            expose_hermes_tools=False,
            default_permission_profile=None,
        )
        # Error surfaced
        assert any("simulated disk full" in e for e in report.errors)
        # And no leaked temp file
        leftover = [p.name for p in tmp_path.iterdir()
                    if p.name.startswith(".config.toml.")]
        assert leftover == [], f"temp files leaked: {leftover}"

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError):
            _format_toml_value(object())


class TestRenderToml:
    def test_starts_with_marker(self):
        out = render_codex_toml_section({})
        assert out.startswith(MIGRATION_MARKER)

    def test_empty_servers_emits_placeholder(self):
        out = render_codex_toml_section({})
        assert "no MCP servers" in out

    def test_servers_sorted_alphabetically(self):
        out = render_codex_toml_section({
            "zoo": {"command": "z"},
            "alpha": {"command": "a"},
            "middle": {"command": "m"},
        })
        # Find the section header positions and confirm order
        a_pos = out.find("[mcp_servers.alpha]")
        m_pos = out.find("[mcp_servers.middle]")
        z_pos = out.find("[mcp_servers.zoo]")
        assert 0 < a_pos < m_pos < z_pos

    def test_server_with_args_and_env(self):
        out = render_codex_toml_section({
            "fs": {
                "command": "npx",
                "args": ["-y", "filesystem"],
                "env": {"PATH": "/usr/bin"},
            }
        })
        assert "[mcp_servers.fs]" in out
        assert 'command = "npx"' in out
        assert 'args = ["-y", "filesystem"]' in out
        # Env emitted as inline table
        assert 'env = { PATH = "/usr/bin" }' in out


# ---- existing-block stripping ----

class TestStripExistingManagedBlock:
    def test_no_managed_block_unchanged(self):
        text = "[other]\nfoo = 1\n"
        assert _strip_existing_managed_block(text) == text

    def test_strips_managed_block_alone(self):
        text = (
            f"{MIGRATION_MARKER}\n"
            "\n"
            "[mcp_servers.fs]\n"
            'command = "npx"\n'
        )
        assert _strip_existing_managed_block(text).strip() == ""

    def test_preserves_user_content_above_managed_block(self):
        text = (
            "[model]\n"
            'name = "gpt-5.5"\n'
            "\n"
            f"{MIGRATION_MARKER}\n"
            "[mcp_servers.fs]\n"
            'command = "x"\n'
        )
        out = _strip_existing_managed_block(text)
        assert "[model]" in out
        assert 'name = "gpt-5.5"' in out
        assert "mcp_servers.fs" not in out

    def test_preserves_unrelated_section_after_managed_block(self):
        text = (
            f"{MIGRATION_MARKER}\n"
            "[mcp_servers.fs]\n"
            'command = "x"\n'
            "\n"
            "[providers]\n"
            'foo = "bar"\n'
        )
        out = _strip_existing_managed_block(text)
        assert "mcp_servers.fs" not in out
        assert "[providers]" in out
        assert 'foo = "bar"' in out


# ---- end-to-end migrate(, expose_hermes_tools=False) ----

class TestMigrate:
    def test_no_servers_no_plugins_no_perms_writes_placeholder(self, tmp_path):
        report = migrate({}, codex_home=tmp_path,
                         discover_plugins=False,
                         default_permission_profile=None, expose_hermes_tools=False)
        assert report.written
        text = (tmp_path / "config.toml").read_text()
        assert MIGRATION_MARKER in text
        assert "no MCP servers" in text or "no MCP servers, plugins, or permissions" in text

    def test_no_servers_still_writes_permissions_default(self, tmp_path):
        """Even with zero MCP servers, enabling the runtime should write the
        default permissions profile so users don't get prompted on every
        write attempt. This is the fix for quirk #2."""
        report = migrate({}, codex_home=tmp_path, discover_plugins=False, expose_hermes_tools=False)
        assert report.written
        text = (tmp_path / "config.toml").read_text()
        # Codex's schema: top-level `default_permissions` keying a built-in
        # profile name (prefixed with ":"). NOT a [permissions] section
        # (which is for *user-defined* profiles with structured fields).
        assert 'default_permissions = ":workspace"' in text
        assert report.wrote_permissions_default == ":workspace"

    def test_explicit_none_permissions_skips_block(self, tmp_path):
        report = migrate({"mcp_servers": {"x": {"command": "y"}}},
                         codex_home=tmp_path,
                         discover_plugins=False,
                         default_permission_profile=None, expose_hermes_tools=False)
        text = (tmp_path / "config.toml").read_text()
        assert "default_permissions" not in text
        assert "[permissions]" not in text
        assert report.wrote_permissions_default is None

    def test_plugin_discovery_writes_plugin_blocks(self, tmp_path, monkeypatch):
        """Discovered curated plugins land as [plugins."<name>@<marketplace>"]
        blocks. This is what OpenClaw calls 'migrate native codex plugins.'"""
        from hermes_cli import codex_runtime_plugin_migration as crpm

        def fake_query(codex_home=None, timeout=8.0):
            return [
                {"name": "google-calendar", "marketplace": "openai-curated",
                 "enabled": True},
                {"name": "github", "marketplace": "openai-curated",
                 "enabled": True},
            ], None
        monkeypatch.setattr(crpm, "_query_codex_plugins", fake_query)

        report = migrate({}, codex_home=tmp_path, discover_plugins=True, expose_hermes_tools=False)
        text = (tmp_path / "config.toml").read_text()
        assert '[plugins."github@openai-curated"]' in text
        assert '[plugins."google-calendar@openai-curated"]' in text
        assert "enabled = true" in text
        assert "google-calendar@openai-curated" in report.migrated_plugins
        assert "github@openai-curated" in report.migrated_plugins

    def test_plugin_discovery_failure_non_fatal(self, tmp_path, monkeypatch):
        """If codex isn't installed or RPC fails, MCP migration still
        completes. The error surfaces in the report but doesn't abort."""
        from hermes_cli import codex_runtime_plugin_migration as crpm

        def fake_query_fails(codex_home=None, timeout=8.0):
            return [], "codex CLI not available"
        monkeypatch.setattr(crpm, "_query_codex_plugins", fake_query_fails)

        report = migrate({"mcp_servers": {"x": {"command": "y"}}},
                         codex_home=tmp_path, discover_plugins=True, expose_hermes_tools=False)
        assert report.written
        assert report.migrated == ["x"]
        assert report.plugin_query_error == "codex CLI not available"
        assert report.migrated_plugins == []

    def test_discover_plugins_false_skips_query(self, tmp_path, monkeypatch):
        """Tests and restricted environments can opt out of the subprocess
        spawn entirely."""
        from hermes_cli import codex_runtime_plugin_migration as crpm

        called = {"yes": False}
        def boom(*a, **kw):
            called["yes"] = True
            return [], None
        monkeypatch.setattr(crpm, "_query_codex_plugins", boom)

        migrate({"mcp_servers": {"x": {"command": "y"}}},
                codex_home=tmp_path, discover_plugins=False, expose_hermes_tools=False)
        assert called["yes"] is False

    def test_dry_run_skips_plugin_query(self, tmp_path, monkeypatch):
        """Dry run should never spawn codex. Even with discover_plugins=True
        the query is skipped because dry_run takes precedence."""
        from hermes_cli import codex_runtime_plugin_migration as crpm

        called = {"yes": False}
        def boom(*a, **kw):
            called["yes"] = True
            return [], None
        monkeypatch.setattr(crpm, "_query_codex_plugins", boom)

        migrate({"mcp_servers": {"x": {"command": "y"}}},
                codex_home=tmp_path, dry_run=True, discover_plugins=True, expose_hermes_tools=False)
        assert called["yes"] is False

    def test_re_run_replaces_plugin_block(self, tmp_path, monkeypatch):
        """Plugin blocks are managed and re-runs should replace them
        cleanly — same idempotency contract as MCP servers."""
        from hermes_cli import codex_runtime_plugin_migration as crpm

        # First run: only github
        monkeypatch.setattr(crpm, "_query_codex_plugins",
                            lambda codex_home=None, timeout=8.0: (
                                [{"name": "github", "marketplace": "openai-curated", "enabled": True}],
                                None,
                            ))
        migrate({}, codex_home=tmp_path, discover_plugins=True,
                default_permission_profile=None, expose_hermes_tools=False)
        first = (tmp_path / "config.toml").read_text()
        assert "github@openai-curated" in first

        # Second run: only canva (github went away)
        monkeypatch.setattr(crpm, "_query_codex_plugins",
                            lambda codex_home=None, timeout=8.0: (
                                [{"name": "canva", "marketplace": "openai-curated", "enabled": True}],
                                None,
                            ))
        migrate({}, codex_home=tmp_path, discover_plugins=True,
                default_permission_profile=None, expose_hermes_tools=False)
        second = (tmp_path / "config.toml").read_text()
        assert "github@openai-curated" not in second
        assert "canva@openai-curated" in second

    def test_expose_hermes_tools_writes_callback_mcp_entry(self, tmp_path):
        """When expose_hermes_tools=True (production default), an
        [mcp_servers.hermes-tools] entry is written so codex calls back
        into Hermes for browser/web/delegate_task/vision/memory tools.

        This is the fix for 'all other tools that codex doesn't provide
        should be useable by hermes' — quirk #7."""
        report = migrate({}, codex_home=tmp_path,
                         discover_plugins=False,
                         default_permission_profile=None,
                         expose_hermes_tools=True)
        text = (tmp_path / "config.toml").read_text()
        assert "[mcp_servers.hermes-tools]" in text
        assert "hermes_tools_mcp_server" in text
        # Must include startup + tool timeouts so codex doesn't give up
        assert "startup_timeout_sec" in text
        assert "tool_timeout_sec" in text
        # And the entry is reported
        assert "hermes-tools" in report.migrated

    def test_expose_hermes_tools_disabled_skips_entry(self, tmp_path):
        """expose_hermes_tools=False suppresses the callback registration."""
        migrate({}, codex_home=tmp_path,
                discover_plugins=False,
                default_permission_profile=None,
                expose_hermes_tools=False)
        text = (tmp_path / "config.toml").read_text()
        assert "[mcp_servers.hermes-tools]" not in text
        assert "hermes_tools_mcp_server" not in text

    def test_dry_run_doesnt_write(self, tmp_path):
        report = migrate({"mcp_servers": {"x": {"command": "y"}}},
                         codex_home=tmp_path, dry_run=True, expose_hermes_tools=False)
        assert report.dry_run is True
        assert not (tmp_path / "config.toml").exists()
        assert "x" in report.migrated

    def test_full_migration_round_trip(self, tmp_path):
        hermes_cfg = {
            "mcp_servers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem"],
                },
                "github": {
                    "url": "https://api.github.com/mcp",
                    "headers": {"Authorization": "Bearer x"},
                },
            }
        }
        report = migrate(hermes_cfg, codex_home=tmp_path, expose_hermes_tools=False)
        assert report.written
        text = (tmp_path / "config.toml").read_text()
        assert "[mcp_servers.filesystem]" in text
        assert "[mcp_servers.github]" in text
        assert 'command = "npx"' in text
        assert 'url = "https://api.github.com/mcp"' in text

    def test_idempotent_re_run_replaces_managed_block(self, tmp_path):
        # First migration
        migrate({"mcp_servers": {"a": {"command": "x"}}}, codex_home=tmp_path, expose_hermes_tools=False)
        first_text = (tmp_path / "config.toml").read_text()
        assert "[mcp_servers.a]" in first_text
        # Second migration with different servers
        migrate({"mcp_servers": {"b": {"command": "y"}}}, codex_home=tmp_path, expose_hermes_tools=False)
        second_text = (tmp_path / "config.toml").read_text()
        assert "[mcp_servers.a]" not in second_text
        assert "[mcp_servers.b]" in second_text

    def test_preserves_user_codex_config_above_marker(self, tmp_path):
        target = tmp_path / "config.toml"
        target.write_text(
            "[model]\n"
            'profile = "default"\n'
            "\n"
            "[providers.openai]\n"
            'api_key = "sk-test"\n'
        )
        migrate({"mcp_servers": {"a": {"command": "x"}}}, codex_home=tmp_path, expose_hermes_tools=False)
        new_text = target.read_text()
        # User's codex config preserved
        assert "[model]" in new_text
        assert 'profile = "default"' in new_text
        assert "[providers.openai]" in new_text
        # And new MCP block appended
        assert "[mcp_servers.a]" in new_text
        assert MIGRATION_MARKER in new_text

    def test_preserves_user_mcp_server_outside_managed_block(self, tmp_path):
        """Quirk #6: when a user adds their own MCP server entry directly
        to ~/.codex/config.toml outside Hermes' managed block, re-running
        migration must preserve it. Tested both above and below the
        managed block."""
        target = tmp_path / "config.toml"
        target.write_text(
            "[mcp_servers.user-above]\n"
            'command = "/usr/bin/above-server"\n'
            'args = ["--above"]\n'
        )
        # First migrate — adds managed block below user content
        migrate({"mcp_servers": {"hermes-mcp": {"command": "npx"}}},
                codex_home=tmp_path, discover_plugins=False,
                expose_hermes_tools=False)
        text = target.read_text()
        assert "user-above" in text, "user MCP server above managed block got nuked"
        assert 'command = "/usr/bin/above-server"' in text

        # Append another user entry below the managed block
        target.write_text(
            text + "\n[mcp_servers.user-below]\ncommand = \"below-server\"\n"
        )
        # Re-migrate — both should survive
        migrate({"mcp_servers": {"hermes-mcp": {"command": "npx"}}},
                codex_home=tmp_path, discover_plugins=False,
                expose_hermes_tools=False)
        final = target.read_text()
        assert "user-above" in final
        assert "user-below" in final
        # And our managed block is still there with the new content
        assert "[mcp_servers.hermes-mcp]" in final

    def test_skipped_keys_reported(self, tmp_path):
        report = migrate({
            "mcp_servers": {
                "x": {
                    "command": "y",
                    "sampling": {"enabled": True},  # codex has no equivalent
                }
            }
        }, codex_home=tmp_path, expose_hermes_tools=False)
        assert "x" in report.skipped_keys_per_server
        assert any("sampling" in s for s in report.skipped_keys_per_server["x"])

    def test_invalid_mcp_servers_value(self, tmp_path):
        report = migrate({"mcp_servers": "notadict"}, codex_home=tmp_path, expose_hermes_tools=False)
        assert any("not a dict" in e for e in report.errors)

    def test_server_without_transport_skipped_with_error(self, tmp_path):
        report = migrate({
            "mcp_servers": {"broken": {"description": "no command/url"}}
        }, codex_home=tmp_path, expose_hermes_tools=False)
        assert "broken" not in report.migrated
        assert any("broken" in e for e in report.errors)

    def test_summary_reports_migration_count(self, tmp_path):
        report = migrate({
            "mcp_servers": {"a": {"command": "x"}, "b": {"command": "y"}}
        }, codex_home=tmp_path, expose_hermes_tools=False)
        summary = report.summary()
        assert "Migrated 2 MCP server(s)" in summary
        assert "- a" in summary
        assert "- b" in summary

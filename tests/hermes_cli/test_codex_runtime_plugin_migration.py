"""Tests for the codex MCP plugin migration helper."""

from __future__ import annotations



from hermes_cli.codex_runtime_plugin_migration import (
    MIGRATION_MARKER,
    MIGRATION_END_MARKER,
    _build_hermes_tools_mcp_entry,
    _strip_existing_managed_block,
    _strip_unmanaged_plugin_tables,
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




    def test_unknown_keys_warned(self):
        cfg, skipped = _translate_one_server("x", {
            "command": "y",
            "totally_made_up_key": "value",
        })
        assert "totally_made_up_key" not in cfg
        assert any("totally_made_up_key" in s for s in skipped)


# ---- TOML rendering ----

class TestTomlValueFormatter:







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
        import utils

        def failing_replace(tmp, target):
            raise OSError("simulated disk full")

        monkeypatch.setattr(utils, "atomic_replace", failing_replace)
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



class TestRenderToml:


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


# ---- end-to-end migrate(, expose_hermes_tools=False) ----

class TestMigrate:



    def test_plugin_discovery_writes_plugin_blocks(self, tmp_path, monkeypatch):
        """Discovered curated plugins land as [plugins."<name>@<marketplace>"]
        blocks. This is what OpenClaw calls 'migrate native codex plugins.'
        The discovery spawn must use the configured ``model.codex_bin`` (#61360)."""
        from hermes_cli import codex_runtime_plugin_migration as crpm

        seen: dict = {}

        def fake_query(codex_home=None, timeout=8.0, codex_bin="codex"):
            seen["codex_bin"] = codex_bin
            return [
                {"name": "google-calendar", "marketplace": "openai-curated",
                 "enabled": True},
                {"name": "github", "marketplace": "openai-curated",
                 "enabled": True},
            ], None
        monkeypatch.setattr(crpm, "_query_codex_plugins", fake_query)

        report = migrate({"model": {"codex_bin": "/opt/codex-app/codex"}},
                         codex_home=tmp_path, discover_plugins=True)
        assert seen["codex_bin"] == "/opt/codex-app/codex"
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

        def fake_query_fails(codex_home=None, timeout=8.0, codex_bin="codex"):
            return [], "codex CLI not available"
        monkeypatch.setattr(crpm, "_query_codex_plugins", fake_query_fails)

        report = migrate({"mcp_servers": {"x": {"command": "y"}}},
                         codex_home=tmp_path, discover_plugins=True, expose_hermes_tools=False)
        assert report.written
        assert report.migrated == ["x"]
        assert report.plugin_query_error == "codex CLI not available"
        assert report.migrated_plugins == []







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





# ---- Bug B: duplicate [plugins.X] tables ----


class TestStripUnmanagedPluginTables:
    """Regression tests for issue #26250 Bug B.

    When codex itself writes ``[plugins."<name>@<marketplace>"]`` tables
    (via the user running ``codex plugins enable`` directly), re-running
    ``hermes codex-runtime migrate`` would re-emit them inside the managed
    block and the resulting duplicate-table-header would crash codex.
    """

    def test_strips_plugin_tables_outside_managed_block(self):
        text = (
            'model = "gpt-5.5"\n'
            "\n"
            "[mcp_servers.user-thing]\n"
            'command = "x"\n'
            "\n"
            '[plugins."tasks@openai-curated"]\n'
            "enabled = true\n"
            "\n"
            '[plugins."web-search@openai-curated"]\n'
            "enabled = true\n"
            "\n"
            "[features]\n"
            "terminal_resize_reflow = true\n"
        )
        stripped = _strip_unmanaged_plugin_tables(text)
        assert "[plugins." not in stripped
        # Non-plugin content preserved
        assert "[mcp_servers.user-thing]" in stripped
        assert "[features]" in stripped
        assert "terminal_resize_reflow = true" in stripped


    def test_multi_line_array_in_plugin_table_does_not_leak(self):
        """A multi-line TOML array inside a [plugins.X] table whose
        continuation lines start with ``[`` (e.g. nested arrays) must NOT
        prematurely exit the strip region — otherwise array fragments
        leak into top-level output and produce invalid TOML on the next
        codex startup. Regression guard for #26260 review.
        """
        text = (
            '[plugins."tasks@openai-curated"]\n'
            "allowed = [\n"
            '  "a",\n'
            '  ["nested"],\n'
            "]\n"
            "[features]\n"
            "x = 1\n"
        )
        stripped = _strip_unmanaged_plugin_tables(text)
        # Everything inside the plugin table — including the multi-line
        # array's continuation lines starting with `[` — should be gone.
        assert '["nested"]' not in stripped
        assert "allowed" not in stripped
        # Sibling user table survives intact.
        assert "[features]" in stripped
        assert "x = 1" in stripped
        # Result is still valid TOML.
        import tomllib
        tomllib.loads(stripped)

    def test_migrate_dedups_codex_owned_plugin_tables(self, tmp_path, monkeypatch):
        """End-to-end: codex's pre-existing [plugins.X] tables get replaced by
        the managed block's re-emission rather than duplicated."""
        target = tmp_path / "config.toml"
        target.write_text(
            "[mcp_servers.user-server]\n"
            'command = "x"\n'
            "\n"
            '[plugins."tasks@openai-curated"]\n'
            "enabled = true\n"
        )

        # Simulate codex's plugin/list reporting the same plugin tasks@openai-curated.
        def fake_query(codex_home=None, timeout=8.0, codex_bin="codex"):
            return (
                [{"name": "tasks", "marketplace": "openai-curated", "enabled": True}],
                None,
            )

        monkeypatch.setattr(
            "hermes_cli.codex_runtime_plugin_migration._query_codex_plugins",
            fake_query,
        )
        migrate({}, codex_home=tmp_path, discover_plugins=True, expose_hermes_tools=False)
        new_text = target.read_text()
        # Only ONE [plugins."tasks@openai-curated"] header should remain — inside
        # the managed block — not the original outside-the-block copy.
        assert new_text.count('[plugins."tasks@openai-curated"]') == 1
        # And the surviving one is inside our managed section.
        managed_start = new_text.index(MIGRATION_MARKER)
        managed_end = new_text.index(MIGRATION_END_MARKER)
        plugin_idx = new_text.index('[plugins."tasks@openai-curated"]')
        assert managed_start < plugin_idx < managed_end
        # File parses cleanly as TOML (the original duplicate-key error is gone).
        import tomllib
        tomllib.loads(new_text)


# ---- Bug C: HERMES_HOME tempdir leak into ~/.codex/config.toml ----


class TestHermesHomeLeakGuard:
    """Regression tests for issue #26250 Bug C.

    Previously ``_build_hermes_tools_mcp_entry()`` read ``HERMES_HOME``
    directly from ``os.environ``, so a pytest ``monkeypatch.setenv`` would
    leak a transient tempdir path into the user's real ``~/.codex/config.toml``
    once codex spawned the hermes-tools MCP subprocess.
    """




    def test_real_hermes_home_propagates(self, monkeypatch, tmp_path):
        """A legitimate HERMES_HOME (not a tempdir path) DOES propagate so the
        MCP subprocess sees the same config as the parent CLI."""
        # Use a path that looks real — under /Users or /home, not /var/folders.
        # We can't easily create one in the test, so just use a stable path
        # outside any tempdir-detector needle. The detector checks for tempdir
        # markers, not for path existence.
        real_path = "/Users/alice/.hermes"
        monkeypatch.setenv("HERMES_HOME", real_path)
        entry = _build_hermes_tools_mcp_entry()
        env = entry.get("env", {})
        assert env.get("HERMES_HOME") == real_path

    def test_unset_hermes_home_omits_env_key(self, monkeypatch):
        """When HERMES_HOME is unset in the environment, the MCP entry MUST
        NOT bake in a resolved-default path. The codex subprocess should
        inherit whatever HERMES_HOME its launcher (systemd, gateway, shell)
        sets at runtime, rather than being pinned to migrate-time defaults.
        Regression guard for issue #26250 follow-up review."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        entry = _build_hermes_tools_mcp_entry()
        env = entry.get("env", {})
        assert "HERMES_HOME" not in env, (
            f"HERMES_HOME should not be set when env var is unset, got: "
            f"{env.get('HERMES_HOME')!r}"
        )


# ---- same-name user-owned [mcp_servers.X] tables (issue #79023) ----


class TestSameNameUserMcpTable:
    """Issue #79023: a Hermes server whose name the user already declares outside the managed
    block must not be emitted twice (duplicate table header = TOML codex refuses to load)."""

    def test_user_table_wins_and_output_stays_valid_toml(self, tmp_path):
        import tomllib

        target = tmp_path / "config.toml"
        target.write_text('[mcp_servers.gbrain]\ncommand = "existing-gbrain"\n', encoding="utf-8")
        report = migrate(
            {"mcp_servers": {"gbrain": {"command": "projected-gbrain"}, "other": {"command": "o"}}},
            codex_home=tmp_path, discover_plugins=False, expose_hermes_tools=False,
            default_permission_profile=None)
        text = target.read_text(encoding="utf-8")
        parsed = tomllib.loads(text)  # would raise "Cannot declare ... twice" before the fix
        assert text.count("[mcp_servers.gbrain]") == 1
        assert parsed["mcp_servers"]["gbrain"]["command"] == "existing-gbrain"
        assert "other" in parsed["mcp_servers"]
        assert report.preserved_user_servers == ["gbrain"]
        assert report.migrated == ["other"]
        assert "gbrain" in report.summary()

    def test_inline_table_user_server_is_preserved(self, tmp_path):
        """User declarations in other valid TOML shapes (`[mcp_servers]` + inline table) are
        theirs too: skip the projection instead of refusing to write on a duplicate key."""
        import tomllib

        target = tmp_path / "config.toml"
        target.write_text('[mcp_servers]\ngbrain = { command = "existing-gbrain" }\n', encoding="utf-8")
        report = migrate(
            {"mcp_servers": {"gbrain": {"command": "projected-gbrain"}, "other": {"command": "o"}}},
            codex_home=tmp_path, discover_plugins=False, expose_hermes_tools=False,
            default_permission_profile=None)
        parsed = tomllib.loads(target.read_text(encoding="utf-8"))
        assert report.written and report.errors == []
        assert report.preserved_user_servers == ["gbrain"]
        assert parsed["mcp_servers"]["gbrain"]["command"] == "existing-gbrain"
        assert parsed["mcp_servers"]["other"]["command"] == "o"


    def test_cli_migrate_dry_run_json_reports_without_writing(self, tmp_path, monkeypatch, capsys):
        """`hermes codex-runtime migrate --dry-run --json` is the supported automation seam:
        drive it through the real ``hermes`` argparse tree so the subcommand registration in
        hermes_cli/main.py stays pinned, and honour ``CODEX_HOME`` like every codex sibling."""
        import json

        import hermes_cli.main as main

        codex_home = tmp_path / "alt-codex"
        codex_home.mkdir()
        target = codex_home / "config.toml"
        target.write_text('[mcp_servers.gbrain]\ncommand = "existing-gbrain"\n', encoding="utf-8")
        monkeypatch.setenv("CODEX_HOME", str(codex_home))
        monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"mcp_servers": {"gbrain": {"command": "projected"}, "other": {"command": "o"}}})
        parser, _subparsers = main._build_cli_parser()
        args = parser.parse_args(["codex-runtime", "migrate", "--dry-run", "--json"])
        rc = args.func(args)
        payload = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert payload["dry_run"] is True and payload["written"] is False
        assert payload["preserved_user_servers"] == ["gbrain"]
        assert payload["target_path"] == str(target)
        assert target.read_text(encoding="utf-8") == '[mcp_servers.gbrain]\ncommand = "existing-gbrain"\n'

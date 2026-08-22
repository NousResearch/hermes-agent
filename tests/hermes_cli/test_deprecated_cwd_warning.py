"""Tests for warn_deprecated_cwd_env_vars() migration warning."""

import types


def _write_env(monkeypatch, tmp_path, content):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / ".env").write_text(content, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


class TestDeprecatedCwdWarning:
    """Warn when MESSAGING_CWD or TERMINAL_CWD is set in .env."""

    def test_process_environment_does_not_trigger_warning(
        self, monkeypatch, tmp_path, capsys
    ):
        _write_env(monkeypatch, tmp_path, "# TERMINAL_CWD=.\n")
        monkeypatch.setenv("MESSAGING_CWD", "/process/message-path")
        monkeypatch.setenv("TERMINAL_CWD", "/process/terminal-path")

        from hermes_cli.config import warn_deprecated_cwd_env_vars

        warn_deprecated_cwd_env_vars()

        assert capsys.readouterr().err == ""

    def test_both_deprecated_vars_in_dotenv_warn(
        self, monkeypatch, tmp_path, capsys
    ):
        _write_env(
            monkeypatch,
            tmp_path,
            "MESSAGING_CWD=/msg/path\nTERMINAL_CWD=/term/path\n",
        )
        monkeypatch.delenv("MESSAGING_CWD", raising=False)
        monkeypatch.delenv("TERMINAL_CWD", raising=False)

        from hermes_cli.config import warn_deprecated_cwd_env_vars

        warn_deprecated_cwd_env_vars()

        captured = capsys.readouterr()
        assert "MESSAGING_CWD" in captured.err
        assert "TERMINAL_CWD" in captured.err
        assert "deprecated" in captured.err.lower()
        assert "config.yaml" in captured.err

    def test_dotenv_terminal_cwd_warns_with_explicit_config(
        self, monkeypatch, tmp_path, capsys
    ):
        hermes_home = _write_env(
            monkeypatch, tmp_path, "TERMINAL_CWD=/legacy/path\n"
        )
        (hermes_home / "config.yaml").write_text(
            "terminal:\n  cwd: /current/path\n", encoding="utf-8"
        )

        from hermes_cli.config import warn_deprecated_cwd_env_vars

        warn_deprecated_cwd_env_vars()

        assert "TERMINAL_CWD=/legacy/path" in capsys.readouterr().err

    def test_commented_and_empty_dotenv_values_do_not_warn(
        self, monkeypatch, tmp_path, capsys
    ):
        _write_env(
            monkeypatch,
            tmp_path,
            "# MESSAGING_CWD=/commented\nTERMINAL_CWD=\n",
        )
        monkeypatch.setenv("TERMINAL_CWD", "/process/bridge")

        from hermes_cli.config import warn_deprecated_cwd_env_vars

        warn_deprecated_cwd_env_vars()

        assert capsys.readouterr().err == ""

    def test_dotenv_read_failure_is_silent(self, monkeypatch, capsys):
        import hermes_cli.config as config_module

        def raise_read_error():
            raise OSError("permission denied")

        monkeypatch.setattr(config_module, "load_env", raise_read_error)

        config_module.warn_deprecated_cwd_env_vars()

        assert capsys.readouterr().err == ""

    def test_migration_hint_uses_real_newlines_not_literal_backslash_n(
        self, monkeypatch, tmp_path, capsys
    ):
        """The YAML hint must be readable, not a one-liner with a literal \\n.

        The hint is a copy-paste snippet; emitting the two-character sequence
        backslash-n inside it makes the suggested YAML invalid if pasted.
        """
        _write_env(monkeypatch, tmp_path, "MESSAGING_CWD=/msg/path\n")
        monkeypatch.delenv("MESSAGING_CWD", raising=False)

        from hermes_cli.config import warn_deprecated_cwd_env_vars

        warn_deprecated_cwd_env_vars()

        err = capsys.readouterr().err
        assert "\\n" not in err
        # terminal: and cwd: land on separate real lines.
        lines = [ln.strip() for ln in err.splitlines()]
        assert "\033[2mterminal:\033[0m" in lines or any(
            ln.endswith("terminal:\033[0m") for ln in lines
        )
        assert any("cwd: /your/project/path" in ln for ln in lines)


class TestStartupWarningAnsiRouting:
    """ANSI escapes must survive prompt_toolkit's patch_stdout StdoutProxy.

    prompt_toolkit's ``Vt100_Output.write()`` does ``data.replace("\\x1b", "?")``,
    so a raw ``sys.stderr.write`` of colored text inside the live TUI surfaces
    as garbled ``?[33m…?[0m``.  When an Application is running the warning must
    be routed through ``cli._cprint`` instead (#2262).
    """

    def _fake_tui(self, monkeypatch, recorded):
        """Pretend a prompt_toolkit Application is running with cli imported."""
        import sys
        import types

        import prompt_toolkit.application.current as pt_current

        monkeypatch.setattr(pt_current, "get_app_or_none", lambda: object())

        fake_cli = types.ModuleType("cli")
        fake_cli._cprint = recorded.append
        monkeypatch.setitem(sys.modules, "cli", fake_cli)

    def test_deprecation_warning_routes_through_cprint_in_tui(
        self, monkeypatch, tmp_path, capsys
    ):
        _write_env(monkeypatch, tmp_path, "MESSAGING_CWD=/msg/path\n")
        monkeypatch.delenv("MESSAGING_CWD", raising=False)

        recorded: list = []
        self._fake_tui(monkeypatch, recorded)

        from hermes_cli.config import warn_deprecated_cwd_env_vars

        warn_deprecated_cwd_env_vars()

        assert recorded, "warning was not routed through _cprint"
        payload = "".join(recorded)
        assert "\033[33m" in payload, "real ESC must reach prompt_toolkit"
        assert "MESSAGING_CWD=/msg/path" in payload
        # Nothing may bypass the renderer and hit stderr raw.
        assert capsys.readouterr().err == ""

    def test_config_warnings_route_through_cprint_in_tui(
        self, monkeypatch, capsys
    ):
        import hermes_cli.config as config_module

        recorded: list = []
        self._fake_tui(monkeypatch, recorded)

        issue = types.SimpleNamespace(
            severity="error", message="providers.bogus is not a mapping"
        )
        monkeypatch.setattr(
            config_module, "validate_config_structure", lambda cfg=None: [issue]
        )

        config_module.print_config_warnings()

        assert recorded, "config warning was not routed through _cprint"
        payload = "".join(recorded)
        assert "\033[31m" in payload
        assert "providers.bogus is not a mapping" in payload
        assert capsys.readouterr().err == ""

    def test_headless_still_writes_plain_stderr(
        self, monkeypatch, tmp_path, capsys
    ):
        """No Application running -> keep the plain stderr path (gateway boot)."""
        import sys

        _write_env(monkeypatch, tmp_path, "MESSAGING_CWD=/msg/path\n")
        monkeypatch.delenv("MESSAGING_CWD", raising=False)
        monkeypatch.delitem(sys.modules, "cli", raising=False)

        import prompt_toolkit.application.current as pt_current

        monkeypatch.setattr(pt_current, "get_app_or_none", lambda: None)

        from hermes_cli.config import warn_deprecated_cwd_env_vars

        warn_deprecated_cwd_env_vars()

        assert "MESSAGING_CWD=/msg/path" in capsys.readouterr().err

    def test_tui_detection_never_imports_cli(self, monkeypatch, tmp_path, capsys):
        """A headless gateway must not drag the 1MB cli module into memory."""
        import builtins
        import sys

        _write_env(monkeypatch, tmp_path, "MESSAGING_CWD=/msg/path\n")
        monkeypatch.delenv("MESSAGING_CWD", raising=False)
        monkeypatch.delitem(sys.modules, "cli", raising=False)

        real_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name == "cli" or name.startswith("cli."):
                raise AssertionError(f"headless path imported {name!r}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guard)

        from hermes_cli.config import warn_deprecated_cwd_env_vars

        warn_deprecated_cwd_env_vars()

        assert "MESSAGING_CWD=/msg/path" in capsys.readouterr().err

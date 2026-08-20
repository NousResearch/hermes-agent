"""Tests for warn_deprecated_cwd_env_vars() migration warning."""


class TestDeprecatedCwdWarning:
    """Warn when MESSAGING_CWD or TERMINAL_CWD is set in .env."""

    def test_messaging_cwd_triggers_warning(self, monkeypatch, capsys, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MESSAGING_CWD=/some/path\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("MESSAGING_CWD", "/some/path")
        monkeypatch.delenv("TERMINAL_CWD", raising=False)

        from hermes_cli.config import warn_deprecated_cwd_env_vars
        warn_deprecated_cwd_env_vars(config={})

        captured = capsys.readouterr()
        assert "MESSAGING_CWD" in captured.err
        assert "deprecated" in captured.err.lower()
        assert "config.yaml" in captured.err

    def test_both_deprecated_vars_warn(self, monkeypatch, capsys, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MESSAGING_CWD=/msg/path\nTERMINAL_CWD=/term/path\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("MESSAGING_CWD", "/msg/path")
        monkeypatch.setenv("TERMINAL_CWD", "/term/path")

        from hermes_cli.config import warn_deprecated_cwd_env_vars
        warn_deprecated_cwd_env_vars(config={})

        captured = capsys.readouterr()
        assert "MESSAGING_CWD" in captured.err
        assert "TERMINAL_CWD" in captured.err

    def test_no_warning_when_dotenv_does_not_contain_terminal_cwd(self, monkeypatch, capsys, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=sk-test\n# TERMINAL_CWD=.\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TERMINAL_CWD", "/home/user")
        monkeypatch.delenv("MESSAGING_CWD", raising=False)

        from hermes_cli.config import warn_deprecated_cwd_env_vars
        warn_deprecated_cwd_env_vars(config={"terminal": {"cwd": "."}})

        captured = capsys.readouterr()
        assert "TERMINAL_CWD" not in captured.err
        assert "Deprecated .env settings" not in captured.err

    def test_no_warning_when_dotenv_file_missing(self, monkeypatch, capsys, tmp_path):
        # tmp_path has no .env file
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TERMINAL_CWD", "/home/user")
        monkeypatch.setenv("MESSAGING_CWD", "/msg/path")

        from hermes_cli.config import warn_deprecated_cwd_env_vars
        warn_deprecated_cwd_env_vars(config={"terminal": {"cwd": "."}})

        captured = capsys.readouterr()
        assert "TERMINAL_CWD" not in captured.err
        assert "MESSAGING_CWD" not in captured.err
        assert "Deprecated .env settings" not in captured.err

    def test_warning_when_dotenv_contains_terminal_cwd(self, monkeypatch, capsys, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("TERMINAL_CWD=/some/legacy/path\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TERMINAL_CWD", "/some/legacy/path")
        monkeypatch.delenv("MESSAGING_CWD", raising=False)

        from hermes_cli.config import warn_deprecated_cwd_env_vars
        warn_deprecated_cwd_env_vars(config={"terminal": {"cwd": "."}})

        captured = capsys.readouterr()
        assert "TERMINAL_CWD" in captured.err
        assert "Deprecated .env settings" in captured.err

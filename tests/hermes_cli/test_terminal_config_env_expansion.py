"""Tests for apply_terminal_config_to_env — terminal.cwd ${VAR} expansion.

Regression test for #101659.
"""

import pytest
from hermes_cli.config import apply_terminal_config_to_env


class TestTerminalConfigEnvExpansion:
    def _stub_read_raw(self, monkeypatch, raw):
        """Stub read_raw_config() to control should_override."""
        import hermes_cli.config as _cfg
        monkeypatch.setattr(_cfg, "read_raw_config", lambda: raw)

    def test_cwd_expands_env_var_reference(self, monkeypatch, tmp_path):
        self._stub_read_raw(monkeypatch, {"terminal": {"cwd": "${DEFAULT_CWD}"}})

        cwd_target = str(tmp_path / "workspace")
        monkeypatch.setenv("DEFAULT_CWD", cwd_target)

        env = {}
        apply_terminal_config_to_env(env=env, config={"terminal": {"cwd": "${DEFAULT_CWD}"}})

        assert env.get("TERMINAL_CWD") == cwd_target

    def test_cwd_unresolved_var_kept_verbatim(self, monkeypatch, tmp_path):
        """Unresolved ${VAR} references are kept as-is, surfacing the same
        runtime_cwd warning as before the fix (no silent path corruption)."""
        self._stub_read_raw(monkeypatch, {"terminal": {"cwd": "${DEFINITELY_NOT_SET_12345}"}})
        monkeypatch.delenv("DEFINITELY_NOT_SET_12345", raising=False)

        env = {}
        apply_terminal_config_to_env(
            env=env,
            config={"terminal": {"cwd": "${DEFINITELY_NOT_SET_12345}"}},
        )

        assert env.get("TERMINAL_CWD") == "${DEFINITELY_NOT_SET_12345}"

    def test_cwd_absolute_path_unchanged(self, monkeypatch, tmp_path):
        self._stub_read_raw(monkeypatch, {"terminal": {"cwd": "/var/log"}})

        env = {}
        apply_terminal_config_to_env(env=env, config={"terminal": {"cwd": "/var/log"}})

        assert env.get("TERMINAL_CWD") == "/var/log"

    def test_cwd_tilde_expanded(self, monkeypatch, tmp_path):
        """~ still expands to home directory."""
        self._stub_read_raw(monkeypatch, {"terminal": {"cwd": "~"}})

        env = {}
        apply_terminal_config_to_env(env=env, config={"terminal": {"cwd": "~"}})

        # home dir varies; just assert it's not literal "~"
        assert env.get("TERMINAL_CWD") != "~"
        assert env.get("TERMINAL_CWD") is not None

    def test_cwd_auto_placeholder_skipped(self, monkeypatch, tmp_path):
        for placeholder in (".", "auto", "cwd"):
            self._stub_read_raw(monkeypatch, {"terminal": {"cwd": placeholder}})

            env = {}
            apply_terminal_config_to_env(env=env, config={"terminal": {"cwd": placeholder}})

            assert "TERMINAL_CWD" not in env, f"placeholder '{placeholder}' should not set TERMINAL_CWD"
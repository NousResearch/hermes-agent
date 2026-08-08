"""Regression tests for `hermes memory setup` required-field validation.

Pins the contract from issue #30821: when a schema field is marked
``required: True``, the wizard MUST NOT silently accept blank input and
write ``memory.provider`` anyway.

Covers:
- `_prompt_field` behavior in isolation (default/required/non-required).
- `cmd_setup` integration — the wizard aborts without persisting
  ``memory.provider`` when the user keeps hitting Enter on a required
  field with no default.
"""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest

from hermes_cli import memory_setup


def _feed_stdin(monkeypatch, lines):
    """Replace stdin with a StringIO of the given lines."""
    monkeypatch.setattr(memory_setup.sys, "stdin", io.StringIO("\n".join(lines) + "\n"))


# ── _prompt_field unit tests ───────────────────────────────────────────────

class TestPromptField:
    def test_returns_typed_value_for_optional_field(self, monkeypatch, capsys):
        _feed_stdin(monkeypatch, ["my-value"])
        val, aborted = memory_setup._prompt_field({"key": "x"}, "Label")
        assert (val, aborted) == ("my-value", False)

    def test_returns_blank_for_optional_blank_input(self, monkeypatch, capsys):
        _feed_stdin(monkeypatch, [""])
        val, aborted = memory_setup._prompt_field({"key": "x"}, "Label")
        assert (val, aborted) == ("", False)

    def test_required_blank_then_value_returns_value(self, monkeypatch, capsys):
        _feed_stdin(monkeypatch, ["", "second-try"])
        val, aborted = memory_setup._prompt_field(
            {"key": "api_key", "required": True}, "API key"
        )
        assert (val, aborted) == ("second-try", False)
        out = capsys.readouterr().out
        assert "'api_key' is required" in out

    def test_required_all_blank_aborts(self, monkeypatch, capsys):
        _feed_stdin(monkeypatch, [""] * (memory_setup._REQUIRED_RETRIES + 1))
        val, aborted = memory_setup._prompt_field(
            {"key": "api_key", "required": True}, "API key"
        )
        assert (val, aborted) == ("", True)
        out = capsys.readouterr().out
        assert "Missing required field 'api_key'" in out
        assert "setup aborted" in out

    def test_required_with_default_accepts_blank(self, monkeypatch, capsys):
        _feed_stdin(monkeypatch, [""])
        val, aborted = memory_setup._prompt_field(
            {"key": "base_url", "required": True}, "Base URL", default="https://x"
        )
        assert (val, aborted) == ("https://x", False)


# ── cmd_setup integration ──────────────────────────────────────────────────

class _FakeProvider:
    """Minimal stand-in for a memory provider plugin."""

    def __init__(self, schema):
        self._schema = schema
        self.saved = None

    def get_config_schema(self):
        return list(self._schema)

    def save_config(self, cfg, hermes_home):
        self.saved = (dict(cfg), hermes_home)


@pytest.fixture
def setup_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir(parents=True, exist_ok=True)
    state = {"loaded": {}, "saved": None}

    def fake_load_config():
        return state["loaded"]

    def fake_save_config(c):
        state["saved"] = dict(c)

    monkeypatch.setattr("hermes_cli.config.load_config", fake_load_config)
    monkeypatch.setattr("hermes_cli.config.save_config", fake_save_config)
    monkeypatch.setattr(memory_setup, "_install_dependencies", lambda *_a, **_kw: None)
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *_a, **_kw: 0)
    return state


class TestCmdSetupRequired:
    def test_blank_required_field_aborts_without_writing_provider(
        self, setup_env, monkeypatch, capsys
    ):
        provider = _FakeProvider(
            [{"key": "DB_PATH", "description": "path", "required": True}]
        )
        monkeypatch.setattr(
            memory_setup,
            "_get_available_providers",
            lambda: [("fake", "local", provider)],
        )
        _feed_stdin(monkeypatch, [""] * (memory_setup._REQUIRED_RETRIES + 1))

        memory_setup.cmd_setup(args=None)

        assert setup_env["saved"] is None
        assert provider.saved is None
        out = capsys.readouterr().out
        assert "Missing required field 'DB_PATH'" in out

    def test_filled_required_field_writes_provider(
        self, setup_env, monkeypatch, capsys
    ):
        provider = _FakeProvider(
            [{"key": "DB_PATH", "description": "path", "required": True}]
        )
        monkeypatch.setattr(
            memory_setup,
            "_get_available_providers",
            lambda: [("fake", "local", provider)],
        )
        _feed_stdin(monkeypatch, ["/tmp/db"])

        memory_setup.cmd_setup(args=None)

        assert setup_env["saved"] == {"memory": {"provider": "fake"}}
        assert provider.saved is not None
        saved_cfg, _ = provider.saved
        assert saved_cfg == {"DB_PATH": "/tmp/db"}

"""Tests for the env-write denylist on the memory-setup ``.env`` writer.

``hermes_cli.memory_setup._write_env_vars`` persists provider plugin
credentials to ``~/.hermes/.env``. Until this fix it called
``Path.write_text`` directly, bypassing the
``_ENV_VAR_NAME_DENYLIST`` gate that ``save_env_value`` enforces (added
in #32277 by teknium1 for the dashboard ``PUT /api/env`` surface).

A memory provider plugin schema declaring ``env_var: "LD_PRELOAD"``
(or any other subprocess-influencing or Hermes-runtime-location name)
could otherwise plant a value into ``.env`` via the interactive
memory-setup wizard. The next Hermes process would load it through the
``env_loader.py`` ``.env → os.environ`` chain and execute attacker
code before ``main()``.

The fix routes through ``save_env_value`` so the same gate fires.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.config import ensure_hermes_home, load_env
from hermes_cli.memory_setup import _write_env_vars


@pytest.fixture(autouse=True)
def _hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ensure_hermes_home()
    return tmp_path


@pytest.mark.parametrize(
    "denied_key",
    [
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "PYTHONPATH",
        "PYTHONHOME",
        "NODE_OPTIONS",
        "PATH",
        "EDITOR",
        "GIT_SSH_COMMAND",
        "HERMES_HOME",
        "HERMES_PROFILE",
        "HERMES_CONFIG",
        "HERMES_ENV",
    ],
)
def test_denylisted_key_is_skipped(denied_key, capsys):
    """Each denylisted name must not land in .env even though the
    memory-setup wizard accepted it from a (hypothetically malicious)
    provider schema. The wizard prints a warning and continues."""
    _write_env_vars({denied_key: "/tmp/evil.so"})

    env = load_env()
    assert denied_key not in env

    captured = capsys.readouterr()
    assert denied_key in captured.out
    assert "denylist" in captured.out.lower() or "Skipping" in captured.out


def test_denylisted_key_does_not_block_other_writes(capsys):
    """If a single batch contains one denylisted key plus legitimate
    integration credentials, the denylisted one is skipped but the
    legitimate ones still land. The wizard must not abort mid-batch."""
    _write_env_vars({
        "LD_PRELOAD": "/tmp/evil.so",
        "HERMES_LANGFUSE_PUBLIC_KEY": "pk-test-123",
        "OPENROUTER_API_KEY": "sk-or-test-456",
    })

    env = load_env()
    assert "LD_PRELOAD" not in env
    assert env["HERMES_LANGFUSE_PUBLIC_KEY"] == "pk-test-123"
    assert env["OPENROUTER_API_KEY"] == "sk-or-test-456"


def test_legitimate_hermes_integration_key_still_writable():
    """``HERMES_*`` overall is NOT blocked — only the four runtime
    location names (HOME/PROFILE/CONFIG/ENV). Integration credentials
    following the ``HERMES_*`` convention (HERMES_LANGFUSE_*,
    HERMES_SPOTIFY_*, HERMES_QWEN_BASE_URL, ...) must keep working or
    the memory-setup wizard regresses for every plugin that follows
    the convention."""
    _write_env_vars({
        "HERMES_LANGFUSE_PUBLIC_KEY": "pk-lf-789",
        "HERMES_QWEN_BASE_URL": "https://example.com/v1",
    })

    env = load_env()
    assert env["HERMES_LANGFUSE_PUBLIC_KEY"] == "pk-lf-789"
    assert env["HERMES_QWEN_BASE_URL"] == "https://example.com/v1"


def test_malformed_key_name_is_skipped(capsys):
    """The canonical writer also enforces ``_ENV_VAR_NAME_RE`` —
    identifiers must match ``[A-Za-z_][A-Za-z0-9_]*``. A plugin schema
    declaring ``env_var: "FOO BAR"`` (space) was previously persisted
    verbatim, producing a malformed ``.env`` line."""
    _write_env_vars({"FOO BAR": "value"})

    env = load_env()
    assert "FOO BAR" not in env
    assert "FOO" not in env  # not silently truncated either

    captured = capsys.readouterr()
    assert "Skipping" in captured.out or "FOO BAR" in captured.out


def test_legitimate_value_writes_round_trip():
    """Negative control — the gate must not regress on a normal write."""
    _write_env_vars({"MEM0_API_KEY": "m0-test-key-abc"})

    env = load_env()
    assert env["MEM0_API_KEY"] == "m0-test-key-abc"


def test_value_with_embedded_newline_is_stripped():
    """``save_env_value`` strips CR/LF from the value to prevent
    .env-file structure injection (a value containing ``\\n`` would
    otherwise split the line and inject an arbitrary follow-on key).
    Routing through it gives the memory-setup wizard the same
    protection."""
    _write_env_vars({"MEM0_API_KEY": "key1\nEVIL=injected\n"})

    env = load_env()
    # CR/LF stripped, value still lands intact (minus the newlines)
    assert env["MEM0_API_KEY"] == "key1EVIL=injected"
    # And no smuggled key landed
    assert "EVIL" not in env

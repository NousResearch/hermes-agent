"""Every ``.env`` writer must go through the encryption-aware facade.

The regression these tests pin down: a writer that does its own
``read_text``/``write_text`` round-trip rewrites an encrypted ``.env`` as
**plaintext** — storing the API key it was asked to save in the clear — while
leaving the ``#HERMES-ENCRYPTED-V1`` marker on line 1, so every later reader
tries to decrypt the now-corrupt file and fails. The credential leaks *and* the
``.env`` is bricked.

The memory-provider setups (``hermes_cli.memory_setup`` and the bundled
openviking plugin) each carried such a writer. These tests drive both through
their real entry points with encryption enabled and assert the secret never
appears in the bytes on disk.

No subprocess needed here: credential encryption is decided per call
(``credentials_encryption_active()`` re-reads config), unlike the SQLCipher
rebind in ``hermes_state`` which is fixed at import time.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from hermes_constants import get_hermes_home
from hermes_crypto import detect, migrate

SECRET = "sk-do-not-write-me-in-plaintext"


def _env_path():
    return get_hermes_home() / ".env"


def _assert_sealed_and_readable(key: str, value: str) -> None:
    """The .env is still an envelope, the secret is not in the clear, and it reads back."""
    import hermes_cli.config as config

    raw = _env_path().read_bytes()
    assert detect.is_encrypted(raw), "writer rewrote the encrypted .env as plaintext"
    assert value.encode() not in raw, "secret was written to disk in the clear"

    config.invalidate_env_cache()
    assert config.load_env()[key] == value


# ── The facade itself ────────────────────────────────────────────────────────


def test_save_env_values_keeps_env_encrypted():
    import hermes_cli.config as config

    _env_path().write_text("EXISTING=keep-me\n", encoding="utf-8")
    migrate.enable("keyfile")

    config.save_env_values({"FAKEMEM_API_KEY": SECRET})

    _assert_sealed_and_readable("FAKEMEM_API_KEY", SECRET)
    config.invalidate_env_cache()
    assert config.load_env()["EXISTING"] == "keep-me"


def test_save_env_values_write_wins_over_remove():
    """A key that is both written and removed is written — the precedence the
    openviking 'replace the whole OPENVIKING_* namespace' call relies on."""
    import hermes_cli.config as config

    _env_path().write_text("A=old\nB=doomed\n", encoding="utf-8")

    config.save_env_values({"A": "new"}, remove=("A", "B"))

    config.invalidate_env_cache()
    loaded = config.load_env()
    assert loaded["A"] == "new"
    assert "B" not in loaded


def test_save_env_values_validates_every_key_before_writing():
    """A bad key in the batch must abort the whole write, not half-apply it."""
    import hermes_cli.config as config

    _env_path().write_text("UNTOUCHED=1\n", encoding="utf-8")
    before = _env_path().read_bytes()

    with pytest.raises(ValueError):
        config.save_env_values({"GOOD_KEY": "x", "BAD NAME": "y"})
    # PYTHONPATH is on the subprocess-execution denylist.
    with pytest.raises(ValueError):
        config.save_env_values({"GOOD_KEY": "x", "PYTHONPATH": "/evil"})

    assert _env_path().read_bytes() == before


# ── hermes_cli.memory_setup ──────────────────────────────────────────────────


def test_memory_setup_keeps_env_encrypted(monkeypatch):
    """Drive the real cmd_setup wizard: it must not plaintext the .env."""
    import hermes_cli.memory_setup as memory_setup

    provider = SimpleNamespace(
        get_config_schema=lambda: [
            {
                "key": "api_key",
                "description": "API key",
                "secret": True,
                "env_var": "FAKEMEM_API_KEY",
            }
        ]
    )
    # No post_setup attribute — otherwise cmd_setup delegates and returns early.
    assert not hasattr(provider, "post_setup")

    monkeypatch.setattr(
        memory_setup, "_get_available_providers", lambda: [("fakemem", "desc", provider)]
    )
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *a, **k: 0)
    monkeypatch.setattr(memory_setup, "_install_dependencies", lambda *a, **k: None)
    monkeypatch.setattr(memory_setup, "_clear_interactive_transition", lambda *a, **k: None)
    monkeypatch.setattr(memory_setup, "_prompt", lambda *a, **k: SECRET)

    _env_path().write_text("EXISTING=keep-me\n", encoding="utf-8")
    migrate.enable("keyfile")

    memory_setup.cmd_setup(argparse.Namespace())

    _assert_sealed_and_readable("FAKEMEM_API_KEY", SECRET)


# ── plugins/memory/openviking ────────────────────────────────────────────────


def test_openviking_writer_keeps_env_encrypted():
    """The plugin's own writer — an independent copy of the same bug."""
    import plugins.memory.openviking as openviking

    _env_path().write_text("OPENVIKING_ACCOUNT=stale\nOTHER=keep-me\n", encoding="utf-8")
    migrate.enable("keyfile")

    openviking._write_env_vars(
        _env_path(),
        {"OPENVIKING_API_KEY": SECRET},
        remove_keys=openviking._OPENVIKING_ENV_KEYS,
    )

    _assert_sealed_and_readable("OPENVIKING_API_KEY", SECRET)

    import hermes_cli.config as config

    config.invalidate_env_cache()
    loaded = config.load_env()
    # remove_keys cleared the stale namespace but left unrelated keys alone.
    assert "OPENVIKING_ACCOUNT" not in loaded
    assert loaded["OTHER"] == "keep-me"

"""Every ``.env`` reader must decrypt before it parses.

None of these leak plaintext — they fail the other way. A raw ``read_text()``
or a bare ``dotenv.load_dotenv()`` on an encrypted ``.env`` sees the
``#HERMES-ENCRYPTED-V1`` marker (a comment, to dotenv) followed by one base64
blob, so it parses **zero** ``KEY=VALUE`` pairs and concludes the file defines
nothing at all. The caller then confidently reports the wrong thing:

* ``cron/scheduler.py``      — re-reads .env each run to pick up rotated keys;
                               loaded nothing, so rotated keys never landed.
* ``hermes_cli/doctor.py``   — "No API key found in ~/.hermes/.env".
* ``agent/credential_sources.py`` — a key that IS in .env is reported as
                               "still set in your shell".
* ``hermes_cli/main.py``     — profile import: every env var "needs setting".
"""

from __future__ import annotations

import os

import pytest

from hermes_constants import get_hermes_home
from hermes_crypto import detect, migrate

SECRET = "sk-encrypted-reader-value"


def _sealed_env(**pairs: str):
    """Write a .env with *pairs*, then encrypt it. Returns the path."""
    env_path = get_hermes_home() / ".env"
    env_path.write_text(
        "".join(f"{k}={v}\n" for k, v in pairs.items()), encoding="utf-8"
    )
    migrate.enable("keyfile")
    assert detect.is_encrypted(env_path.read_bytes())
    return env_path


# ── config.read_env_text — the shared facade ─────────────────────────────────


def test_read_env_text_decrypts():
    from hermes_cli.config import read_env_text

    env_path = _sealed_env(OPENAI_API_KEY=SECRET)

    text = read_env_text(env_path)
    assert f"OPENAI_API_KEY={SECRET}" in text
    # The raw bytes are still an envelope — we decrypted in memory only.
    assert detect.is_encrypted(env_path.read_bytes())


def test_env_file_defines_returns_none_when_undecryptable(monkeypatch):
    """Another profile's .env is sealed under its own keystore: 'cannot verify'.

    Never False — reporting a configured key as missing is the nag this guards.
    """
    from hermes_cli.config import env_file_defines

    env_path = _sealed_env(OPENAI_API_KEY=SECRET)
    assert env_file_defines(env_path, "OPENAI_API_KEY") is True

    # Simulate a foreign keystore: the DEK for this file is unobtainable.
    import hermes_crypto

    def _no_key():
        raise hermes_crypto.KeystoreError("foreign profile keystore")

    monkeypatch.setattr(hermes_crypto, "get_data_key", _no_key)

    assert env_file_defines(env_path, "OPENAI_API_KEY") is None


def test_env_file_defines_missing_file_is_false():
    from hermes_cli.config import env_file_defines

    assert env_file_defines(get_hermes_home() / "nope.env", "ANY") is False


# ── cron/scheduler — rotated keys must reach scheduled jobs ──────────────────


def test_cron_env_reload_sees_encrypted_keys(monkeypatch):
    """The per-run .env reload must load the key, not silently load nothing.

    Also pins *why* the old code was wrong: a bare ``dotenv.load_dotenv()`` on
    the very same file loads nothing at all, without raising — which is what
    made this silent.
    """
    from dotenv import load_dotenv

    from hermes_cli.env_loader import load_dotenv_file

    env_path = _sealed_env(OPENAI_API_KEY=SECRET)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # What cron used to do: succeeds, loads zero variables.
    load_dotenv(str(env_path), override=True, encoding="utf-8")
    assert "OPENAI_API_KEY" not in os.environ

    # What cron does now.
    load_dotenv_file(env_path, override=True)
    assert os.environ["OPENAI_API_KEY"] == SECRET


def test_cron_env_reload_overrides_a_rotated_key(monkeypatch):
    """The whole point of the per-run reload: a rotated key must win."""
    from hermes_cli.env_loader import load_dotenv_file

    env_path = _sealed_env(OPENAI_API_KEY="sk-rotated-new")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-stale-old")

    load_dotenv_file(env_path, override=True)

    assert os.environ["OPENAI_API_KEY"] == "sk-rotated-new"


# ── agent/credential_sources — key source must not be mis-attributed ─────────


def test_credential_source_sees_key_in_encrypted_dotenv(monkeypatch):
    """A key in an encrypted .env must not be reported as 'shell exported'."""
    import agent.credential_sources as cs

    _sealed_env(OPENAI_API_KEY=SECRET)
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)

    result = cs._remove_env_source(
        "openai", type("R", (), {"source": "env:OPENAI_API_KEY"})()
    )

    # It was found in .env and cleared, so no "still set in your shell" hint.
    assert any("Cleared OPENAI_API_KEY" in c for c in result.cleaned), result.cleaned
    assert not any("still set in your shell" in h for h in result.hints), result.hints


# ── hermes_cli/doctor — no false "No API key found" ──────────────────────────


def test_doctor_finds_api_key_in_encrypted_env(monkeypatch, capsys):
    """doctor must not tell the user to re-run setup on a good encrypted .env."""
    import sys
    import types
    from argparse import Namespace

    import hermes_cli.doctor as doctor_mod

    _sealed_env(OPENAI_API_KEY=SECRET)
    monkeypatch.setattr(doctor_mod, "HERMES_HOME", get_hermes_home())
    monkeypatch.setitem(
        sys.modules,
        "model_tools",
        types.SimpleNamespace(
            check_tool_availability=lambda *a, **kw: (_ for _ in ()).throw(SystemExit(0)),
            TOOLSET_REQUIREMENTS={},
        ),
    )

    with pytest.raises(SystemExit):
        doctor_mod.run_doctor(Namespace(fix=False))

    out = capsys.readouterr().out
    assert "API key or custom endpoint configured" in out, out
    assert "No API key found" not in out, out

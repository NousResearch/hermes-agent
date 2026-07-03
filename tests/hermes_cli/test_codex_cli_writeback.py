"""BUILD-52: Hermes writes rotated Codex tokens back to ~/.codex/auth.json.

Covers _writeback_codex_cli_tokens (chain-match guard, schema preservation,
permissions, failure tolerance) and its hook in _refresh_codex_auth_tokens.
"""
import json
import os
import stat

import pytest

from hermes_cli import auth as auth_mod


def _write_cli_auth(path, refresh_token="R_OLD", extra=True):
    payload = {
        "OPENAI_API_KEY": None,
        "tokens": {
            "id_token": "ID_TOKEN_KEEP",
            "access_token": "A_OLD",
            "refresh_token": refresh_token,
            "account_id": "acct_123",
        },
        "last_refresh": "2026-07-01T00:00:00Z",
    }
    if extra:
        payload["custom_field"] = {"keep": "me"}
    path.write_text(json.dumps(payload))
    os.chmod(path, 0o600)
    return payload


@pytest.fixture()
def codex_home(tmp_path, monkeypatch):
    home = tmp_path / "codex"
    home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(home))
    return home


def test_writeback_updates_matching_chain(codex_home):
    auth_path = codex_home / "auth.json"
    _write_cli_auth(auth_path, refresh_token="R_OLD")

    ok = auth_mod._writeback_codex_cli_tokens("R_OLD", "A_NEW", "R_NEW")

    assert ok is True
    payload = json.loads(auth_path.read_text())
    assert payload["tokens"]["access_token"] == "A_NEW"
    assert payload["tokens"]["refresh_token"] == "R_NEW"
    # untouched fields preserved
    assert payload["tokens"]["id_token"] == "ID_TOKEN_KEEP"
    assert payload["tokens"]["account_id"] == "acct_123"
    assert payload["custom_field"] == {"keep": "me"}
    assert payload["last_refresh"] != "2026-07-01T00:00:00Z"
    # perms stay owner-only
    assert stat.S_IMODE(os.stat(auth_path).st_mode) == 0o600
    # no temp file left behind
    assert list(codex_home.glob("*.tmp")) == []


def test_writeback_skips_different_chain(codex_home):
    auth_path = codex_home / "auth.json"
    before = _write_cli_auth(auth_path, refresh_token="R_FRESHER_RELOGIN")

    ok = auth_mod._writeback_codex_cli_tokens("R_OLD", "A_NEW", "R_NEW")

    assert ok is False
    assert json.loads(auth_path.read_text()) == before


def test_writeback_noop_when_file_missing(codex_home):
    assert auth_mod._writeback_codex_cli_tokens("R_OLD", "A_NEW", "R_NEW") is False
    assert not (codex_home / "auth.json").exists()


def test_writeback_noop_on_corrupt_json(codex_home):
    auth_path = codex_home / "auth.json"
    auth_path.write_text("{not json")

    assert auth_mod._writeback_codex_cli_tokens("R_OLD", "A_NEW", "R_NEW") is False
    assert auth_path.read_text() == "{not json"


def test_writeback_noop_on_empty_args(codex_home):
    auth_path = codex_home / "auth.json"
    before = _write_cli_auth(auth_path)
    assert auth_mod._writeback_codex_cli_tokens("", "A", "R") is False
    assert auth_mod._writeback_codex_cli_tokens("R_OLD", "", "R") is False
    assert json.loads(auth_path.read_text()) == before


def test_refresh_path_invokes_writeback(codex_home, monkeypatch):
    auth_path = codex_home / "auth.json"
    _write_cli_auth(auth_path, refresh_token="R_OLD")

    monkeypatch.setattr(
        auth_mod,
        "refresh_codex_oauth_pure",
        lambda access, refresh, timeout_seconds: {
            "access_token": "A_NEW",
            "refresh_token": "R_NEW",
        },
    )
    saved = {}
    monkeypatch.setattr(auth_mod, "_save_codex_tokens", lambda tokens: saved.update(tokens))

    updated = auth_mod._refresh_codex_auth_tokens(
        {"access_token": "A_OLD", "refresh_token": "R_OLD"}, timeout_seconds=5.0
    )

    assert updated["refresh_token"] == "R_NEW"
    assert saved["refresh_token"] == "R_NEW"  # hermes store saved
    payload = json.loads(auth_path.read_text())
    assert payload["tokens"]["refresh_token"] == "R_NEW"  # CLI healed in lockstep

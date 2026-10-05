"""Sole-credential exhausted-status display clamp (#119163 follow-up)."""

from __future__ import annotations

import json
import time

from agent.credential_pool import PooledCredential
from hermes_cli.auth_commands import _format_exhausted_status


def _exhausted_429(*, reset_at: float, code=429, reason="rate_limit",
                   message="rate limit exceeded", status_at=None) -> PooledCredential:
    now = time.time() if status_at is None else status_at
    return PooledCredential(
        provider="openrouter", id="k1", label="key-one", auth_type="api_key",
        priority=0, source="manual", access_token="sk-test",
        last_status="exhausted", last_status_at=now,
        last_error_code=code, last_error_reason=reason,
        last_error_message=message, last_error_reset_at=reset_at,
    )


def test_sole_far_future_nonbilling_429_shows_short_wait():
    now = time.time()
    entry = _exhausted_429(reset_at=now + 30 * 24 * 3600, status_at=now)
    status = _format_exhausted_status(entry, sole_credential=True)
    assert "1m" in status
    assert "30d" not in status


def test_two_entry_pool_preserves_provider_reset():
    now = time.time()
    entry = _exhausted_429(reset_at=now + 30 * 24 * 3600, status_at=now)
    status = _format_exhausted_status(entry, sole_credential=False)
    assert "30d" in status


def test_sole_billing_429_preserves_provider_reset():
    now = time.time()
    entry = _exhausted_429(
        reset_at=now + 30 * 24 * 3600, status_at=now,
        code=402, reason="billing", message="payment required",
    )
    status = _format_exhausted_status(entry, sole_credential=True)
    assert "30d" in status


def test_auth_failed_label_unchanged_by_sole_flag():
    now = time.time()
    entry = _exhausted_429(
        reset_at=now + 3600, status_at=now,
        code=401, reason="invalid_token", message="unauthorized",
    )
    assert _format_exhausted_status(entry, sole_credential=True) == \
        _format_exhausted_status(entry, sole_credential=False)
    assert "re-auth may be required" in _format_exhausted_status(entry, sole_credential=True)


def _persisted_exhausted_entry(*, entry_id: str, label: str, reset_at: float,
                              status_at: float) -> dict:
    return {
        "id": entry_id,
        "label": label,
        "auth_type": "api_key",
        "priority": 0,
        "source": "manual",
        "access_token": "sk-test",
        "last_status": "exhausted",
        "last_status_at": status_at,
        "last_error_code": 429,
        "last_error_reason": "rate_limit",
        "last_error_message": "rate limit exceeded",
        "last_error_reset_at": reset_at,
    }


def _write_pool_store(tmp_path, monkeypatch, entries: list[dict]) -> None:
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "auth.json").write_text(json.dumps(
        {"version": 1, "providers": {},
         "credential_pool": {"openrouter": entries}},
        indent=2))


def _list_openrouter_output(capsys) -> str:
    from hermes_cli.auth_commands import auth_list_command
    auth_list_command(type("Args", (), {"provider": "openrouter"})())
    return capsys.readouterr().out


def test_auth_list_sole_pool_clamps_displayed_reset(tmp_path, monkeypatch, capsys):
    now = time.time()
    _write_pool_store(tmp_path, monkeypatch, [
        _persisted_exhausted_entry(entry_id="k1", label="key-one",
                                   reset_at=now + 30 * 24 * 3600, status_at=now),
    ])
    out = _list_openrouter_output(capsys)
    assert "1m" in out
    assert "30d" not in out


def test_auth_list_two_entry_pool_preserves_displayed_reset(tmp_path, monkeypatch, capsys):
    now = time.time()
    _write_pool_store(tmp_path, monkeypatch, [
        _persisted_exhausted_entry(entry_id="k1", label="key-one",
                                   reset_at=now + 30 * 24 * 3600, status_at=now),
        _persisted_exhausted_entry(entry_id="k2", label="key-two",
                                   reset_at=now + 30 * 24 * 3600, status_at=now),
    ])
    out = _list_openrouter_output(capsys)
    assert "30d" in out


def test_interactive_remove_sole_pool_clamps_displayed_reset(tmp_path, monkeypatch, capsys):
    now = time.time()
    _write_pool_store(tmp_path, monkeypatch, [
        _persisted_exhausted_entry(entry_id="k1", label="key-one",
                                   reset_at=now + 30 * 24 * 3600, status_at=now),
    ])
    import hermes_cli.auth_commands as auth_commands_mod
    monkeypatch.setattr(auth_commands_mod, "_pick_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(auth_commands_mod, "_ask", lambda *a, **k: "")
    auth_commands_mod._interactive_remove()
    out = capsys.readouterr().out
    assert "1m" in out
    assert "30d" not in out

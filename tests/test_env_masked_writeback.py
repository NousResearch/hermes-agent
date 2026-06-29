"""Regression tests for issue #54708.

The dashboard Keys page pre-populates editable inputs with the *masked*
``redacted_value`` returned by ``GET /api/env``. Saving an otherwise-unchanged
key then sent that mask (e.g. ``xoxb...pJHJ`` or ``***``) to ``PUT /api/env``,
which wrote it verbatim over the real credential in ``.env`` — silently
destroying it. ``set_env_var`` now refuses masked write-backs.

These tests exercise the pure guard (``_is_masked_writeback``) and the route
(``set_env_var``) directly, without standing up a server.
"""

import asyncio

import pytest

from hermes_cli import web_server
from hermes_cli.web_server import EnvVarUpdate, _is_masked_writeback, set_env_var

try:  # FastAPI may be optional in some envs
    from fastapi import HTTPException
except Exception:  # pragma: no cover
    from hermes_cli.web_server import HTTPException  # type: ignore


def test_masked_writeback_detected_against_stored_value(monkeypatch):
    real = "xoxb-1234567890-realtokenmaterial-pJHJ"
    masked = web_server.redact_key(real)  # head...tail form shown by the UI
    assert masked != real  # sanity: it really is masked

    monkeypatch.setattr("hermes_cli.config.get_env_value", lambda key: real)
    # The masked echo must be flagged...
    assert _is_masked_writeback("SLACK_BOT_TOKEN", masked) is True
    # ...while the genuine value is allowed through.
    assert _is_masked_writeback("SLACK_BOT_TOKEN", real) is False


def test_generic_sentinels_flagged_without_stored_value(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.get_env_value", lambda key: None)
    assert _is_masked_writeback("ANY_KEY", "***") is True
    assert _is_masked_writeback("ANY_KEY", "sk-p...7890") is True
    assert _is_masked_writeback("ANY_KEY", "\u00abredacted:ghp_\u2026\u00bb") is True
    # A normal credential is not a sentinel.
    assert _is_masked_writeback("ANY_KEY", "sk-proj-abcdef1234567890") is False
    # Empty value is a no-op (handled elsewhere), never a mask.
    assert _is_masked_writeback("ANY_KEY", "") is False


def test_put_env_rejects_masked_writeback(monkeypatch):
    real = "xoxb-1234567890-realtokenmaterial-pJHJ"
    masked = web_server.redact_key(real)

    saved = {}
    monkeypatch.setattr("hermes_cli.config.get_env_value", lambda key: real)
    monkeypatch.setattr(
        web_server, "save_env_value", lambda k, v: saved.__setitem__(k, v)
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(set_env_var(EnvVarUpdate(key="SLACK_BOT_TOKEN", value=masked)))
    assert exc.value.status_code == 400
    # Crucially, the real credential was NOT overwritten.
    assert "SLACK_BOT_TOKEN" not in saved


def test_put_env_allows_real_value(monkeypatch):
    saved = {}
    monkeypatch.setattr("hermes_cli.config.get_env_value", lambda key: "old-but-real-value")
    monkeypatch.setattr(
        web_server, "save_env_value", lambda k, v: saved.__setitem__(k, v)
    )

    result = asyncio.run(
        set_env_var(EnvVarUpdate(key="SLACK_BOT_TOKEN", value="xoxb-brand-new-token-9999"))
    )
    assert result == {"ok": True, "key": "SLACK_BOT_TOKEN"}
    assert saved["SLACK_BOT_TOKEN"] == "xoxb-brand-new-token-9999"

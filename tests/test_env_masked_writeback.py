"""Regression tests for issue #54708.

The dashboard Keys page pre-populates editable inputs with the *masked*
``redacted_value`` returned by ``GET /api/env``. Saving an otherwise-unchanged
key then sent that mask (e.g. ``xoxb...pJHJ``) to ``PUT /api/env``, which wrote
it verbatim over the real credential in ``.env`` — silently destroying it.
``set_env_var`` now refuses a write when the value exactly matches the redacted
form of the key's current on-disk (``.env``) value.

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

    monkeypatch.setattr(web_server, "load_env", lambda: {"SLACK_BOT_TOKEN": real})
    # The masked echo must be flagged...
    assert _is_masked_writeback("SLACK_BOT_TOKEN", masked) is True
    # ...while the genuine value is allowed through.
    assert _is_masked_writeback("SLACK_BOT_TOKEN", real) is False


def test_no_stored_value_allows_any_write(monkeypatch):
    # With no credential on disk there is nothing to corrupt, so the exact-match
    # guard never fires -- even for strings that look like display sentinels.
    monkeypatch.setattr(web_server, "load_env", lambda: {})
    assert _is_masked_writeback("ANY_KEY", "***") is False
    assert _is_masked_writeback("ANY_KEY", "sk-proj-abcdef1234567890") is False
    # Empty value is a no-op (handled elsewhere), never a mask.
    assert _is_masked_writeback("ANY_KEY", "") is False


def test_only_own_mask_flagged_not_other_keys_mask(monkeypatch):
    # A value equal to the mask of a *different* key's secret is still a real
    # write for this key -- the on-disk value for THIS key is what matters.
    real = "xoxb-1234567890-realtokenmaterial-pJHJ"
    monkeypatch.setattr(web_server, "load_env", lambda: {"OTHER_KEY": real})
    assert _is_masked_writeback("SLACK_BOT_TOKEN", web_server.redact_key(real)) is False


def test_stale_os_environ_does_not_bypass_guard(monkeypatch):
    # Regression for teknium1's review: get_env_value() preferred os.environ,
    # so a stale inherited env value could differ from the profile .env value
    # that GET masks and PUT writes -- letting a masked write-back through.
    # Reading via load_env() (the .env source) keeps mask/read/write aligned.
    disk_real = "xoxb-ondisk-profile-realtoken-pJHJ"
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-STALE-inherited-environ-9999")
    monkeypatch.setattr(web_server, "load_env", lambda: {"SLACK_BOT_TOKEN": disk_real})
    # The mask the UI shows is derived from the on-disk .env value...
    masked = web_server.redact_key(disk_real)
    # ...and echoing it back must be rejected regardless of os.environ.
    assert _is_masked_writeback("SLACK_BOT_TOKEN", masked) is True


def test_put_env_rejects_masked_writeback(monkeypatch):
    real = "xoxb-1234567890-realtokenmaterial-pJHJ"
    masked = web_server.redact_key(real)

    saved = {}
    monkeypatch.setattr(web_server, "load_env", lambda: {"SLACK_BOT_TOKEN": real})
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
    monkeypatch.setattr(web_server, "load_env", lambda: {"SLACK_BOT_TOKEN": "old-but-real-value"})
    monkeypatch.setattr(
        web_server, "save_env_value", lambda k, v: saved.__setitem__(k, v)
    )

    result = asyncio.run(
        set_env_var(EnvVarUpdate(key="SLACK_BOT_TOKEN", value="xoxb-brand-new-token-9999"))
    )
    assert result == {"ok": True, "key": "SLACK_BOT_TOKEN"}
    assert saved["SLACK_BOT_TOKEN"] == "xoxb-brand-new-token-9999"

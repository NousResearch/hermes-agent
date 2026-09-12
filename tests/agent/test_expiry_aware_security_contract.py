"""Regression tests for the documented expiry-aware security contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import account_usage
from agent.agent_init import (
    SessionCredentialBindingError,
    _resolve_pinned_session_credential,
    _session_credential_binding,
)
from agent.credential_pool import CodexAccountUsageWindows, select_expiry_aware_entry
from hermes_state import SessionDB


_NOW = 1_700_000_000.0


def _usage(account_id: str, remaining: float) -> CodexAccountUsageWindows:
    return CodexAccountUsageWindows(
        account_id=account_id,
        remaining_fraction=remaining,
        reset_at=_NOW + 3600,
        observed_at=_NOW,
        short_window_remaining_fraction=1.0,
    )


def test_equal_weekly_resets_rank_remaining_allowance_then_stable_entry_id():
    """The public FEFO contract defines both deterministic tie breakers."""
    entries = [
        SimpleNamespace(id="z-entry", extra={"account_id": "account-z"}),
        SimpleNamespace(id="a-entry", extra={"account_id": "account-a"}),
    ]
    usage = {
        "z-entry": _usage("account-z", 0.20),
        "a-entry": _usage("account-a", 0.80),
    }

    selected = select_expiry_aware_entry(entries, lambda entry: usage[entry.id], now=_NOW)

    assert selected is entries[1]


@pytest.mark.parametrize(
    "base_url",
    [
        "https://chatgpt.com:443/backend-api/codex",
        "https://user@chatgpt.com/backend-api/codex",
        "https://chatgpt.com/backend-api/codex/extra",
        "https://chatgpt.com/backend-api/codex?redirect=https://example.invalid",
        "https://chatgpt.com/backend-api/codex#fragment",
        "http://chatgpt.com/backend-api/codex",
        "https://chatgpt.com.evil.invalid/backend-api/codex",
    ],
)
def test_usage_authorization_accepts_only_the_exact_official_origin(base_url: str):
    entry = SimpleNamespace(runtime_base_url=base_url, base_url=base_url)

    assert account_usage._entry_codex_usage_origin(entry) is None


def test_usage_http_client_disables_redirects_before_sending_authorization(monkeypatch):
    client_kwargs: dict[str, object] = {}

    class Client:
        def __init__(self, **kwargs):
            client_kwargs.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get(self, *_args, **_kwargs):
            raise AssertionError("request should not be needed for this constructor regression")

    monkeypatch.setattr(account_usage.httpx, "Client", Client)
    with pytest.raises(AssertionError):
        account_usage._get_json("https://chatgpt.com/backend-api/wham/usage", {"Authorization": "Bearer synthetic"}, timeout=1)

    assert client_kwargs["follow_redirects"] is False


def test_pinned_account_mismatch_error_never_discloses_persisted_account_id():
    persisted_account_id = "private-account-id-should-not-escape"
    binding = {
        "provider": "openai-codex",
        "entry_id": "entry-one",
        "account_id": persisted_account_id,
    }
    credential = SimpleNamespace(id="entry-one", account_id="different-account")
    pool = SimpleNamespace(select_exact=lambda _entry_id: credential)

    with pytest.raises(SessionCredentialBindingError) as exc:
        _resolve_pinned_session_credential(pool, "session-one", binding)

    assert persisted_account_id not in str(exc.value)
    assert "different-account" not in str(exc.value)


def test_sqlite_session_reader_rejects_whitespace_mutated_identity(tmp_path):
    binding = {
        "provider": "openai-codex",
        "entry_id": " entry-one",
        "account_id": "account-one",
    }
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("session-one", "cli", model_config={"credential_binding": binding})

        with pytest.raises(SessionCredentialBindingError):
            _session_credential_binding(db, "session-one")

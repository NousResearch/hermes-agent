"""The in-chat /status line identifies an active Nous free-tier route."""

import base64
import json
import time
from datetime import datetime

import pytest

from agent.i18n import t
from gateway.config import Platform
from gateway.session import SessionEntry, build_session_key
from hermes_cli import anon_auth
from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store
from tests.gateway.test_status_command import _make_event, _make_runner, _make_source


def _runner():
    source = _make_source()
    entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-free-tier-status",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    return _make_runner(entry)


def _jwt(**claims) -> str:
    def segment(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()

    payload = {
        "sub": "nas_user:status",
        "client_id": "nas-anonymous",
        "account_tier": "anonymous",
        "scope": "inference:invoke tool:invoke",
        "exp": int(time.time()) + 900,
        **claims,
    }
    return f"{segment({'alg': 'RS256'})}.{segment(payload)}.sig"


def _seed_nous(state: dict) -> None:
    with _auth_store_lock():
        store = _load_auth_store()
        store.setdefault("providers", {})["nous"] = state
        store["active_provider"] = "nous"
        _save_auth_store(store)


def _free_tier_state() -> dict:
    return {
        "auth_method": anon_auth.ANON_AUTH_METHOD,
        "account_tier": "anonymous",
        "anon_token": "anon_status",
        "access_token": _jwt(),
        "expires_at": "2999-01-01T00:00:00+00:00",
        "inference_base_url": "https://welcome-api.nousresearch.com/v1",
    }


def _account_state() -> dict:
    return {
        "auth_method": "oauth_device_code",
        "access_token": _jwt(client_id="hermes-cli", account_tier="standard"),
        "refresh_token": "refresh-status",
        "expires_at": "2999-01-01T00:00:00+00:00",
    }


@pytest.fixture(autouse=True)
def isolated_auth_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-store"))
    monkeypatch.setenv("HERMES_GUEST_ONBOARDING", "1")


@pytest.mark.asyncio
async def test_status_names_the_free_tier_and_the_slash_command_when_the_free_tier_carries_inference(
):
    runner = _runner()
    _seed_nous(_free_tier_state())

    result = await runner._handle_message(_make_event("/status"))

    assert anon_auth.FREE_TIER_STATUS_LINE in result


@pytest.mark.asyncio
async def test_status_omits_the_line_for_a_real_account():
    runner = _runner()
    _seed_nous(_account_state())

    result = await runner._handle_message(_make_event("/status"))

    assert anon_auth.FREE_TIER_STATUS_LINE not in result


@pytest.mark.asyncio
async def test_a_status_gate_failure_never_breaks_status(monkeypatch):
    runner = _runner()
    _seed_nous(_free_tier_state())
    monkeypatch.setattr(anon_auth, "guest_carries_inference", lambda: False)
    expected = await runner._handle_message(_make_event("/status"))

    def broken_store():
        raise RuntimeError("broken store")

    monkeypatch.setattr(anon_auth, "guest_carries_inference", broken_store)

    result = await runner._handle_message(_make_event("/status"))

    assert result == expected


def test_the_line_comes_from_the_catalog_in_every_language():
    assert t("gateway.status.free_tier") == anon_auth.FREE_TIER_STATUS_LINE
    # Every catalog carries its own translation; the product name, the route and the verb stay verbatim.
    for lang in ("ja", "de"):
        line = t("gateway.status.free_tier", lang=lang)
        assert line != anon_auth.FREE_TIER_STATUS_LINE
        assert line.startswith("Nous · ") and " · nous/welcome · " in line and "/login" in line

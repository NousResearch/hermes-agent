"""Regression coverage for the devseunggwan review of #131333/#131338.

Two blocked behaviors, both rooted in the same lookup inside
``_ladder_credential_rungs``: the recovery ladder used
``getattr(client, "api_key", "")`` to name the credential that made the failed
request. An Anthropic OAuth client built by :func:`build_anthropic_client`
carries its bearer in ``auth_token`` and leaves ``api_key`` ``None``, so the
lookup yields ``""`` and both identity-keyed lanes misfire:

1. Auth-refresh rung (401): ``refresh_kwargs = {"failed_api_key": ""}`` →
   ``_refresh_anthropic_credentials("")`` returns False before any pool
   lookup, so ``try_refresh_matching`` is never reached and the OAuth token
   never rotates.
2. Pool-rotation rung (429): ``failed_api_key=""`` →
   ``mark_exhausted_and_rotate`` cannot identify the sender and benches the
   pool's current entry — the healthy one — instead of the stale account A
   that sent the request.

Both tests drive the REAL ladder generator end to end (only the token-endpoint
POST and the final fallback rung are substituted), with the client built by
the real ``build_anthropic_client`` over two real persisted OAuth pool rows
under an isolated ``HERMES_HOME``. The disputed call sites compute the
identity inside the production code — the tests never pass a hint themselves,
which is exactly what the ``SimpleNamespace(api_key=...)`` stubs hid.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import httpx
import pytest
import hermes_constants

import agent.auxiliary_client as aux
from agent.anthropic_adapter import build_anthropic_client

# OAuth-shaped tokens: sk-ant-oat routes build_anthropic_client into the oauth
# auth style, where the SDK client carries auth_token and api_key stays None.
_STALE_ACCESS = "sk-ant-oat01-stale-access-aaaa"
_STALE_REFRESH = "sk-ant-ort01-stale-refresh-aaaa"
_CURRENT_ACCESS = "sk-ant-oat01-current-access-bbbb"
_CURRENT_REFRESH = "sk-ant-ort01-current-refresh-bbbb"
_ANTHROPIC_URL = "https://api.anthropic.com"
_MODEL = "claude-sonnet-4"


@pytest.fixture
def oauth_pool_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with two persisted OAuth pool rows.

    account-B (priority 0) is the pool's current/healthy entry; account-A
    (priority 1) holds the stale token the requests were sent with. A
    no-identity rotation selects priority-0 B — the innocent entry the old
    expression used to bench.
    """
    home = tmp_path / "hermes-home"
    home.mkdir()
    (tmp_path / "fakehome").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "fakehome"))
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    from hermes_cli import auth as auth_mod
    monkeypatch.setattr(auth_mod, "_global_auth_store_cache", None)

    expired_ms = int((time.time() - 3600) * 1000)
    store = {
        "version": 1,
        "providers": {},
        "credential_pool": {
            "anthropic": [
                {
                    "id": "acctB", "label": "account-B", "auth_type": "oauth",
                    "priority": 0, "source": "manual",
                    "access_token": _CURRENT_ACCESS, "refresh_token": _CURRENT_REFRESH,
                    "expires_at_ms": expired_ms, "base_url": _ANTHROPIC_URL,
                },
                {
                    "id": "acctA", "label": "account-A", "auth_type": "oauth",
                    "priority": 1, "source": "manual",
                    "access_token": _STALE_ACCESS, "refresh_token": _STALE_REFRESH,
                    "expires_at_ms": expired_ms, "base_url": _ANTHROPIC_URL,
                },
            ],
        },
    }
    (home / "auth.json").write_text(json.dumps(store), encoding="utf-8")

    def pool_rows():
        on_disk = json.loads((home / "auth.json").read_text(encoding="utf-8"))
        return (on_disk.get("credential_pool") or {}).get("anthropic") or []

    return {"home": home, "rows": pool_rows}


def _status_by_label(oauth_pool_home):
    return {row["label"]: row.get("last_status") for row in oauth_pool_home["rows"]()}


def _hermetic_ladder(monkeypatch):
    """Keep the last rung (provider fallback) out of the way; record its error."""
    seen = []

    def _no_fallback(first_err, route):
        seen.append(first_err)
        yield from ()
        return None

    monkeypatch.setattr(aux, "_ladder_provider_fallback", _no_fallback)
    return seen


def _drive_ladder_to_recovery(monkeypatch, first_err, client, retry_err=None):
    """Drive the real ladder; every in-ladder request fails until the ladder's
    own recovery produces a ``retry_same_provider`` step, whose success ends the
    walk (``_call_llm_impl``'s performer would rebuild and resend here). Raising
    instead would run the ladder tail's second ``_recover_provider_pool`` — the
    no-hint marker for the rotated-to entry — and bench a second, innocent
    entry the rungs under test never chose.
    """
    ladder = aux._aux_recovery_ladder(
        first_err,
        client=client,
        kwargs={"model": _MODEL},
        task="compression",
        async_mode=False,
        base_info=f"{_ANTHROPIC_URL}/v1",
        resolved_provider="anthropic",
        resolved_model=_MODEL,
        resolved_base_url=_ANTHROPIC_URL,
        resolved_api_key=_STALE_ACCESS,
        resolved_api_mode=None,
        final_model=_MODEL,
        max_tokens=None,
        main_runtime=None,
        route_info={},
    )
    fail = retry_err or first_err

    def perform(step):
        if step.kind == "retry_same_provider":
            return "recovered-response"
        raise fail

    assert aux._drive_ladder(ladder, perform) == "recovered-response"


def test_oauth_client_429_marks_only_the_sending_entry(
        oauth_pool_home, monkeypatch):
    """A 429 on the stale OAuth client (A) while B is current: only A is benched.

    The ladder's pool-rotation rung computes the sender identity itself; with
    the old ``getattr(client, "api_key", "")`` expression that identity is
    empty, the entry cannot be identified, and the rotation benches the
    healthy current entry (B) instead.
    """
    _hermetic_ladder(monkeypatch)
    client = build_anthropic_client(_STALE_ACCESS, _ANTHROPIC_URL)
    assert client.api_key is None, "OAuth clients must not regress to the stub shape"
    assert client.auth_token == _STALE_ACCESS

    _drive_ladder_to_recovery(monkeypatch, _anthropic_429(), client)

    assert _status_by_label(oauth_pool_home) == {
        "account-A": "exhausted", "account-B": None,
    }, _status_by_label(oauth_pool_home)


def test_oauth_client_429_without_identity_benches_the_current_entry(
        oauth_pool_home, monkeypatch):
    """Failure-mode control: the identity-less call the old expression produced.

    ``failed_api_key=""`` must fall back to the pool's current entry and bench
    the innocent priority-0 account-B — the documented pre-fix behavior this
    fix exists to prevent.
    """
    _hermetic_ladder(monkeypatch)
    client = SimpleNamespace(api_key=None, auth_token=None,
                             base_url=f"{_ANTHROPIC_URL}/v1")

    _drive_ladder_to_recovery(monkeypatch, _anthropic_429(), client)

    status = _status_by_label(oauth_pool_home)
    assert status["account-B"] == "exhausted", status


def test_oauth_client_401_refresh_rung_reaches_try_refresh_matching(
        oauth_pool_home, monkeypatch):
    """A 401 on an Anthropic OAuth client must refresh the matching pool entry.

    The auth-refresh rung builds ``refresh_kwargs`` itself; with the old
    ``getattr(client, "api_key", "")`` expression the hint is empty and
    ``_refresh_anthropic_credentials("")`` returns False before any pool
    lookup — ``try_refresh_matching`` is never called. Only the token-endpoint
    POST is substituted; pool selection, matching and the pair rotation stay
    real.
    """
    refresh_posts = []
    refresh_hints = []

    def fake_refresh_pure(refresh_token, *, use_json=False):
        refresh_posts.append(refresh_token)
        return {
            "access_token": "sk-ant-oat01-rotated-access",
            "refresh_token": f"sk-ant-ort01-rotated-{len(refresh_posts)}",
            "expires_at_ms": int((time.time() + 3600) * 1000),
        }

    from agent.credential_pool import CredentialPool
    from agent import anthropic_credentials as ac

    real_try_refresh_matching = CredentialPool.try_refresh_matching

    def recording_try_refresh_matching(self, *args, **kwargs):
        hint = kwargs.get("api_key_hint") if "api_key_hint" in kwargs else (
            args[0] if args else None)
        if hint is not None:
            refresh_hints.append(hint)
        return real_try_refresh_matching(self, *args, **kwargs)

    monkeypatch.setattr(ac, "refresh_anthropic_oauth_pure", fake_refresh_pure)
    monkeypatch.setattr(CredentialPool, "try_refresh_matching",
                        recording_try_refresh_matching)
    _hermetic_ladder(monkeypatch)

    client = build_anthropic_client(_STALE_ACCESS, _ANTHROPIC_URL)
    _drive_ladder_to_recovery(
        monkeypatch, _anthropic_401(), client, retry_err=_anthropic_429())

    # The rung matched the OAuth entry by its auth_token and spent ITS refresh
    # token — not account-B's, and not nothing.
    assert refresh_hints == [_STALE_ACCESS], refresh_hints
    assert refresh_posts[:1] == [_STALE_REFRESH], refresh_posts


def _anthropic_429() -> Exception:
    request = httpx.Request("POST", f"{_ANTHROPIC_URL}/v1/messages")
    exc = Exception("Error code: 429 - rate_limit_error")
    exc.status_code = 429
    exc.response = httpx.Response(429, request=request)
    return exc


def _anthropic_401() -> Exception:
    request = httpx.Request("POST", f"{_ANTHROPIC_URL}/v1/messages")
    exc = Exception("Error code: 401 - authentication_error")
    exc.status_code = 401
    exc.response = httpx.Response(401, request=request)
    return exc

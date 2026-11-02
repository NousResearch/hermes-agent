"""A Nous 401 on a still-valid agent key must not kill the turn in ~1.5 s.

Nov 1 2026 incident: Nous answered ``401 {"error": "authentication required"}`` in bursts for
unexpired agent keys. Two invariants pinned here:

1. The forced refresh after a 401 compares the store against the bearer the request actually
   sent. A pool row that is older than the agent's key (the pre-expiry adoption swaps the agent's
   key without touching its pool row) must not "adopt" the store's key when that key IS the one
   that just failed: that replayed the dead bearer with no refresh at all.
2. One Nous 401 episode redeems the refresh token at most once; later 401s on the fresh,
   unexpired key back off and retry instead of re-minting or giving up after three tries.
"""

import time

import hermes_cli.auth as auth_mod
from hermes_cli import auth_nous
from agent.agent_runtime_helpers import _recover_auth_failure
from agent.credential_pool import load_pool

from tests.hermes_cli.test_auth_nous_provider import _invoke_jwt, _setup_nous_auth


def _future(seconds: int) -> str:
    return auth_mod.datetime.fromtimestamp(time.time() + seconds, tz=auth_mod.timezone.utc).isoformat()


def _seed(hermes_home, token: str, refresh_token: str) -> None:
    _setup_nous_auth(hermes_home, access_token=token, refresh_token=refresh_token,
                     scope=auth_mod.DEFAULT_NOUS_SCOPE, expires_at=_future(3000), expires_in=3000)


def _count_refresh_posts(monkeypatch) -> list[str]:
    posts: list[str] = []

    def _fake_refresh_access_token(*, client, portal_base_url, client_id, refresh_token):
        posts.append(refresh_token)
        return {"access_token": _invoke_jwt(seconds=3600), "refresh_token": f"rt-minted-{len(posts)}",
                "expires_in": 3600, "token_type": "Bearer", "scope": auth_mod.DEFAULT_NOUS_SCOPE}

    monkeypatch.setattr(auth_mod, "_refresh_access_token", _fake_refresh_access_token)
    monkeypatch.setattr(auth_nous, "_refresh_access_token", _fake_refresh_access_token)
    return posts


def test_forced_refresh_never_adopts_the_bearer_that_just_failed(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared"))
    _seed(hermes_home, _invoke_jwt(seconds=2000), "rt-old")
    pool = load_pool("nous")  # this process's pool row: the older key
    (entry,) = pool.entries()
    # Another agent in the process adopted the rotated key from the store; the request sent it
    # and Nous rejected it.
    failing = _invoke_jwt(seconds=3000)
    _seed(hermes_home, failing, "rt-current")
    posts = _count_refresh_posts(monkeypatch)

    refreshed = pool.try_refresh_matching(api_key_hint=failing, credential_id=entry.id)

    assert refreshed is not None
    assert refreshed.runtime_api_key != failing, "the 'refresh' handed back the bearer that just 401'd"
    assert posts == ["rt-current"], f"expected exactly one real refresh POST, got {posts}"


class _Agent:
    provider = "nous"
    api_mode = "chat_completions"
    log_prefix = ""
    _interrupt_requested = False

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.swapped: list[str] = []

    def _is_entitlement_failure(self, *_a) -> bool:
        return False

    def _swap_credential(self, entry) -> bool:
        self.api_key = entry.runtime_api_key
        self.swapped.append(entry.runtime_api_key)
        return True

    def _vprint(self, *_a, **_k) -> None:
        pass

    def _touch_activity(self, *_a) -> None:
        pass


def test_one_refresh_per_episode_then_bounded_backoff(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared"))
    key = _invoke_jwt(seconds=3000)
    _seed(hermes_home, key, "rt-seed")
    pool = load_pool("nous")
    (entry,) = pool.entries()
    posts = _count_refresh_posts(monkeypatch)
    try:  # absent on base: let the refresh-POST count, not an ImportError, be what fails there
        from agent import credential_auth_recovery
        monkeypatch.setattr(credential_auth_recovery, "TRANSIENT_AUTH_BACKOFF_S", (0.01, 0.02, 0.03))
    except ImportError:
        pass
    agent = _Agent(key)
    agent._credential_pool_entry_id = entry.id

    def _on_401():
        return _recover_auth_failure(
            agent, pool, status_code=401, has_retried_429=False,
            error_context={"message": "authentication required"}, api_key_hint=agent.api_key,
            credential_id=entry.id, rotate_and_swap=lambda *_a: False)

    # Nous keeps refusing a key that is valid: refresh once, then three backoff retries.
    outcomes = [_on_401()[0] for _ in range(5)]

    assert outcomes == [True, True, True, True, False], outcomes
    assert len(posts) == 1, f"one 401 episode redeemed the refresh token {len(posts)}x"
    assert agent.swapped and set(agent.swapped[1:]) == {agent.swapped[0]}, (
        "backoff retries must keep the freshly minted key, not re-mint")

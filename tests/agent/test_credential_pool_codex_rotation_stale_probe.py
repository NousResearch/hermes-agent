"""A Codex 429 on the primary account must rotate to a healthy sibling.

The early-reopen quota probe (#43747) caches its verdict per token for
``CODEX_QUOTA_PROBE_MIN_INTERVAL_SECONDS``. A "restored" verdict cached before
the account hit its cap survives the 429 that follows. Rotation then consults
the cache, lifts the bench it just wrote, and hands back the failed priority-0
entry. The #97315 no-recovery guard turns that into ``None``, so the turn falls
over to ``fallback_providers`` while a healthy second account sits idle.
"""

from __future__ import annotations

import base64
import json
import threading
import time

import pytest

import hermes_cli.auth as auth_mod
import hermes_cli.auth_codex as auth_codex


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    auth_mod._codex_quota_probe_cache.clear()
    yield
    auth_mod._codex_quota_probe_cache.clear()


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _refuse(**_kwargs):
        raise AssertionError("the quota probe must not reach the network in this test")

    monkeypatch.setattr(auth_codex, "_codex_http_client", _refuse)


def _jwt(claims: dict) -> str:
    def _part(payload: dict) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{_part({'alg': 'none'})}.{_part(claims)}.sig"


def _token(account: str) -> str:
    return _jwt({"exp": time.time() + 7 * 86400, "sub": account})


def _entry(idx: int, token: str) -> dict:
    return {
        "id": f"codex-{idx}",
        "label": f"codex-login-{idx}",
        "auth_type": "oauth",
        "priority": idx - 1,
        "source": "manual:device_code",
        "access_token": token,
        "refresh_token": f"rf-{idx}",
    }


def _load(tmp_path, monkeypatch, entries: list[dict]):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(
        json.dumps({"version": 1, "credential_pool": {"openai-codex": entries}})
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    from agent.credential_pool import load_pool

    return load_pool("openai-codex"), hermes_home / "auth.json"


def _prime_probe_cache(token: str, restored: bool) -> None:
    """A probe that ran shortly before the 429, while the window still had room."""
    key = auth_codex._codex_quota_probe_cache_key(token)
    auth_mod._codex_quota_probe_cache[key] = (time.monotonic() - 60, restored)


def _mark_usage_limit(pool, credential_id: str):
    return pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={
            "reason": "usage_limit_reached",
            "message": "The usage limit has been reached",
            "reset_at": time.time() + 3600,
        },
        credential_id=credential_id,
        failure_reason="rate_limit",
        model="gpt-5.6-sol",
    )


def _persisted_status(auth_path, credential_id: str):
    rows = json.loads(auth_path.read_text())["credential_pool"]["openai-codex"]
    return next(r for r in rows if r["id"] == credential_id).get("last_status")


def test_429_rotates_to_second_account_despite_stale_restored_probe(tmp_path, monkeypatch):
    primary, secondary = _token("primary"), _token("secondary")
    pool, auth_path = _load(tmp_path, monkeypatch, [_entry(1, primary), _entry(2, secondary)])
    _prime_probe_cache(primary, restored=True)

    nxt = _mark_usage_limit(pool, "codex-1")

    assert nxt is not None and nxt.id == "codex-2"
    # The bench this rotation wrote must survive it, on disk too.
    assert _persisted_status(auth_path, "codex-1") == "exhausted"


def test_rotation_never_unbenches_the_just_marked_entry(tmp_path, monkeypatch):
    """Whatever lifts the bench mid-selection (probe false positive, token
    adoption), rotation moves on to the sibling instead of giving up."""
    pool, auth_path = _load(
        tmp_path, monkeypatch, [_entry(1, _token("primary")), _entry(2, _token("secondary"))]
    )
    monkeypatch.setattr(pool, "_codex_quota_restored_upstream", lambda entry: True)

    nxt = _mark_usage_limit(pool, "codex-1")

    assert nxt is not None and nxt.id == "codex-2"
    assert _persisted_status(auth_path, "codex-1") == "exhausted"


def test_next_turn_selection_does_not_bounce_back_to_exhausted_primary(tmp_path, monkeypatch):
    primary, secondary = _token("primary"), _token("secondary")
    pool, _auth_path = _load(tmp_path, monkeypatch, [_entry(1, primary), _entry(2, secondary)])
    _prime_probe_cache(primary, restored=True)

    _mark_usage_limit(pool, "codex-1")

    # The 429 is newer, first-hand evidence; it supersedes the cached verdict for the
    # throttle window instead of re-admitting the primary on the next turn.
    assert auth_mod._probe_codex_quota_restored(primary) is False
    selected = pool.select(model="gpt-5.6-sol")
    assert selected is not None and selected.id == "codex-2"


def test_in_flight_probe_cannot_restore_account_after_newer_429(tmp_path, monkeypatch):
    primary, secondary = _token("primary"), _token("secondary")
    pool, _auth_path = _load(tmp_path, monkeypatch, [_entry(1, primary), _entry(2, secondary)])
    started, release = threading.Event(), threading.Event()
    results = []

    class _UsageClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get(self, *_args, **_kwargs):
            started.set()
            if not release.wait(5):
                raise AssertionError("probe was not released")
            return self

        status_code = 200

        def json(self):
            return {"rate_limit": {"primary_window": {"used_percent": 0}}}

    monkeypatch.setattr(auth_codex, "_codex_http_client", lambda **_kwargs: _UsageClient())
    probe = threading.Thread(target=lambda: results.append(auth_codex._probe_codex_quota_restored(primary)))
    probe.start()
    try:
        assert started.wait(5)
        rotated = _mark_usage_limit(pool, "codex-1")
        assert rotated is not None and rotated.id == "codex-2"
    finally:
        release.set()
        probe.join(5)

    assert not probe.is_alive()
    assert results == [False]
    assert auth_mod._probe_codex_quota_restored(primary) is False
    selected = pool.select(model="gpt-5.6-sol")
    assert selected is not None and selected.id == "codex-2"


def test_sole_entry_still_surfaces_no_recovery(tmp_path, monkeypatch):
    primary = _token("primary")
    pool, _auth_path = _load(tmp_path, monkeypatch, [_entry(1, primary)])
    _prime_probe_cache(primary, restored=True)

    assert _mark_usage_limit(pool, "codex-1") is None

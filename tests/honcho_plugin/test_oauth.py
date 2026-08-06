"""Tests for plugins/memory/honcho/oauth.py — OAuth grant storage + refresh."""

import json
from pathlib import Path

import pytest

from plugins.memory.honcho import oauth
from plugins.memory.honcho.oauth import OAuthCredential


def _host_block(refresh="hch-rt-old", expires_at=10_000):
    return {
        "apiKey": "hch-at-old",
        "oauth": {
            "refreshToken": refresh,
            "expiresAt": expires_at,
            "clientId": "hermes-desktop",
            "tokenEndpoint": "http://localhost:8000/oauth/token",
            "scope": "write",
            "tokenType": "Bearer",
        },
    }


def _write(path: Path, raw: dict) -> None:
    path.write_text(json.dumps(raw), encoding="utf-8")


class TestTokenDetection:
    def test_access_token_prefix(self):
        assert oauth.is_oauth_access_token("hch-at-abc")
        assert not oauth.is_oauth_access_token("hch-v3-abc")
        assert not oauth.is_oauth_access_token("hch-rt-abc")
        assert not oauth.is_oauth_access_token(None)


class TestCredentialModel:
    def test_roundtrip(self):
        cred = OAuthCredential.from_host_block(_host_block())
        assert cred is not None
        block = cred.oauth_block()
        assert block["refreshToken"] == "hch-rt-old"
        assert block["expiresAt"] == 10_000
        assert block["clientId"] == "hermes-desktop"

    def test_incomplete_block_returns_none(self):
        # plain API key (no oauth sub-block)
        assert OAuthCredential.from_host_block({"apiKey": "hch-v3-x"}) is None
        # oauth block missing refreshToken
        bad = _host_block()
        del bad["oauth"]["refreshToken"]
        assert OAuthCredential.from_host_block(bad) is None

    def test_is_expired_respects_skew(self):
        cred = OAuthCredential.from_host_block(_host_block(expires_at=1000))
        assert not cred.is_expired(now=800, skew=120)  # 1000-120=880 > 800
        assert cred.is_expired(now=900, skew=120)  # 900 >= 880


class TestEnsureFreshToken:
    def test_no_oauth_credential_is_noop(self, tmp_path):
        path = tmp_path / "honcho.json"
        _write(path, {"hosts": {"hermes": {"apiKey": "hch-v3-static"}}})
        token, refreshed = oauth.ensure_fresh_token(path, "hermes", now=0)
        assert token is None and refreshed is False

    def test_fresh_token_skips_refresh(self, tmp_path, monkeypatch):
        path = tmp_path / "honcho.json"
        _write(path, {"hosts": {"hermes": _host_block(expires_at=10_000)}})
        monkeypatch.setattr(
            oauth, "_http_post_form",
            lambda *a, **k: pytest.fail("refresh must not be called when fresh"),
        )
        token, refreshed = oauth.ensure_fresh_token(path, "hermes", now=0)
        assert token == "hch-at-old" and refreshed is False

    def test_fresh_token_served_from_cache_without_disk(self, tmp_path, monkeypatch):
        path = tmp_path / "honcho.json"
        _write(path, {"hosts": {"hermes": _host_block(expires_at=10_000)}})
        oauth._expiry_cache.clear()
        # First call seeds the cache from disk.
        oauth.ensure_fresh_token(path, "hermes", now=0)
        # Second call must not touch disk while the token is well clear of expiry.
        monkeypatch.setattr(
            oauth, "_read_config",
            lambda *a, **k: pytest.fail("disk must not be read while token is fresh"),
        )
        token, refreshed = oauth.ensure_fresh_token(path, "hermes", now=100)
        assert token == "hch-at-old" and refreshed is False

    def test_expired_token_refreshes_and_persists_rotation(self, tmp_path, monkeypatch):
        path = tmp_path / "honcho.json"
        _write(path, {"hosts": {"hermes": _host_block(expires_at=100)}})

        def fake_post(url, data, timeout):
            assert data["grant_type"] == "refresh_token"
            assert data["refresh_token"] == "hch-rt-old"
            assert data["client_id"] == "hermes-desktop"
            return {
                "access_token": "hch-at-new",
                "refresh_token": "hch-rt-new",
                "expires_in": 3600,
                "scope": "write",
                "token_type": "Bearer",
            }

        monkeypatch.setattr(oauth, "_http_post_form", fake_post)
        token, refreshed = oauth.ensure_fresh_token(path, "hermes", now=1000)
        assert token == "hch-at-new" and refreshed is True

        # Rotated refresh token + new access token + absolute expiry persisted.
        saved = json.loads(path.read_text())["hosts"]["hermes"]
        assert saved["apiKey"] == "hch-at-new"
        assert saved["oauth"]["refreshToken"] == "hch-rt-new"
        assert saved["oauth"]["expiresAt"] == 1000 + 3600

    def test_refresh_failure_fails_open(self, tmp_path, monkeypatch):
        path = tmp_path / "honcho.json"
        _write(path, {"hosts": {"hermes": _host_block(expires_at=100)}})

        def boom(*a, **k):
            raise RuntimeError("network down")

        monkeypatch.setattr(oauth, "_http_post_form", boom)
        token, refreshed = oauth.ensure_fresh_token(path, "hermes", now=1000)
        # Stale token returned, no crash, file untouched.
        assert token == "hch-at-old" and refreshed is False
        assert json.loads(path.read_text())["hosts"]["hermes"]["apiKey"] == "hch-at-old"

    def test_double_check_uses_disk_when_already_rotated(self, tmp_path, monkeypatch):
        # Simulates a concurrent thread that rotated the token on disk after our
        # stale in-memory snapshot: the locked re-read must skip the HTTP call.
        path = tmp_path / "honcho.json"
        _write(path, {"hosts": {"hermes": _host_block(refresh="hch-rt-fresh", expires_at=10_000)}})
        stale_raw = {"hosts": {"hermes": _host_block(refresh="hch-rt-old", expires_at=100)}}
        stale_raw["hosts"]["hermes"]["apiKey"] = "hch-at-stale"
        monkeypatch.setattr(
            oauth, "_http_post_form",
            lambda *a, **k: pytest.fail("must not refresh; disk token is fresh"),
        )
        token, refreshed = oauth.ensure_fresh_token(path, "hermes", stale_raw, now=1000)
        assert token == "hch-at-old"  # the on-disk fresh credential's access token

    def test_refresh_holds_cross_process_lock(self, tmp_path, monkeypatch):
        # A second opener must not grab <config>.lock mid-refresh — proving the
        # rotation is serialized machine-wide so peers can't replay the token.
        fcntl = pytest.importorskip("fcntl")
        path = tmp_path / "honcho.json"
        _write(path, {"hosts": {"hermes": _host_block(expires_at=100)}})
        seen = {}

        def fake_post(url, data, timeout):
            with open(f"{path}.lock", "a+b") as other:
                try:
                    fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(other.fileno(), fcntl.LOCK_UN)
                    seen["held"] = False
                except OSError:
                    seen["held"] = True
            return {"access_token": "hch-at-new", "refresh_token": "hch-rt-new",
                    "expires_in": 3600, "scope": "write", "token_type": "Bearer"}

        monkeypatch.setattr(oauth, "_http_post_form", fake_post)
        token, refreshed = oauth.ensure_fresh_token(path, "hermes", now=1000)
        assert refreshed is True and seen.get("held") is True
        # Released afterward: a non-blocking acquire now succeeds.
        with open(f"{path}.lock", "a+b") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def test_refresh_degrades_when_lock_unavailable(self, tmp_path, monkeypatch):
        # No flock (unsupported FS/platform) must not block refresh — it falls
        # back to in-process serialization only.
        fcntl = pytest.importorskip("fcntl")
        path = tmp_path / "honcho.json"
        _write(path, {"hosts": {"hermes": _host_block(expires_at=100)}})

        def no_flock(*a, **k):
            raise OSError("flock unsupported")

        monkeypatch.setattr(fcntl, "flock", no_flock)
        monkeypatch.setattr(
            oauth, "_http_post_form",
            lambda *a, **k: {"access_token": "hch-at-new", "refresh_token": "hch-rt-new",
                             "expires_in": 3600, "scope": "write", "token_type": "Bearer"},
        )
        token, refreshed = oauth.ensure_fresh_token(path, "hermes", now=1000)
        assert token == "hch-at-new" and refreshed is True


class TestInstallGrant:
    def test_deep_merges_config_and_preserves_other_hosts(self, tmp_path):
        path = tmp_path / "honcho.json"
        _write(path, {
            "apiKey": "hch-v3-root",  # root static key preserved
            "hosts": {
                "obsidian": {"workspace": "obsidian"},
                "hermes": {"workspace": "hermes", "saveMessages": False},
            },
        })
        grant = {
            "access_token": "hch-at-fresh",
            "refresh_token": "hch-rt-fresh",
            "expires_in": 3600,
            "scope": "write",
            "config": {
                "environment": "production",
                "hosts": {"hermes": {"saveMessages": True, "recallMode": "hybrid"}},
            },
        }
        cred = oauth.install_grant(
            path, "hermes", grant,
            client_id="hermes-desktop",
            token_endpoint="http://localhost:8000/oauth/token",
            now=1000,
        )
        assert cred.expires_at == 1000 + 3600

        saved = json.loads(path.read_text())
        assert saved["apiKey"] == "hch-v3-root"  # untouched
        assert saved["hosts"]["obsidian"] == {"workspace": "obsidian"}  # untouched
        h = saved["hosts"]["hermes"]
        assert h["apiKey"] == "hch-at-fresh"
        assert h["oauth"]["refreshToken"] == "hch-rt-fresh"
        assert h["saveMessages"] is True  # grant config won the deep-merge
        assert h["recallMode"] == "hybrid"  # new key added
        assert h["workspace"] == "hermes"  # pre-existing key preserved
        assert saved["environment"] == "production"  # root key from grant

    def test_rejects_grant_without_tokens(self, tmp_path):
        path = tmp_path / "honcho.json"
        _write(path, {})
        with pytest.raises(ValueError):
            oauth.install_grant(
                path, "hermes", {"access_token": "hch-at-x"},  # no refresh_token
                client_id="c", token_endpoint="e",
            )


class TestApplyTokenToClient:
    def test_mutates_live_bearer(self):
        class FakeHttp:
            api_key = "hch-at-old"

        class FakeClient:
            _http = FakeHttp()

        client = FakeClient()
        assert oauth.apply_token_to_client(client, "hch-at-new") is True
        assert client._http.api_key == "hch-at-new"

    def test_returns_false_when_shape_unknown(self):
        assert oauth.apply_token_to_client(object(), "hch-at-new") is False

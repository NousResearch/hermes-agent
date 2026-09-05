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
            oauth, "_http_post_form_status",
            lambda *a, **k: pytest.fail("refresh must not be called when fresh"),
        )
        token, refreshed = oauth.ensure_fresh_token(path, "hermes", now=0)
        assert token == "hch-at-old" and refreshed is False


    def test_expired_token_refreshes_and_persists_rotation(self, tmp_path, monkeypatch):
        path = tmp_path / "honcho.json"
        _write(path, {"hosts": {"hermes": _host_block(expires_at=100)}})

        def fake_post(url, data, timeout):
            assert data["grant_type"] == "refresh_token"
            assert data["refresh_token"] == "hch-rt-old"
            assert data["client_id"] == "hermes-desktop"
            return 200, {
                "access_token": "hch-at-new",
                "refresh_token": "hch-rt-new",
                "expires_in": 3600,
                "scope": "write",
                "token_type": "Bearer",
            }

        monkeypatch.setattr(oauth, "_http_post_form_status", fake_post)
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
        monkeypatch.setattr(oauth, "_REFRESH_RETRY_DELAY_SECONDS", 0)

        calls = []

        def boom(*a, **k):
            calls.append(a)
            raise RuntimeError("network down")

        monkeypatch.setattr(oauth, "_http_post_form_status", boom)
        token, refreshed = oauth.ensure_fresh_token(path, "hermes", now=1000)
        # Stale token returned, no crash, file untouched. Transient failures
        # retry exactly once, then fail open.
        assert token == "hch-at-old" and refreshed is False
        assert len(calls) == 2
        assert json.loads(path.read_text())["hosts"]["hermes"]["apiKey"] == "hch-at-old"

    def test_double_check_uses_disk_when_already_rotated(self, tmp_path, monkeypatch):
        # Simulates a concurrent thread that rotated the token on disk after our
        # stale in-memory snapshot: the locked re-read must skip the HTTP call.
        path = tmp_path / "honcho.json"
        _write(path, {"hosts": {"hermes": _host_block(refresh="hch-rt-fresh", expires_at=10_000)}})
        stale_raw = {"hosts": {"hermes": _host_block(refresh="hch-rt-old", expires_at=100)}}
        stale_raw["hosts"]["hermes"]["apiKey"] = "hch-at-stale"
        monkeypatch.setattr(
            oauth, "_http_post_form_status",
            lambda *a, **k: pytest.fail("must not refresh; disk token is fresh"),
        )
        token, refreshed = oauth.ensure_fresh_token(path, "hermes", stale_raw, now=1000)
        assert token == "hch-at-old"  # the on-disk fresh credential's access token


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

    def test_reads_and_writes_under_the_refresh_locks(self, tmp_path, monkeypatch):
        path = tmp_path / "honcho.json"
        _write(path, {"hosts": {"other": _host_block()}})
        seen = {}
        real_read, real_persist = oauth._read_config_strict, oauth._persist_credential
        locked_paths = []

        @oauth.contextmanager
        def spy_file_lock(p):
            locked_paths.append(p)
            yield

        def spy_read(p):
            seen["read_locked"] = oauth._refresh_lock.locked()
            return real_read(p)

        def spy_persist(p, host, cred, raw=None):
            seen["persist_locked"] = oauth._refresh_lock.locked()
            return real_persist(p, host, cred, raw)

        monkeypatch.setattr(oauth, "_config_refresh_lock", spy_file_lock)
        monkeypatch.setattr(oauth, "_read_config_strict", spy_read)
        monkeypatch.setattr(oauth, "_persist_credential", spy_persist)
        oauth.install_grant(
            path, "hermes", {"access_token": "hch-at-new", "refresh_token": "hch-rt-new", "expires_in": 60},
            client_id="c", token_endpoint="e",
        )
        assert seen == {"read_locked": True, "persist_locked": True}
        assert locked_paths == [path]
        assert not oauth._refresh_lock.locked()
        assert "other" in json.loads(path.read_text())["hosts"]


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


class TestPersistReadFailure:
    """One unreadable honcho.json must never become a single-host store.

    ``_persist_credential`` promises "leaving all else intact", but it seeded
    the write from a reader that returned ``{}`` on ANY read failure, and
    ``_atomic_write_config`` replaces the whole file via ``os.replace`` —
    which needs only a writable parent, so a present-but-unreadable store did
    not stop the overwrite. Same class as the auth-store fix in #75206.
    """

    @staticmethod
    def _seed_two_hosts(path: Path) -> bytes:
        _write(path, {
            "hosts": {
                "keep.example": _host_block(refresh="hch-rt-keep"),
                "rotate.example": _host_block(refresh="hch-rt-rot"),
            },
            "defaults": {"workspace": "w1"},
        })
        return path.read_bytes()

    def _unreadable(self, monkeypatch, target: Path):
        real = Path.read_text

        def boom(self, *a, **kw):
            if self.name == target.name:
                raise PermissionError(13, "Permission denied")
            return real(self, *a, **kw)

        monkeypatch.setattr(Path, "read_text", boom)

    def test_persist_raises_and_preserves_store_when_unreadable(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "honcho.json"
        before = self._seed_two_hosts(path)
        cred = OAuthCredential.from_host_block(_host_block(refresh="hch-rt-new"))

        with monkeypatch.context() as m:
            self._unreadable(m, path)
            with pytest.raises(OSError):
                oauth._persist_credential(path, "rotate.example", cred)

        assert path.read_bytes() == before

    def test_rotate_refuses_before_spending_the_refresh_token(
        self, tmp_path, monkeypatch
    ):
        # Rotation is single-use: an exchange whose result cannot be persisted
        # loses the grant, so an unreadable store must fail the refresh BEFORE
        # the exchange, not after.
        path = tmp_path / "honcho.json"
        before = self._seed_two_hosts(path)
        cred = OAuthCredential.from_host_block(_host_block(refresh="hch-rt-rot"))
        key = (str(path), "rotate.example")
        oauth._refresh_failure_at.pop(key, None)

        def no_exchange(*a, **kw):  # pragma: no cover - must not run
            raise AssertionError("exchange ran against an unreadable store")

        with monkeypatch.context() as m:
            m.setattr(oauth, "_exchange_with_retry", no_exchange)
            self._unreadable(m, path)
            assert oauth._rotate_and_persist(
                path, "rotate.example", key, cred, now=1.0
            ) is None

        assert key in oauth._refresh_failure_at
        oauth._refresh_failure_at.pop(key, None)
        assert path.read_bytes() == before

    def test_corrupt_store_refuses_to_persist(self, tmp_path):
        path = tmp_path / "honcho.json"
        truncated = json.dumps(
            {"hosts": {"keep.example": _host_block(refresh="hch-rt-keep")}}
        )[:-5]
        path.write_text(truncated, encoding="utf-8")
        cred = OAuthCredential.from_host_block(_host_block(refresh="hch-rt-new"))

        with pytest.raises(ValueError):
            oauth._read_config_strict(path)
        with pytest.raises(ValueError):
            oauth._persist_credential(path, "keep.example", cred)
        with pytest.raises(ValueError):
            oauth.install_grant(
                path, "new.example",
                {"access_token": "hch-at-new", "refresh_token": "hch-rt-new", "expires_in": 3600},
                client_id="hermes-desktop", token_endpoint="http://localhost:8000/oauth/token",
            )
        assert path.read_text(encoding="utf-8") == truncated
        assert not path.with_name(path.name + ".corrupt").exists()

    def test_corrupt_store_fails_the_refresh_before_the_exchange(self, tmp_path, monkeypatch):
        path = tmp_path / "honcho.json"
        path.write_text("{not json", encoding="utf-8")
        cred = OAuthCredential.from_host_block(_host_block(refresh="hch-rt-rot"))
        key = (str(path), "rotate.example")
        oauth._refresh_failure_at.pop(key, None)

        def no_exchange(*a, **kw):  # pragma: no cover - must not run
            raise AssertionError("exchange ran against a corrupt store")

        monkeypatch.setattr(oauth, "_exchange_with_retry", no_exchange)
        assert oauth._rotate_and_persist(path, "rotate.example", key, cred, now=1.0) is None
        assert key in oauth._refresh_failure_at
        oauth._refresh_failure_at.pop(key, None)
        assert path.read_text(encoding="utf-8") == "{not json"

    def test_missing_store_still_bootstraps_empty(self, tmp_path):
        assert oauth._read_config_strict(tmp_path / "honcho.json") == {}

    def test_bom_prefixed_store_is_not_corruption(self, tmp_path):
        path = tmp_path / "honcho.json"
        path.write_text(
            json.dumps({"hosts": {"keep.example": _host_block()}}),
            encoding="utf-8-sig",
        )
        assert "keep.example" in oauth._read_config(path).get("hosts", {})
        assert "keep.example" in oauth._read_config_strict(path).get("hosts", {})

    def test_install_grant_raises_and_preserves_store_when_unreadable(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "honcho.json"
        before = self._seed_two_hosts(path)

        with monkeypatch.context() as m:
            self._unreadable(m, path)
            with pytest.raises(OSError):
                oauth.install_grant(
                    path, "new.example",
                    {"access_token": "hch-at-new", "refresh_token": "hch-rt-new",
                     "expires_in": 3600},
                    client_id="hermes-desktop",
                    token_endpoint="http://localhost:8000/oauth/token",
                )

        assert path.read_bytes() == before

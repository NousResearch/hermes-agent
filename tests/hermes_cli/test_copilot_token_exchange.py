"""Tests for Copilot token exchange (raw GitHub token → Copilot API token)."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clear_jwt_cache():
    """Reset the module-level JWT + failure caches before each test."""
    import hermes_cli.copilot_auth as mod
    mod._jwt_cache.clear()
    mod._exchange_failure_cache.clear()
    yield
    mod._jwt_cache.clear()
    mod._exchange_failure_cache.clear()


class TestExchangeCopilotToken:
    """Tests for exchange_copilot_token()."""

    def _mock_urlopen(self, token="tid=abc;exp=123;sku=copilot_individual", expires_at=None):
        """Create a mock urlopen context manager returning a token response."""
        if expires_at is None:
            expires_at = time.time() + 1800
        resp_data = json.dumps({"token": token, "expires_at": expires_at}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = resp_data
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch("urllib.request.urlopen")
    def test_exchanges_token_successfully(self, mock_urlopen):
        from hermes_cli.copilot_auth import exchange_copilot_token

        mock_urlopen.return_value = self._mock_urlopen(token="tid=abc;exp=999")
        api_token, expires_at, base_url = exchange_copilot_token("gho_test123")

        assert api_token == "tid=abc;exp=999"
        assert isinstance(expires_at, float)
        assert base_url is None  # no proxy-ep in this token

        # Verify request was made with correct headers
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.get_header("Authorization") == "token gho_test123"
        assert "GitHubCopilotChat" in req.get_header("User-agent")



    @patch("urllib.request.urlopen")
    def test_raises_on_empty_token(self, mock_urlopen):
        from hermes_cli.copilot_auth import exchange_copilot_token

        resp_data = json.dumps({"token": "", "expires_at": 0}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = resp_data
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        with pytest.raises(ValueError, match="empty token"):
            exchange_copilot_token("gho_test123")


class TestGetCopilotApiToken:
    """Tests for get_copilot_api_token() — the fallback wrapper."""

    @patch("hermes_cli.copilot_auth.exchange_copilot_token")
    def test_returns_exchanged_token(self, mock_exchange):
        from hermes_cli.copilot_auth import get_copilot_api_token

        mock_exchange.return_value = ("exchanged_jwt", time.time() + 1800, None)
        api_token, base_url = get_copilot_api_token("gho_raw")
        assert api_token == "exchanged_jwt"
        assert base_url is None


class TestTokenFingerprint:
    """Tests for _token_fingerprint()."""

    def test_consistent(self):
        from hermes_cli.copilot_auth import _token_fingerprint

        fp1 = _token_fingerprint("gho_abc123")
        fp2 = _token_fingerprint("gho_abc123")
        assert fp1 == fp2


class TestCallerIntegration:
    """Test that callers correctly use token exchange."""

    @patch("hermes_cli.copilot_auth.resolve_copilot_token", return_value=("gho_raw", "GH_TOKEN"))
    @patch("hermes_cli.copilot_auth.get_copilot_api_token", return_value=("exchanged_jwt", None))
    def test_auth_resolve_uses_exchange(self, mock_exchange, mock_resolve):
        from hermes_cli.auth import _resolve_api_key_provider_secret

        # Create a minimal pconfig mock
        pconfig = MagicMock()
        token, source = _resolve_api_key_provider_secret("copilot", pconfig)
        assert token == "exchanged_jwt"
        assert source == "GH_TOKEN"
        mock_exchange.assert_called_once_with("gho_raw")


class TestDeriveBaseUrlFromProxyEp:
    """Tests for _derive_base_url_from_proxy_ep()."""

    def test_extracts_enterprise_url(self):
        from hermes_cli.copilot_auth import _derive_base_url_from_proxy_ep

        token = "tid=abc;exp=999;proxy-ep=proxy.enterprise.githubcopilot.com;sku=copilot_enterprise"
        assert _derive_base_url_from_proxy_ep(token) == "https://api.enterprise.githubcopilot.com"




    @patch("urllib.request.urlopen")
    def test_exchange_returns_none_base_url_for_individual(self, mock_urlopen, _clear_jwt_cache):
        """exchange_copilot_token returns None base_url for individual accounts."""
        from hermes_cli.copilot_auth import exchange_copilot_token

        token_no_ep = "tid=abc;exp=999;sku=copilot_individual"
        expires_at = time.time() + 1800
        resp_data = json.dumps({"token": token_no_ep, "expires_at": expires_at}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = resp_data
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        api_token, _, base_url = exchange_copilot_token("gho_test")
        assert base_url is None


class TestJwtDiskStoreBounds:
    """The on-disk JWT store must go through one bounded read everywhere."""

    def _store_path(self, tmp_path, monkeypatch):
        import hermes_cli.copilot_auth as mod
        path = tmp_path / mod._JWT_DISK_FILENAME
        monkeypatch.setattr(mod, "_jwt_disk_path", lambda: path)
        return path

    def test_read_jwt_store_rejects_oversized_file(self, tmp_path, monkeypatch):
        import hermes_cli.copilot_auth as mod

        path = self._store_path(tmp_path, monkeypatch)
        path.write_text("x" * (mod._JWT_DISK_MAX_BYTES + 1))
        assert mod._read_jwt_store(path) is None
        # Load path treats it as unusable → caller re-exchanges.
        assert mod._load_jwt_from_disk("deadbeef") is None

    def test_read_jwt_store_rejects_non_dict_and_malformed(self, tmp_path, monkeypatch):
        import hermes_cli.copilot_auth as mod

        path = self._store_path(tmp_path, monkeypatch)
        path.write_text("[1, 2, 3]")
        assert mod._read_jwt_store(path) is None
        path.write_text("{not json")
        assert mod._read_jwt_store(path) is None

    def test_evict_ignores_oversized_store(self, tmp_path, monkeypatch):
        """Eviction on an oversized store must not parse or rewrite it."""
        import hermes_cli.copilot_auth as mod

        path = self._store_path(tmp_path, monkeypatch)
        blob = "x" * (mod._JWT_DISK_MAX_BYTES + 1)
        path.write_text(blob)
        mod.evict_cached_exchanged_token("gho_whatever")
        # Untouched — bounded read refused it before any rewrite.
        assert path.read_text() == blob

    def test_save_discards_oversized_store_instead_of_merging(self, tmp_path, monkeypatch):
        """Saving over a corrupt/oversized store starts fresh rather than
        re-serializing the oversized content back out."""
        import json as _json
        import time as _time
        import hermes_cli.copilot_auth as mod

        path = self._store_path(tmp_path, monkeypatch)
        path.write_text("x" * (mod._JWT_DISK_MAX_BYTES + 1))
        mod._save_jwt_to_disk("fp1", "tid=fresh", _time.time() + 1800, None)
        store = _json.loads(path.read_text())
        assert set(store) == {"fp1"}
        assert store["fp1"]["api_token"] == "tid=fresh"


class TestExchangeFailureFastPath:
    """Auth rejections must not sleep, and failures must not repeat network hits.

    Regression tests for the /model picker stall: a 403-rejected token made
    every load_pool("copilot") burn ~4.5s in retry backoff, turning provider
    discovery (picker open, delegation spawns, dashboard) into a 7s wait.
    """

    def _http_error(self, code):
        import urllib.error
        return urllib.error.HTTPError(
            url="https://api.github.com/copilot_internal/v2/token",
            code=code, msg="err", hdrs=None, fp=None,
        )

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_403_fails_fast_without_retry_or_sleep(self, mock_urlopen, mock_sleep):
        from hermes_cli.copilot_auth import exchange_copilot_token

        mock_urlopen.side_effect = self._http_error(403)
        with pytest.raises(ValueError):
            exchange_copilot_token("gho_rejected")
        assert mock_urlopen.call_count == 1  # no retries on auth rejection
        mock_sleep.assert_not_called()

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_negative_cache_skips_network_on_second_call(self, mock_urlopen, mock_sleep):
        from hermes_cli.copilot_auth import exchange_copilot_token

        mock_urlopen.side_effect = self._http_error(403)
        with pytest.raises(ValueError):
            exchange_copilot_token("gho_rejected")
        with pytest.raises(ValueError, match="recently failed"):
            exchange_copilot_token("gho_rejected")
        assert mock_urlopen.call_count == 1  # second call never hit the network

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_transient_failure_still_retries_then_caches(self, mock_urlopen, mock_sleep):
        import hermes_cli.copilot_auth as mod
        from hermes_cli.copilot_auth import exchange_copilot_token, _token_fingerprint

        mock_urlopen.side_effect = OSError("network unreachable")
        with pytest.raises(ValueError):
            exchange_copilot_token("gho_flaky")
        assert mock_urlopen.call_count == mod._EXCHANGE_MAX_ATTEMPTS
        fp = _token_fingerprint("gho_flaky")
        until = mod._exchange_failure_cache.get(fp, 0)
        # Transient TTL, not the 30-min permanent one.
        assert 0 < until - time.time() <= mod._EXCHANGE_FAILURE_TTL_TRANSIENT_SECONDS + 1

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_success_clears_negative_cache(self, mock_urlopen, mock_sleep):
        import hermes_cli.copilot_auth as mod
        from hermes_cli.copilot_auth import exchange_copilot_token, _token_fingerprint

        fp = _token_fingerprint("gho_recovering")
        # Simulate an expired negative-cache entry so the call proceeds.
        mod._exchange_failure_cache[fp] = time.time() - 1

        resp_data = json.dumps(
            {"token": "tid=ok;exp=1", "expires_at": time.time() + 1800}
        ).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = resp_data
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        api_token, _, _ = exchange_copilot_token("gho_recovering")
        assert api_token == "tid=ok;exp=1"
        assert fp not in mod._exchange_failure_cache

    def test_evict_clears_negative_cache(self):
        import hermes_cli.copilot_auth as mod
        from hermes_cli.copilot_auth import evict_cached_exchanged_token, _token_fingerprint

        fp = _token_fingerprint("gho_stale")
        mod._exchange_failure_cache[fp] = time.time() + 999
        evict_cached_exchanged_token("gho_stale")
        assert fp not in mod._exchange_failure_cache

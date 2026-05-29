"""Probe failure classification + decision logic for the custom-provider flow.

Task 7 / #3263: the endpoint probe used to swallow every error and fall back to
a single "could not verify, saving anyway" message. It now classifies the
failure (``error_class``) and a pure ``decide_probe_action`` weighs it into a
per-class behavior (save quietly / soft confirm / warn+re-enter / fail closed).
"""

from __future__ import annotations

import socket
import urllib.error
import io

import pytest

from hermes_cli.models import (
    classify_probe_error,
    decide_probe_action,
    _probe_host_is_local,
    probe_api_models,
)


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://x/models", code=code, msg="x", hdrs=None, fp=io.BytesIO(b"")
    )


def _url_error(reason) -> urllib.error.URLError:
    return urllib.error.URLError(reason)


# ── _probe_host_is_local ───────────────────────────────────────────────────


class TestHostIsLocal:
    @pytest.mark.parametrize("url", [
        "http://localhost:11434/v1",
        "http://127.0.0.1:8080/v1",
        "http://0.0.0.0:5000",
        "http://192.168.1.50:1234/v1",
        "http://10.0.0.4/v1",
        "http://[::1]:8000/v1",
        "http://my-box.local/v1",
    ])
    def test_local_hosts(self, url):
        assert _probe_host_is_local(url) is True

    @pytest.mark.parametrize("url", [
        "https://api.openai.com/v1",
        "https://openrouter.ai/api/v1",
        "http://8.8.8.8/v1",
        "",
    ])
    def test_remote_hosts(self, url):
        assert _probe_host_is_local(url) is False


# ── classify_probe_error ───────────────────────────────────────────────────


class TestClassifyProbeError:
    def test_404_is_no_catalog(self):
        ec, _ = classify_probe_error(_http_error(404), "https://api.x.com/v1")
        assert ec == "no_catalog"

    @pytest.mark.parametrize("code", [401, 403])
    def test_auth(self, code):
        ec, _ = classify_probe_error(_http_error(code), "https://api.x.com/v1")
        assert ec == "auth"

    @pytest.mark.parametrize("code", [500, 502, 503])
    def test_server_error(self, code):
        ec, _ = classify_probe_error(_http_error(code), "https://api.x.com/v1")
        assert ec == "server_error"

    def test_other_4xx_is_http_error(self):
        ec, _ = classify_probe_error(_http_error(418), "https://api.x.com/v1")
        assert ec == "http_error"

    def test_connection_refused_local(self):
        ec, _ = classify_probe_error(
            _url_error(ConnectionRefusedError(61, "refused")), "http://localhost:11434/v1"
        )
        assert ec == "local_refused"

    def test_connection_refused_remote(self):
        ec, _ = classify_probe_error(
            _url_error(ConnectionRefusedError(61, "refused")), "https://api.x.com/v1"
        )
        assert ec == "remote_unreachable"

    def test_dns_failure(self):
        ec, _ = classify_probe_error(
            _url_error(socket.gaierror(-2, "Name or service not known")),
            "https://typo.nonexist/v1",
        )
        assert ec == "dns"

    def test_timeout(self):
        ec, _ = classify_probe_error(_url_error(socket.timeout("timed out")), "https://api.x.com/v1")
        assert ec == "timeout"

    def test_bare_timeout_error(self):
        ec, _ = classify_probe_error(TimeoutError("timed out"), "https://api.x.com/v1")
        assert ec == "timeout"

    def test_unknown(self):
        ec, detail = classify_probe_error(ValueError("weird"), "https://api.x.com/v1")
        assert ec == "unknown"
        assert "weird" in detail


# ── decide_probe_action ────────────────────────────────────────────────────


class TestDecideProbeAction:
    def test_skip_validation_always_saves(self):
        for ec in ["", "auth", "dns", "local_refused", "unknown"]:
            d = decide_probe_action(ec, interactive=True, skip_validation=True)
            assert d.save is True and d.action == "save" and d.exit_nonzero is False

    @pytest.mark.parametrize("ec", ["", "no_catalog"])
    def test_ok_classes_save_quietly(self, ec):
        d = decide_probe_action(ec, interactive=True, skip_validation=False)
        assert d.save is True and d.action == "save" and d.exit_nonzero is False

    def test_404_counts_as_success_non_interactive(self):
        d = decide_probe_action("no_catalog", interactive=False, skip_validation=False)
        assert d.save is True and d.exit_nonzero is False

    @pytest.mark.parametrize("ec", ["auth", "dns", "local_refused", "timeout", "remote_unreachable", "unknown"])
    def test_non_interactive_fails_closed(self, ec):
        d = decide_probe_action(ec, interactive=False, skip_validation=False)
        assert d.save is False and d.action == "abort" and d.exit_nonzero is True

    @pytest.mark.parametrize("ec", ["local_refused", "timeout", "server_error"])
    def test_soft_classes_default_yes(self, ec):
        d = decide_probe_action(ec, interactive=True, skip_validation=False)
        assert d.action == "soft_confirm" and d.confirm_default is True and d.save is False

    def test_auth_offers_reenter_key_default_no(self):
        d = decide_probe_action("auth", interactive=True, skip_validation=False)
        assert d.action == "reenter_key" and d.confirm_default is False and d.save is False

    @pytest.mark.parametrize("ec", ["dns", "remote_unreachable"])
    def test_url_errors_reprompt_url_default_no(self, ec):
        d = decide_probe_action(ec, interactive=True, skip_validation=False)
        assert d.action == "reprompt_url" and d.confirm_default is False

    @pytest.mark.parametrize("ec", ["http_error", "unknown"])
    def test_block_misc_confirm_default_no(self, ec):
        d = decide_probe_action(ec, interactive=True, skip_validation=False)
        assert d.action == "soft_confirm" and d.confirm_default is False


# ── probe_api_models integration (no real network) ─────────────────────────


class TestProbeApiModelsClassification:
    def test_returns_error_class_on_failure(self, monkeypatch):
        """A connection-refused against a remote host surfaces error_class."""
        def _raise(*a, **k):
            raise urllib.error.URLError(ConnectionRefusedError(61, "refused"))

        monkeypatch.setattr("urllib.request.urlopen", _raise)
        result = probe_api_models("k", "https://api.example.com/v1", timeout=0.1)
        assert result["models"] is None
        assert result["error_class"] == "remote_unreachable"
        assert result["error_detail"]

    def test_404_prefers_http_error_over_connection(self, monkeypatch):
        """An HTTP 404 (reachable) is preferred over a connection error."""
        def _raise(*a, **k):
            raise _http_error(404)

        monkeypatch.setattr("urllib.request.urlopen", _raise)
        result = probe_api_models("k", "https://api.example.com/v1", timeout=0.1)
        assert result["error_class"] == "no_catalog"

    def test_success_sets_error_class_none(self, monkeypatch):
        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b'{"data": [{"id": "m1"}]}'

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())
        result = probe_api_models("k", "https://api.example.com/v1", timeout=0.1)
        assert result["models"] == ["m1"]
        assert result["error_class"] is None

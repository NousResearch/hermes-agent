"""Tests for short-circuiting probes to endpoints that blackhole TCP connects.

A routable-but-dead endpoint (e.g. a corp LAN address while off-VPN) drops SYNs
without a RST or ICMP error, so each probe waits out its full timeout. Once one
probe has observed that, the rest must not repeat it.

Covers:
- _endpoint_blackholed / _note_endpoint_blackholed host:port keying and TTL
- detect_local_server_type aborting its waterfall on the first connect timeout
- fetch_endpoint_model_metadata skipping its candidate loop once blackholed
- _query_ollama_api_show_uncached / _query_local_context_length_uncached
  honouring and recording the blackhole
- non-timeout failures (refused, no route) leaving the waterfall untouched
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _clear_caches():
    """Module-level caches must not leak between tests."""
    from agent import model_metadata
    model_metadata._endpoint_blackhole_cache.clear()
    model_metadata._endpoint_probe_path_cache.clear()
    model_metadata._endpoint_model_metadata_cache.clear()
    model_metadata._endpoint_model_metadata_cache_time.clear()
    model_metadata._LOCAL_CTX_PROBE_CACHE.clear()
    yield
    model_metadata._endpoint_blackhole_cache.clear()
    model_metadata._endpoint_probe_path_cache.clear()
    model_metadata._endpoint_model_metadata_cache.clear()
    model_metadata._endpoint_model_metadata_cache_time.clear()
    model_metadata._LOCAL_CTX_PROBE_CACHE.clear()


def _client_mock(side_effect):
    client = MagicMock()
    client.__enter__ = lambda s: client
    client.__exit__ = MagicMock(return_value=False)
    client.get.side_effect = side_effect
    client.post.side_effect = side_effect
    return client


class TestBlackholeCache:
    def test_unseen_endpoint_is_not_blackholed(self):
        from agent.model_metadata import _endpoint_blackholed

        assert _endpoint_blackholed("http://10.0.0.9:30080/v1") is False

    def test_note_then_detected(self):
        from agent.model_metadata import _endpoint_blackholed, _note_endpoint_blackholed

        _note_endpoint_blackholed("http://10.0.0.9:30080/v1")
        assert _endpoint_blackholed("http://10.0.0.9:30080/v1") is True

    def test_keyed_on_host_port_not_path(self):
        """Every probe path for one server shares a single entry."""
        from agent.model_metadata import _endpoint_blackholed, _note_endpoint_blackholed

        _note_endpoint_blackholed("http://10.0.0.9:30080")
        assert _endpoint_blackholed("http://10.0.0.9:30080/v1") is True
        assert _endpoint_blackholed("http://10.0.0.9:30080/api/v1") is True

    def test_different_port_is_independent(self):
        from agent.model_metadata import _endpoint_blackholed, _note_endpoint_blackholed

        _note_endpoint_blackholed("http://10.0.0.9:30080/v1")
        assert _endpoint_blackholed("http://10.0.0.9:11434/v1") is False

    def test_entry_expires_after_ttl(self):
        """A recovered endpoint (VPN back up) is probed again without a restart."""
        from agent import model_metadata
        from agent.model_metadata import _endpoint_blackholed, _note_endpoint_blackholed

        _note_endpoint_blackholed("http://10.0.0.9:30080/v1")
        stale = (
            model_metadata._endpoint_blackhole_cache["10.0.0.9:30080"]
            - model_metadata._ENDPOINT_BLACKHOLE_TTL_SECONDS
            - 1
        )
        model_metadata._endpoint_blackhole_cache["10.0.0.9:30080"] = stale
        assert _endpoint_blackholed("http://10.0.0.9:30080/v1") is False

    def test_ttl_zero_disables_short_circuit(self):
        from agent import model_metadata
        from agent.model_metadata import _endpoint_blackholed, _note_endpoint_blackholed

        _note_endpoint_blackholed("http://10.0.0.9:30080/v1")
        with patch.object(model_metadata, "_ENDPOINT_BLACKHOLE_TTL_SECONDS", 0.0):
            assert _endpoint_blackholed("http://10.0.0.9:30080/v1") is False


class TestDetectLocalServerTypeBlackhole:
    URL = "http://10.0.0.9:30080/v1"

    def test_connect_timeout_aborts_waterfall_after_one_probe(self):
        """Four sequential 2s probes against a dead host must collapse to one."""
        from agent.model_metadata import _endpoint_blackholed, detect_local_server_type

        client = _client_mock(httpx.ConnectTimeout("timed out"))
        with patch("httpx.Client", return_value=client):
            assert detect_local_server_type(self.URL) is None

        assert client.get.call_count == 1
        assert _endpoint_blackholed(self.URL) is True

    def test_second_call_makes_no_request_at_all(self):
        from agent.model_metadata import detect_local_server_type

        client = _client_mock(httpx.ConnectTimeout("timed out"))
        with patch("httpx.Client", return_value=client):
            detect_local_server_type(self.URL)
            first_count = client.get.call_count
            assert detect_local_server_type(self.URL) is None

        assert client.get.call_count == first_count

    def test_refused_does_not_blackhole_and_runs_full_waterfall(self):
        """Refused answers instantly, so skipping buys nothing and must not fire.

        This is the common "local server not started yet" path.
        """
        from agent.model_metadata import _endpoint_blackholed, detect_local_server_type

        client = _client_mock(httpx.ConnectError("connection refused"))
        with patch("httpx.Client", return_value=client):
            assert detect_local_server_type(self.URL) is None

        assert client.get.call_count > 1
        assert _endpoint_blackholed(self.URL) is False

    def test_read_timeout_does_not_blackhole(self):
        """A read timeout means the connection was accepted — not a blackhole."""
        from agent.model_metadata import _endpoint_blackholed, detect_local_server_type

        client = _client_mock(httpx.ReadTimeout("slow"))
        with patch("httpx.Client", return_value=client):
            detect_local_server_type(self.URL)

        assert _endpoint_blackholed(self.URL) is False


class TestFetchEndpointModelMetadataBlackhole:
    URL = "http://10.0.0.9:30080/v1"

    def test_connect_timeout_skips_remaining_candidates(self):
        """A timeout condemns the host, not the URL suffix — one stall, not two."""
        from agent.model_metadata import _endpoint_blackholed, fetch_endpoint_model_metadata

        with patch("agent.model_metadata.detect_local_server_type", return_value=None), \
             patch(
                 "agent.model_metadata.requests.get",
                 side_effect=requests.exceptions.ConnectTimeout("timed out"),
             ) as get:
            assert fetch_endpoint_model_metadata(self.URL) == {}

        assert get.call_count == 1
        assert _endpoint_blackholed(self.URL) is True

    def test_refused_tries_every_candidate_and_does_not_blackhole(self):
        from agent.model_metadata import _endpoint_blackholed, fetch_endpoint_model_metadata

        with patch("agent.model_metadata.detect_local_server_type", return_value=None), \
             patch(
                 "agent.model_metadata.requests.get",
                 side_effect=requests.exceptions.ConnectionError("refused"),
             ) as get:
            assert fetch_endpoint_model_metadata(self.URL) == {}

        assert get.call_count == 2  # /v1-suffixed and bare candidates
        assert _endpoint_blackholed(self.URL) is False

    def test_blackholed_endpoint_issues_no_request(self):
        """force_refresh bypasses the metadata cache, so only the guard can stop it."""
        from agent.model_metadata import _note_endpoint_blackholed, fetch_endpoint_model_metadata

        _note_endpoint_blackholed(self.URL)
        with patch("agent.model_metadata.detect_local_server_type", return_value=None), \
             patch("agent.model_metadata.requests.get") as get:
            assert fetch_endpoint_model_metadata(self.URL, force_refresh=True) == {}

        get.assert_not_called()


class TestQueryOllamaApiShowBlackhole:
    URL = "http://10.0.0.9:30080/v1"

    def test_connect_timeout_records_blackhole(self):
        from agent.model_metadata import _endpoint_blackholed, _query_ollama_api_show_uncached

        client = _client_mock(httpx.ConnectTimeout("timed out"))
        with patch("httpx.Client", return_value=client):
            assert _query_ollama_api_show_uncached("some-model", self.URL) is None

        assert client.post.call_count == 1
        assert _endpoint_blackholed(self.URL) is True

    def test_blackholed_endpoint_issues_no_request(self):
        from agent.model_metadata import _note_endpoint_blackholed, _query_ollama_api_show_uncached

        _note_endpoint_blackholed(self.URL)
        with patch("httpx.Client") as client_cls:
            assert _query_ollama_api_show_uncached("some-model", self.URL) is None

        client_cls.assert_not_called()

    def test_read_timeout_does_not_blackhole(self):
        from agent.model_metadata import _endpoint_blackholed, _query_ollama_api_show_uncached

        client = _client_mock(httpx.ReadTimeout("slow"))
        with patch("httpx.Client", return_value=client):
            assert _query_ollama_api_show_uncached("some-model", self.URL) is None

        assert _endpoint_blackholed(self.URL) is False


class TestQueryLocalContextLengthBlackhole:
    URL = "http://10.0.0.9:30080/v1"

    def test_connect_timeout_records_blackhole(self):
        from agent.model_metadata import (
            _endpoint_blackholed,
            _query_local_context_length_uncached,
        )

        client = _client_mock(httpx.ConnectTimeout("timed out"))
        with patch("agent.model_metadata.detect_local_server_type", return_value=None), \
             patch("httpx.Client", return_value=client):
            assert _query_local_context_length_uncached("some-model", self.URL) is None

        assert _endpoint_blackholed(self.URL) is True

    def test_blackholed_endpoint_skips_detection_and_requests(self):
        """The guard sits before detect_local_server_type — nothing runs at all."""
        from agent.model_metadata import (
            _note_endpoint_blackholed,
            _query_local_context_length_uncached,
        )

        _note_endpoint_blackholed(self.URL)
        with patch("agent.model_metadata.detect_local_server_type") as detect, \
             patch("httpx.Client") as client_cls:
            assert _query_local_context_length_uncached("some-model", self.URL) is None

        detect.assert_not_called()
        client_cls.assert_not_called()

    def test_read_timeout_does_not_blackhole(self):
        from agent.model_metadata import (
            _endpoint_blackholed,
            _query_local_context_length_uncached,
        )

        client = _client_mock(httpx.ReadTimeout("slow"))
        with patch("agent.model_metadata.detect_local_server_type", return_value=None), \
             patch("httpx.Client", return_value=client):
            assert _query_local_context_length_uncached("some-model", self.URL) is None

        assert _endpoint_blackholed(self.URL) is False


class TestIsConnectTimeout:
    def test_httpx_connect_timeout(self):
        from agent.model_metadata import _is_connect_timeout

        assert _is_connect_timeout(httpx.ConnectTimeout("x")) is True

    def test_requests_connect_timeout(self):
        from requests.exceptions import ConnectTimeout

        from agent.model_metadata import _is_connect_timeout

        assert _is_connect_timeout(ConnectTimeout("x")) is True

    def test_unrelated_errors_are_not_connect_timeouts(self):
        from agent.model_metadata import _is_connect_timeout

        assert _is_connect_timeout(httpx.ReadTimeout("x")) is False
        assert _is_connect_timeout(httpx.ConnectError("x")) is False
        assert _is_connect_timeout(ValueError("x")) is False

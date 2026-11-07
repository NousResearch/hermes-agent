"""Tests for the local-endpoint probe disk L2 cache in agent/model_metadata.

Contract:
- only SUCCESSFUL probes are persisted (failures never pin a verdict);
- fresh disk entries serve cross-process (in-proc caches cleared);
- entries expire after _LOCAL_PROBE_DISK_TTL_SECONDS;
- corrupted cache files degrade to a miss;
- detect_local_server_type and query_ollama_num_ctx both use it.
"""

import json
import time
from unittest.mock import MagicMock, patch

import httpx

import agent.model_metadata as MM
from agent.model_metadata_server_type import SERVER_TYPE_DISK_TTL_SECONDS


def _clear_in_proc():
    MM._endpoint_probe_path_cache.clear()
    MM._LOCAL_CTX_PROBE_CACHE.clear()


def _cache_file():
    return MM._local_probe_disk_cache_path()


class TestDiskHelpers:

    def test_expired_entry_is_miss(self):
        MM._local_probe_disk_put("server_type", "http://127.0.0.1:11434", "ollama")
        path = _cache_file()
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data.values():
            entry["ts"] = time.time() - MM._LOCAL_PROBE_DISK_TTL_SECONDS - 1
        path.write_text(json.dumps(data), encoding="utf-8")
        assert MM._local_probe_disk_get("server_type", "http://127.0.0.1:11434") is None

    def test_corrupted_file_is_miss(self):
        path = _cache_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json{{{", encoding="utf-8")
        assert MM._local_probe_disk_get("server_type", "anything") is None
        # And put() recovers by rewriting cleanly
        MM._local_probe_disk_put("server_type", "k", "vllm")
        assert MM._local_probe_disk_get("server_type", "k") == "vllm"



class TestDetectLocalServerTypeDiskL2:
    def test_disk_hit_revalidates_with_one_cheap_request(self):
        _clear_in_proc()
        MM._local_probe_disk_put("server_type", "http://127.0.0.1:9999", "ollama")
        ok = MagicMock(status_code=200)
        ok.json.return_value = {"models": []}
        client = MagicMock()
        client.__enter__ = lambda s: client
        client.__exit__ = lambda s, *a: False
        client.get.return_value = ok
        with patch("httpx.Client", return_value=client):
            result = MM.detect_local_server_type("http://127.0.0.1:9999/v1")
        assert result == "ollama"
        assert client.get.call_count == 1  # the certified leg only; no waterfall

    def test_successful_probe_persists_to_disk(self):
        _clear_in_proc()
        ok = MagicMock(status_code=200)
        ok.json.return_value = {"models": []}
        notfound = MagicMock(status_code=404, text="")

        def route(url, *a, **k):
            return ok if url.endswith("/api/tags") else notfound

        client = MagicMock()
        client.get.side_effect = route
        client.__enter__ = lambda s: client
        client.__exit__ = lambda s, *a: False
        with patch("httpx.Client", return_value=client):
            result = MM.detect_local_server_type("http://127.0.0.1:8888/v1")
        assert result == "ollama"
        assert MM._local_probe_disk_get("server_type", "http://127.0.0.1:8888") == "ollama"

    def test_failed_probe_not_persisted(self):
        _clear_in_proc()
        client = MagicMock()
        client.get.side_effect = Exception("connection refused")
        client.__enter__ = lambda s: client
        client.__exit__ = lambda s, *a: False
        with patch("httpx.Client", return_value=client):
            result = MM.detect_local_server_type("http://127.0.0.1:7777/v1")
        assert result is None
        assert MM._local_probe_disk_get("server_type", "http://127.0.0.1:7777") is None

    def test_positive_verdict_survives_the_5min_probe_ttl(self):
        """A verdict older than _LOCAL_PROBE_DISK_TTL_SECONDS but within the
        server-type TTL must still skip the waterfall (>=5-min idle windows
        were re-running the serial 404 burst on every first message); only the
        single revalidation leg runs."""
        _clear_in_proc()
        MM._local_probe_disk_put("server_type", "http://127.0.0.1:9001", "vllm")
        path = MM._local_probe_disk_cache_path()
        data = json.loads(path.read_text(encoding="utf-8"))
        data["server_type:http://127.0.0.1:9001"]["ts"] = (
            time.time() - MM._LOCAL_PROBE_DISK_TTL_SECONDS - 60
        )
        path.write_text(json.dumps(data), encoding="utf-8")
        ok = MagicMock(status_code=200)
        ok.json.return_value = {"version": "0.1"}
        client = MagicMock()
        client.__enter__ = lambda s: client
        client.__exit__ = lambda s, *a: False
        client.get.return_value = ok
        with patch("httpx.Client", return_value=client):
            result = MM.detect_local_server_type("http://127.0.0.1:9001/v1")
        assert result == "vllm"
        assert client.get.call_count == 1  # revalidation only, not the waterfall

    def test_positive_verdict_expires_at_the_server_type_ttl(self):
        """Beyond the day-long verdict TTL the waterfall must run again
        (a server swap under the same base_url is eventually re-detected)."""
        _clear_in_proc()
        MM._local_probe_disk_put("server_type", "http://127.0.0.1:9002", "vllm")
        path = MM._local_probe_disk_cache_path()
        data = json.loads(path.read_text(encoding="utf-8"))
        data["server_type:http://127.0.0.1:9002"]["ts"] = (
            time.time() - SERVER_TYPE_DISK_TTL_SECONDS - 60
        )
        path.write_text(json.dumps(data), encoding="utf-8")
        client = MagicMock()
        client.get.side_effect = Exception("connection refused")
        client.__enter__ = lambda s: client
        client.__exit__ = lambda s, *a: False
        with patch("httpx.Client", return_value=client):
            result = MM.detect_local_server_type("http://127.0.0.1:9002/v1")
        assert result is None
        assert client.get.called

    def test_runtime_window_kinds_keep_the_short_ttl(self):
        """ollama_num_ctx is a restart-sensitive value: it must still expire at 300 s."""
        MM._local_probe_disk_put("ollama_num_ctx", "http://127.0.0.1:9003|m", 131072)
        path = MM._local_probe_disk_cache_path()
        data = json.loads(path.read_text(encoding="utf-8"))
        data["ollama_num_ctx:http://127.0.0.1:9003|m"]["ts"] = (
            time.time() - MM._LOCAL_PROBE_DISK_TTL_SECONDS - 60
        )
        path.write_text(json.dumps(data), encoding="utf-8")
        assert MM._local_probe_disk_get("ollama_num_ctx", "http://127.0.0.1:9003|m") is None

    def test_short_ttl_write_keeps_day_long_verdicts(self):
        """Every write prunes the shared file: a 300 s-kind write must not drop a server_type
        verdict that is past 300 s but still inside its day (another host, or a model switch)."""
        MM._local_probe_disk_put("server_type", "http://127.0.0.1:9004", "vllm")
        path = MM._local_probe_disk_cache_path()
        data = json.loads(path.read_text(encoding="utf-8"))
        data["server_type:http://127.0.0.1:9004"]["ts"] = time.time() - 600
        path.write_text(json.dumps(data), encoding="utf-8")
        MM._local_probe_disk_put("ollama_num_ctx", "http://127.0.0.1:9005|m", 8192)
        assert MM._local_probe_disk_get("server_type", "http://127.0.0.1:9004", SERVER_TYPE_DISK_TTL_SECONDS) == "vllm"


class TestServerTypeRevalidation:
    """A day-long disk verdict must not outlive the server it describes.

    A disk hit is revalidated with the one waterfall leg that certified the kind: a swap
    under the same base_url self-heals on next use (1 extra request, not the 5-leg burst);
    a network blip keeps the verdict; a confirmed match refreshes the entry's freshness."""

    URL = "http://127.0.0.1:9100"

    def _client(self, get_fn):
        client = MagicMock()
        client.__enter__ = lambda s: client
        client.__exit__ = lambda s, *a: False
        client.get.side_effect = get_fn
        return client

    def _resp(self, status, body=None):
        r = MagicMock()
        r.status_code = status
        r.json.return_value = body if isinstance(body, dict) else {}
        r.text = body if isinstance(body, str) else ""
        return r

    def test_swap_under_same_url_self_heals(self):
        """Disk says vllm, the port now answers like ollama: revalidation mismatches, the
        waterfall re-runs, and the disk entry is REWRITTEN with the new kind."""
        _clear_in_proc()
        MM._local_probe_disk_put("server_type", self.URL, "vllm")

        def get(url):
            if url.endswith("/api/tags"):
                return self._resp(200, {"models": []})
            return self._resp(404)

        client = self._client(get)
        with patch("httpx.Client", return_value=client):
            assert MM.detect_local_server_type(f"{self.URL}/v1") == "ollama"
        assert MM._local_probe_disk_get("server_type", self.URL) == "ollama"
        # 1 revalidation leg (/version) + lm-studio + ollama; vllm leg never reached (break on match).
        assert client.get.call_count == 3

    def test_match_is_one_request_and_refreshes_the_verdict(self):
        _clear_in_proc()
        MM._local_probe_disk_put("server_type", self.URL, "vllm")
        path = MM._local_probe_disk_cache_path()
        data = json.loads(path.read_text(encoding="utf-8"))
        stale_ts = time.time() - 120
        data[f"server_type:{self.URL}"]["ts"] = stale_ts
        path.write_text(json.dumps(data), encoding="utf-8")

        client = self._client(
            lambda url: self._resp(200, {"version": "0.1"}) if url.endswith("/version") else self._resp(404)
        )
        with patch("httpx.Client", return_value=client):
            assert MM.detect_local_server_type(f"{self.URL}/v1") == "vllm"
        assert client.get.call_count == 1  # no waterfall
        refreshed = json.loads(path.read_text(encoding="utf-8"))[f"server_type:{self.URL}"]["ts"]
        assert refreshed > stale_ts  # confirmed match buys the verdict another day

    def test_network_blip_keeps_the_verdict_and_does_not_blackhole(self):
        """A down/flaky server during revalidation must neither void the verdict nor poison
        the host: the next healthy call revalidates and succeeds."""
        _clear_in_proc()
        MM._local_probe_disk_put("server_type", self.URL, "vllm")

        def boom(url):
            raise httpx.ConnectTimeout("connect timeout")

        client = self._client(boom)
        with patch("httpx.Client", return_value=client):
            assert MM.detect_local_server_type(f"{self.URL}/v1") == "vllm"
        assert client.get.call_count == 1  # no waterfall storm behind a blip

        _clear_in_proc()
        ok = self._client(lambda url: self._resp(200, {"version": "0.1"}))
        with patch("httpx.Client", return_value=ok):
            assert MM.detect_local_server_type(f"{self.URL}/v1") == "vllm"
        assert ok.get.call_count == 1  # revalidated, host was never marked blackholed

    def test_non_json_200_on_the_certified_leg_is_a_swap_not_a_blip(self):
        """Another server answering /version with an HTML 200 must re-run the waterfall, not
        keep the stale verdict as if the host were merely unreachable."""
        _clear_in_proc()
        MM._local_probe_disk_put("server_type", self.URL, "vllm")

        def get(url):
            if url.endswith("/version"):
                r = self._resp(200, "<html>not vllm</html>")
                r.json.side_effect = ValueError("not json")
                return r
            if url.endswith("/api/tags"):
                return self._resp(200, {"models": []})
            return self._resp(404)

        client = self._client(get)
        with patch("httpx.Client", return_value=client):
            assert MM.detect_local_server_type(f"{self.URL}/v1") == "ollama"
        assert MM._local_probe_disk_get("server_type", self.URL) == "ollama"


class TestOllamaNumCtxDiskL2:
    def test_disk_hit_skips_api_show(self):
        _clear_in_proc()
        MM._local_probe_disk_put("server_type", "http://127.0.0.1:11434", "ollama")
        MM._local_probe_disk_put(
            "ollama_num_ctx", "http://127.0.0.1:11434|llama3", 131072
        )
        ok = MagicMock(status_code=200)
        ok.json.return_value = {"models": []}
        client = MagicMock()
        client.__enter__ = lambda s: client
        client.__exit__ = lambda s, *a: False
        client.get.return_value = ok
        with patch("httpx.Client", return_value=client):
            ctx = MM.query_ollama_num_ctx("llama3", "http://127.0.0.1:11434/v1")
        assert ctx == 131072
        client.post.assert_not_called()  # num_ctx disk hit still skips /api/show
        assert client.get.call_count == 1  # server-type revalidation leg only

    def test_successful_show_persists(self):
        _clear_in_proc()
        MM._local_probe_disk_put("server_type", "http://127.0.0.1:11434", "ollama")
        resp = MagicMock(status_code=200)
        resp.json.return_value = {
            "parameters": "",
            "model_info": {"llama.context_length": 8192},
        }
        ok = MagicMock(status_code=200)
        ok.json.return_value = {"models": []}
        client = MagicMock()
        client.post.return_value = resp
        client.get.return_value = ok
        client.__enter__ = lambda s: client
        client.__exit__ = lambda s, *a: False
        with patch("httpx.Client", return_value=client):
            ctx = MM.query_ollama_num_ctx("llama3", "http://127.0.0.1:11434/v1")
        assert ctx == 8192
        assert (
            MM._local_probe_disk_get("ollama_num_ctx", "http://127.0.0.1:11434|llama3")
            == 8192
        )

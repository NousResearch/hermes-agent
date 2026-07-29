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

import agent.model_metadata as MM


def _clear_in_proc():
    MM._endpoint_probe_path_cache.clear()
    MM._LOCAL_CTX_PROBE_CACHE.clear()


def _cache_file():
    return MM._local_probe_disk_cache_path()


class TestDiskHelpers:
    def test_put_get_roundtrip(self):
        MM._local_probe_disk_put("server_type", "http://127.0.0.1:11434", "ollama")
        assert MM._local_probe_disk_get("server_type", "http://127.0.0.1:11434") == "ollama"

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

    def test_stale_entries_pruned_on_put(self):
        MM._local_probe_disk_put("server_type", "old", "ollama")
        path = _cache_file()
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data.values():
            entry["ts"] = time.time() - MM._LOCAL_PROBE_DISK_TTL_SECONDS - 1
        path.write_text(json.dumps(data), encoding="utf-8")
        MM._local_probe_disk_put("server_type", "new", "vllm")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "server_type:old" not in data
        assert "server_type:new" in data


class TestDetectLocalServerTypeDiskL2:
    def test_disk_hit_skips_http_entirely(self):
        _clear_in_proc()
        MM._local_probe_disk_put("server_type", "http://127.0.0.1:9999", "ollama")
        with patch("httpx.Client") as mock_client:
            result = MM.detect_local_server_type("http://127.0.0.1:9999/v1")
        assert result == "ollama"
        mock_client.assert_not_called()

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


class TestOllamaNumCtxDiskL2:
    def test_disk_hit_skips_api_show(self):
        _clear_in_proc()
        MM._local_probe_disk_put("server_type", "http://127.0.0.1:11434", "ollama")
        MM._local_probe_disk_put(
            "ollama_num_ctx", "http://127.0.0.1:11434|llama3", 131072
        )
        with patch("httpx.Client") as mock_client:
            ctx = MM.query_ollama_num_ctx("llama3", "http://127.0.0.1:11434/v1")
        assert ctx == 131072
        mock_client.assert_not_called()

    def test_successful_show_persists(self):
        _clear_in_proc()
        MM._local_probe_disk_put("server_type", "http://127.0.0.1:11434", "ollama")
        resp = MagicMock(status_code=200)
        resp.json.return_value = {
            "parameters": "",
            "model_info": {"llama.context_length": 8192},
        }
        client = MagicMock()
        client.post.return_value = resp
        client.__enter__ = lambda s: client
        client.__exit__ = lambda s, *a: False
        with patch("httpx.Client", return_value=client):
            ctx = MM.query_ollama_num_ctx("llama3", "http://127.0.0.1:11434/v1")
        assert ctx == 8192
        assert (
            MM._local_probe_disk_get("ollama_num_ctx", "http://127.0.0.1:11434|llama3")
            == 8192
        )

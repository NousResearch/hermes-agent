# Regression coverage for hermes-agent#31000.

import json
import threading
import time

import pytest

from plugins.memory.openviking import OpenVikingMemoryProvider


class _FakeVikingClient:
    def __init__(self, *, fail_post: bool = False):
        self._lock = threading.Lock()
        self.calls = []
        self.fail_post = fail_post

    def post(self, path, payload=None, **kwargs):
        with self._lock:
            self.calls.append(("post", path, dict(payload or {}), dict(kwargs)))
        if self.fail_post:
            raise RuntimeError("mirror write failed")
        if path == "/api/v1/content/write":
            return {
                "status": "ok",
                "result": {
                    "uri": (payload or {}).get("uri", ""),
                    "written_bytes": len((payload or {}).get("content", "")),
                },
            }
        return {"status": "ok", "result": {}}

    def delete(self, path, **kwargs):
        with self._lock:
            self.calls.append(("delete", path, None, dict(kwargs)))
        return {"status": "ok", "result": {"uri": kwargs.get("params", {}).get("uri", "")}}

    def snapshot(self):
        with self._lock:
            return list(self.calls)


def _provider(tmp_path, client):
    provider = OpenVikingMemoryProvider()
    provider._hermes_home = str(tmp_path)
    provider._agent = "hermes"
    provider._client = client
    provider._ensure_client = lambda: client
    provider._new_client = lambda: client
    return provider


def _wait_for(predicate, *, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("timed out waiting for OpenViking mirror worker")


def _registry_path(tmp_path):
    return tmp_path / "openviking" / "memory_mirror_registry.json"


def test_replace_updates_the_same_openviking_uri_and_registry(tmp_path):
    client = _FakeVikingClient()
    provider = _provider(tmp_path, client)

    provider.on_memory_write("add", "user", "Preferred provider is DeepInfra")
    _wait_for(lambda: len(client.snapshot()) == 1)
    create = client.snapshot()[0]
    uri = create[2]["uri"]

    provider.on_memory_write(
        "replace",
        "user",
        "Preferred provider is OpenRouter",
        metadata={"old_text": "DeepInfra"},
    )
    _wait_for(lambda: len(client.snapshot()) == 2)

    replace = client.snapshot()[1]
    assert replace[0:2] == ("post", "/api/v1/content/write")
    assert replace[2]["uri"] == uri
    assert replace[2]["content"] == "Preferred provider is OpenRouter"
    assert replace[2]["mode"] == "replace"
    assert replace[2]["wait"] is True

    registry = json.loads(_registry_path(tmp_path).read_text(encoding="utf-8"))
    assert registry == {
        "version": 1,
        "entries": [
            {
                "target": "user",
                "uri": uri,
                "content": "Preferred provider is OpenRouter",
            }
        ],
    }
    provider.shutdown()


def test_remove_deletes_exact_mapped_uri_and_registry_entry(tmp_path):
    client = _FakeVikingClient()
    provider = _provider(tmp_path, client)

    provider.on_memory_write("add", "memory", "Project alpha is active")
    _wait_for(lambda: len(client.snapshot()) == 1)
    uri = client.snapshot()[0][2]["uri"]

    provider.on_memory_write(
        "remove",
        "memory",
        "",
        metadata={"old_text": "alpha is active"},
    )
    _wait_for(lambda: len(client.snapshot()) == 2)

    delete = client.snapshot()[1]
    assert delete[0:2] == ("delete", "/api/v1/fs")
    assert delete[3]["params"] == {"uri": uri, "recursive": False, "wait": True}

    registry = json.loads(_registry_path(tmp_path).read_text(encoding="utf-8"))
    assert registry == {"version": 1, "entries": []}
    provider.shutdown()


def test_mapping_survives_provider_restart_for_true_replace(tmp_path):
    client = _FakeVikingClient()
    first = _provider(tmp_path, client)

    first.on_memory_write("add", "user", "Employment status is job seeking")
    _wait_for(lambda: len(client.snapshot()) == 1)
    uri = client.snapshot()[0][2]["uri"]
    first.shutdown()

    second = _provider(tmp_path, client)
    second.on_memory_write(
        "replace",
        "user",
        "Employment status is employed",
        metadata={"old_text": "job seeking"},
    )
    _wait_for(lambda: len(client.snapshot()) == 2)

    replace = client.snapshot()[1]
    assert replace[2]["uri"] == uri
    assert replace[2]["mode"] == "replace"
    assert replace[2]["wait"] is True
    assert replace[2]["content"] == "Employment status is employed"
    second.shutdown()


def test_rapid_add_replace_remove_is_processed_in_fifo_order(tmp_path):
    client = _FakeVikingClient()
    provider = _provider(tmp_path, client)

    provider.on_memory_write("add", "user", "Device owned: Tablet A")
    provider.on_memory_write(
        "replace",
        "user",
        "Device owned: Tablet B",
        metadata={"old_text": "Tablet A"},
    )
    provider.on_memory_write(
        "remove",
        "user",
        "",
        metadata={"old_text": "Tablet B"},
    )

    _wait_for(lambda: len(client.snapshot()) == 3)
    calls = client.snapshot()
    uri = calls[0][2]["uri"]

    assert [(call[0], call[1]) for call in calls] == [
        ("post", "/api/v1/content/write"),
        ("post", "/api/v1/content/write"),
        ("delete", "/api/v1/fs"),
    ]
    assert calls[0][2]["mode"] == "create"
    assert calls[1][2]["mode"] == "replace"
    assert calls[1][2]["wait"] is True
    assert calls[1][2]["uri"] == uri
    assert calls[2][3]["params"] == {"uri": uri, "recursive": False, "wait": True}

    registry = json.loads(_registry_path(tmp_path).read_text(encoding="utf-8"))
    assert registry == {"version": 1, "entries": []}
    provider.shutdown()


def test_unmapped_replace_fails_closed_with_warning(tmp_path, caplog):
    client = _FakeVikingClient()
    provider = _provider(tmp_path, client)

    with caplog.at_level("WARNING", logger="plugins.memory.openviking"):
        provider.on_memory_write(
            "replace",
            "user",
            "Employment status is employed",
            metadata={"old_text": "job seeking"},
        )
        provider.shutdown()

    assert client.snapshot() == []
    assert any(
        "no stable OpenViking URI mapping" in record.message
        for record in caplog.records
    )


def test_ambiguous_replace_fails_closed_without_guessing(tmp_path, caplog):
    client = _FakeVikingClient()
    provider = _provider(tmp_path, client)

    provider.on_memory_write("add", "user", "Device owned: Tablet A")
    provider.on_memory_write("add", "user", "Device owned: Tablet B")
    _wait_for(lambda: len(client.snapshot()) == 2)

    with caplog.at_level("WARNING", logger="plugins.memory.openviking"):
        provider.on_memory_write(
            "replace",
            "user",
            "Device owned: Tablet C",
            metadata={"old_text": "Tablet"},
        )
        provider.shutdown()

    assert len(client.snapshot()) == 2
    assert any(
        "matched 2 OpenViking URI mappings" in record.message
        for record in caplog.records
    )


def test_final_mirror_failure_is_visible_at_warning_level(tmp_path, caplog):
    client = _FakeVikingClient(fail_post=True)
    provider = _provider(tmp_path, client)

    with caplog.at_level("WARNING", logger="plugins.memory.openviking"):
        provider.on_memory_write("add", "user", "Visible failure")
        _wait_for(lambda: len(client.snapshot()) == 1)
        _wait_for(
            lambda: any(
                record.levelname == "WARNING" and "OpenViking memory mirror failed" in record.message
                for record in caplog.records
            )
        )

    assert not _registry_path(tmp_path).exists()
    provider.shutdown()

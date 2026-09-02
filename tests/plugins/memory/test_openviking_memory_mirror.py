# Regression coverage for hermes-agent#31000.

import json
import threading
import time

import pytest

from plugins.memory.openviking import OpenVikingMemoryProvider
from plugins.memory.openviking.native_memory_mirror import NativeMemoryMirror


class _FakeVikingClient:
    def __init__(
        self,
        *,
        endpoint: str = "http://openviking.test",
        api_key: str = "test-key",
        account: str = "default",
        user: str = "default",
        agent: str = "hermes",
        fail_post: bool = False,
        fail_delete: bool = False,
    ):
        self._lock = threading.Lock()
        self.calls = []
        self.fail_post = fail_post
        self.fail_delete = fail_delete
        self._endpoint = endpoint
        self._api_key = api_key
        self._account = account
        self._user = user
        self._agent = agent

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

    def get(self, path, params=None, **kwargs):
        if path == "/api/v1/system/status":
            return {"status": "ok", "result": {"user": "default"}}
        return {"status": "ok", "result": {}}

    def delete(self, path, **kwargs):
        with self._lock:
            self.calls.append(("delete", path, None, dict(kwargs)))
        if self.fail_delete:
            raise RuntimeError("mirror delete failed")
        return {
            "status": "ok",
            "result": {"uri": kwargs.get("params", {}).get("uri", "")},
        }

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

    provider.on_memory_write("add", "user", "  Preferred provider is DeepInfra  ")
    _wait_for(lambda: len(client.snapshot()) == 1)
    create = client.snapshot()[0]
    uri = create[2]["uri"]
    assert create[2]["content"] == "Preferred provider is DeepInfra"

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

    _wait_for(
        lambda: json.loads(_registry_path(tmp_path).read_text(encoding="utf-8"))[
            "entries"
        ][0]["content"]
        == "Preferred provider is OpenRouter"
    )
    registry = json.loads(_registry_path(tmp_path).read_text(encoding="utf-8"))
    connection = registry["entries"][0]["connection"]
    assert len(connection) == 64
    assert registry == {
        "version": 2,
        "entries": [
            {
                "connection": connection,
                "target": "user",
                "uri": uri,
                "content": "Preferred provider is OpenRouter",
            }
        ],
    }
    provider.shutdown()


def test_add_requires_server_confirmed_user_without_fallback(tmp_path, caplog):
    client = _FakeVikingClient(user="configured-fallback")

    def fail_identity_probe(path, params=None, **kwargs):
        assert path == "/api/v1/system/status"
        raise TimeoutError("identity probe timed out")

    client.get = fail_identity_probe
    provider = _provider(tmp_path, client)

    with caplog.at_level("WARNING", logger="plugins.memory.openviking"):
        provider.on_memory_write("add", "user", "Do not misroute this memory")
        provider.shutdown()

    assert client.snapshot() == []
    assert not _registry_path(tmp_path).exists()
    assert any(
        "did not confirm the current user identity" in record.message
        for record in caplog.records
    )


def test_add_identity_probe_uses_normal_request_timeout(tmp_path):
    client = _FakeVikingClient()
    identity_probe_kwargs = []

    def record_identity_probe(path, params=None, **kwargs):
        assert path == "/api/v1/system/status"
        identity_probe_kwargs.append(dict(kwargs))
        return {"status": "ok", "result": {"user": "default"}}

    client.get = record_identity_probe
    provider = _provider(tmp_path, client)

    provider.on_memory_write("add", "user", "Use the normal identity timeout")
    _wait_for(lambda: len(client.snapshot()) == 1)

    assert identity_probe_kwargs == [{"timeout": 3.0}]
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

    _wait_for(
        lambda: json.loads(_registry_path(tmp_path).read_text(encoding="utf-8"))[
            "entries"
        ]
        == []
    )
    registry = json.loads(_registry_path(tmp_path).read_text(encoding="utf-8"))
    assert registry == {"version": 2, "entries": []}
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

    provider.shutdown()
    registry = json.loads(_registry_path(tmp_path).read_text(encoding="utf-8"))
    assert registry == {"version": 2, "entries": []}


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
                record.levelname == "WARNING"
                and "OpenViking memory mirror failed" in record.message
                for record in caplog.records
            )
        )

    assert not _registry_path(tmp_path).exists()
    provider.shutdown()


@pytest.mark.parametrize(
    ("registry_text", "expected_warning"),
    [
        ("{not valid JSON", "cannot read mirror registry"),
        ("[]", "invalid mirror registry format: expected a JSON object"),
        (
            json.dumps({"entries": []}),
            "invalid mirror registry format: missing version",
        ),
        (
            json.dumps({"version": 3, "entries": []}),
            "unsupported mirror registry version 3: expected 2",
        ),
    ],
)
def test_invalid_registry_blocks_add_without_overwriting_it(
    tmp_path, caplog, registry_text, expected_warning
):
    path = _registry_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(registry_text, encoding="utf-8")
    original = path.read_bytes()
    client = _FakeVikingClient()
    provider = _provider(tmp_path, client)

    with caplog.at_level("WARNING", logger="plugins.memory.openviking"):
        provider.on_memory_write("add", "user", "Do not orphan this memory")
        provider.shutdown()

    assert client.snapshot() == []
    assert path.read_bytes() == original
    assert any(expected_warning in record.message for record in caplog.records)


def test_failed_replace_keeps_previous_registry_mapping(tmp_path, caplog):
    client = _FakeVikingClient()
    provider = _provider(tmp_path, client)
    provider.on_memory_write("add", "user", "Preferred shell is zsh")
    _wait_for(lambda: len(client.snapshot()) == 1)
    previous = json.loads(_registry_path(tmp_path).read_text(encoding="utf-8"))
    client.fail_post = True

    with caplog.at_level("WARNING", logger="plugins.memory.openviking"):
        provider.on_memory_write(
            "replace",
            "user",
            "Preferred shell is fish",
            metadata={"old_text": "zsh"},
        )
        _wait_for(lambda: len(client.snapshot()) == 2)
        provider.shutdown()

    assert json.loads(_registry_path(tmp_path).read_text(encoding="utf-8")) == previous
    assert any(
        "OpenViking memory mirror failed" in record.message for record in caplog.records
    )


def test_failed_remove_keeps_registry_mapping(tmp_path, caplog):
    client = _FakeVikingClient()
    provider = _provider(tmp_path, client)
    provider.on_memory_write("add", "memory", "Project delta is active")
    _wait_for(lambda: len(client.snapshot()) == 1)
    previous = json.loads(_registry_path(tmp_path).read_text(encoding="utf-8"))
    client.fail_delete = True

    with caplog.at_level("WARNING", logger="plugins.memory.openviking"):
        provider.on_memory_write(
            "remove",
            "memory",
            "",
            metadata={"old_text": "delta is active"},
        )
        _wait_for(lambda: len(client.snapshot()) == 2)
        provider.shutdown()

    assert json.loads(_registry_path(tmp_path).read_text(encoding="utf-8")) == previous
    assert any(
        "OpenViking memory mirror failed" in record.message for record in caplog.records
    )


def test_registry_save_failure_is_reported_after_remote_acceptance(
    tmp_path, caplog, monkeypatch
):
    client = _FakeVikingClient()
    provider = _provider(tmp_path, client)

    def fail_save(_path, _registry):
        raise OSError("registry disk full")

    monkeypatch.setattr(
        NativeMemoryMirror,
        "_save_registry",
        staticmethod(fail_save),
    )
    with caplog.at_level("WARNING", logger="plugins.memory.openviking"):
        provider.on_memory_write("add", "user", "Remote write without mapping")
        _wait_for(lambda: len(client.snapshot()) == 1)
        _wait_for(
            lambda: any(
                "registry disk full" in record.message for record in caplog.records
            )
        )
        provider.shutdown()

    assert not _registry_path(tmp_path).exists()
    assert client.snapshot()[0][1] == "/api/v1/content/write"


def test_registry_mapping_is_isolated_by_connection(tmp_path):
    first_client = _FakeVikingClient(api_key="key-a")
    second_client = _FakeVikingClient(api_key="key-b")
    first = _provider(tmp_path, first_client)
    second = _provider(tmp_path, second_client)

    first.on_memory_write("add", "user", "Preferred editor is Helix")
    second.on_memory_write("add", "user", "Preferred editor is Helix")
    _wait_for(lambda: len(first_client.snapshot()) == 1)
    _wait_for(lambda: len(second_client.snapshot()) == 1)
    _wait_for(
        lambda: (
            _registry_path(tmp_path).exists()
            and len(
                json.loads(_registry_path(tmp_path).read_text(encoding="utf-8"))[
                    "entries"
                ]
            )
            == 2
        )
    )

    first_uri = first_client.snapshot()[0][2]["uri"]
    second_uri = second_client.snapshot()[0][2]["uri"]
    registry_text = _registry_path(tmp_path).read_text(encoding="utf-8")
    registry = json.loads(registry_text)
    assert len(registry["entries"]) == 2
    assert len({entry["connection"] for entry in registry["entries"]}) == 2
    assert "key-a" not in registry_text
    assert "key-b" not in registry_text

    first.on_memory_write(
        "replace",
        "user",
        "Preferred editor is Neovim",
        metadata={"old_text": "Helix"},
    )
    _wait_for(lambda: len(first_client.snapshot()) == 2)

    replace = first_client.snapshot()[1]
    assert replace[2]["uri"] == first_uri
    assert replace[2]["uri"] != second_uri
    assert len(second_client.snapshot()) == 1
    first.shutdown()
    second.shutdown()


def test_provider_instances_serialize_shared_registry_updates(tmp_path, monkeypatch):
    first_client = _FakeVikingClient(api_key="key-a")
    second_client = _FakeVikingClient(api_key="key-b")
    first = _provider(tmp_path, first_client)
    second = _provider(tmp_path, second_client)
    first_save_started = threading.Event()
    release_first_save = threading.Event()
    save_count = 0
    save_count_lock = threading.Lock()
    original_save = NativeMemoryMirror._save_registry

    def blocking_save(path, registry):
        nonlocal save_count
        with save_count_lock:
            save_count += 1
            is_first = save_count == 1
        if is_first:
            first_save_started.set()
            assert release_first_save.wait(timeout=2.0)
        original_save(path, registry)

    monkeypatch.setattr(
        NativeMemoryMirror,
        "_save_registry",
        staticmethod(blocking_save),
    )

    try:
        first.on_memory_write("add", "user", "First profile fact")
        assert first_save_started.wait(timeout=2.0)

        second.on_memory_write("add", "user", "Second profile fact")
        time.sleep(0.05)
        assert second_client.snapshot() == []

        release_first_save.set()
        _wait_for(lambda: len(second_client.snapshot()) == 1)
        _wait_for(
            lambda: (
                len(
                    json.loads(_registry_path(tmp_path).read_text(encoding="utf-8"))[
                        "entries"
                    ]
                )
                == 2
            )
        )
    finally:
        release_first_save.set()
        first.shutdown()
        second.shutdown()


def test_concurrent_first_writes_share_one_mirror_worker(tmp_path, monkeypatch):
    client = _FakeVikingClient()
    provider = _provider(tmp_path, client)
    constructor_started = threading.Event()
    release_constructor = threading.Event()
    constructed = 0
    constructed_lock = threading.Lock()
    original_init = NativeMemoryMirror.__init__

    def blocking_init(self, owner):
        nonlocal constructed
        with constructed_lock:
            constructed += 1
            is_first = constructed == 1
        if is_first:
            constructor_started.set()
            assert release_constructor.wait(timeout=2.0)
        original_init(self, owner)

    monkeypatch.setattr(NativeMemoryMirror, "__init__", blocking_init)
    first = threading.Thread(
        target=provider.on_memory_write,
        args=("add", "user", "First concurrent fact"),
    )
    second = threading.Thread(
        target=provider.on_memory_write,
        args=("add", "user", "Second concurrent fact"),
    )

    first.start()
    assert constructor_started.wait(timeout=2.0)
    second.start()
    release_constructor.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)
    _wait_for(lambda: len(client.snapshot()) == 2)

    assert constructed == 1
    provider.shutdown()

"""Optional peer identity must agree across setup, requests and memory writes."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import yaml

import plugins.memory.openviking as ov


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    for key in (*ov._OPENVIKING_ENV_KEYS, "OPENVIKING_CLI_CONFIG_FILE"):
        # Track absent keys too, so setup's direct environment writes are undone.
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("source", ["env", "yaml", "actor_peer_id", "agent_id"])
@pytest.mark.parametrize("peer", ["", "hermes", "work-assistant"])
def test_configured_peer_routing_is_preserved(tmp_path, monkeypatch, source, peer):
    config = {}
    if source == "env":
        monkeypatch.setenv("OPENVIKING_AGENT", peer)
    elif source == "yaml":
        config["agent"] = peer
    else:
        path = tmp_path / "ovcli.conf"
        path.write_text(
            json.dumps({"url": "http://localhost:1933", source: peer}), encoding="utf-8"
        )
        config = {"use_ovcli_config": True, "ovcli_config_path": str(path)}

    settings = ov._resolve_connection_settings(config)
    client = ov._VikingClient("http://localhost:1933", agent=settings["agent"])
    monkeypatch.setattr(client, "get", lambda *a, **kw: {"result": {"user": "alice"}})
    provider = ov.OpenVikingMemoryProvider()
    uri = provider._build_memory_uri("preferences", client=client)

    assert settings["agent"] == peer
    assert client._headers().get("X-OpenViking-Actor-Peer", "") == peer
    prefix = f"peers/{peer}/" if peer else ""
    assert uri.startswith(f"viking://user/alice/{prefix}memories/preferences/mem_")


def test_unconfigured_client_and_schema_do_not_supply_a_peer():
    settings = ov._resolve_connection_settings({})
    client = ov._VikingClient("http://localhost:1933")
    schema = {
        field["key"]: field
        for field in ov.OpenVikingMemoryProvider().get_config_schema()
    }

    assert settings["agent"] == ""
    assert schema["agent"]["default"] == ""
    assert "X-OpenViking-Actor-Peer" not in client._headers()
    assert "X-OpenViking-Actor-Peer" not in client._multipart_headers()


@pytest.mark.parametrize("peer", ["", "hermes"])
def test_linked_profile_status_only_shows_a_configured_peer(tmp_path, peer):
    path = tmp_path / "ovcli.conf"
    path.write_text(
        json.dumps({"url": "http://localhost:1933", "actor_peer_id": peer}),
        encoding="utf-8",
    )
    display = ov.OpenVikingMemoryProvider().get_status_config({
        "use_ovcli_config": True,
        "ovcli_config_path": str(path),
    })

    if peer:
        assert display["agent"] == peer
    else:
        assert "agent" not in display


@pytest.mark.parametrize("peer", ["", "hermes"])
def test_memory_uri_uses_captured_peer_even_when_empty(monkeypatch, peer):
    client = ov._VikingClient("http://localhost:1933", agent=peer)
    monkeypatch.setattr(client, "get", lambda *a, **kw: {"result": {"user": "alice"}})
    provider = ov.OpenVikingMemoryProvider()
    provider._agent = "later-peer"

    uri = provider._build_memory_uri("preferences", client=client)

    prefix = f"peers/{peer}/" if peer else ""
    assert uri.startswith(f"viking://user/alice/{prefix}memories/preferences/mem_")
    assert "later-peer" not in uri


@pytest.mark.parametrize("save_to_store", [False, True])
@pytest.mark.parametrize("credential", ["dev", "user", "root", "service"])
@pytest.mark.parametrize("stale_env", [False, True])
def test_new_setup_does_not_ask_for_or_save_peer(
    tmp_path,
    monkeypatch,
    save_to_store,
    credential,
    stale_env,
):
    from hermes_cli import memory_setup

    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text(
        "OPENVIKING_AGENT=old-peer\nOTHER_KEY=keep\n", encoding="utf-8"
    )
    config = {"memory": {"openviking": {"agent": "old-peer", "recall_limit": 9}}}
    if stale_env:
        for key in ov._OPENVIKING_ENV_KEYS:
            monkeypatch.setenv(
                key, "old-peer" if key == "OPENVIKING_AGENT" else "old-value"
            )
    monkeypatch.setenv("OTHER_KEY", "keep")
    validations = []

    def validate(values, **kwargs):
        validations.append(dict(values))
        role = (
            "root"
            if credential == "root"
            else "user"
            if values.get("api_key")
            else None
        )
        return True, "", role

    def prompt(label, default=None, secret=False):
        values = {
            "OpenViking server URL": "http://localhost:1933",
            "OpenViking user API key": "test-user-key",
            "OpenViking root API key": "test-root-key",
            "OpenViking API key": "test-service-key",
            "OpenViking account": "account",
            "OpenViking user": "alice",
            "OpenViking profile name": "personal",
        }
        assert label in values, f"Unexpected setup question: {label}"
        return values[label]

    def select(title, options, **kwargs):
        choices = {
            "  OpenViking connection": 0 if credential == "service" else 1,
            "  OpenViking credential": {"dev": 2, "user": 0, "root": 1}.get(
                credential, 0
            ),
            "  Save OpenViking config": int(save_to_store),
        }
        assert title in choices, f"Unexpected setup menu: {title}"
        return choices[title]

    monkeypatch.setattr(memory_setup, "_prompt", prompt)
    monkeypatch.setattr(memory_setup, "_curses_select", select)
    monkeypatch.setattr(ov, "_validate_openviking_reachability", lambda *a: (True, ""))
    monkeypatch.setattr(ov, "_validate_openviking_setup_values", validate)

    ov.OpenVikingMemoryProvider().post_setup(str(home), config)

    assert validations
    assert all(values["agent"] == "" for values in validations)
    assert "OPENVIKING_AGENT" not in (home / ".env").read_text(encoding="utf-8")
    assert "OTHER_KEY=keep" in (home / ".env").read_text(encoding="utf-8")
    saved_config = ov._load_hermes_openviking_config()
    assert saved_config["recall_limit"] == 9
    settings = ov._resolve_connection_settings(saved_config)
    assert settings["agent"] == ""
    assert settings == {
        key: validations[-1][key]
        for key in ("endpoint", "api_key", "account", "user", "agent")
    }
    assert "OPENVIKING_AGENT" not in os.environ
    assert os.environ["OTHER_KEY"] == "keep"
    if save_to_store:
        saved = json.loads(
            Path(saved_config["ovcli_config_path"]).read_text(encoding="utf-8")
        )
        assert "actor_peer_id" not in saved
        assert "agent_id" not in saved


@pytest.mark.parametrize("peer", ["", "work-assistant"])
def test_hermes_only_save_uses_the_same_clean_values_in_file_and_process(
    tmp_path, peer
):
    from dotenv import dotenv_values

    env_path = tmp_path / ".env"
    ov._setup._save_hermes_only_config(
        config={"memory": {}},
        provider_config={},
        env_path=env_path,
        values={
            "endpoint": "http://localhost:29333",
            "api_key": "test\r\n-key\x00",
            "agent": peer,
        },
    )

    expected = {
        "OPENVIKING_ENDPOINT": "http://localhost:29333",
        "OPENVIKING_API_KEY": "test-key",
    }
    if peer:
        expected["OPENVIKING_AGENT"] = peer
    assert dict(dotenv_values(env_path)) == expected
    assert {
        key: os.environ[key] for key in ov._OPENVIKING_ENV_KEYS if key in os.environ
    } == expected


def test_hermes_only_save_failure_leaves_process_environment_unchanged(
    tmp_path, monkeypatch
):
    for key in ov._OPENVIKING_ENV_KEYS:
        monkeypatch.setenv(key, "old-value")

    def fail_write(*args, **kwargs):
        raise OSError("test write failure")

    monkeypatch.setattr(ov, "_write_env_vars", fail_write)
    with pytest.raises(OSError, match="test write failure"):
        ov._setup._save_hermes_only_config(
            config={"memory": {}},
            provider_config={},
            env_path=tmp_path / ".env",
            values={"endpoint": "http://localhost:29333", "api_key": "test-key"},
        )

    assert all(os.environ[key] == "old-value" for key in ov._OPENVIKING_ENV_KEYS)


@pytest.mark.parametrize("peer", ["", "hermes"])
def test_wire_requests_keep_writes_and_session_messages_in_the_selected_scope(
    tmp_path,
    monkeypatch,
    peer,
):
    records = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.respond({"status": "ok", "healthy": True, "version": "test"})
            elif self.path == "/api/v1/system/status":
                self.respond({"result": {"user": "alice"}})
            else:
                self.send_error(404)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            records.append((self.path, dict(self.headers), payload))
            self.respond({"status": "ok", "result": {"written_bytes": 10}})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    home = tmp_path / "hermes"
    home.mkdir()
    provider_config = {"endpoint": f"http://127.0.0.1:{server.server_port}"}
    if peer:
        provider_config["agent"] = peer
    (home / "config.yaml").write_text(
        yaml.safe_dump({
            "memory": {"provider": "openviking", "openviking": provider_config}
        }),
        encoding="utf-8",
    )
    provider = ov.OpenVikingMemoryProvider()
    try:
        provider.initialize("peer-test", hermes_home=str(home))
        assert provider._client is not None
        result = json.loads(
            provider.handle_tool_call("viking_remember", {"content": "I like tea"})
        )
        assert result["status"] == "submitted"
        provider.on_memory_write("add", "user", "I like coffee")
        provider.sync_turn("hello", "hi", session_id="peer-test")
        assert provider._drain_writers("peer-test", timeout=5.0)
        provider.sync_turn(
            "next",
            "reply",
            session_id="peer-test",
            messages=[
                {"role": "user", "content": "next"},
                {"role": "assistant", "content": "reply"},
            ],
        )
        assert provider._drain_writers("peer-test", timeout=5.0)
        provider.on_session_end([])
    finally:
        provider.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3.0)

    assert records
    for _path, headers, _payload in records:
        if peer:
            assert headers["X-OpenViking-Actor-Peer"] == peer
        else:
            assert "X-OpenViking-Actor-Peer" not in headers
    writes = [
        payload for path, _, payload in records if path == "/api/v1/content/write"
    ]
    prefix = f"peers/{peer}/" if peer else ""
    assert {write["content"] for write in writes} == {"I like coffee"}
    assert all(
        write["uri"].startswith(f"viking://user/alice/{prefix}memories/")
        for write in writes
    )
    remember_messages = [
        (path, payload)
        for path, _, payload in records
        if path.startswith("/api/v1/sessions/hermes-remember-")
        and path.endswith("/messages")
    ]
    assert len(remember_messages) == 1
    remember_path, remember_message = remember_messages[0]
    remember_session = remember_path.removesuffix("/messages")
    assert remember_message == {
        "role": "user",
        "parts": [{"type": "text", "text": "I like tea"}],
    }
    assert any(path == f"{remember_session}/commit" for path, _, _ in records)
    batches = [
        payload["messages"]
        for path, _, payload in records
        if path.endswith("/messages/batch")
    ]
    assert len(batches) == 2
    for batch in batches:
        assert "peer_id" not in batch[0]
        if peer:
            assert batch[1]["peer_id"] == peer
        else:
            assert "peer_id" not in batch[1]
    assert any(path.endswith("/commit") for path, _, _ in records)

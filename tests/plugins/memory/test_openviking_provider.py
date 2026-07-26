import json
import os
import socket
import stat
import threading
import time
import zipfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import plugins.memory.openviking as openviking_module
from plugins.memory.openviking import (
    OpenVikingMemoryProvider,
    _DEFERRED_COMMIT_TIMEOUT,
    _VikingClient,
)


def _clear_openviking_tenant_env(monkeypatch):
    for name in ("OPENVIKING_ACCOUNT", "OPENVIKING_USER", "OPENVIKING_AGENT"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _isolate_openviking_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(openviking_module.Path, "home", staticmethod(lambda: home))


def _clear_openviking_env(monkeypatch):
    for key in (
        "OPENVIKING_ENDPOINT",
        "OPENVIKING_API_KEY",
        "OPENVIKING_ACCOUNT",
        "OPENVIKING_USER",
        "OPENVIKING_AGENT",
        "OPENVIKING_CLI_CONFIG_FILE",
        "OPENVIKING_PROFILE_TOKEN_BUDGET",
    ):
        monkeypatch.delenv(key, raising=False)


def _prompt_from_values(values: dict[str, str], *, forbidden: set[str] | None = None):
    forbidden = forbidden or set()

    def _prompt(label, default=None, secret=False):
        if label in forbidden:
            raise AssertionError(f"{label} should not be prompted")
        return values.get(label, default or "")

    return _prompt


def _allow_setup_validation(monkeypatch, *, root_access: bool = False):
    monkeypatch.setattr(
        openviking_module,
        "_validate_openviking_reachability",
        lambda endpoint: (True, ""),
        raising=False,
    )
    monkeypatch.setattr(
        openviking_module,
        "_validate_openviking_auth",
        lambda values: (True, ""),
        raising=False,
    )
    monkeypatch.setattr(
        openviking_module,
        "_validate_openviking_root_access",
        lambda values: (root_access, "" if root_access else "Requires role: root"),
        raising=False,
    )
    monkeypatch.setattr(
        openviking_module,
        "_validate_openviking_setup_values",
        lambda values, *, require_api_key=False: (
            True,
            "",
            "root" if root_access else ("user" if values.get("api_key") else None),
        ),
        raising=False,
    )


def test_openviking_provider_config_loader_uses_readonly_config(monkeypatch):
    import hermes_cli.config as config_mod

    calls = []
    backing_config = {
        "memory": {
            "openviking": {
                "endpoint": "http://127.0.0.1:19472",
                "api_key": "test-key",
            }
        }
    }

    def load_config_readonly():
        calls.append("readonly")
        return backing_config

    def load_config():
        raise AssertionError("OpenViking config loader should use readonly config")

    monkeypatch.setattr(config_mod, "load_config_readonly", load_config_readonly)
    monkeypatch.setattr(config_mod, "load_config", load_config)

    config = openviking_module._load_hermes_openviking_config()

    assert calls == ["readonly"]
    assert config == {
        "endpoint": "http://127.0.0.1:19472",
        "api_key": "test-key",
    }
    assert config is not backing_config["memory"]["openviking"]


def test_linked_ovcli_config_is_read_at_runtime(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    ovcli_path = tmp_path / "ovcli.conf"
    ovcli_path.write_text(
        json.dumps({
            "url": "http://openviking-one.local",
            "api_key": "key-one",
            "account": "acct-one",
            "user": "alice",
            "agent_id": "agent-one",
        }),
        encoding="utf-8",
    )
    provider_config = {"use_ovcli_config": True, "ovcli_config_path": str(ovcli_path)}

    settings = openviking_module._resolve_connection_settings(provider_config)

    assert settings == {
        "endpoint": "http://openviking-one.local",
        "api_key": "key-one",
        "account": "",
        "user": "",
        "agent": "agent-one",
    }

    ovcli_path.write_text(
        json.dumps({
            "url": "http://openviking-two.local",
            "api_key": "key-two",
            "agent_id": "agent-two",
        }),
        encoding="utf-8",
    )

    settings = openviking_module._resolve_connection_settings(provider_config)

    assert settings == {
        "endpoint": "http://openviking-two.local",
        "api_key": "key-two",
        "account": "",
        "user": "",
        "agent": "agent-two",
    }


def test_connection_values_omit_stale_identity_for_user_key_with_root_key():
    values = openviking_module._connection_values_from_ovcli({
        "url": "https://openviking.example",
        "api_key": "user-key",
        "root_api_key": "root-key",
        "account": "stale-account",
        "user": "stale-user",
    })

    assert values["api_key"] == "user-key"
    assert values["account"] == ""
    assert values["user"] == ""


def test_link_ovcli_profile_removes_stale_inline_config(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENVIKING_ENDPOINT=http://old.local\nOTHER_KEY=keep\n", encoding="utf-8")
    config = {"memory": {}}
    provider_config = {
        "use_ovcli_config": False,
        "endpoint": "http://stale.local",
        "api_key": "stale-key",
        "account": "default",
        "user": "default",
        "agent": "stale-agent",
        "api_key_type": "root",
    }
    ovcli_path = tmp_path / "ovcli.conf.VPS_ROOT"

    openviking_module._link_ovcli_profile(
        config=config,
        provider_config=provider_config,
        env_path=env_path,
        ovcli_path=ovcli_path,
    )

    assert config["memory"]["openviking"] == {
        "use_ovcli_config": True,
        "ovcli_config_path": str(ovcli_path),
    }
    assert "OPENVIKING_ENDPOINT" not in env_path.read_text(encoding="utf-8")
    assert "OTHER_KEY=keep" in env_path.read_text(encoding="utf-8")


def test_post_setup_existing_profile_picker_validates_and_links_saved_profile(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    env_path = hermes_home / ".env"
    env_path.write_text("OPENVIKING_ENDPOINT=http://old.local\nOTHER_KEY=keep\n", encoding="utf-8")
    openviking_home = tmp_path / ".openviking"
    openviking_home.mkdir()
    active_path = openviking_home / "ovcli.conf"
    saved_path = openviking_home / "ovcli.conf.VPS"
    active_path.write_text(json.dumps({"url": "http://active.local"}), encoding="utf-8")
    saved_path.write_text(
        json.dumps({"url": "https://vps.example", "api_key": "user-key"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(openviking_module.Path, "home", staticmethod(lambda: tmp_path))

    from hermes_cli import memory_setup

    validate_calls = []

    def validate_values(values, *, require_api_key=False):
        validate_calls.append(dict(values))
        return True, "", "user"

    monkeypatch.setattr(
        openviking_module,
        "_validate_openviking_setup_values",
        validate_values,
        raising=False,
    )
    choices = iter([0, 0])
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: next(choices))
    config = {"memory": {}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    assert validate_calls == [{
        "endpoint": "https://vps.example",
        "api_key": "user-key",
        "root_api_key": "",
        "account": "",
        "user": "",
        "agent": "",
    }]
    assert config["memory"]["provider"] == "openviking"
    assert config["memory"]["openviking"] == {
        "use_ovcli_config": True,
        "ovcli_config_path": str(saved_path),
    }
    env_text = env_path.read_text(encoding="utf-8")
    assert "OPENVIKING_" not in env_text
    assert "OTHER_KEY=keep" in env_text


def test_start_local_openviking_server_uses_endpoint_host_and_port(monkeypatch):
    popen_calls = []

    def fake_popen(args, **kwargs):
        popen_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(openviking_module, "_local_openviking_port_is_open", lambda host, port: False)
    monkeypatch.setattr(openviking_module.shutil, "which", lambda name: "/usr/local/bin/openviking-server")
    monkeypatch.setattr(openviking_module.subprocess, "Popen", fake_popen)

    started, message = openviking_module._start_local_openviking_server("http://127.0.0.1:1934")

    assert started is True
    assert "127.0.0.1:1934" in message
    args, kwargs = popen_calls[0]
    assert args == ["/usr/local/bin/openviking-server", "--host", "127.0.0.1", "--port", "1934"]
    assert kwargs["start_new_session"] is True


def test_start_local_openviking_server_does_not_spawn_when_port_already_open(monkeypatch):
    """A live listener means a second server would just die on DataDirectoryLocked."""
    probed = []

    def fake_probe(host, port):
        probed.append((host, port))
        return True

    monkeypatch.setattr(openviking_module, "_local_openviking_port_is_open", fake_probe)
    monkeypatch.setattr(openviking_module.shutil, "which", lambda name: "/usr/local/bin/openviking-server")
    monkeypatch.setattr(
        openviking_module.subprocess,
        "Popen",
        MagicMock(side_effect=AssertionError("must not spawn while a server is already listening")),
    )

    started, message = openviking_module._start_local_openviking_server("http://127.0.0.1:1934")

    assert started is True
    assert "already running" in message
    assert probed == [("127.0.0.1", 1934)]


def test_start_local_openviking_server_reports_running_server_without_cli_on_path(monkeypatch):
    """The port probe outranks PATH: a reachable server is started, whoever launched it."""
    monkeypatch.setattr(openviking_module, "_local_openviking_port_is_open", lambda host, port: True)
    monkeypatch.setattr(openviking_module.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        openviking_module.subprocess,
        "Popen",
        MagicMock(side_effect=AssertionError("must not spawn")),
    )

    started, message = openviking_module._start_local_openviking_server("http://127.0.0.1:1934")

    assert started is True
    assert "already running" in message


def test_start_local_openviking_server_rejects_unparseable_url_before_probing(monkeypatch):
    monkeypatch.setattr(
        openviking_module,
        "_local_openviking_port_is_open",
        MagicMock(side_effect=AssertionError("must not probe an unparseable endpoint")),
    )

    started, message = openviking_module._start_local_openviking_server("http://127.0.0.1:not-a-port")

    assert started is False
    assert "Could not parse local OpenViking URL" in message


def test_local_openviking_port_is_open_detects_listener_and_closed_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        _host, port = listener.getsockname()
        assert openviking_module._local_openviking_port_is_open("127.0.0.1", port) is True

    # Socket closed: the same port no longer accepts connections.
    assert openviking_module._local_openviking_port_is_open("127.0.0.1", port) is False


def test_https_local_endpoint_is_not_runtime_autostart_eligible(monkeypatch):
    _clear_openviking_env(monkeypatch)
    monkeypatch.setenv("OPENVIKING_ENDPOINT", "https://localhost:1934")

    class FakeVikingClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            assert endpoint == "https://localhost:1934"

        def health(self):
            return False

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeVikingClient)
    monkeypatch.setattr(
        openviking_module,
        "_start_local_openviking_server",
        MagicMock(side_effect=AssertionError("https localhost endpoint should not auto-start")),
    )

    warnings = []
    provider = OpenVikingMemoryProvider()
    provider.initialize("session-1", platform="cli", warning_callback=warnings.append)

    assert provider._client is None
    assert warnings == [
        "Remote OpenViking server at https://localhost:1934 is not reachable. "
        "OpenViking memory disabled; will retry on a later access or when "
        "the config changes. "
        "Check the configured endpoint and network connectivity."
    ]


def test_runtime_does_not_autostart_when_local_server_reports_unhealthy(monkeypatch):
    _clear_openviking_env(monkeypatch)
    monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://localhost:1934")

    class FakeVikingClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            assert endpoint == "http://localhost:1934"

        def health(self):
            return False

        def health_payload(self):
            return {"healthy": False}

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeVikingClient)
    monkeypatch.setattr(
        openviking_module,
        "_start_local_openviking_server",
        MagicMock(side_effect=AssertionError("responding unhealthy server should not auto-start another process")),
    )

    warnings = []
    provider = OpenVikingMemoryProvider()
    provider.initialize("session-1", platform="cli", warning_callback=warnings.append)

    assert provider._client is None
    assert warnings == [
        "OpenViking server at http://localhost:1934 responded but reported unhealthy status. "
        "OpenViking memory disabled; will retry on a later access or when the config changes."
    ]


def test_handle_unreachable_endpoint_waits_long_enough_after_autostart(monkeypatch, capsys):
    wait_calls = []

    monkeypatch.setattr(
        openviking_module,
        "_start_local_openviking_server",
        lambda endpoint: (True, "Started openviking-server on 127.0.0.1:1934 in the background."),
    )
    monkeypatch.setattr(
        openviking_module,
        "_wait_for_openviking_health",
        lambda endpoint, *, timeout_seconds=0: wait_calls.append((endpoint, timeout_seconds)) or True,
    )

    result = openviking_module._handle_unreachable_endpoint(
        "http://127.0.0.1:1934",
        "OpenViking server is not reachable.",
        lambda *args, **kwargs: 0,
        -1,
    )

    assert result is True
    assert wait_calls == [("http://127.0.0.1:1934", 60.0)]
    output = capsys.readouterr().out
    assert "Waiting for OpenViking server to become reachable..." in output


def test_initialize_autostarts_local_openviking_in_background_when_runtime_health_fails(monkeypatch):
    _clear_openviking_env(monkeypatch)
    monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://127.0.0.1:1934")
    health_calls = []
    start_calls = []
    waiter_calls = []

    class FakeVikingClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            assert endpoint == "http://127.0.0.1:1934"

        def health(self):
            health_calls.append("health")
            return False

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeVikingClient)
    monkeypatch.setattr(
        openviking_module,
        "_start_local_openviking_server",
        lambda endpoint: start_calls.append(endpoint) or (True, "started"),
    )
    monkeypatch.setattr(
        openviking_module,
        "_wait_for_openviking_health",
        MagicMock(side_effect=AssertionError("runtime init should not wait synchronously")),
    )

    provider = OpenVikingMemoryProvider()
    monkeypatch.setattr(
        provider,
        "_start_runtime_openviking_waiter",
        lambda **kwargs: waiter_calls.append(kwargs),
        raising=False,
    )
    statuses = []
    provider.initialize("session-1", platform="cli", status_callback=statuses.append)

    assert provider._client is None
    assert health_calls == ["health"]
    assert start_calls == ["http://127.0.0.1:1934"]
    assert len(waiter_calls) == 1
    assert waiter_calls[0]["status_callback"] == statuses.append
    assert any("starting in the background" in message for message in statuses)


def test_tool_search_sorts_by_raw_score_across_buckets():
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.post.return_value = {
        "result": {
            "memories": [
                {"uri": "viking://memories/1", "score": 0.9003, "abstract": "memory result"},
            ],
            "resources": [
                {"uri": "viking://resources/1", "score": 0.9004, "abstract": "resource result"},
            ],
            "skills": [
                {"uri": "viking://skills/1", "score": 0.8999, "abstract": "skill result"},
            ],
            "total": 3,
        }
    }

    result = json.loads(provider._tool_search({"query": "ranking"}))

    assert [entry["uri"] for entry in result["results"]] == [
        "viking://resources/1",
        "viking://memories/1",
        "viking://skills/1",
    ]
    assert [entry["score"] for entry in result["results"]] == [0.9, 0.9, 0.9]
    assert result["total"] == 3


def test_tool_add_resource_rejects_hermes_credential_file_upload(tmp_path, monkeypatch):
    import agent.file_safety as fs

    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()
    auth_json = hermes_home / "auth.json"
    auth_json.write_text('{"OPENROUTER_API_KEY":"sk-test-secret"}', encoding="utf-8")
    monkeypatch.setattr(fs, "_hermes_home_path", lambda: hermes_home)

    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()

    result = json.loads(provider._tool_add_resource({"url": str(auth_json)}))

    assert "error" in result
    assert "credential store" in result["error"]
    provider._client.upload_temp_file.assert_not_called()
    provider._client.post.assert_not_called()


def test_get_tool_schemas_omits_profile_and_keeps_narrow_forget_tools():
    provider = OpenVikingMemoryProvider()

    names = [schema["name"] for schema in provider.get_tool_schemas()]

    assert "viking_profile" not in names
    assert "viking_forget" in names


def test_viking_client_delete_uses_identity_headers(monkeypatch):
    client = _VikingClient(
        "https://example.com",
        api_key="test-key",
        account="acct",
        user="alice",
        agent="hermes",
    )
    captured = {}

    def capture_delete(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"status": "ok", "result": {"uri": "viking://user/memories/x.md"}},
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(client._httpx, "delete", capture_delete)

    assert client.delete("/api/v1/fs", params={"uri": "viking://user/memories/x.md"}) == {
        "status": "ok",
        "result": {"uri": "viking://user/memories/x.md"},
    }
    assert captured["url"] == "https://example.com/api/v1/fs"
    assert captured["kwargs"]["params"] == {"uri": "viking://user/memories/x.md"}
    assert captured["kwargs"]["headers"]["Authorization"] == "Bearer test-key"
    assert captured["kwargs"]["headers"]["X-OpenViking-Actor-Peer"] == "hermes"


def test_validate_openviking_reachability_uses_health_only(monkeypatch):
    events = []

    class FakeVikingClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            assert endpoint == "https://openviking.example"
            assert api_key == ""

        def health(self):
            events.append("health")
            return True

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeVikingClient)

    ok, message = openviking_module._validate_openviking_reachability(
        "https://openviking.example"
    )

    assert ok is True
    assert message == ""
    assert events == ["health"]


# ---------------------------------------------------------------------------
# on_session_switch — flush + commit + rotate behavior (hermes-agent#28296)
# ---------------------------------------------------------------------------

def _make_provider_with_session(session_id: str, turn_count: int):
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._session_id = session_id
    provider._turn_count = turn_count
    return provider


def test_on_session_switch_commits_old_session_and_rotates_id():
    provider = _make_provider_with_session("old-sid", turn_count=3)

    provider.on_session_switch("new-sid", parent_session_id="old-sid")

    provider._client.post.assert_called_once_with(
        "/api/v1/sessions/old-sid/commit",
        {"keep_recent_count": 0},
    )
    assert provider._session_id == "new-sid"
    assert provider._turn_count == 0


def test_sync_turn_captures_session_id_before_worker_runs():
    """Worker must use the session id snapshotted at sync_turn() call time, not
    re-read self._session_id later — otherwise a delayed worker can write the
    previous turn's messages into the rotated-in NEW session."""
    import threading

    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._endpoint = "http://test"
    provider._api_key = ""
    provider._account = "acct"
    provider._user = "usr"
    provider._agent = "hermes"
    provider._session_id = "old-sid"

    started = threading.Event()
    release = threading.Event()
    captured_paths = []
    captured_payloads = []

    def fake_post(path, payload=None, **kwargs):
        started.set()
        release.wait(timeout=2.0)
        captured_paths.append(path)
        captured_payloads.append(payload)
        return {}

    # Patch _VikingClient inside the worker by stubbing post on a client
    # the constructor will produce. Easiest path: monkeypatch the class.
    real_client_cls = _VikingClient

    class StubClient:
        def __init__(self, *a, **kw):
            pass

        def post(self, path, payload=None, **kwargs):
            return fake_post(path, payload, **kwargs)

    import plugins.memory.openviking as _mod
    _mod._VikingClient = StubClient
    try:
        provider.sync_turn("u", "a")
        # Wait until the worker is parked inside the first post call.
        assert started.wait(timeout=2.0), "worker never entered post()"
        # Rotate the provider's session id while the worker is mid-flight.
        provider._session_id = "new-sid"
        release.set()
        for t in list(provider._inflight_writers.get("old-sid", set())):
            t.join(timeout=2.0)
    finally:
        _mod._VikingClient = real_client_cls

    # The whole turn must target the OLD session id as a single ordered batch.
    assert captured_paths == ["/api/v1/sessions/old-sid/messages/batch"]
    assert captured_payloads == [{
        "messages": [
            {"role": "user", "parts": [{"type": "text", "text": "u"}]},
            {"role": "assistant", "parts": [{"type": "text", "text": "a"}], "peer_id": "hermes"},
        ]
    }]


def _long_structured_turn(assistant_count=204):
    return [
        {"role": "user", "content": "u"},
        *[
            {"role": "assistant", "content": f"assistant-{index}"}
            for index in range(assistant_count)
        ],
    ]


def test_end_then_switch_does_not_double_commit():
    """Mirrors the /new and compression call order: commit_memory_session
    (→ on_session_end) immediately followed by on_session_switch. The switch
    must NOT issue a second commit on the same session id."""
    provider = _make_provider_with_session("old-sid", turn_count=2)

    provider.on_session_end([])
    provider.on_session_switch("new-sid", parent_session_id="old-sid")

    # Exactly one commit call, on the OLD session, fired by on_session_end.
    provider._client.post.assert_called_once_with(
        "/api/v1/sessions/old-sid/commit",
        {"keep_recent_count": 0},
    )
    assert provider._session_id == "new-sid"
    assert provider._turn_count == 0


def test_session_needs_commit_guard_wins_over_stale_turn_count():
    """Regression for hermes-agent#28296 review (M3): once a session is marked
    committed, _session_needs_commit must return False even if turn_count is
    still positive. A racing sync_turn can re-increment _turn_count after the
    commit+reset; without the guard ordering, a follow-up finalizer would
    double-commit the same session. The committed-guard must be checked BEFORE
    the turn_count>0 shortcut."""
    provider = _make_provider_with_session("old-sid", turn_count=5)
    provider._mark_session_committed("old-sid")

    # turn_count is a (stale) 5 but the session is already committed.
    assert provider._session_needs_commit("old-sid", 5) is False
    # An uncommitted session with turns still needs a commit.
    assert provider._session_needs_commit("fresh-sid", 5) is True


# ---------------------------------------------------------------------------
# Hung-writer protection: the sync worker can outlive the bounded join
# because each OpenViking POST has _TIMEOUT=30s and there are two per turn.
# Committing while late writes are still in flight would orphan them past
# the commit boundary — they would never be extracted.
# ---------------------------------------------------------------------------

class _HungThread:
    """Thread stand-in that stays alive across joins."""

    def is_alive(self):
        return True

    def join(self, timeout=None):
        # Pretend the join timed out — worker still running.
        return None


# ---------------------------------------------------------------------------
# Orphaned-writer hazard: commit must wait for ALL writers for the session,
# not just the latest tracked one. sync_turn's bounded rate-limit can drop a
# still-alive previous worker — that dropped writer keeps POSTing under the
# old sid and would otherwise land its writes past the commit boundary.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="POSIX advisory locks")
@pytest.mark.parametrize("owner_run_id", ["dead-owner", ""])
def test_concurrent_providers_claim_unlocked_pending_owner_once(
    tmp_path,
    monkeypatch,
    owner_run_id,
):
    """Only one provider may recover a missing or legacy owner lock."""
    import threading

    pytest.importorskip("fcntl")
    _clear_openviking_env(monkeypatch)

    pending_dir = tmp_path / openviking_module._PENDING_SESSIONS_RELATIVE_DIR
    pending_dir.mkdir(parents=True)
    marker = pending_dir / "old-sid.json"
    marker.write_text(
        json.dumps({"session_id": "old-sid", "owner_run_id": owner_run_id}),
        encoding="utf-8",
    )

    posts = []
    posts_lock = threading.Lock()
    commit_started = threading.Event()
    release_commit = threading.Event()

    class StubClient:
        def post(self, path, payload=None, **kwargs):
            with posts_lock:
                posts.append((path, payload))
            commit_started.set()
            release_commit.wait(timeout=5.0)
            return {}

    providers = [OpenVikingMemoryProvider(), OpenVikingMemoryProvider()]
    scan_barrier = threading.Barrier(len(providers))
    for provider in providers:
        provider._client = StubClient()
        provider._hermes_home = str(tmp_path)
        pending_sessions = provider._pending_sessions

        def _scan_together(scan=pending_sessions):
            sessions = scan()
            scan_barrier.wait(timeout=2.0)
            return sessions

        provider._pending_sessions = _scan_together

    recovery_threads = [
        threading.Thread(target=provider._recover_pending_sessions)
        for provider in providers
    ]
    for thread in recovery_threads:
        thread.start()
    for thread in recovery_threads:
        thread.join(timeout=2.0)
        assert not thread.is_alive()

    assert commit_started.wait(timeout=2.0), "recovery commit did not start"
    release_commit.set()
    assert all(provider._drain_finalizers(timeout=2.0) for provider in providers)

    assert posts.count((
        "/api/v1/sessions/old-sid/commit",
        {"keep_recent_count": 0},
    )) == 1


# ---------------------------------------------------------------------------
# on_memory_write: explicit memory writes use content/write and stay outside
# the session transcript/commit boundary.
# ---------------------------------------------------------------------------


def test_shutdown_waits_for_memory_write_worker(monkeypatch):
    import threading

    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._endpoint = "http://test"
    provider._api_key = ""
    provider._account = "acct"
    provider._user = "usr"
    provider._agent = "hermes"

    worker_started = threading.Event()
    release_worker = threading.Event()
    worker_finished = threading.Event()
    shutdown_returned = threading.Event()

    class StubClient:
        def __init__(self, *a, **kw):
            pass

        def post(self, path, payload=None, **kwargs):
            assert path == "/api/v1/content/write"
            worker_started.set()
            release_worker.wait(timeout=2.0)
            worker_finished.set()
            return {}

    monkeypatch.setattr(openviking_module, "_VikingClient", StubClient)

    provider.on_memory_write("add", "user", "remember this")
    assert worker_started.wait(timeout=2.0), "worker never entered post()"

    shutdown_thread = threading.Thread(
        target=lambda: (provider.shutdown(), shutdown_returned.set()),
        daemon=True,
    )
    shutdown_thread.start()

    returned_before_worker_finished = shutdown_returned.wait(timeout=0.1)
    release_worker.set()
    assert shutdown_returned.wait(timeout=2.0), "shutdown did not return after worker finished"
    shutdown_thread.join(timeout=2.0)

    assert not returned_before_worker_finished
    assert worker_finished.is_set()
    assert provider._memory_write_threads == set()


def _make_prefetch_provider() -> OpenVikingMemoryProvider:
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._endpoint = "http://test"
    provider._api_key = ""
    provider._account = "acct"
    provider._user = "usr"
    provider._agent = "hermes"
    return provider


_SESSION_START_LIST_PARAMS = {
    "output": "agent",
    "recursive": True,
    "abs_limit": 512,
    "node_limit": 512,
}


def _memory_listing(*entries):
    return list(entries)


def _mock_session_start_reads(
    provider: OpenVikingMemoryProvider,
    responses: dict[tuple[str, str], object],
):
    calls = []

    def fake_get(path, params=None, **kwargs):
        request_params = dict(params or {})
        uri = request_params.get("uri", "")
        calls.append((path, request_params, kwargs.get("timeout")))
        response = responses.get((path, uri), "")
        if isinstance(response, Exception):
            raise response
        return {"result": response}

    provider._client.get.side_effect = fake_get
    return calls


def test_session_start_token_estimator_matches_shared_openviking_contract():
    provider = OpenVikingMemoryProvider

    assert provider._estimate_tokens("abcd") == 1
    assert provider._estimate_tokens("设") == 2
    assert provider._estimate_tokens("设置") == 3
    assert provider._estimate_tokens("设置ab") == 4


def test_prefetch_prepends_session_start_memory_context_once_per_session():
    provider = _make_prefetch_provider()
    calls = _mock_session_start_reads(
        provider,
        {
            ("/api/v1/content/read", "viking://user/memories/profile.md"): (
                "User prefers concise answers."
            ),
            ("/api/v1/fs/ls", "viking://user/memories/preferences"): _memory_listing(
                {"isDir": True, "rel_path": "owner"},
                {
                    "isDir": False,
                    "rel_path": "owner/z-last.md",
                    "abstract": "  Keep   replies compact.  ",
                },
                {
                    "isDir": False,
                    "rel_path": "owner/a-first.md",
                    "abstract": "Verify source before editing.",
                },
                {"isDir": False, "rel_path": "owner/ignored.txt", "abstract": "ignore"},
            ),
            ("/api/v1/fs/ls", "viking://user/memories/entities"): _memory_listing(
                {
                    "isDir": False,
                    "rel_path": "people/ada.md",
                    "abstract": "Ada Lovelace is a collaborator.",
                },
            ),
        },
    )
    provider._search_prefetch_context = MagicMock(return_value="- [events]\n  recalled context")

    first = provider.prefetch("What should we recall?", session_id="sid-123")
    second = provider.prefetch("What should we recall?", session_id="sid-123")

    assert '<user-profile uri="viking://user/memories/profile.md">' in first
    assert "User prefers concise answers." in first
    assert "<available-memories>" in first
    assert "viking://user/memories/preferences/" in first
    assert "owner/z-last.md — Keep replies compact." in first
    assert first.index("owner/a-first.md") < first.index("owner/z-last.md")
    assert "viking://user/memories/entities/" in first
    assert "people/ada.md — Ada Lovelace is a collaborator." in first
    assert "owner/ignored.txt" not in first
    assert "<preferences" not in first
    assert "<entities" not in first
    assert "recalled context" in first
    assert "<user-profile" not in second
    assert "recalled context" in second
    assert [(path, params) for path, params, _timeout in calls] == [
        ("/api/v1/content/read", {"uri": "viking://user/memories/profile.md"}),
        (
            "/api/v1/fs/ls",
            {"uri": "viking://user/memories/preferences", **_SESSION_START_LIST_PARAMS},
        ),
        (
            "/api/v1/fs/ls",
            {"uri": "viking://user/memories/entities", **_SESSION_START_LIST_PARAMS},
        ),
    ]
    assert provider._search_prefetch_context.call_count == 2


def test_prefetch_reinjects_after_in_place_compression_same_session():
    provider = _make_prefetch_provider()
    provider._session_id = "sid-123"
    profiles = iter(["Profile before compression.", "Profile after compression."])

    def fake_get(path, params=None, **kwargs):
        uri = (params or {}).get("uri", "")
        if uri == "viking://user/memories/profile.md":
            return {"result": next(profiles)}
        return {"result": []}

    provider._client.get.side_effect = fake_get
    provider._search_prefetch_context = MagicMock(return_value="should not run")

    first = provider.prefetch("hi", session_id="sid-123")
    provider._turn_count = 3
    provider.on_session_switch("sid-123", reason="compression")
    second = provider.prefetch("hi", session_id="sid-123")

    assert "Profile before compression." in first
    assert "Profile after compression." in second


def test_queue_prefetch_is_noop_for_openviking_recall(monkeypatch):
    provider = _make_prefetch_provider()
    constructed_clients = []

    class StubClient:
        def __init__(self, *a, **kw):
            constructed_clients.append((a, kw))

    monkeypatch.setattr(openviking_module, "_VikingClient", StubClient)

    provider.queue_prefetch("anything", session_id="sid-123")

    assert constructed_clients == []


def test_prefetch_sends_contract_safe_memory_context_payload(monkeypatch):
    provider = _make_prefetch_provider()

    captured_calls = []

    class StubClient:
        def __init__(self, *a, **kw):
            pass

        def post(self, path, payload=None, **kwargs):
            captured_calls.append((path, payload))
            return {"result": {"memories": [], "resources": []}}

    monkeypatch.setattr(openviking_module, "_VikingClient", StubClient)

    provider.prefetch("anything")

    assert captured_calls == [
        (
            "/api/v1/search/find",
            {
                "query": "anything",
                "limit": 24,
                "score_threshold": 0,
                "context_type": "memory",
            },
        )
    ]
    payload = captured_calls[0][1]
    assert "top_k" not in payload
    assert "mode" not in payload
    assert "target_uri" not in payload


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
from hermes_cli import __version__ as _HERMES_VERSION
from plugins.memory.openviking import (
    OpenVikingMemoryProvider,
    _DEFERRED_COMMIT_TIMEOUT,
    _VikingClient,
)

_EXPECTED_USER_AGENT = f"openviking-memory-hermes/{_HERMES_VERSION}"


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


def test_connection_settings_read_dashboard_config_file(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        """\
memory:
  provider: openviking
  openviking:
    endpoint: http://saved.test:1933
    account: saved-account
    user: saved-user
    agent: saved-agent
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    settings = openviking_module._resolve_connection_settings(
        openviking_module._load_hermes_openviking_config()
    )

    assert settings["endpoint"] == "http://saved.test:1933"
    assert settings["account"] == "saved-account"
    assert settings["user"] == "saved-user"
    assert settings["agent"] == "saved-agent"
    assert settings["api_key"] == ""


def test_linked_ovcli_config_is_read_at_runtime(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    ovcli_path = tmp_path / "ovcli.conf"
    ovcli_path.write_text(
        json.dumps({
            "url": "http://openviking-one.test",
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
        "endpoint": "http://openviking-one.test",
        "api_key": "key-one",
        "account": "",
        "user": "",
        "agent": "agent-one",
    }

    ovcli_path.write_text(
        json.dumps({
            "url": "http://openviking-two.test",
            "api_key": "key-two",
            "agent_id": "agent-two",
        }),
        encoding="utf-8",
    )

    settings = openviking_module._resolve_connection_settings(provider_config)

    assert settings == {
        "endpoint": "http://openviking-two.test",
        "api_key": "key-two",
        "account": "",
        "user": "",
        "agent": "agent-two",
    }


def test_linked_ovcli_without_url_falls_through_to_dashboard_endpoint(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    ovcli_path = tmp_path / "ovcli.conf"
    ovcli_path.write_text(json.dumps({"api_key": "linked-key"}), encoding="utf-8")

    settings = openviking_module._resolve_connection_settings({
        "use_ovcli_config": True,
        "ovcli_config_path": str(ovcli_path),
        "endpoint": "http://saved.test:1933",
    })

    assert settings["endpoint"] == "http://saved.test:1933"
    assert settings["api_key"] == "linked-key"


def test_profile_discovery_warns_when_skipping_unsafe_ovcli_endpoint(tmp_path, caplog):
    profile_path = tmp_path / "ovcli.conf.blocked"
    profile_path.write_text(
        json.dumps({"url": "http://169.254.169.254/latest/meta-data"}),
        encoding="utf-8",
    )

    with caplog.at_level("WARNING", logger=openviking_module.__name__):
        assert (
            openviking_module._load_profile(
                profile_path,
                source="saved",
                name="blocked",
            )
            is None
        )

    assert "Skipping invalid OpenViking CLI config" in caplog.text
    assert str(profile_path) in caplog.text


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
    env_path.write_text("OPENVIKING_ENDPOINT=http://old.test\nOTHER_KEY=keep\n", encoding="utf-8")
    config = {"memory": {}}
    provider_config = {
        "use_ovcli_config": False,
        "endpoint": "http://stale.test",
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


@pytest.mark.parametrize("peer_key", [None, "actor_peer_id", "agent_id"])
def test_post_setup_existing_profile_picker_validates_and_links_saved_profile(
    tmp_path, monkeypatch, peer_key,
):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    env_path = hermes_home / ".env"
    env_path.write_text("OPENVIKING_ENDPOINT=http://old.test\nOTHER_KEY=keep\n", encoding="utf-8")
    openviking_home = tmp_path / ".openviking"
    openviking_home.mkdir()
    active_path = openviking_home / "ovcli.conf"
    saved_path = openviking_home / "ovcli.conf.VPS"
    active_path.write_text(json.dumps({"url": "http://active.test"}), encoding="utf-8")
    saved_values = {"url": "https://vps.example", "api_key": "user-key"}
    if peer_key:
        saved_values[peer_key] = "existing-peer"
    saved_path.write_text(json.dumps(saved_values), encoding="utf-8")
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
        "agent": "existing-peer" if peer_key else "",
    }]
    assert config["memory"]["provider"] == "openviking"
    assert config["memory"]["openviking"] == {
        "use_ovcli_config": True,
        "ovcli_config_path": str(saved_path),
    }
    env_text = env_path.read_text(encoding="utf-8")
    assert "OPENVIKING_" not in env_text
    assert "OTHER_KEY=keep" in env_text
    settings = openviking_module._resolve_connection_settings(config["memory"]["openviking"])
    assert settings["agent"] == ("existing-peer" if peer_key else "")
    assert json.loads(saved_path.read_text(encoding="utf-8")) == saved_values


def test_local_setup_recommends_user_api_key_before_unauthenticated_mode(monkeypatch):
    monkeypatch.setattr(
        openviking_module,
        "_validate_openviking_reachability",
        lambda endpoint: (True, ""),
    )
    monkeypatch.setattr(
        openviking_module,
        "_validate_openviking_setup_values",
        lambda values, *, require_api_key=False: (True, "", "user"),
    )
    credential_menu = {}

    def select(title, options, *, default=0, cancel_returns=None):
        assert title == "  OpenViking credential"
        credential_menu["options"] = options
        credential_menu["default"] = default
        return 0

    def prompt(label, default=None, secret=False):
        if label == "OpenViking server URL":
            return default
        if label == "OpenViking user API key":
            assert secret is True
            return "user-key"
        raise AssertionError(f"Unexpected prompt: {label}")

    values = openviking_module._prompt_manual_connection_values(
        prompt,
        select,
        -1,
    )

    assert [label for label, _description in credential_menu["options"]] == [
        "User API key",
        "Root API key",
        "No API key",
    ]
    assert credential_menu["default"] == 0
    assert values["api_key"] == "user-key"
    assert values["api_key_type"] == "user"


def test_start_local_openviking_server_uses_endpoint_host_and_port(monkeypatch):
    popen_calls = []

    def fake_popen(args, **kwargs):
        popen_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(openviking_module, "_local_openviking_port_is_open", lambda host, port: False)
    monkeypatch.setattr(openviking_module.shutil, "which", lambda name: "/usr/local/bin/openviking-server")
    monkeypatch.setattr(openviking_module.subprocess, "Popen", fake_popen)

    state, message = openviking_module._start_local_openviking_server("http://127.0.0.1:1934")

    assert state == openviking_module._LOCAL_SERVER_STARTED
    assert "127.0.0.1:1934" in message
    args, kwargs = popen_calls[0]
    assert args == ["/usr/local/bin/openviking-server", "--host", "127.0.0.1", "--port", "1934"]
    assert kwargs["start_new_session"] is True


def test_start_local_openviking_server_strips_pythonpath_from_child_env(monkeypatch):
    """The spawned server must not inherit Hermes's PYTHONPATH (#78153).

    Inheriting it makes openviking-server import packages from the Hermes
    venv instead of its own, and on Windows locks Hermes venv DLLs so the
    venv cannot be rebuilt during `hermes update`.
    """
    popen_calls = []

    def fake_popen(args, **kwargs):
        popen_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(openviking_module, "_local_openviking_port_is_open", lambda host, port: False)
    monkeypatch.setattr(openviking_module.shutil, "which", lambda name: "/usr/local/bin/openviking-server")
    monkeypatch.setattr(openviking_module.subprocess, "Popen", fake_popen)
    monkeypatch.setenv("PYTHONPATH", "/opt/hermes/.venv/Lib/site-packages")
    monkeypatch.setenv("HERMES_PROFILE", "test-profile")

    state, _message = openviking_module._start_local_openviking_server("http://127.0.0.1:1934")

    assert state == openviking_module._LOCAL_SERVER_STARTED
    _, kwargs = popen_calls[0]
    child_env = kwargs["env"]
    assert child_env is not None
    assert "PYTHONPATH" not in child_env
    assert child_env.get("HERMES_PROFILE") == "test-profile"


def test_start_local_openviking_server_does_not_spawn_when_port_already_open(monkeypatch):
    """A live listener means a second server would just die on DataDirectoryLocked."""
    probed = []

    def fake_probe(host, port):
        probed.append((host, port))
        return True

    monkeypatch.setattr(openviking_module, "_local_openviking_port_is_open", fake_probe)
    monkeypatch.setattr(
        openviking_module,
        "_describe_local_port_listener",
        lambda host, port: "python-test-server (PID 4242)",
    )
    monkeypatch.setattr(openviking_module.shutil, "which", lambda name: "/usr/local/bin/openviking-server")
    monkeypatch.setattr(
        openviking_module.subprocess,
        "Popen",
        MagicMock(side_effect=AssertionError("must not spawn while a server is already listening")),
    )

    state, message = openviking_module._start_local_openviking_server("http://127.0.0.1:1934")

    assert state == openviking_module._LOCAL_SERVER_OCCUPIED
    assert "python-test-server (PID 4242)" in message
    assert "not passed OpenViking's /health check" in message
    assert "already running" not in message
    assert probed == [("127.0.0.1", 1934)]


def test_start_local_openviking_server_reports_occupied_port_without_cli_on_path(monkeypatch):
    """The port probe outranks PATH but never claims the listener is OpenViking."""
    monkeypatch.setattr(openviking_module, "_local_openviking_port_is_open", lambda host, port: True)
    monkeypatch.setattr(
        openviking_module,
        "_describe_local_port_listener",
        lambda host, port: "an unidentified process",
    )
    monkeypatch.setattr(openviking_module.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        openviking_module.subprocess,
        "Popen",
        MagicMock(side_effect=AssertionError("must not spawn")),
    )

    state, message = openviking_module._start_local_openviking_server("http://127.0.0.1:1934")

    assert state == openviking_module._LOCAL_SERVER_OCCUPIED
    assert "unidentified process" in message


def test_start_local_openviking_server_rejects_unparseable_url_before_probing(monkeypatch):
    monkeypatch.setattr(
        openviking_module,
        "_local_openviking_port_is_open",
        MagicMock(side_effect=AssertionError("must not probe an unparseable endpoint")),
    )

    state, message = openviking_module._start_local_openviking_server("http://127.0.0.1:not-a-port")

    assert state == openviking_module._LOCAL_SERVER_FAILED
    assert "Could not parse local OpenViking URL" in message


def test_local_openviking_port_is_open_detects_listener_and_closed_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        _host, port = listener.getsockname()
        assert openviking_module._local_openviking_port_is_open("127.0.0.1", port) is True

    # Socket closed: the same port no longer accepts connections.
    assert openviking_module._local_openviking_port_is_open("127.0.0.1", port) is False


def test_describe_local_port_listener_reports_process(monkeypatch):
    import psutil

    connection = SimpleNamespace(
        status=psutil.CONN_LISTEN,
        laddr=SimpleNamespace(ip="0.0.0.0", port=1934),
        pid=4242,
    )
    monkeypatch.setattr(psutil, "net_connections", lambda *, kind: [connection])
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda pid: SimpleNamespace(name=lambda: "postgres"),
    )

    assert openviking_module._describe_local_port_listener("127.0.0.1", 1934) == (
        "postgres (PID 4242)"
    )


def test_runtime_reports_occupied_port_and_does_not_wait_or_spawn(monkeypatch):
    monkeypatch.setattr(
        openviking_module,
        "_start_local_openviking_server",
        lambda endpoint: (
            openviking_module._LOCAL_SERVER_OCCUPIED,
            "Port 127.0.0.1:1934 is occupied by postgres (PID 99).",
        ),
    )
    provider = OpenVikingMemoryProvider()
    provider._endpoint = "http://127.0.0.1:1934"
    provider._start_runtime_openviking_waiter = MagicMock()
    warnings = []

    provider._handle_runtime_openviking_unreachable(warning_callback=warnings.append)

    provider._start_runtime_openviking_waiter.assert_not_called()
    assert provider._client is None
    assert len(warnings) == 1
    assert "postgres (PID 99)" in warnings[0]
    assert "temporarily unavailable" in warnings[0]


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
        "OpenViking memory is temporarily unavailable; Hermes will retry on a later access or when "
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
        "Service at http://localhost:1934 responded but reported unhealthy OpenViking status. "
        "OpenViking memory is temporarily unavailable; Hermes will retry on a later access "
        "or when the config changes."
    ]


def test_handle_unreachable_endpoint_waits_long_enough_after_autostart(monkeypatch, capsys):
    wait_calls = []

    monkeypatch.setattr(
        openviking_module,
        "_start_local_openviking_server",
        lambda endpoint: (
            openviking_module._LOCAL_SERVER_STARTED,
            "Started openviking-server on 127.0.0.1:1934 in the background.",
        ),
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
        lambda endpoint: start_calls.append(endpoint)
        or (openviking_module._LOCAL_SERVER_STARTED, "started"),
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
            json=lambda: {"status": "ok", "result": {"uri": "viking://~/memories/x.md"}},
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(client._httpx, "delete", capture_delete)

    assert client.delete("/api/v1/fs", params={"uri": "viking://~/memories/x.md"}) == {
        "status": "ok",
        "result": {"uri": "viking://~/memories/x.md"},
    }
    assert captured["url"] == "https://example.com/api/v1/fs"
    assert captured["kwargs"]["params"] == {"uri": "viking://~/memories/x.md"}
    assert captured["kwargs"]["headers"]["Authorization"] == "Bearer test-key"
    assert captured["kwargs"]["headers"]["X-OpenViking-Actor-Peer"] == "hermes"
    assert captured["kwargs"]["headers"]["User-Agent"] == _EXPECTED_USER_AGENT


def test_viking_client_upload_uses_user_agent_without_json_content_type(
    tmp_path,
    monkeypatch,
):
    client = _VikingClient(
        "https://example.com",
        api_key="test-key",
        account="acct",
        user="alice",
        agent="hermes",
    )
    upload = tmp_path / "notes.txt"
    upload.write_text("notes", encoding="utf-8")
    captured = {}

    def capture_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"result": {"temp_file_id": "temp-1"}},
        )

    monkeypatch.setattr(client._httpx, "post", capture_post)

    assert client.upload_temp_file(upload) == "temp-1"
    assert captured["url"] == "https://example.com/api/v1/resources/temp_upload"
    headers = captured["kwargs"]["headers"]
    assert headers["User-Agent"] == _EXPECTED_USER_AGENT
    assert "Content-Type" not in headers


def test_openviking_identity_probes_are_anonymous_before_authenticated_requests(monkeypatch):
    calls = []

    def response(payload):
        return SimpleNamespace(status_code=200, text="", json=lambda: payload)

    def fake_get(url, **kwargs):
        calls.append((url, kwargs["headers"]))
        if url.endswith("/health"):
            return response({"status": "ok"})
        if url.endswith("/openapi.json"):
            return response({"info": {"title": "OpenViking API"}})
        if url.endswith("/api/v1/system/status"):
            return response({"status": "ok"})
        if url.endswith("/api/v1/admin/accounts"):
            return response({"status": "ok", "result": []})
        raise AssertionError(f"unexpected request: {url}")

    monkeypatch.setattr(
        openviking_module,
        "_get_httpx",
        lambda: SimpleNamespace(get=fake_get),
    )

    valid, message, role = openviking_module._validate_openviking_setup_values({
        "endpoint": "https://openviking.example",
        "api_key": "secret-key",
        "account": "acct",
        "user": "alice",
        "agent": "hermes",
    })

    assert (valid, message, role) == (True, "", "root")
    assert [url.removeprefix("https://openviking.example") for url, _headers in calls] == [
        "/health",
        "/openapi.json",
        "/api/v1/system/status",
        "/api/v1/admin/accounts",
    ]
    expected_anonymous_headers = {
        "Accept": "application/json",
    }
    assert calls[0][1] == expected_anonymous_headers
    assert calls[1][1] == expected_anonymous_headers
    for _url, headers in calls[2:]:
        assert headers["X-API-Key"] == "secret-key"
        assert headers["Authorization"] == "Bearer secret-key"


def test_repeated_openviking_health_probes_never_send_credentials_or_tenant_headers(
    monkeypatch,
):
    captured_headers = []
    client = _VikingClient(
        "https://openviking.example",
        api_key="secret-key",
        account="acct",
        user="alice",
        agent="hermes",
    )

    def fake_get(_url, **kwargs):
        captured_headers.append(kwargs["headers"])
        return SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"status": "ok", "healthy": True, "version": "0.2.10"},
        )

    monkeypatch.setattr(client._httpx, "get", fake_get)

    assert client.health() is True
    assert client.health() is True
    assert captured_headers == [
        {"Accept": "application/json"},
        {"Accept": "application/json"},
    ]


def test_cloud_health_retries_with_api_key_after_anonymous_auth_error(monkeypatch):
    """Hosted OpenViking may require auth on GET /health (#78410)."""
    calls = []
    client = _VikingClient(
        "https://api.vikingdb.cn-beijing.volces.com/openviking",
        api_key="account.user.0123456789abcdef0123456789abcdef",
        agent="hermes",
    )
    modern = {"status": "ok", "healthy": True, "version": "0.3.0"}

    def fake_get(url, **kwargs):
        headers = kwargs["headers"]
        calls.append(dict(headers))
        if "Authorization" not in headers:
            return SimpleNamespace(
                status_code=401,
                text='{"error":{"code":"AuthenticationError","message":"The API key in the request is missing or invalid."}}',
                json=lambda: {
                    "error": {
                        "code": "AuthenticationError",
                        "message": "The API key in the request is missing or invalid.",
                    }
                },
            )
        return SimpleNamespace(status_code=200, text="", json=lambda: modern)

    monkeypatch.setattr(client._httpx, "get", fake_get)

    payload = client.health_payload()
    assert payload == modern
    assert client.health() is True
    assert calls[0] == {
        "Accept": "application/json",
    }
    assert "Authorization" in calls[1]
    assert calls[1]["Authorization"].startswith("Bearer account.user.")
    assert "X-API-Key" in calls[1]
    # No tenant headers on health.
    assert "X-OpenViking-Account" not in calls[1]
    assert "X-OpenViking-User" not in calls[1]


def test_cloud_health_does_not_send_key_without_api_key(monkeypatch):
    client = _VikingClient(
        "https://api.vikingdb.cn-beijing.volces.com/openviking",
        api_key="",
        agent="hermes",
    )
    calls = []

    def fake_get(url, **kwargs):
        calls.append(kwargs["headers"])
        return SimpleNamespace(
            status_code=401,
            text="AuthenticationError",
            json=lambda: {
                "error": {
                    "code": "AuthenticationError",
                    "message": "The API key in the request is missing or invalid.",
                }
            },
        )

    monkeypatch.setattr(client._httpx, "get", fake_get)

    with pytest.raises(openviking_module._OpenVikingHTTPError):
        client.health_payload()
    assert calls == [{"Accept": "application/json"}]


def test_health_non_auth_errors_do_not_retry_with_credentials(monkeypatch):
    client = _VikingClient(
        "https://openviking.example",
        api_key="secret-key",
        agent="hermes",
    )
    calls = []

    def fake_get(url, **kwargs):
        calls.append(kwargs["headers"])
        return SimpleNamespace(
            status_code=503,
            text="unavailable",
            json=lambda: {"error": {"code": "UNAVAILABLE", "message": "down"}},
        )

    monkeypatch.setattr(client._httpx, "get", fake_get)

    with pytest.raises(openviking_module._OpenVikingHTTPError):
        client.health_payload()
    assert calls == [{"Accept": "application/json"}]


def test_modern_openviking_identity_does_not_probe_openapi():
    client = MagicMock()
    client.health_payload.return_value = {
        "status": "ok",
        "healthy": True,
        "version": "0.2.10",
    }

    state, health = openviking_module._probe_openviking_identity(client)

    assert state == "modern"
    assert health["version"] == "0.2.10"
    client.openapi_payload.assert_not_called()


def test_legacy_health_requires_openviking_openapi_identity_before_auth(monkeypatch):
    events = []

    class ForeignServiceClient:
        def __init__(self, *args, **kwargs):
            pass

        def health_payload(self):
            events.append("health")
            return {"status": "ok"}

        def openapi_payload(self):
            events.append("openapi")
            return {"info": {"title": "Unrelated Service"}}

        def validate_auth(self):
            raise AssertionError("credentials must not be sent before identity is verified")

    monkeypatch.setattr(openviking_module, "_VikingClient", ForeignServiceClient)

    valid, message, role = openviking_module._validate_openviking_setup_values({
        "endpoint": "https://foreign.example",
        "api_key": "secret-key",
    })

    assert valid is False
    assert role is None
    assert "0.2.6 or earlier" in message
    assert "0.2.10 or newer" in message
    assert events == ["health", "openapi"]


def test_verified_legacy_openviking_is_healthy_for_reachability_and_runtime(monkeypatch):
    events = []

    class LegacyOpenVikingClient:
        def __init__(self, *args, **kwargs):
            pass

        def health_payload(self):
            events.append("health")
            return {"status": "ok"}

        def openapi_payload(self):
            events.append("openapi")
            return {"info": {"title": "OpenViking API"}}

    monkeypatch.setattr(openviking_module, "_VikingClient", LegacyOpenVikingClient)

    reachable, message = openviking_module._validate_openviking_reachability(
        "https://legacy.example"
    )
    runtime_state, runtime_message = openviking_module._classify_runtime_openviking_health(
        LegacyOpenVikingClient(),
        "https://legacy.example",
    )

    assert (reachable, message) == (True, "")
    assert (runtime_state, runtime_message) == ("healthy", "")
    assert events == ["health", "openapi", "health", "openapi"]


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

    # The whole turn must target the OLD session id: session registration
    # first (see _ensure_ov_session), then the single ordered batch.
    assert captured_paths == [
        "/api/v1/sessions",
        "/api/v1/sessions/old-sid/messages/batch",
    ]
    assert captured_payloads == [
        {"session_id": "old-sid"},
        {
            "messages": [
                {"role": "user", "parts": [{"type": "text", "text": "u"}]},
                {"role": "assistant", "parts": [{"type": "text", "text": "a"}], "peer_id": "hermes"},
            ]
        },
    ]


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


def test_memory_write_uses_one_connection_for_identity_uri_and_post(monkeypatch):
    import threading

    provider = OpenVikingMemoryProvider()
    provider._agent = "alice-agent"
    provider._ensure_client = lambda: True

    identity_started = threading.Event()
    release_identity = threading.Event()
    write_finished = threading.Event()
    writes = []

    class StubClient:
        def __init__(self, user, agent):
            self._user = user
            self._agent = agent

        def get(self, path, **kwargs):
            assert path == "/api/v1/system/status"
            identity_started.set()
            assert release_identity.wait(timeout=2.0)
            return {"status": "ok", "result": {"user": self._user}}

        def post(self, path, payload=None, **kwargs):
            writes.append((self._user, self._agent, path, payload))
            write_finished.set()
            return {"status": "ok"}

    alice = StubClient("alice", "alice-agent")
    bob = StubClient("bob", "bob-agent")
    provider._client = alice
    monkeypatch.setattr(provider, "_new_client", lambda: alice)

    provider.on_memory_write("add", "user", "remember this")
    assert identity_started.wait(timeout=2.0), "identity probe did not start"

    # Simulate a profile reload while the write worker is resolving identity.
    provider._client = bob
    provider._agent = "bob-agent"
    release_identity.set()

    assert write_finished.wait(timeout=2.0), "memory write did not finish"
    for worker in list(provider._memory_write_threads):
        worker.join(timeout=2.0)

    assert len(writes) == 1
    user, agent, path, payload = writes[0]
    assert (user, agent, path) == ("alice", "alice-agent", "/api/v1/content/write")
    assert payload["uri"].startswith(
        "viking://user/alice/peers/alice-agent/memories/preferences/mem_"
    )
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
        if path == "/api/v1/system/status":
            return {"status": "ok", "result": {"user": "default"}}
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
            ("/api/v1/content/read", "viking://user/default/memories/profile.md"): (
                "User prefers concise answers."
            ),
            ("/api/v1/fs/ls", "viking://user/default/memories/preferences"): _memory_listing(
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
            ("/api/v1/fs/ls", "viking://user/default/memories/entities"): _memory_listing(
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

    assert '<user-profile uri="viking://user/default/memories/profile.md">' in first
    assert "User prefers concise answers." in first
    assert "<available-memories>" in first
    assert "viking://user/default/memories/preferences/" in first
    assert "owner/z-last.md — Keep replies compact." in first
    assert first.index("owner/a-first.md") < first.index("owner/z-last.md")
    assert "viking://user/default/memories/entities/" in first
    assert "people/ada.md — Ada Lovelace is a collaborator." in first
    assert "owner/ignored.txt" not in first
    assert "<preferences" not in first
    assert "<entities" not in first
    assert "recalled context" in first
    assert "<user-profile" not in second
    assert "recalled context" in second
    assert [(path, params) for path, params, _timeout in calls] == [
        # The user-space probe runs once, before the first URI is built.
        ("/api/v1/system/status", {}),
        ("/api/v1/content/read", {"uri": "viking://user/default/memories/profile.md"}),
        (
            "/api/v1/fs/ls",
            {"uri": "viking://user/default/memories/preferences", **_SESSION_START_LIST_PARAMS},
        ),
        (
            "/api/v1/fs/ls",
            {"uri": "viking://user/default/memories/entities", **_SESSION_START_LIST_PARAMS},
        ),
    ]
    assert provider._search_prefetch_context.call_count == 2


def test_session_start_reuses_one_fallback_user_after_status_probe_failure():
    provider = _make_prefetch_provider()
    provider._user = "configured-user"
    provider._client._user = "configured-user"
    provider._search_prefetch_context = MagicMock(return_value="")
    status_calls = 0
    status_timeouts = []
    read_uris = []

    def fake_get(path, params=None, **kwargs):
        nonlocal status_calls
        if path == "/api/v1/system/status":
            status_calls += 1
            status_timeouts.append(kwargs.get("timeout"))
            if status_calls == 1:
                raise RuntimeError("temporary status failure")
            return {"status": "ok", "result": {"user": "alice"}}

        uri = (params or {}).get("uri", "")
        read_uris.append(uri)
        if path == "/api/v1/content/read":
            return {"result": "Configured-user profile."}
        return {"result": []}

    provider._client.get.side_effect = fake_get

    block = provider.prefetch("What should we recall?", session_id="sid-fallback")

    assert status_calls == 1
    assert len(status_timeouts) == 1
    assert 0 < status_timeouts[0] <= 3.0
    assert read_uris == [
        "viking://user/configured-user/memories/profile.md",
        "viking://user/configured-user/memories/preferences",
        "viking://user/configured-user/memories/entities",
    ]
    assert (
        '<user-profile uri="viking://user/configured-user/memories/profile.md">'
        in block
    )


def test_prefetch_reinjects_after_in_place_compression_same_session():
    provider = _make_prefetch_provider()
    provider._session_id = "sid-123"
    profiles = iter(["Profile before compression.", "Profile after compression."])

    def fake_get(path, params=None, **kwargs):
        uri = (params or {}).get("uri", "")
        if path == "/api/v1/system/status":
            return {"status": "ok", "result": {"user": "default"}}
        if uri == "viking://user/default/memories/profile.md":
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

def test_in_place_compression_rearms_commit_guard():
    """Post-compression turns must still be committable (#74695).

    ``compress_context()`` commits before rewriting the transcript, which
    latches the per-sid guard. In-place mode (the default) keeps the SAME sid,
    so the latch then rejected every later commit for a still-live session —
    the next compression, /new, normal session end and startup recovery all
    silently did nothing, and post-compression turns were never extracted.
    """
    provider = _make_provider_with_session("sid-123", turn_count=4)
    provider._ensure_client = lambda: True

    # Compression commits the live session, latching the guard.
    provider._mark_session_committed("sid-123")
    assert provider._session_needs_commit("sid-123", 4) is False

    # In-place compression: same id in, no rotation.
    provider.on_session_switch("sid-123", reason="compression")

    # The session is still live, so new turns must be committable again.
    assert provider._has_committed_session("sid-123") is False
    assert provider._turn_count == 0
    assert provider._session_needs_commit("sid-123", 2) is True


def test_rotating_compression_keeps_old_session_latched():
    """Rotation mode must keep the guard, which dedupes the old id's finalize.

    With ``compression.in_place: false`` a fresh child id is minted. The old id
    stays committed so its ``_finalize_session_async`` does not double-commit
    what compression already committed — the behavior the guard exists for.
    """
    provider = _make_provider_with_session("old-sid", turn_count=4)
    provider._ensure_client = lambda: True
    provider._finalize_session_async = MagicMock()

    provider._mark_session_committed("old-sid")
    provider.on_session_switch("new-sid", reason="compression")

    assert provider._has_committed_session("old-sid") is True
    assert provider._session_needs_commit("old-sid", 4) is False


def test_undo_rewind_does_not_rearm_commit_guard():
    """Only compression re-arms; a same-session /undo must not."""
    provider = _make_provider_with_session("sid-123", turn_count=4)
    provider._ensure_client = lambda: True

    provider._mark_session_committed("sid-123")
    provider.on_session_switch("sid-123", rewound=True)

    assert provider._has_committed_session("sid-123") is True


def test_in_place_compression_lifecycle_allows_a_later_commit():
    """End-to-end wiring, not a hand-set latch (#74695).

    Drives the real sequence a session goes through: commit at the compression
    boundary, same-id ``on_session_switch``, a post-compression turn via
    ``sync_turn``, then a later commit. Before the fix the second commit never
    reached the server, so every turn after the first compression was lost.
    """
    provider = _make_provider_with_session("sid-123", turn_count=3)
    provider._ensure_client = lambda: True
    provider._new_client = lambda: provider._client

    def _commit_calls():
        return [
            c for c in provider._client.post.call_args_list
            if c.args and str(c.args[0]).endswith("/commit")
        ]

    # 1. Compression commits the live session through the real path.
    provider.on_session_end([{"role": "user", "content": "before"}])
    assert len(_commit_calls()) == 1
    assert provider._has_committed_session("sid-123") is True

    # 2. In-place compression: same id back in, no rotation.
    provider.on_session_switch("sid-123", reason="compression")

    # No new turns means no duplicate extraction at an immediate boundary.
    provider.on_session_end([])
    assert len(_commit_calls()) == 1

    # 3. A genuinely new turn lands on the still-live session.
    provider.sync_turn("after compression", "reply", session_id="sid-123")
    assert provider._drain_writers("sid-123", timeout=5.0)
    assert provider._turn_count > 0
    assert any(
        call.args and str(call.args[0]).endswith("/messages/batch")
        for call in provider._client.post.call_args_list
    )

    # 4. That turn must still be committable.
    provider.on_session_end([{"role": "user", "content": "after"}])
    assert len(_commit_calls()) == 2, (
        "post-compression turns were never committed: "
        f"{provider._client.post.call_args_list}"
    )

def test_resolve_connection_settings_reads_config_yaml_non_secret_fields(monkeypatch):
    """#68209: non-secret fields saved to config.yaml feed the resolution chain."""
    _clear_openviking_env(monkeypatch)
    provider_config = {
        "endpoint": "http://saved.test:1933",
        "account": "cfg-account",
        "user": "cfg-user",
        "agent": "cfg-agent",
    }

    settings = openviking_module._resolve_connection_settings(provider_config)

    assert settings["endpoint"] == "http://saved.test:1933"
    assert settings["account"] == "cfg-account"
    assert settings["user"] == "cfg-user"
    assert settings["agent"] == "cfg-agent"


def test_env_overrides_config_yaml_non_secret_fields(monkeypatch):
    """env still wins over config.yaml (env -> ovcli -> config.yaml -> default)."""
    _clear_openviking_env(monkeypatch)
    monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://env.test")
    monkeypatch.setenv("OPENVIKING_AGENT", "env-agent")

    settings = openviking_module._resolve_connection_settings(
        {"endpoint": "http://saved.test", "agent": "cfg-agent"}
    )

    assert settings["endpoint"] == "http://env.test"
    assert settings["agent"] == "env-agent"


def test_blocked_endpoint_does_not_fall_back_or_construct_client(monkeypatch, tmp_path):
    _clear_openviking_env(monkeypatch)
    monkeypatch.setenv(
        "OPENVIKING_ENDPOINT",
        "http://169.254.169.254/latest/meta-data/temporary-credential",
    )
    monkeypatch.setattr(
        openviking_module,
        "_VikingClient",
        MagicMock(side_effect=AssertionError("blocked endpoint must not construct a client")),
    )
    warnings = []
    provider = OpenVikingMemoryProvider()

    provider.initialize(
        "session-1",
        hermes_home=str(tmp_path),
        platform="cli",
        warning_callback=warnings.append,
    )

    assert provider._client is None
    assert provider._endpoint == ""
    assert len(warnings) == 1
    assert "blocked metadata address" in warnings[0]
    assert "temporary-credential" not in warnings[0]
    assert openviking_module._DEFAULT_ENDPOINT not in warnings[0]


@pytest.mark.parametrize(
    "health_payload",
    [
        {"status": "ok", "healthy": True},
        ["not", "openviking"],
    ],
)
def test_runtime_rejects_unrelated_json_health_response(
    monkeypatch, tmp_path, health_payload
):
    _clear_openviking_env(monkeypatch)
    monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://localhost:1934")

    class UnrelatedJsonService:
        def __init__(self, *args, **kwargs):
            pass

        def health_payload(self):
            return health_payload

    monkeypatch.setattr(openviking_module, "_VikingClient", UnrelatedJsonService)
    monkeypatch.setattr(
        openviking_module,
        "_local_openviking_port_is_open",
        lambda host, port: True,
    )
    monkeypatch.setattr(
        openviking_module,
        "_describe_local_port_listener",
        lambda host, port: "python-http-server (PID 4242)",
    )
    monkeypatch.setattr(
        openviking_module,
        "_start_local_openviking_server",
        MagicMock(side_effect=AssertionError("responding non-OpenViking service must not auto-start")),
    )
    warnings = []
    provider = OpenVikingMemoryProvider()

    provider.initialize(
        "session-1",
        hermes_home=str(tmp_path),
        platform="cli",
        warning_callback=warnings.append,
    )

    assert provider._client is None
    assert len(warnings) == 1
    assert "/health response is not valid OpenViking" in warnings[0]
    assert "python-http-server (PID 4242)" in warnings[0]


def test_is_available_true_for_config_yaml_endpoint(monkeypatch):
    """#68209: a config.yaml endpoint (no env, no ovcli) counts as available."""
    _clear_openviking_env(monkeypatch)
    monkeypatch.setattr(
        openviking_module,
        "_load_hermes_openviking_config",
        lambda: {"endpoint": "http://saved.test:1933"},
    )
    assert OpenVikingMemoryProvider().is_available() is True


def test_is_available_false_without_any_endpoint(monkeypatch):
    _clear_openviking_env(monkeypatch)
    monkeypatch.setattr(
        openviking_module, "_load_hermes_openviking_config", lambda: {}
    )
    assert OpenVikingMemoryProvider().is_available() is False


class TestOpenVikingEnvWriter:
    """``_write_env_vars`` copies existing .env lines through on every update,
    so how it *reads* them decides whether a credential update lands.

    f1ea4a56c ("cover the remaining setup-time .env reads with utf-8-sig")
    swept this class; this writer was missed.
    """

    def test_bom_prefixed_env_updates_in_place(self, tmp_path):
        from plugins.memory.openviking import _write_env_vars

        env = tmp_path / ".env"
        env.write_bytes(b"\xef\xbb\xbfOPENAI_API_KEY=old\nOTHER=1\n")

        _write_env_vars(env, {"OPENAI_API_KEY": "new"})

        lines = [l for l in env.read_text(encoding="utf-8-sig").splitlines() if l]
        # The stale value must be gone, not left as a duplicate. Hermes and
        # python-dotenv use the last occurrence, but the file must have one value.
        assert lines.count("OPENAI_API_KEY=new") == 1
        assert not any(l.endswith("=old") for l in lines)
        assert "OTHER=1" in lines

    def test_non_utf8_env_preserves_unrelated_bytes(self, tmp_path):
        from plugins.memory.openviking import _write_env_vars

        env = tmp_path / ".env"
        env.write_bytes(b"NAME=caf\xe9\nOPENAI_API_KEY=old\n")

        _write_env_vars(env, {"OPENAI_API_KEY": "new"})

        assert env.read_bytes() == b"NAME=caf\xe9\nOPENAI_API_KEY=new\n"

    def test_plain_env_is_unchanged_apart_from_the_write(self, tmp_path):
        from plugins.memory.openviking import _write_env_vars

        env = tmp_path / ".env"
        env.write_text("A=1\nOPENAI_API_KEY=old\nB=2\n", encoding="utf-8")

        _write_env_vars(env, {"OPENAI_API_KEY": "new"})

        assert env.read_text(encoding="utf-8").splitlines() == [
            "A=1", "OPENAI_API_KEY=new", "B=2",
        ]


def _registration_provider(sid):
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._endpoint = "http://test"
    provider._api_key = ""
    provider._account = "acct"
    provider._user = "usr"
    provider._agent = "hermes"
    provider._session_id = sid
    return provider


def test_sync_turn_registers_session_once_per_sid():
    """The sid is registered with POST /api/v1/sessions before its first
    batch and never re-registered on later turns."""
    provider = _registration_provider("sid-reg")
    captured = []

    class StubClient:
        def __init__(self, *a, **kw):
            pass

        def post(self, path, payload=None, **kwargs):
            captured.append(path)
            return {}

    import plugins.memory.openviking as _mod
    real_client_cls = _mod._VikingClient
    _mod._VikingClient = StubClient
    try:
        provider.sync_turn("u1", "a1")
        assert provider._drain_writers("sid-reg", timeout=2.0)
        provider.sync_turn("u2", "a2")
        assert provider._drain_writers("sid-reg", timeout=2.0)
    finally:
        _mod._VikingClient = real_client_cls

    assert captured[0] == "/api/v1/sessions"
    assert captured.count("/api/v1/sessions") == 1
    assert captured.count("/api/v1/sessions/sid-reg/messages/batch") == 2


def test_sync_turn_treats_already_exists_registration_as_registered():
    """An ALREADY_EXISTS answer (racing writer / previous run) counts as
    registered: the batch proceeds and the sid is not re-registered."""
    provider = _registration_provider("sid-exists")
    captured = []

    class StubClient:
        def __init__(self, *a, **kw):
            pass

        def post(self, path, payload=None, **kwargs):
            captured.append(path)
            if path == "/api/v1/sessions":
                raise RuntimeError(
                    "ALREADY_EXISTS: Session 'sid-exists' already exists"
                )
            return {}

    import plugins.memory.openviking as _mod
    real_client_cls = _mod._VikingClient
    _mod._VikingClient = StubClient
    try:
        provider.sync_turn("u1", "a1")
        assert provider._drain_writers("sid-exists", timeout=2.0)
        provider.sync_turn("u2", "a2")
        assert provider._drain_writers("sid-exists", timeout=2.0)
    finally:
        _mod._VikingClient = real_client_cls

    assert captured.count("/api/v1/sessions") == 1
    assert captured.count("/api/v1/sessions/sid-exists/messages/batch") == 2


def test_overlapping_first_writers_register_session_once():
    """Two writers for ONE sid in flight at the same time produce exactly one
    POST /api/v1/sessions: registration is single-flight per sid (review
    finding on #72146 — the unsynchronized check/post/add sequence let both
    first turns register)."""
    provider = _registration_provider("sid-race")
    captured = []
    cap_lock = threading.Lock()
    reg_entered = threading.Event()
    release_reg = threading.Event()

    class StubClient:
        def __init__(self, *a, **kw):
            pass

        def post(self, path, payload=None, **kwargs):
            with cap_lock:
                captured.append(path)
            if path == "/api/v1/sessions":
                reg_entered.set()
                # Hold the first registration POST open: this is exactly the
                # window in which an unsynchronized second writer would pass
                # the registered-set check and POST again.
                release_reg.wait(timeout=2.0)
            return {}

    import plugins.memory.openviking as _mod
    real_client_cls = _mod._VikingClient
    _mod._VikingClient = StubClient
    try:
        provider.sync_turn("u1", "a1")          # writer 1: enters registration
        assert reg_entered.wait(timeout=2.0)
        provider.sync_turn("u2", "a2")          # writer 2: spawned mid-POST
        # Give writer 2 time to reach the registration seam while writer 1
        # still holds the POST open.
        time.sleep(0.3)
        release_reg.set()
        assert provider._drain_writers("sid-race", timeout=5.0)
    finally:
        _mod._VikingClient = real_client_cls

    with cap_lock:
        assert captured.count("/api/v1/sessions") == 1
        assert captured.count("/api/v1/sessions/sid-race/messages/batch") == 2

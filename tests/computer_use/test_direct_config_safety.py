"""Current direct-CUA config, never loader fallback, authorizes desktop input."""

import json
import socket
import subprocess

import pytest

from hermes_cli import config
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.computer_use import cua_backend, tool as cu
from tools.computer_use.backend import ActionResult
from tools.computer_use.cua_backend import CuaDriverBackend
from tools.computer_use_tool import registry


REMOTE = "  remote:\n    enabled: true\n    url: https://desktop.example.test\n"
LOCAL = "  remote:\n    enabled: false\n"
BROKEN_AGENT = "max_turns: 5\nagent: malformed\n"


@pytest.fixture
def desktop(profile_home, monkeypatch):
    """Exercise real construction/registry dispatch; replace only desktop I/O."""
    events = []
    prompts = []

    def denied(*args, **kwargs):
        pytest.fail("config safety test must not connect or launch a process")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(subprocess, "Popen", denied)

    def record(backend, action):
        target = backend._remote_config.url if backend._remote_config else "local"
        events.append((action, target))

    def click(backend, **kwargs):
        record(backend, "click")
        return ActionResult(ok=True, action="click")

    monkeypatch.setattr(CuaDriverBackend, "start", lambda self: record(self, "start"))
    monkeypatch.setattr(CuaDriverBackend, "stop", lambda self: None)
    monkeypatch.setattr(CuaDriverBackend, "list_apps", lambda self: record(self, "list_apps") or [])
    monkeypatch.setattr(CuaDriverBackend, "click", click)
    monkeypatch.setattr(cu, "_approval_callback", lambda *args: prompts.append(args[0]) or "approve_once")
    # A local driver being installed must not rescue invalid remote intent.
    monkeypatch.setattr("tools.computer_use.cua_backend_driver.cua_driver_binary_available", lambda: True)
    yield events, prompts
    cu.reset_backend_for_tests()


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_COMPUTER_USE_BACKEND", raising=False)
    monkeypatch.delenv("HERMES_MANAGED_DIR", raising=False)
    monkeypatch.setenv("HERMES_CUA_REMOTE_TOKEN", "t" * 64)
    token = set_hermes_home_override(tmp_path)
    cu.reset_backend_for_tests()
    yield tmp_path
    cu.reset_backend_for_tests()
    reset_hermes_home_override(token)


@pytest.mark.parametrize("initial,current,preload", [
    (None, REMOTE, False),
    (None, REMOTE, True),
    (LOCAL, REMOTE, True),
    (REMOTE, REMOTE.replace("desktop.example.test", "new.example.test"), True),
    (REMOTE, "  remote:\n    enabled: true\n", False),
    (REMOTE, "  cua_telemetry: false\n", True),
    (REMOTE, REMOTE.replace("enabled: true", "enabled: 1"), False),
    (LOCAL, LOCAL.replace("enabled: false", "enabled: 0"), True),
], ids=["cold", "cold-preloaded", "warm-local", "warm-remote", "deleted-url", "deleted-block", "integer-one", "integer-zero"])
@pytest.mark.parametrize("action", ["list_apps", "click"])
def test_loader_fallback_cannot_authorize_a_desktop(profile_home, desktop, initial, current, preload, action):
    path = profile_home / "config.yaml"
    if initial is not None:
        path.write_text("computer_use:\n" + initial)
        assert config.load_config()["computer_use"]["remote"]["enabled"] is (initial == REMOTE)
    path.write_text("computer_use:\n" + current + BROKEN_AGENT)
    if preload:
        config.load_config()  # another consumer may have cached the fallback already
    events, _ = desktop
    for _ in range(2):
        result = registry.dispatch(
            "computer_use", {"action": action, "element": 1}, session_id="invalid-config",
        )
        assert isinstance(result, str)
        result = json.loads(result)
        assert not events, {"desktop_effects": events, "result": result}
        assert "error" in result, result
    assert cu.check_computer_use_requirements() is False


@pytest.mark.parametrize("selection,managed,expected", [
    (None, None, "local"),
    (LOCAL, None, "local"),
    ("  remote:\n    enabled: false\n    url: https://disabled.example.test\n", None, "local"),
    (REMOTE.replace("https://desktop.example.test", "${CU_TEST_URL}"), None, "https://desktop.example.test/mcp"),
    ("  remote:\n    url: ${CU_TEST_URL}\n", "  remote:\n    enabled: true\n", "https://desktop.example.test/mcp"),
    (REMOTE, "  remote:\n    url: ${CU_MANAGED_URL}\n", "https://managed.example.test/mcp"),
], ids=["absent", "disabled", "explicit-disabled-url", "expanded", "managed-enable-inherits-url", "managed-url-wins"])
@pytest.mark.parametrize("action", ["list_apps", "click"])
def test_normal_config_preserves_direct_target_and_approval(profile_home, desktop, monkeypatch, selection, managed, expected, action):
    monkeypatch.setenv("CU_TEST_URL", "https://desktop.example.test")
    monkeypatch.setenv("CU_MANAGED_URL", "https://managed.example.test")
    if selection is not None:
        (profile_home / "config.yaml").write_text("computer_use:\n" + selection)
    if managed is not None:
        managed_dir = profile_home / "managed"
        managed_dir.mkdir()
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_dir))
        (managed_dir / "config.yaml").write_text("computer_use:\n" + managed)
    result = registry.dispatch(
        "computer_use", {"action": action, "element": 1}, session_id="valid-config",
    )
    assert isinstance(result, str)
    result = json.loads(result)
    events, prompts = desktop
    assert "error" not in result, result
    assert events == [("start", expected), (action, expected)]
    assert prompts == (["click"] if action == "click" else [])
    assert cu.check_computer_use_requirements() is True


@pytest.mark.parametrize("initial,current", [(REMOTE, LOCAL), (LOCAL, REMOTE)], ids=["bound-remote", "bound-local"])
def test_backend_availability_stays_bound_after_config_edit(profile_home, desktop, monkeypatch, initial, current):
    path = profile_home / "config.yaml"
    path.write_text("computer_use:\n" + initial)
    backend = CuaDriverBackend()
    path.write_text("computer_use:\n" + current)
    probes = []
    monkeypatch.setattr(cua_backend, "cua_driver_binary_available", lambda: probes.append("local") or False)
    assert backend.is_available() is (initial == REMOTE)
    assert probes == ([] if initial == REMOTE else ["local"])
    assert not desktop[0]


@pytest.mark.parametrize("initial,current", [(REMOTE, LOCAL), (LOCAL, REMOTE)], ids=["bound-remote", "bound-local"])
def test_empty_discovery_diagnosis_stays_bound_after_config_edit(profile_home, desktop, monkeypatch, initial, current):
    path = profile_home / "config.yaml"
    path.write_text("computer_use:\n" + initial)
    backend = CuaDriverBackend()
    path.write_text("computer_use:\n" + current)
    probes = []
    monkeypatch.setattr(cua_backend, "_linux_session_locked", lambda: probes.append("local") or True)
    reason = cua_backend._empty_discovery_reason(remote=backend._remote_config is not None)
    assert ("remote desktop returned no windows" in reason) is (initial == REMOTE)
    assert probes == ([] if initial == REMOTE else ["local"])
    assert not desktop[0]


@pytest.mark.parametrize("text", [
    "computer_use:\n  provider: remote\n",
    "computer_use:\n  provider: local\n" + REMOTE,
    "computer_use:\n  provider: do-not-leak\n",
    "computer_use:\n  provider: null\n",
    "computer_use:\n  remote: null\n",
    "computer_use:\n  remote: []\n",
    "computer_use: [do-not-leak\n",
    "- do-not-leak\n",
    "computer_use: null\n",
    None,
    "computer_use:\n" + REMOTE.replace("https://desktop.example.test", "https://user:do-not-leak@desktop.example.test"),
], ids=[
    "provider-remote", "provider-local-conflict", "unknown-provider", "null-provider",
    "null-remote", "list-remote", "yaml", "root", "null-block", "unreadable", "url-credentials",
])
@pytest.mark.parametrize("managed", [False, True])
def test_unsupported_or_malformed_intent_cannot_select_a_desktop(profile_home, desktop, monkeypatch, text, managed):
    path = profile_home / "config.yaml"
    if managed:
        path.write_text("computer_use:\n" + REMOTE)
        managed_dir = profile_home / "managed"
        managed_dir.mkdir()
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_dir))
        path = managed_dir / "config.yaml"
    if text is None:
        path.mkdir()
    else:
        path.write_text(text)
    result = registry.dispatch("computer_use", {"action": "list_apps"}, session_id="bad-shape")
    assert isinstance(result, str)
    assert not desktop[0], {"desktop_effects": desktop[0], "result": result}
    assert "error" in json.loads(result)
    assert "do-not-leak" not in result
    assert cu.check_computer_use_requirements() is False


@pytest.mark.parametrize("mode", ["bounded", "unrestricted"])
def test_direct_remote_constructor_retains_requested_permission_guard(profile_home, desktop, mode):
    (profile_home / "config.yaml").write_text("computer_use:\n" + REMOTE)
    with pytest.raises(RuntimeError, match="standard permission mode only"):
        CuaDriverBackend(permission_mode=mode)
    assert not desktop[0]


def test_explicit_noop_never_constructs_a_desktop(profile_home, desktop, monkeypatch):
    (profile_home / "config.yaml").write_text("computer_use:\n  provider: unsupported\n")
    monkeypatch.setenv("HERMES_COMPUTER_USE_BACKEND", "noop")
    result = registry.dispatch("computer_use", {"action": "click", "element": 1}, session_id="noop")
    assert isinstance(result, str)
    assert "error" not in json.loads(result)
    assert not desktop[0]
    assert isinstance(cu._get_backend("noop"), cu._NoopBackend)


@pytest.mark.parametrize("token", [None, "do-not-leak"])
def test_direct_remote_token_errors_are_sanitized(profile_home, desktop, monkeypatch, token):
    (profile_home / "config.yaml").write_text("computer_use:\n" + REMOTE)
    if token is None:
        monkeypatch.delenv("HERMES_CUA_REMOTE_TOKEN")
    else:
        monkeypatch.setenv("HERMES_CUA_REMOTE_TOKEN", token)
    result = registry.dispatch("computer_use", {"action": "list_apps"}, session_id="invalid-token")
    assert isinstance(result, str)
    assert "HERMES_CUA_REMOTE_TOKEN" in json.loads(result)["error"]
    assert "do-not-leak" not in result
    assert not desktop[0]


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("remote_block", ["{}", "\n    url: https://incomplete.example.test/mcp"])
def test_incomplete_remote_intent_never_selects_local(profile_home, desktop, monkeypatch, managed, remote_block):
    path = profile_home / "config.yaml"
    if managed:
        managed_dir = profile_home / "managed"
        managed_dir.mkdir()
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_dir))
        path = managed_dir / "config.yaml"
    path.write_text("computer_use:\n  remote: " + remote_block + "\n")
    result = registry.dispatch("computer_use", {"action": "click", "element": 1}, session_id="incomplete")
    assert isinstance(result, str)
    assert not desktop[0], {"desktop_effects": desktop[0], "result": result}
    assert "error" in json.loads(result)

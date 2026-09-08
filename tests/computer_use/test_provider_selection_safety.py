"""Real-profile selection must never silently move desktop input."""

import json
import socket
import subprocess

import pytest

from agent import computer_use_registry as providers
from agent.computer_use_provider import ComputerUseProvider
from hermes_cli import config
from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override
from tools.computer_use import tool as cu
from tools.computer_use.backend import ActionResult
from tools.computer_use.cua_backend import CuaDriverBackend
from tools.computer_use.host_provider import HostCuaProvider
from tools.computer_use.remote_provider import RemoteCuaProvider
from tools.computer_use_tool import registry


@pytest.fixture(autouse=True)
def profile(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_COMPUTER_USE_BACKEND", raising=False)
    monkeypatch.setenv("HERMES_CUA_REMOTE_TOKEN", "t" * 64)
    token = set_hermes_home_override(tmp_path)
    cu.reset_backend_for_tests()
    monkeypatch.setattr(CuaDriverBackend, "start", lambda self: None)
    monkeypatch.setattr(CuaDriverBackend, "stop", lambda self: None)
    monkeypatch.setattr(CuaDriverBackend, "list_apps", lambda self: [])
    yield tmp_path
    cu.reset_backend_for_tests()
    reset_hermes_home_override(token)


@pytest.fixture
def inert_desktops(profile, monkeypatch):
    """Keep real selection/dispatch; only desktop I/O is substituted."""
    def denied(*args, **kwargs):
        raise AssertionError("selection test must not connect or launch a process")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(subprocess, "Popen", denied)
    events = []

    class Backend(cu._NoopBackend):
        def __init__(self, target):
            super().__init__()
            self.target = target
            events.append(("create", target))

        def start(self):
            events.append(("start", self.target))

        def list_apps(self):
            events.append(("list_apps", self.target))
            return []

        def click(self, *, element=None, **kwargs):
            events.append(("click", self.target))
            return ActionResult(ok=True, action="click")

    class Provider(ComputerUseProvider):
        @property
        def name(self):
            return "leased-review"

        def is_available(self):
            return True

        def create_backend(self, session_id, permission_mode):
            return Backend(self.name)

    provider = Provider()
    scope = hermes_home_key()
    previous = providers.snapshot_registration(provider.name, scope=scope)
    providers.register_provider(provider, scope=scope)
    monkeypatch.setattr(HostCuaProvider, "create_backend", lambda *args: Backend("local"))
    monkeypatch.setattr(RemoteCuaProvider, "create_backend", lambda *args: Backend("remote"))
    try:
        yield events
    finally:
        cu.reset_backend_for_tests()
        providers.restore_registration(provider.name, provider, previous, scope=scope)


@pytest.mark.parametrize("action", ["list_apps", "click"])
@pytest.mark.parametrize("preload", [False, True])
@pytest.mark.parametrize("warm_provider", [None, "local", "remote"])
@pytest.mark.parametrize("selection", [
    "  provider: leased-review\n",
    "  remote:\n    enabled: true\n    url: https://desktop.example.test\n",
])
def test_loader_fallback_cannot_authorize_a_different_desktop(
    profile, inert_desktops, monkeypatch, action, preload, warm_provider, selection,
):
    path = profile / "config.yaml"
    if warm_provider:
        remote_block = selection if "remote:" in selection else ""
        path.write_text(f"computer_use:\n  provider: {warm_provider}\n" + remote_block)
        assert config.load_config()["computer_use"]["provider"] == warm_provider
    path.write_text("computer_use:\n" + selection + "max_turns: 5\nagent: malformed\n")
    # Exercise a fallback already cached by another config consumer as well as
    # repeated real registry calls. A warm LKG must not become fresh authority.
    if preload:
        config.load_config()
    prompts = []
    monkeypatch.setattr(cu, "_approval_callback", lambda *args: prompts.append(args[0]) or "approve_once")
    for _ in range(2):
        result = json.loads(registry.dispatch(
            "computer_use", {"action": action, "element": 1}, session_id="fallback-test",
        ))
        assert not inert_desktops, {"desktop_effects": inert_desktops, "result": result}
        assert "error" in result, result


@pytest.mark.parametrize("selection,expected", [
    ("  provider: leased-review\n", "leased-review"),
    ("  provider: ${CU_TEST_PROVIDER}\n", "leased-review"),
    ("  provider: '  CUA  '\n", "local"),
    ("  provider: local\n  remote:\n    enabled: true\n", "local"),
    ("  remote:\n    enabled: true\n    url: ${CU_TEST_URL}\n", "remote"),
])
@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("action", ["list_apps", "click"])
def test_normal_loader_preserves_selection_and_approved_actions(
    profile, inert_desktops, monkeypatch, selection, expected, action, managed,
):
    monkeypatch.setenv("CU_TEST_PROVIDER", "leased-review")
    monkeypatch.setenv("CU_TEST_URL", "https://desktop.example.test")
    path = profile / "config.yaml"
    if managed:
        managed_dir = profile / "managed"
        managed_dir.mkdir()
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_dir))
        path.write_text("computer_use:\n  cua_telemetry: false\n" + (
            "  provider: unknown-machine\n" if "provider:" in selection else ""
        ))
        path = managed_dir / "config.yaml"
    path.write_text("computer_use:\n" + selection)
    prompts = []
    monkeypatch.setattr(cu, "_approval_callback", lambda *args: prompts.append(args[0]) or "approve_once")
    result = json.loads(registry.dispatch(
        "computer_use", {"action": action, "element": 1}, session_id="normal-test",
    ))
    assert "error" not in result, result
    assert prompts == (["click"] if action == "click" else [])
    assert inert_desktops == [("create", expected), ("start", expected), (action, expected)]


@pytest.mark.parametrize("initial,current", [
    (
        "  provider: remote\n  remote:\n    enabled: true\n    url: https://old.example.test\n",
        "  provider: remote\n  remote:\n    enabled: true\n",
    ),
    ("  provider: removed-plugin\n", "  cua_telemetry: false\n"),
    (
        "  provider: remote\n  remote:\n    enabled: true\n    url: https://old.example.test\n",
        "  provider: remote\n  remote:\n    enabled: 1\n    url: https://old.example.test\n",
    ),
], ids=["deleted-url", "deleted-provider", "integer-enabled"])
def test_loader_fallback_cannot_restore_desktop_authority(profile, monkeypatch, initial, current):
    def denied(*args, **kwargs):
        raise AssertionError("selection test must not connect or launch a process")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(subprocess, "Popen", denied)
    monkeypatch.delenv("HERMES_MANAGED_DIR", raising=False)
    starts = []
    monkeypatch.setattr(CuaDriverBackend, "start", lambda self: starts.append(
        self._remote_config.url if self._remote_config else "local",
    ))

    class Provider(ComputerUseProvider):
        @property
        def name(self):
            return "removed-plugin"

        def is_available(self):
            return True

        def create_backend(self, session_id, permission_mode):
            starts.append(self.name)
            return cu._NoopBackend()

    provider = Provider()
    scope = hermes_home_key()
    previous = providers.snapshot_registration(provider.name, scope=scope)
    providers.register_provider(provider, scope=scope)
    path = profile / "config.yaml"
    try:
        path.write_text("computer_use:\n" + initial)
        assert config.load_config()["computer_use"]["provider"] in {"remote", provider.name}
        path.write_text("computer_use:\n" + current + "max_turns: 5\nagent: malformed\n")
        # Real loader, registry, remote factory and transport injection. Repeated
        # dispatch also exercises the fallback cached under the broken signature.
        for _ in range(2):
            result = json.loads(registry.dispatch(
                "computer_use", {"action": "list_apps"}, session_id="removed-authority",
            ))
            assert not starts, {"wrong_target_started": starts, "result": result}
            assert "error" in result, result
    finally:
        cu.reset_backend_for_tests()
        providers.restore_registration(provider.name, provider, previous, scope=scope)


def test_managed_partial_overlay_preserves_expanded_inherited_target(profile, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("selection test must not connect or launch a process")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(subprocess, "Popen", denied)
    monkeypatch.setenv("CU_TEST_URL", "https://inherited.example.test")
    managed_dir = profile / "managed"
    managed_dir.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_dir))
    (profile / "config.yaml").write_text(
        "computer_use:\n  provider: remote\n  remote:\n    url: ${CU_TEST_URL}\n",
    )
    (managed_dir / "config.yaml").write_text("computer_use:\n  remote:\n    enabled: true\n")
    starts = []
    monkeypatch.setattr(CuaDriverBackend, "start", lambda self: starts.append(self._remote_config.url))
    result = json.loads(registry.dispatch(
        "computer_use", {"action": "list_apps"}, session_id="inherited-target",
    ))
    assert "error" not in result, result
    assert starts == ["https://inherited.example.test/mcp"]


def test_explicit_local_does_not_inherit_legacy_remote_transport(profile):
    (profile / "config.yaml").write_text(
        "computer_use:\n  provider: local\n  remote:\n"
        "    enabled: true\n    url: https://desktop.example.test/mcp\n"
    )
    backend = cu._get_backend("same-session")
    assert backend._remote_config is None, "explicit local must drive this machine, not the leftover remote block"


@pytest.mark.parametrize("selection,legacy,expected", [
    ("", None, "local"),
    ("  remote:\n    enabled: true\n    url: https://desktop.example.test\n", None, "remote"),
    ("  remote:\n    enabled: true\n    url: https://desktop.example.test\n", "cua", "remote"),
    ("  provider: remote\n  remote:\n    url: https://desktop.example.test\n", None, "remote"),
    ("  provider: local\n  remote:\n    enabled: true\n    url: https://desktop.example.test\n", None, "local"),
])
def test_real_profile_migration_keeps_the_selected_machine(profile, monkeypatch, caplog, selection, legacy, expected):
    if legacy:
        monkeypatch.setenv("HERMES_COMPUTER_USE_BACKEND", legacy)
    # Resolve two homes in one process: a previous local default must not erase B's intent.
    assert cu.active_computer_use_provider().name == "local" if not legacy else True
    other = profile / "other"
    other.mkdir()
    (other / "config.yaml").write_text("computer_use:\n" + (selection or "  cua_telemetry: false\n"))
    token = set_hermes_home_override(other)
    try:
        assert cu.active_computer_use_provider().name == expected
        backend = cu._get_backend("same-session")
        assert (backend._remote_config is not None) == (expected == "remote")
        if expected == "remote":
            assert backend._remote_config.url == "https://desktop.example.test/mcp"
        if "provider:" not in selection and expected == "remote":
            assert "remote.enabled" in caplog.text and "provider" in caplog.text
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize("text,legacy", [
    ("computer_use: [broken]\n", None),
    ("computer_use:\n  provider: []\n", None),
    ("computer_use:\n  provider: null\n", None),
    ("computer_use:\n  provider: ''\n", None),
    ("computer_use:\n  provider: unknown-machine\n", None),
    ("computer_use: [\n", None),
    ("- not-a-mapping\n", None),
    ("computer_use:\n  remote: null\n", None),
    ("computer_use:\n  remote:\n    enabled: 'true'\n", None),
    ("computer_use:\n  remote:\n    url: https://orphan.example.test\n", None),
    ("computer_use:\n  remote:\n    enabled: true\n    url: invalid\n", None),
    ("computer_use:\n  provider: remote\n  remote:\n    enabled: false\n    url: https://desktop.example.test\n", None),
    ("computer_use:\n  provider: remote\n", "local"),
    ("computer_use:\n  provider: local\n", "remote"),
])
def test_malformed_or_conflicting_selection_never_falls_back(profile, monkeypatch, text, legacy):
    import json
    (profile / "config.yaml").write_text(text)
    if legacy:
        monkeypatch.setenv("HERMES_COMPUTER_USE_BACKEND", legacy)
    started = []
    monkeypatch.setattr(CuaDriverBackend, "start", lambda self: started.append(self))
    result = json.loads(cu.handle_computer_use({"action": "list_apps"}, session_id="bad-config"))
    assert "error" in result
    assert not started, "invalid intent must be rejected before starting any desktop backend"


def test_unreadable_config_never_falls_back(profile):
    (profile / "config.yaml").mkdir()  # real unreadable target, no permission assumptions under root
    with pytest.raises(RuntimeError, match="config"):
        cu._get_backend("unreadable")


@pytest.mark.parametrize("mode", ["bounded", "unrestricted"])
def test_remote_provider_checks_requested_permission_mode(profile, monkeypatch, mode):
    (profile / "config.yaml").write_text(
        "computer_use:\n  provider: remote\n  remote:\n    enabled: true\n    url: https://desktop.example.test\n"
    )
    monkeypatch.setattr(cu, "_cua_permission_mode", lambda sid: mode)
    with pytest.raises(RuntimeError, match="standard permission mode only"):
        cu._get_backend("nonstandard")


@pytest.mark.parametrize("mode", ["bounded", "unrestricted"])
def test_injected_remote_transport_cannot_bypass_mode_guard(mode):
    from tools.computer_use.remote import RemoteCuaConfig
    remote = RemoteCuaConfig("https://desktop.example.test/mcp", "t" * 64)
    with pytest.raises(RuntimeError, match="standard permission mode only"):
        CuaDriverBackend(permission_mode=mode, remote_config=remote)

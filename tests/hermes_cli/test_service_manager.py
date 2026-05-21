"""Tests for hermes_cli.service_manager — the abstract ServiceManager
protocol, the detect_service_manager() entry point, and the host-side
adapter wrappers (Systemd / Launchd / Windows).

The s6 backend is added in Phase 3; its tests live alongside the
implementation in this same file once that phase ships.
"""
from __future__ import annotations

import pytest

from hermes_cli.service_manager import (
    LaunchdServiceManager,
    ServiceManager,
    ServiceManagerKind,
    SystemdServiceManager,
    WindowsServiceManager,
    detect_service_manager,
    get_service_manager,
    validate_profile_name,
)


# ---------------------------------------------------------------------------
# validate_profile_name
# ---------------------------------------------------------------------------


def test_validate_profile_name_accepts_valid_names() -> None:
    # Smoke: known-good names should not raise.
    validate_profile_name("coder")
    validate_profile_name("my-profile")
    validate_profile_name("assistant_v2")
    validate_profile_name("a")
    validate_profile_name("0")
    validate_profile_name("0abc")


@pytest.mark.parametrize(
    "bad",
    [
        "",                  # empty
        "Coder",             # uppercase
        "foo/bar",           # path traversal
        "../escape",         # path traversal
        "-leading-dash",     # leading dash (s6 reads as a flag)
        "_leading_underscore",  # leading underscore
        "name with spaces",  # whitespace
        "name.with.dots",    # punctuation
        "a" * 252,           # too long
    ],
)
def test_validate_profile_name_rejects_invalid(bad: str) -> None:
    with pytest.raises(ValueError):
        validate_profile_name(bad)


# ---------------------------------------------------------------------------
# detect_service_manager
# ---------------------------------------------------------------------------


def test_detect_service_manager_returns_known_value() -> None:
    """Without mocking, the function must still return one of the
    advertised literals — anything else means a new platform branch
    was added without updating ServiceManagerKind."""
    result = detect_service_manager()
    assert result in ("systemd", "launchd", "windows", "s6", "none")


# ---------------------------------------------------------------------------
# Backend wrappers — kind + registration unsupported on hosts
# ---------------------------------------------------------------------------


def test_systemd_manager_kind_and_registration_unsupported() -> None:
    mgr = SystemdServiceManager()
    assert mgr.kind == "systemd"
    assert mgr.supports_runtime_registration() is False
    with pytest.raises(NotImplementedError):
        mgr.register_profile_gateway("foo", port=9100)
    with pytest.raises(NotImplementedError):
        mgr.unregister_profile_gateway("foo")
    assert mgr.list_profile_gateways() == []
    # Protocol conformance — runtime_checkable lets us assert this.
    assert isinstance(mgr, ServiceManager)


def test_launchd_manager_kind_and_registration_unsupported() -> None:
    mgr = LaunchdServiceManager()
    assert mgr.kind == "launchd"
    assert mgr.supports_runtime_registration() is False
    with pytest.raises(NotImplementedError):
        mgr.register_profile_gateway("foo", port=9100)
    assert mgr.list_profile_gateways() == []
    assert isinstance(mgr, ServiceManager)


def test_windows_manager_kind_and_registration_unsupported() -> None:
    mgr = WindowsServiceManager()
    assert mgr.kind == "windows"
    assert mgr.supports_runtime_registration() is False
    with pytest.raises(NotImplementedError):
        mgr.register_profile_gateway("foo", port=9100)
    assert isinstance(mgr, ServiceManager)


# ---------------------------------------------------------------------------
# Lifecycle delegation — wrappers must call through to module-level fns
# ---------------------------------------------------------------------------


def test_systemd_manager_lifecycle_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.gateway.systemd_start", lambda: called.append("start"),
    )
    monkeypatch.setattr(
        "hermes_cli.gateway.systemd_stop", lambda: called.append("stop"),
    )
    monkeypatch.setattr(
        "hermes_cli.gateway.systemd_restart", lambda: called.append("restart"),
    )
    monkeypatch.setattr(
        "hermes_cli.gateway._probe_systemd_service_running",
        lambda *a, **kw: (False, True),
    )
    mgr = SystemdServiceManager()
    mgr.start("ignored")
    mgr.stop("ignored")
    mgr.restart("ignored")
    assert called == ["start", "stop", "restart"]
    assert mgr.is_running("ignored") is True


def test_launchd_manager_lifecycle_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.gateway.launchd_start", lambda: called.append("start"),
    )
    monkeypatch.setattr(
        "hermes_cli.gateway.launchd_stop", lambda: called.append("stop"),
    )
    monkeypatch.setattr(
        "hermes_cli.gateway.launchd_restart", lambda: called.append("restart"),
    )
    monkeypatch.setattr(
        "hermes_cli.gateway._probe_launchd_service_running", lambda: False,
    )
    mgr = LaunchdServiceManager()
    mgr.start("ignored")
    mgr.stop("ignored")
    mgr.restart("ignored")
    assert called == ["start", "stop", "restart"]
    assert mgr.is_running("ignored") is False


def test_windows_manager_lifecycle_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    # Force-import the submodule so monkeypatch's attribute lookup
    # against the `hermes_cli` package succeeds — gateway_windows is
    # imported lazily inside the wrapper and may not yet be loaded.
    import hermes_cli.gateway_windows  # noqa: F401

    class _FakeWindowsModule:
        @staticmethod
        def start() -> None: called.append("start")
        @staticmethod
        def stop() -> None: called.append("stop")
        @staticmethod
        def restart() -> None: called.append("restart")
        @staticmethod
        def is_installed() -> bool: return True

    monkeypatch.setattr("hermes_cli.gateway_windows", _FakeWindowsModule)
    monkeypatch.setattr(
        "hermes_cli.gateway.find_gateway_pids",
        lambda **kw: [12345],
    )
    mgr = WindowsServiceManager()
    mgr.start("ignored")
    mgr.stop("ignored")
    mgr.restart("ignored")
    assert called == ["start", "stop", "restart"]
    assert mgr.is_running("ignored") is True


def test_windows_manager_is_running_false_when_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.gateway_windows  # noqa: F401

    class _FakeWindowsModule:
        @staticmethod
        def is_installed() -> bool: return False

    monkeypatch.setattr("hermes_cli.gateway_windows", _FakeWindowsModule)
    monkeypatch.setattr(
        "hermes_cli.gateway.find_gateway_pids",
        lambda **kw: [12345],  # PIDs would otherwise vote "running"
    )
    assert WindowsServiceManager().is_running("ignored") is False


def test_windows_manager_install_forwards_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    import hermes_cli.gateway_windows  # noqa: F401

    class _FakeWindowsModule:
        @staticmethod
        def install(*, force, start_now, start_on_login, elevated_handoff) -> None:
            captured["force"] = force
            captured["start_now"] = start_now
            captured["start_on_login"] = start_on_login
            captured["elevated_handoff"] = elevated_handoff

    monkeypatch.setattr("hermes_cli.gateway_windows", _FakeWindowsModule)
    WindowsServiceManager().install(
        force=True, start_now=True, start_on_login=False, elevated_handoff=True,
    )
    assert captured == {
        "force": True,
        "start_now": True,
        "start_on_login": False,
        "elevated_handoff": True,
    }


# ---------------------------------------------------------------------------
# get_service_manager factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,cls",
    [
        ("systemd", SystemdServiceManager),
        ("launchd", LaunchdServiceManager),
        ("windows", WindowsServiceManager),
    ],
)
def test_get_service_manager_returns_correct_backend(
    monkeypatch: pytest.MonkeyPatch,
    kind: ServiceManagerKind,
    cls: type,
) -> None:
    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager", lambda: kind,
    )
    assert isinstance(get_service_manager(), cls)


def test_get_service_manager_raises_when_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager", lambda: "none",
    )
    with pytest.raises(RuntimeError, match="no supported service manager"):
        get_service_manager()


def test_get_service_manager_raises_for_s6_until_phase_3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The s6 backend ships in Phase 3 — until then the factory raises
    with an explicit message so accidental host code that ends up
    running inside the container surfaces clearly."""
    monkeypatch.setattr(
        "hermes_cli.service_manager.detect_service_manager", lambda: "s6",
    )
    with pytest.raises(RuntimeError, match="s6 backend not yet implemented"):
        get_service_manager()

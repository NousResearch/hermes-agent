"""Interfaces for headed cloud sessions (RFC, no desktop stack)."""

from __future__ import annotations

from typing import Optional

import pytest

from agent.headed_session import (
    BROWSER_HANDOFF_HOST_KIND,
    DisplayTarget,
    HeadedCloudSessionProvider,
    HeadedSessionSurfaces,
    assert_computer_use_targets_session,
)
from agent import headed_session_registry as reg
from tools.computer_use.backend import ComputerUseBackend


class _SurfacesProvider(HeadedCloudSessionProvider):
    name = "test-headed"

    def is_available(self) -> bool:
        return True

    def create_session(self) -> HeadedSessionSurfaces:
        display = DisplayTarget(fingerprint="sess-aaa", label="test")
        return HeadedSessionSurfaces(
            display=display,
            pty_endpoint="/api/pty/sess-aaa",
            webvnc_endpoint="https://vnc.example/sess-aaa",
        )

    def close_session(self, surfaces: HeadedSessionSurfaces) -> None:
        return None


class _Bound:
    def __init__(self, fingerprint: Optional[str]) -> None:
        self._fp = fingerprint

    def bound_display_fingerprint(self) -> Optional[str]:
        return self._fp


def _surfaces(*, fingerprint="sess-aaa", host_kind="headed_session") -> HeadedSessionSurfaces:
    return HeadedSessionSurfaces(
        display=DisplayTarget(fingerprint=fingerprint, host_kind=host_kind),
        pty_endpoint="/api/pty/sess-aaa",
        webvnc_endpoint="https://vnc.example/sess-aaa",
    )


@pytest.fixture(autouse=True)
def _clean_registry():
    reg._reset_for_tests()
    yield
    reg._reset_for_tests()


def test_display_target_rejects_empty_fingerprint():
    with pytest.raises(ValueError, match="fingerprint"):
        DisplayTarget(fingerprint="  ")


def test_surfaces_share_one_display_fingerprint():
    surfaces = _SurfacesProvider().create_session()
    assert surfaces.display.fingerprint == "sess-aaa"
    assert surfaces.pty_endpoint.endswith(surfaces.display.fingerprint)
    assert surfaces.webvnc_endpoint.endswith(surfaces.display.fingerprint)


def test_assert_matches_bound_fingerprint():
    assert_computer_use_targets_session(_Bound("sess-aaa"), _surfaces())


def test_assert_fail_closed_on_fingerprint_mismatch():
    with pytest.raises(ValueError, match="90374"):
        assert_computer_use_targets_session(_Bound("sess-bbb"), _surfaces())


def test_assert_fail_closed_when_unbound():
    with pytest.raises(ValueError, match="90374"):
        assert_computer_use_targets_session(_Bound(None), _surfaces())


def test_assert_refuses_browser_handoff_for_cua():
    surfaces = _surfaces(host_kind=BROWSER_HANDOFF_HOST_KIND)
    with pytest.raises(ValueError, match="92524"):
        assert_computer_use_targets_session(_Bound("sess-aaa"), surfaces)


def test_computer_use_backend_bound_display_fingerprint_defaults_none():
    class _Stub(ComputerUseBackend):
        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def is_available(self) -> bool:
            return False

        def capture(self, mode="som", app=None, pid=None, window_id=None):
            raise NotImplementedError

        def click(self, **kwargs):
            raise NotImplementedError

        def drag(self, **kwargs):
            raise NotImplementedError

        def scroll(self, **kwargs):
            raise NotImplementedError

        def type_text(self, text, **kwargs):
            raise NotImplementedError

        def key(self, keys, **kwargs):
            raise NotImplementedError

        def list_apps(self):
            return []

        def focus_app(self, app, raise_window=False):
            raise NotImplementedError

        def set_value(self, value, element=None):
            raise NotImplementedError

    assert _Stub().bound_display_fingerprint() is None


def test_registry_register_and_get():
    p = _SurfacesProvider()
    reg.register_provider(p)
    assert reg.get_provider("test-headed") is p
    assert reg.get_provider("TEST-HEADED") is p


def test_registry_rejects_non_provider():
    with pytest.raises(TypeError):
        reg.register_provider(object())


def test_registry_scoped_registration_isolated():
    p_global, p_scoped = _SurfacesProvider(), _SurfacesProvider()
    reg.register_provider(p_global)
    reg.register_provider(p_scoped, scope="profile-a")
    assert reg.get_provider("test-headed", scope="profile-a") is p_scoped
    assert reg.get_provider("test-headed", scope="profile-b") is p_global

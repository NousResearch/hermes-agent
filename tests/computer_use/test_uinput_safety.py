"""Tests for ``tools.computer_use.uinput_safety`` — the shared predicate
guarding against trycua/cua#2618: cua-driver <0.13.1 can leak XInput master
pointers on Linux/X11 when ``/dev/uinput`` is not read+write accessible.
Fixed upstream by trycua/cua#2631 (cua-driver-rs-v0.13.1). Hermes #74148.

This module is pure and dependency-injected (no subprocess/env reads) so the
matrix of platform/display/version/uinput-accessibility combinations can be
tested directly, without mocking subprocess calls. Doctor- and backend-level
wiring (which resolve the live values) are covered in their own test files.
"""

from __future__ import annotations

import os

import pytest


# ── version parsing ─────────────────────────────────────────────────────────


class TestParseDriverVersion:
    def test_parses_bare_semver(self):
        from tools.computer_use.uinput_safety import parse_driver_version

        assert parse_driver_version("0.13.1") == (0, 13, 1)
        assert parse_driver_version("0.12.6") == (0, 12, 6)
        assert parse_driver_version("1.0.0") == (1, 0, 0)

    def test_parses_v_prefixed_semver(self):
        from tools.computer_use.uinput_safety import parse_driver_version

        assert parse_driver_version("v0.13.1") == (0, 13, 1)

    def test_returns_none_for_none(self):
        from tools.computer_use.uinput_safety import parse_driver_version

        assert parse_driver_version(None) is None

    def test_returns_none_for_non_string(self):
        from tools.computer_use.uinput_safety import parse_driver_version

        assert parse_driver_version(1.0) is None  # type: ignore[arg-type]
        assert parse_driver_version(("0", "13", "1")) is None  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "malformed",
        ["", "unknown", "dev", "0.13", "0.13.1-beta", "0.13.1.4", "abc.def.ghi", "  "],
    )
    def test_returns_none_for_malformed(self, malformed):
        from tools.computer_use.uinput_safety import parse_driver_version

        assert parse_driver_version(malformed) is None


class TestDriverVersionIsVulnerable:
    def test_below_fixed_version_is_vulnerable(self):
        from tools.computer_use.uinput_safety import driver_version_is_vulnerable

        assert driver_version_is_vulnerable("0.12.6") is True
        assert driver_version_is_vulnerable("0.0.1") is True
        assert driver_version_is_vulnerable("0.13.0") is True

    def test_at_or_above_fixed_version_is_not_vulnerable(self):
        from tools.computer_use.uinput_safety import driver_version_is_vulnerable

        assert driver_version_is_vulnerable("0.13.1") is False
        assert driver_version_is_vulnerable("0.13.2") is False
        assert driver_version_is_vulnerable("0.14.0") is False
        assert driver_version_is_vulnerable("1.0.0") is False

    def test_unparseable_version_is_unknown(self):
        from tools.computer_use.uinput_safety import driver_version_is_vulnerable

        assert driver_version_is_vulnerable(None) is None
        assert driver_version_is_vulnerable("unknown") is None
        assert driver_version_is_vulnerable("") is None


# ── uinput device accessibility ─────────────────────────────────────────────


class TestUinputReadWriteAccessible:
    def test_accessible_when_open_for_rdwr_succeeds(self, tmp_path):
        from tools.computer_use.uinput_safety import uinput_read_write_accessible

        fake_device = tmp_path / "uinput"
        fake_device.write_bytes(b"")
        assert uinput_read_write_accessible(str(fake_device)) is True

    def test_not_accessible_when_missing(self, tmp_path):
        from tools.computer_use.uinput_safety import uinput_read_write_accessible

        missing = tmp_path / "does-not-exist"
        assert uinput_read_write_accessible(str(missing)) is False

    def test_not_accessible_when_permission_denied(self, tmp_path):
        from tools.computer_use.uinput_safety import uinput_read_write_accessible

        if os.name != "posix" or os.geteuid() == 0:
            pytest.skip("requires a non-root POSIX user to test permission denial")
        fake_device = tmp_path / "uinput"
        fake_device.write_bytes(b"")
        fake_device.chmod(0o000)
        try:
            assert uinput_read_write_accessible(str(fake_device)) is False
        finally:
            fake_device.chmod(0o644)

    def test_defaults_to_real_uinput_path(self):
        from tools.computer_use.uinput_safety import (
            UINPUT_DEVICE_PATH,
            uinput_read_write_accessible,
        )

        assert UINPUT_DEVICE_PATH == "/dev/uinput"
        # Must not raise regardless of the real host's device state.
        assert uinput_read_write_accessible() in (True, False)


# ── the combined predicate ──────────────────────────────────────────────────


class TestDetectUinputLeakRisk:
    def _base_kwargs(self, **overrides):
        kwargs = dict(
            platform="linux",
            display=":0",
            driver_version="0.12.6",
            uinput_accessible=False,
        )
        kwargs.update(overrides)
        return kwargs

    def test_flags_the_known_unsafe_combination(self):
        from tools.computer_use.uinput_safety import (
            UINPUT_LEAK_RISK_CODE,
            detect_uinput_leak_risk,
        )

        risk = detect_uinput_leak_risk(**self._base_kwargs())
        assert risk is not None
        assert risk["code"] == UINPUT_LEAK_RISK_CODE
        assert risk["driver_version"] == "0.12.6"
        assert risk["uinput_path"] == "/dev/uinput"

    def test_non_linux_preserves_current_behavior(self):
        from tools.computer_use.uinput_safety import detect_uinput_leak_risk

        for plat in ("darwin", "win32", "freebsd"):
            assert detect_uinput_leak_risk(**self._base_kwargs(platform=plat)) is None

    def test_no_display_preserves_current_behavior(self):
        from tools.computer_use.uinput_safety import detect_uinput_leak_risk

        assert detect_uinput_leak_risk(**self._base_kwargs(display=None)) is None
        assert detect_uinput_leak_risk(**self._base_kwargs(display="")) is None

    def test_fixed_version_preserves_current_behavior(self):
        from tools.computer_use.uinput_safety import detect_uinput_leak_risk

        assert detect_uinput_leak_risk(**self._base_kwargs(driver_version="0.13.1")) is None
        assert detect_uinput_leak_risk(**self._base_kwargs(driver_version="0.14.0")) is None

    def test_unknown_or_malformed_version_preserves_current_behavior(self):
        """Unknown/malformed versions must not be guessed at either way."""
        from tools.computer_use.uinput_safety import detect_uinput_leak_risk

        assert detect_uinput_leak_risk(**self._base_kwargs(driver_version=None)) is None
        assert detect_uinput_leak_risk(**self._base_kwargs(driver_version="unknown")) is None
        assert detect_uinput_leak_risk(**self._base_kwargs(driver_version="dev-build")) is None

    def test_accessible_uinput_preserves_current_behavior(self):
        from tools.computer_use.uinput_safety import detect_uinput_leak_risk

        assert detect_uinput_leak_risk(**self._base_kwargs(uinput_accessible=True)) is None


# ── the device-free precheck ────────────────────────────────────────────────


class TestUinputLeakRiskPossible:
    """The cheap, device-free half of the predicate. Callers use it to skip
    opening ``/dev/uinput`` entirely when the platform/display/version alone
    already rule the risk out."""

    def _base_kwargs(self, **overrides):
        kwargs = dict(platform="linux", display=":0", driver_version="0.12.6")
        kwargs.update(overrides)
        return kwargs

    def test_true_only_for_linux_x11_and_a_vulnerable_version(self):
        from tools.computer_use.uinput_safety import uinput_leak_risk_possible

        assert uinput_leak_risk_possible(**self._base_kwargs()) is True

    def test_false_off_linux(self):
        from tools.computer_use.uinput_safety import uinput_leak_risk_possible

        for plat in ("darwin", "win32", "freebsd"):
            assert uinput_leak_risk_possible(**self._base_kwargs(platform=plat)) is False

    def test_false_without_display(self):
        from tools.computer_use.uinput_safety import uinput_leak_risk_possible

        assert uinput_leak_risk_possible(**self._base_kwargs(display=None)) is False
        assert uinput_leak_risk_possible(**self._base_kwargs(display="")) is False

    def test_false_for_fixed_version(self):
        from tools.computer_use.uinput_safety import uinput_leak_risk_possible

        assert uinput_leak_risk_possible(**self._base_kwargs(driver_version="0.13.1")) is False
        assert uinput_leak_risk_possible(**self._base_kwargs(driver_version="0.14.0")) is False

    def test_false_for_unparseable_version(self):
        from tools.computer_use.uinput_safety import uinput_leak_risk_possible

        assert uinput_leak_risk_possible(**self._base_kwargs(driver_version=None)) is False
        assert uinput_leak_risk_possible(**self._base_kwargs(driver_version="unknown")) is False

    def test_agrees_with_detect_when_the_device_is_inaccessible(self):
        """Precheck is exactly ``detect_uinput_leak_risk``'s device-free part:
        whenever it says "possible", an inaccessible device must flag risk, and
        whenever it says "impossible", no device state can flag risk."""
        from tools.computer_use.uinput_safety import (
            detect_uinput_leak_risk,
            uinput_leak_risk_possible,
        )

        for platform in ("linux", "darwin"):
            for display in (":0", None):
                for version in ("0.12.6", "0.13.1", "unknown", None):
                    kwargs = dict(platform=platform, display=display, driver_version=version)
                    possible = uinput_leak_risk_possible(**kwargs)
                    flagged = detect_uinput_leak_risk(**kwargs, uinput_accessible=False) is not None
                    assert possible is flagged, kwargs
                    if not possible:
                        assert detect_uinput_leak_risk(**kwargs, uinput_accessible=True) is None


# ── message / doctor-check / ActionResult builders ──────────────────────────


class TestUinputLeakRiskMessaging:
    @pytest.fixture
    def risk(self):
        from tools.computer_use.uinput_safety import detect_uinput_leak_risk

        return detect_uinput_leak_risk(
            platform="linux", display=":0", driver_version="0.12.6",
            uinput_accessible=False,
        )

    def test_message_cites_upstream_and_hermes_issues(self, risk):
        from tools.computer_use.uinput_safety import uinput_leak_risk_message

        msg = uinput_leak_risk_message(risk)
        assert "2618" in msg
        assert "2631" in msg
        assert "74148" in msg
        assert "0.12.6" in msg

    def test_message_and_hint_never_recommend_permission_changes(self, risk):
        from tools.computer_use.uinput_safety import (
            uinput_leak_risk_hint,
            uinput_leak_risk_message,
        )

        combined = (uinput_leak_risk_message(risk) + " " + uinput_leak_risk_hint()).lower()
        for forbidden in ("chmod", "sudo", "udev", "add.*group", "777", "666"):
            assert forbidden not in combined, f"unexpected {forbidden!r} in guidance"

    def test_message_and_hint_recommend_upgrade(self, risk):
        from tools.computer_use.uinput_safety import (
            uinput_leak_risk_hint,
            uinput_leak_risk_message,
        )

        combined = uinput_leak_risk_message(risk) + " " + uinput_leak_risk_hint()
        assert "0.13.1" in combined
        assert "upgrade" in combined.lower()

    def test_doctor_check_shape(self, risk):
        from tools.computer_use.uinput_safety import uinput_leak_risk_check

        check = uinput_leak_risk_check(risk)
        assert check["status"] in ("fail", "degraded")
        assert isinstance(check["name"], str) and check["name"]
        assert "2618" in check["message"]
        assert isinstance(check.get("data"), dict)
        assert check["data"]["driver_version"] == "0.12.6"

    def test_action_result_shape(self, risk):
        from tools.computer_use.backend import ActionResult
        from tools.computer_use.uinput_safety import (
            UINPUT_LEAK_RISK_CODE,
            uinput_leak_risk_action_result,
        )

        result = uinput_leak_risk_action_result("click", risk)
        assert isinstance(result, ActionResult)
        assert result.ok is False
        assert result.action == "click"
        assert result.code == UINPUT_LEAK_RISK_CODE
        assert "2618" in result.message

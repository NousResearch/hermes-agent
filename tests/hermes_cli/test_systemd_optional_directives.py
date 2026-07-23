"""Tests for systemd optional-directive normalization (issue #41119).

On older systemd versions that don't support RestartMaxDelaySec /
RestartSteps, the installed unit file has those directives silently
dropped.  Without normalization, systemd_unit_is_current() would
perpetually report the unit as outdated because the strict text
comparison sees a difference.

The fix: _strip_optional_systemd_directives() removes those directives
from both the installed and expected text before comparison.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# _strip_optional_systemd_directives
# ---------------------------------------------------------------------------


class TestStripOptionalSystemdDirectives:
    def test_removes_restart_max_delay_sec(self):
        from hermes_cli.gateway import _strip_optional_systemd_directives
        text = """[Service]
Restart=always
RestartSec=5
RestartMaxDelaySec=300
RestartSteps=5
"""
        result = _strip_optional_systemd_directives(text)
        assert "RestartMaxDelaySec" not in result
        assert "RestartSteps" not in result
        assert "Restart=always" in result
        assert "RestartSec=5" in result

    def test_preserves_other_directives(self):
        from hermes_cli.gateway import _strip_optional_systemd_directives
        text = """[Service]
Type=simple
ExecStart=/usr/bin/python gateway run
Restart=always
RestartSec=5
KillMode=mixed
KillSignal=SIGTERM
"""
        result = _strip_optional_systemd_directives(text)
        assert "Type=simple" in result
        assert "ExecStart=" in result
        assert "KillMode=mixed" in result
        assert "KillSignal=SIGTERM" in result

    def test_handles_empty_string(self):
        from hermes_cli.gateway import _strip_optional_systemd_directives
        assert _strip_optional_systemd_directives("") == ""

    def test_handles_no_optional_directives(self):
        from hermes_cli.gateway import _strip_optional_systemd_directives
        text = "[Service]\nRestart=always\n"
        result = _strip_optional_systemd_directives(text)
        assert "Restart=always" in result
        assert "RestartMaxDelaySec" not in result

    def test_preserves_comments(self):
        from hermes_cli.gateway import _strip_optional_systemd_directives
        text = """[Service]
# RestartMaxDelaySec is set below
RestartMaxDelaySec=300
"""
        result = _strip_optional_systemd_directives(text)
        # The comment line should be preserved
        assert "# RestartMaxDelaySec" in result
        # The actual directive should be removed
        assert "RestartMaxDelaySec=300" not in result

    def test_handles_inline_values_with_equals(self):
        from hermes_cli.gateway import _strip_optional_systemd_directives
        text = "RestartMaxDelaySec=300\n"
        result = _strip_optional_systemd_directives(text)
        assert result == ""

    def test_full_unit_comparison(self):
        """Simulate the full stale-check flow with an older systemd unit."""
        from hermes_cli.gateway import (
            _normalize_service_definition,
            _strip_optional_systemd_directives,
        )
        # What the installed unit looks like on older systemd (directives stripped)
        installed = """[Unit]
Description=Hermes Gateway
After=network-online.target

[Service]
Type=simple
ExecStart=/usr/bin/python -m hermes_cli.main gateway run
Restart=always
RestartSec=5
KillMode=mixed
KillSignal=SIGTERM

[Install]
WantedBy=default.target
"""
        # What generate_systemd_unit produces (with the directives)
        expected = """[Unit]
Description=Hermes Gateway
After=network-online.target

[Service]
Type=simple
ExecStart=/usr/bin/python -m hermes_cli.main gateway run
Restart=always
RestartSec=5
RestartMaxDelaySec=300
RestartSteps=5
KillMode=mixed
KillSignal=SIGTERM

[Install]
WantedBy=default.target
"""
        # Without normalization, they differ
        assert _normalize_service_definition(installed) != _normalize_service_definition(expected)

        # With optional-directive stripping, they match
        norm_installed = _normalize_service_definition(
            _strip_optional_systemd_directives(installed)
        )
        norm_expected = _normalize_service_definition(
            _strip_optional_systemd_directives(expected)
        )
        assert norm_installed == norm_expected


# ---------------------------------------------------------------------------
# systemd_unit_is_current integration
# ---------------------------------------------------------------------------


class TestSystemdUnitIsCurrent:
    def test_unit_without_fatal_config_restart_policy_is_not_current(
        self, tmp_path, monkeypatch,
    ):
        from hermes_cli import gateway as gw

        expected = """[Service]
Restart=always
RestartForceExitStatus=75
RestartPreventExitStatus=78
"""
        installed = expected.replace("RestartPreventExitStatus=78\n", "")
        unit_file = tmp_path / "hermes-gateway.service"
        unit_file.write_text(installed)

        monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: unit_file)
        monkeypatch.setattr(
            gw,
            "generate_systemd_unit",
            lambda system=False, run_as_user=None: expected,
        )

        assert gw.systemd_unit_is_current(system=False) is False

    def test_unit_without_optional_directives_is_current(self, tmp_path, monkeypatch):
        """Installed unit missing RestartMaxDelaySec/RestartSteps should be
        considered current when the generated unit includes them."""
        from hermes_cli import gateway as gw

        installed = """[Unit]
Description=Hermes Gateway

[Service]
Type=simple
ExecStart=/usr/bin/python gateway run
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
"""
        unit_file = tmp_path / "hermes-gateway.service"
        unit_file.write_text(installed)

        monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: unit_file)
        monkeypatch.setattr(
            gw,
            "generate_systemd_unit",
            lambda system=False, run_as_user=None: installed + "\nRestartMaxDelaySec=300\nRestartSteps=5\n",
        )

        assert gw.systemd_unit_is_current(system=False) is True

    def test_unit_with_different_restart_is_not_current(self, tmp_path, monkeypatch):
        """A unit with genuinely different config should still be outdated."""
        from hermes_cli import gateway as gw

        installed = """[Unit]
Description=Hermes Gateway

[Service]
Type=simple
ExecStart=/usr/bin/python gateway run
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
"""
        expected = """[Unit]
Description=Hermes Gateway

[Service]
Type=simple
ExecStart=/usr/bin/python gateway run
Restart=always
RestartSec=5
RestartMaxDelaySec=300
RestartSteps=5

[Install]
WantedBy=default.target
"""
        unit_file = tmp_path / "hermes-gateway.service"
        unit_file.write_text(installed)

        monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: unit_file)
        monkeypatch.setattr(
            gw,
            "generate_systemd_unit",
            lambda system=False, run_as_user=None: expected,
        )

        assert gw.systemd_unit_is_current(system=False) is False

    def test_unit_with_optional_directives_is_current(self, tmp_path, monkeypatch):
        """Installed unit WITH the optional directives should also be current."""
        from hermes_cli import gateway as gw

        unit_text = """[Unit]
Description=Hermes Gateway

[Service]
Type=simple
ExecStart=/usr/bin/python gateway run
Restart=always
RestartSec=5
RestartMaxDelaySec=300
RestartSteps=5

[Install]
WantedBy=default.target
"""
        unit_file = tmp_path / "hermes-gateway.service"
        unit_file.write_text(unit_text)

        monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: unit_file)
        monkeypatch.setattr(
            gw,
            "generate_systemd_unit",
            lambda system=False, run_as_user=None: unit_text,
        )

        assert gw.systemd_unit_is_current(system=False) is True

    def test_nonexistent_unit_is_not_current(self, tmp_path, monkeypatch):
        from hermes_cli import gateway as gw
        unit_file = tmp_path / "nonexistent.service"
        monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: unit_file)
        assert gw.systemd_unit_is_current(system=False) is False

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
from gateway import systemd_identity
from gateway import systemd_unit_render
from gateway import systemd_unit_state



# ---------------------------------------------------------------------------
# _strip_optional_systemd_directives
# ---------------------------------------------------------------------------


class TestStripOptionalSystemdDirectives:
    def test_removes_restart_max_delay_sec(self):
        from gateway.systemd_unit_render import strip_optional_systemd_directives as _strip_optional_systemd_directives
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

        monkeypatch.setattr(systemd_identity, "unit_path", lambda system=False: unit_file)
        monkeypatch.setattr(
            systemd_unit_render,
            "generate_systemd_unit",
            lambda system=False, run_as_user=None: expected,
        )

        assert systemd_unit_state.unit_is_current(system=False) is False

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

        monkeypatch.setattr(systemd_identity, "unit_path", lambda system=False: unit_file)
        monkeypatch.setattr(
            systemd_unit_render,
            "generate_systemd_unit",
            lambda system=False, run_as_user=None: installed + "\nRestartMaxDelaySec=300\nRestartSteps=5\n",
        )

        assert systemd_unit_state.unit_is_current(system=False) is True


    def test_nonexistent_unit_is_not_current(self, tmp_path, monkeypatch):
        from hermes_cli import gateway as gw
        unit_file = tmp_path / "nonexistent.service"
        monkeypatch.setattr(systemd_identity, "unit_path", lambda system=False: unit_file)
        assert systemd_unit_state.unit_is_current(system=False) is False

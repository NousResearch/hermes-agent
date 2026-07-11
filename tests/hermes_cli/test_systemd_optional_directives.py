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


# ---------------------------------------------------------------------------
# PATH normalization (WSL Windows-interop staleness churn)
# ---------------------------------------------------------------------------


class TestSystemdUnitPathNormalization:
    """The generated unit's Environment="PATH=..." is assembled from the
    invoking shell. Under WSL it also carries volatile Windows-interop /mnt/*
    entries harvested from os.environ, which differ between the login shell
    that installed the unit and the non-login shell that later runs
    status/restart. A unit that differs ONLY by its PATH payload must be
    treated as current — otherwise every restart rewrites it + daemon-reloads.
    """

    def test_masks_only_volatile_mnt_segments(self):
        from hermes_cli.gateway import _normalize_systemd_unit_for_comparison
        # Same deterministic entry (/a/bin), differing volatile /mnt/* payloads.
        a = 'Environment="PATH=/a/bin:/mnt/c/WINDOWS/system32"'
        b = 'Environment="PATH=/a/bin:/mnt/c/Users/user/AppData/Local/Microsoft/WindowsApps"'
        assert _normalize_systemd_unit_for_comparison(a) == _normalize_systemd_unit_for_comparison(b)

    def test_non_mnt_path_difference_survives_normalization(self):
        """A change to a deterministic (non-/mnt) PATH entry must NOT be masked
        away — otherwise real service-PATH updates (Node dir, service bins) would
        be silently treated as current. This is the case the reviewer flagged."""
        from hermes_cli.gateway import _normalize_systemd_unit_for_comparison
        a = 'Environment="PATH=/opt/node-v20/bin:/mnt/c/WINDOWS/system32"'
        b = 'Environment="PATH=/opt/node-v22/bin:/mnt/c/WINDOWS/system32"'
        assert _normalize_systemd_unit_for_comparison(a) != _normalize_systemd_unit_for_comparison(b)

    def test_mask_helper_drops_mnt_keeps_service_entries(self):
        from hermes_cli.gateway import _mask_volatile_wsl_interop_path
        value = "/home/a/venv/bin:/mnt/c/WINDOWS/system32:/usr/bin:/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/"
        assert _mask_volatile_wsl_interop_path(value) == "/home/a/venv/bin:/usr/bin"

    def test_non_path_differences_still_visible(self):
        from hermes_cli.gateway import _normalize_systemd_unit_for_comparison
        a = 'Environment="PATH=/x"\nRestartSec=5'
        b = 'Environment="PATH=/y"\nRestartSec=10'
        assert _normalize_systemd_unit_for_comparison(a) != _normalize_systemd_unit_for_comparison(b)

    def test_other_environment_lines_untouched(self):
        from hermes_cli.gateway import _normalize_systemd_unit_for_comparison
        a = 'Environment="PATH=/x"\nEnvironment="HERMES_HOME=/home/a/.hermes"'
        b = 'Environment="PATH=/y"\nEnvironment="HERMES_HOME=/home/b/.hermes"'
        # PATH is masked away, but the differing HERMES_HOME must remain visible
        assert _normalize_systemd_unit_for_comparison(a) != _normalize_systemd_unit_for_comparison(b)

    def test_unit_differing_only_by_wsl_interop_path_is_current(self, tmp_path, monkeypatch):
        """Reproduces the live WSL churn: installed unit has extra Windows-interop
        /mnt/* PATH entries that the regenerated unit lacks; still current."""
        from hermes_cli import gateway as gw

        base = """[Unit]
Description=Hermes Gateway

[Service]
Type=simple
ExecStart=/home/a/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main gateway run
WorkingDirectory=/home/a/.hermes
Environment="PATH={path}"
Environment="VIRTUAL_ENV=/home/a/.hermes/hermes-agent/venv"
Environment="HERMES_HOME=/home/a/.hermes"
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
"""
        installed = base.format(
            path="/home/a/.hermes/hermes-agent/venv/bin:/mnt/c/WINDOWS/system32:"
            "/mnt/c/Users/user/AppData/Local/Microsoft/WindowsApps:/usr/bin:/bin"
        )
        regenerated = base.format(
            path="/home/a/.hermes/hermes-agent/venv/bin:/mnt/c/WINDOWS/system32:/usr/bin:/bin"
        )
        unit_file = tmp_path / "hermes-gateway.service"
        unit_file.write_text(installed)

        monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: unit_file)
        monkeypatch.setattr(
            gw,
            "generate_systemd_unit",
            lambda system=False, run_as_user=None: regenerated,
        )

        assert gw.systemd_unit_is_current(system=False) is True

    def test_unit_differing_only_by_non_mnt_path_is_outdated(self, tmp_path, monkeypatch):
        """Regression for the reviewer's concern: when the ONLY difference is a
        deterministic (non-/mnt) PATH entry — everything else, including all
        other directives, identical — the unit must still be flagged stale so
        refresh_systemd_unit_if_needed() rewrites and daemon-reloads it."""
        from hermes_cli import gateway as gw

        base = """[Unit]
Description=Hermes Gateway

[Service]
Type=simple
ExecStart=/home/a/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main gateway run
WorkingDirectory=/home/a/.hermes
Environment="PATH={path}"
Environment="VIRTUAL_ENV=/home/a/.hermes/hermes-agent/venv"
Environment="HERMES_HOME=/home/a/.hermes"
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
"""
        # Identical volatile /mnt/* payload; only the deterministic Node dir moved.
        installed = base.format(
            path="/opt/node-v20/bin:/mnt/c/WINDOWS/system32:/usr/bin:/bin"
        )
        regenerated = base.format(
            path="/opt/node-v22/bin:/mnt/c/WINDOWS/system32:/usr/bin:/bin"
        )
        unit_file = tmp_path / "hermes-gateway.service"
        unit_file.write_text(installed)

        monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: unit_file)
        monkeypatch.setattr(
            gw,
            "generate_systemd_unit",
            lambda system=False, run_as_user=None: regenerated,
        )

        assert gw.systemd_unit_is_current(system=False) is False

    def test_unit_differing_by_hermes_home_still_outdated(self, tmp_path, monkeypatch):
        """PATH masking must not hide a genuinely changed non-PATH directive."""
        from hermes_cli import gateway as gw

        installed = """[Unit]
Description=Hermes Gateway

[Service]
Type=simple
ExecStart=/usr/bin/python gateway run
Environment="PATH=/a:/mnt/c/WINDOWS/system32"
Environment="HERMES_HOME=/home/a/.hermes"
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
"""
        regenerated = installed.replace(
            'HERMES_HOME=/home/a/.hermes', 'HERMES_HOME=/home/a/.hermes/profiles/coder'
        ).replace('PATH=/a:/mnt/c/WINDOWS/system32', 'PATH=/a:/usr/bin')
        unit_file = tmp_path / "hermes-gateway.service"
        unit_file.write_text(installed)

        monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: unit_file)
        monkeypatch.setattr(
            gw,
            "generate_systemd_unit",
            lambda system=False, run_as_user=None: regenerated,
        )

        assert gw.systemd_unit_is_current(system=False) is False

"""Operator customisations in the launchd plist survive regeneration (#82046).

`hermes gateway status` tells the operator to run `hermes gateway start` when the
service definition is stale. That path regenerated the plist from a fixed
template, dropping any `*ResourceLimits` block or extra `EnvironmentVariables`
entry an operator had added — silently, and with no backup. These tests pin the
preservation, the backup, and the messaging.
"""

import plistlib
from types import SimpleNamespace

import pytest

import hermes_cli.gateway as gateway_cli


GENERATED = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ai.hermes.gateway</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/bin:/bin</string>
        <key>VIRTUAL_ENV</key>
        <string>/Users/alice/.hermes/venv</string>
        <key>HERMES_HOME</key>
        <string>/Users/alice/.hermes</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
"""


def _customised(plist_text=GENERATED, *, limits=True, env=True):
    """Return ``plist_text`` with the customisations an operator would add."""
    parsed = plistlib.loads(plist_text.encode("utf-8"))
    if limits:
        parsed["SoftResourceLimits"] = {"NumberOfFiles": 4096}
        parsed["HardResourceLimits"] = {"NumberOfFiles": 8192}
    if env:
        parsed["EnvironmentVariables"]["HERMES_MEMORY_DAEMON_IDLE_TIMEOUT"] = "900"
    return plistlib.dumps(parsed).decode("utf-8")


def _generated_with_resource_limits(plist_text=GENERATED, nfiles=10240):
    """Simulate a template that owns Soft/HardResourceLimits (e.g. #80748)."""
    parsed = plistlib.loads(plist_text.encode("utf-8"))
    parsed["SoftResourceLimits"] = {"NumberOfFiles": nfiles}
    parsed["HardResourceLimits"] = {"NumberOfFiles": nfiles}
    return plistlib.dumps(parsed).decode("utf-8")


class TestLaunchdPlistCustomisations:
    def test_generated_plist_reports_no_customisations(self):
        # Pass generated= so ownership is evaluated against this fixture, not
        # whatever the live template happens to emit on main.
        assert gateway_cli.launchd_plist_customisations(
            GENERATED, generated=GENERATED
        ) == ({}, {})

    def test_detects_resource_limits_and_extra_env(self):
        extra_top_level, extra_env = gateway_cli.launchd_plist_customisations(
            _customised(), generated=GENERATED
        )

        assert set(extra_top_level) == {"SoftResourceLimits", "HardResourceLimits"}
        assert extra_top_level["SoftResourceLimits"] == {"NumberOfFiles": 4096}
        assert extra_env == {"HERMES_MEMORY_DAEMON_IDLE_TIMEOUT": "900"}

    def test_unparseable_plist_reports_no_customisations(self):
        # Hand-edited plists can be malformed. Detection must degrade, not raise.
        assert gateway_cli.launchd_plist_customisations("<plist>old content</plist>") == ({}, {})
        assert gateway_cli.launchd_plist_customisations("") == ({}, {})

    def test_managed_keys_are_never_treated_as_customisations(self):
        # A managed key the operator edited is still managed — regeneration is
        # supposed to update it. Only unrecognised keys are carried across.
        edited = GENERATED.replace("<key>KeepAlive</key>\n    <true/>", "<key>KeepAlive</key>\n    <false/>")
        extra_top_level, _ = gateway_cli.launchd_plist_customisations(
            edited, generated=GENERATED
        )

        assert "KeepAlive" not in extra_top_level

    def test_template_owned_resource_limits_are_not_operator_customisations(self):
        # #80748 coexistence: once the template emits Soft/HardResourceLimits,
        # a stock install of that template must not be reported as customised.
        template = _generated_with_resource_limits()
        assert gateway_cli.launchd_plist_customisations(
            template, generated=template
        ) == ({}, {})


class TestMergeLaunchdCustomisations:
    def test_merge_preserves_resource_limits(self):
        merged = gateway_cli._merge_launchd_customisations(GENERATED, _customised())
        parsed = plistlib.loads(merged.encode("utf-8"))

        assert parsed["SoftResourceLimits"] == {"NumberOfFiles": 4096}
        assert parsed["HardResourceLimits"] == {"NumberOfFiles": 8192}

    def test_merge_preserves_extra_environment_variables(self):
        merged = gateway_cli._merge_launchd_customisations(GENERATED, _customised())
        parsed = plistlib.loads(merged.encode("utf-8"))

        assert parsed["EnvironmentVariables"]["HERMES_MEMORY_DAEMON_IDLE_TIMEOUT"] == "900"

    def test_merge_keeps_generated_values_for_managed_keys(self):
        installed = _customised(GENERATED.replace("/usr/bin:/bin", "/stale/path"))
        merged = gateway_cli._merge_launchd_customisations(GENERATED, installed)
        parsed = plistlib.loads(merged.encode("utf-8"))

        assert parsed["EnvironmentVariables"]["PATH"] == "/usr/bin:/bin"

    def test_template_owned_resource_limits_are_not_double_spliced(self):
        # #80748 coexistence: when the template already emits Soft/HardResourceLimits,
        # merging a stock installed copy of that template must be a pure noop —
        # no second SoftResourceLimits block, no "stale forever" drift.
        template = _generated_with_resource_limits()
        merged = gateway_cli._merge_launchd_customisations(template, template)

        assert merged == template
        assert merged.count("<key>SoftResourceLimits</key>") == 1
        assert merged.count("<key>HardResourceLimits</key>") == 1

    def test_operator_limits_yield_to_template_when_template_owns_them(self):
        # Once the template owns the limit keys, a hand-raised value is a
        # managed-key edit (regeneration is supposed to reset it) — not an
        # operator customisation to carry across. Extra env still survives.
        template = _generated_with_resource_limits(nfiles=10240)
        installed = _customised(template, limits=True, env=True)
        # _customised() overwrites limits to 4096/8192; confirm that.
        installed_parsed = plistlib.loads(installed.encode("utf-8"))
        assert installed_parsed["SoftResourceLimits"] == {"NumberOfFiles": 4096}

        merged = gateway_cli._merge_launchd_customisations(template, installed)
        parsed = plistlib.loads(merged.encode("utf-8"))

        assert parsed["SoftResourceLimits"] == {"NumberOfFiles": 10240}
        assert parsed["HardResourceLimits"] == {"NumberOfFiles": 10240}
        assert parsed["EnvironmentVariables"]["HERMES_MEMORY_DAEMON_IDLE_TIMEOUT"] == "900"
        assert merged.count("<key>SoftResourceLimits</key>") == 1

    def test_merge_is_a_noop_without_customisations(self):
        # The overwhelmingly common case must stay byte-identical to the template.
        assert gateway_cli._merge_launchd_customisations(GENERATED, GENERATED) == GENERATED

    def test_merge_tolerates_unparseable_installed_plist(self):
        assert gateway_cli._merge_launchd_customisations(GENERATED, "not a plist") == GENERATED

    def test_merge_output_is_valid_plist(self):
        merged = gateway_cli._merge_launchd_customisations(GENERATED, _customised())

        assert plistlib.loads(merged.encode("utf-8"))["Label"] == "ai.hermes.gateway"

    def test_merge_is_idempotent(self):
        # Not cosmetic: launchd_plist_is_current() compares the installed plist
        # against the merged generation. If merging its own output drifted, the
        # plist would be reported stale forever and every start would rewrite it.
        once = gateway_cli._merge_launchd_customisations(GENERATED, _customised())
        twice = gateway_cli._merge_launchd_customisations(GENERATED, once)

        assert twice == once


class TestLaunchdPlistIsCurrentWithCustomisations:
    def test_customised_plist_is_not_perpetually_stale(self, tmp_path, monkeypatch):
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
        monkeypatch.setattr(gateway_cli, "generate_launchd_plist", lambda: GENERATED)

        # What refresh would write for an operator-customised plist...
        plist_path.write_text(
            gateway_cli._merge_launchd_customisations(GENERATED, _customised()),
            encoding="utf-8",
        )

        # ...must then read back as current, or `status` nags forever.
        assert gateway_cli.launchd_plist_is_current() is True

    def test_stale_customised_plist_is_still_detected_as_stale(self, tmp_path, monkeypatch):
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
        monkeypatch.setattr(gateway_cli, "generate_launchd_plist", lambda: GENERATED)
        plist_path.write_text(
            _customised(GENERATED.replace("<true/>\n\n    <key>KeepAlive</key>", "<false/>\n\n    <key>KeepAlive</key>")),
            encoding="utf-8",
        )

        assert gateway_cli.launchd_plist_is_current() is False


class TestServiceFileBackup:
    def test_backup_copies_content_to_timestamped_sibling(self, tmp_path):
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        plist_path.write_text(GENERATED, encoding="utf-8")

        backup = gateway_cli._backup_service_file(plist_path)

        assert backup is not None
        assert backup.parent == plist_path.parent
        assert backup.read_text(encoding="utf-8") == GENERATED

    def test_backup_is_not_named_plist(self, tmp_path):
        # launchd must never mistake a backup for a second service definition.
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        plist_path.write_text(GENERATED, encoding="utf-8")

        backup = gateway_cli._backup_service_file(plist_path)

        assert backup.suffix != ".plist"
        assert ".bak-" in backup.name

    def test_backup_of_missing_file_is_none(self, tmp_path):
        assert gateway_cli._backup_service_file(tmp_path / "absent.plist") is None

    def test_backups_are_pruned_to_the_retention_window(self, tmp_path, monkeypatch):
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        plist_path.write_text(GENERATED, encoding="utf-8")
        monkeypatch.setattr(gateway_cli, "_SERVICE_BACKUP_RETENTION", 3)

        for _ in range(6):
            gateway_cli._backup_service_file(plist_path)

        assert len(list(tmp_path.glob("ai.hermes.gateway.plist.bak-*"))) == 3


class TestRefreshPreservesCustomisations:
    @pytest.fixture
    def refresh_env(self, tmp_path, monkeypatch):
        """Stub out everything after the write so only the write path is exercised."""
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
        monkeypatch.setattr(gateway_cli, "generate_launchd_plist", lambda: GENERATED)
        monkeypatch.setattr(gateway_cli, "_refuse_temp_home_service_write", lambda *a, **k: False)
        monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway")
        monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/501")
        monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: tmp_path)
        # No running gateway → refresh takes the simple in-process reload branch.
        monkeypatch.setitem(
            __import__("sys").modules,
            "gateway.status",
            __import__("gateway.status", fromlist=["status"]),
        )
        monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: None)
        # launchctl reports the label as supervising a live process, so the
        # reload's retry loop settles on the first pass.
        monkeypatch.setattr(
            gateway_cli.subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(returncode=0, stdout='"PID" = 4242;', stderr=""),
        )
        return plist_path

    def test_refresh_preserves_customisations_and_backs_up(self, refresh_env, capsys):
        plist_path = refresh_env
        plist_path.write_text(_customised(), encoding="utf-8")

        assert gateway_cli.refresh_launchd_plist_if_needed() is True

        written = plistlib.loads(plist_path.read_text(encoding="utf-8").encode("utf-8"))
        assert written["SoftResourceLimits"] == {"NumberOfFiles": 4096}
        assert written["EnvironmentVariables"]["HERMES_MEMORY_DAEMON_IDLE_TIMEOUT"] == "900"

        backups = list(plist_path.parent.glob("ai.hermes.gateway.plist.bak-*"))
        assert len(backups) == 1
        assert "SoftResourceLimits" in backups[0].read_text(encoding="utf-8")

        out = capsys.readouterr().out
        assert "SoftResourceLimits" in out
        assert "HERMES_MEMORY_DAEMON_IDLE_TIMEOUT" in out
        assert "Backed up existing launchd plist to:" in out

    def test_refresh_of_uncustomised_plist_writes_the_plain_template(self, refresh_env):
        plist_path = refresh_env
        plist_path.write_text(GENERATED.replace("<true/>", "<false/>", 1), encoding="utf-8")

        assert gateway_cli.refresh_launchd_plist_if_needed() is True
        assert plist_path.read_text(encoding="utf-8") == GENERATED

    def test_refresh_is_skipped_when_already_current(self, refresh_env):
        plist_path = refresh_env
        plist_path.write_text(
            gateway_cli._merge_launchd_customisations(GENERATED, _customised()),
            encoding="utf-8",
        )

        assert gateway_cli.refresh_launchd_plist_if_needed() is False
        assert list(plist_path.parent.glob("*.bak-*")) == []


class TestStatusRecommendation:
    @pytest.fixture
    def status_env(self, tmp_path, monkeypatch):
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
        monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway")
        monkeypatch.setattr(gateway_cli, "generate_launchd_plist", lambda: GENERATED)
        monkeypatch.setattr(gateway_cli, "_launchd_unsupported_marker_exists", lambda: False)
        monkeypatch.setattr(
            gateway_cli.subprocess,
            "run",
            lambda *a, **k: type("R", (), {"returncode": 0, "stdout": '"PID" = 4242;'})(),
        )
        monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: 4242)
        return plist_path

    def test_stale_status_names_the_customisations_it_will_keep(self, status_env, capsys):
        status_env.write_text(
            _customised(GENERATED.replace("<true/>", "<false/>", 1)), encoding="utf-8"
        )

        gateway_cli.launchd_status()

        out = capsys.readouterr().out
        assert "stale" in out
        assert "SoftResourceLimits" in out
        assert "HERMES_MEMORY_DAEMON_IDLE_TIMEOUT" in out
        assert "carried across" in out
        assert "backed up" in out

    def test_stale_status_does_not_recommend_restart_as_the_remedy(self, status_env, capsys):
        # `restart` never refreshes the definition, so it cannot clear this
        # warning. Status must not imply it is an equivalent remedy.
        status_env.write_text(GENERATED.replace("<true/>", "<false/>", 1), encoding="utf-8")

        gateway_cli.launchd_status()

        out = capsys.readouterr().out
        assert "Run: hermes gateway start" in out
        assert "Run: hermes gateway restart" not in out
        assert "only recycles the running process" in out

    def test_current_status_still_reports_present_customisations(self, status_env, capsys):
        status_env.write_text(
            gateway_cli._merge_launchd_customisations(GENERATED, _customised()),
            encoding="utf-8",
        )

        gateway_cli.launchd_status()

        out = capsys.readouterr().out
        assert "matches the current Hermes install" in out
        assert "SoftResourceLimits" in out

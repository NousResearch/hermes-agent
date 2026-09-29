# -*- coding: utf-8 -*-
"""Regression tests for the launchd plist scan's malformed-file tolerance.

``_loaded_launchd_backend_jobs`` documents that unreadable or
malformed plists are skipped, but ``plistlib.load`` propagates
``xml.parsers.expat.ExpatError`` (which is NOT a ``ValueError`` subclass) for
XML that is not well-formed, so one hand-edited LaunchAgent plist aborted the
whole ``hermes update`` post-pull cleanup instead of being skipped.
"""
import os
from unittest import mock

import pytest

from hermes_cli import main_dashboard

# ``_loaded_launchd_backend_jobs`` reads ``sys.platform`` directly (no host seam), so
# the scan runs only on a real macOS host — never by faking the platform.
pytestmark = pytest.mark.platforms("macos")


# A plist that is not well-formed XML yet launchd itself tolerates: a raw `&&`
# in ProgramArguments is exactly what an operator writes when trying to chain
# two commands in one job.
MALFORMED_PLIST = (
    '<?xml version="1.0" encoding="UTF-8"?>\n<plist version="1.0">\n<dict>\n'
    "  <key>Label</key>\n  <string>com.example.bad</string>\n"
    "  <key>ProgramArguments</key>\n  <array>\n"
    "    <string>/usr/local/bin/hermes</string>\n    <string>&&</string>\n"
    "  </array>\n</dict>\n</plist>\n"
)

GOOD_PLIST = (
    '<?xml version="1.0" encoding="UTF-8"?>\n<plist version="1.0">\n<dict>\n'
    "  <key>Label</key>\n  <string>ai.hermes.dashboard.test</string>\n"
    "  <key>ProgramArguments</key>\n  <array>\n"
    "    <string>/usr/local/bin/hermes</string>\n    <string>dashboard</string>\n"
    "    <string>--port</string>\n    <string>9119</string>\n"
    "  </array>\n</dict>\n</plist>\n"
)


def _good_plist_with_home(home: str) -> str:
    return GOOD_PLIST.replace(
        "  <key>ProgramArguments</key>",
        "  <key>EnvironmentVariables</key>\n  <dict>\n"
        f"    <key>HERMES_HOME</key>\n    <string>{home}</string>\n"
        "  </dict>\n  <key>ProgramArguments</key>",
    )


def test_malformed_plist_is_skipped_not_fatal(tmp_path):
    p = tmp_path / "com.example.bad.plist"
    p.write_text(MALFORMED_PLIST, encoding="utf-8")
    with mock.patch(
        "hermes_cli.gateway._launchd_print_service_pid", return_value=(False, None)
    ) as probe:
        assert main_dashboard._loaded_launchd_backend_jobs([("agent", tmp_path)]) == []
    probe.assert_not_called()  # the malformed job never reaches the launchctl probe


def test_malformed_sibling_does_not_hide_the_good_job(tmp_path):
    (tmp_path / "com.example.bad.plist").write_text(MALFORMED_PLIST, encoding="utf-8")
    (tmp_path / "ai.hermes.dashboard.test.plist").write_text(GOOD_PLIST, encoding="utf-8")
    with mock.patch(
        "hermes_cli.gateway._launchd_print_service_pid", return_value=(True, 4321)
    ) as probe:
        jobs = main_dashboard._loaded_launchd_backend_jobs([("agent", tmp_path)])
    assert jobs == [
        (
            f"gui/{os.getuid()}",  # windows-footgun: ok — platforms("macos") file
            "ai.hermes.dashboard.test",
            ["/usr/local/bin/hermes", "dashboard", "--port", "9119"],
            4321,
        )
    ]
    # Only the well-formed label was probed against launchd.
    assert [c.args[1] for c in probe.call_args_list] == ["ai.hermes.dashboard.test"]


def test_update_scope_excludes_loaded_job_pinned_to_foreign_home(tmp_path):
    plist = tmp_path / "ai.hermes.dashboard.test.plist"
    plist.write_text(_good_plist_with_home("/work/other/.hermes"), encoding="utf-8")

    with mock.patch(
        "hermes_cli.gateway._launchd_print_service_pid", return_value=(True, 4321)
    ) as probe:
        jobs = main_dashboard._loaded_launchd_backend_jobs(
            [("agent", tmp_path)], scope_homes={"/home/u/.hermes"}
        )

    assert jobs == []
    probe.assert_not_called()


def test_update_scope_keeps_loaded_job_pinned_to_owned_profile(tmp_path):
    owned = "/home/u/.hermes/profiles/work"
    plist = tmp_path / "ai.hermes.dashboard.test.plist"
    plist.write_text(_good_plist_with_home(owned), encoding="utf-8")

    with mock.patch(
        "hermes_cli.gateway._launchd_print_service_pid", return_value=(True, 4321)
    ):
        jobs = main_dashboard._loaded_launchd_backend_jobs(
            [("agent", tmp_path)], scope_homes={"/home/u/.hermes", owned}
        )

    assert len(jobs) == 1


def test_scoped_unpinned_job_cannot_claim_different_pid_by_argv(tmp_path):
    plist = tmp_path / "ai.hermes.dashboard.test.plist"
    plist.write_text(GOOD_PLIST, encoding="utf-8")

    with mock.patch(
        "hermes_cli.gateway._launchd_print_service_pid", return_value=(True, 9999)
    ):
        jobs = main_dashboard._loaded_launchd_backend_jobs(
            [("agent", tmp_path)], scope_homes={"/home/u/.hermes"}
        )

    argv = ["/usr/local/bin/hermes", "dashboard", "--port", "9119"]
    assert main_dashboard._launchd_job_owning_backend(1234, argv, jobs) is None
    assert main_dashboard._launchd_job_owning_backend(9999, argv, jobs) == (
        f"gui/{os.getuid()}", "ai.hermes.dashboard.test", 9999
    )


def test_scoped_pinned_job_argv_fallback_requires_same_exact_home(tmp_path):
    root = "/home/u/.hermes"
    profile = f"{root}/profiles/work"
    plist = tmp_path / "ai.hermes.dashboard.test.plist"
    plist.write_text(_good_plist_with_home(profile), encoding="utf-8")

    with mock.patch(
        "hermes_cli.gateway._launchd_print_service_pid", return_value=(True, None)
    ):
        jobs = main_dashboard._loaded_launchd_backend_jobs(
            [("agent", tmp_path)], scope_homes={root, profile}
        )

    argv = ["/usr/local/bin/hermes", "dashboard", "--port", "9119"]
    assert main_dashboard._launchd_job_owning_backend(
        1234, argv, jobs, hermes_home=root
    ) is None
    assert main_dashboard._launchd_job_owning_backend(
        1234, argv, jobs, hermes_home=profile
    ) == (f"gui/{os.getuid()}", "ai.hermes.dashboard.test", None)

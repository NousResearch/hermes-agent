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


# A LaunchAgent whose argv is a wrapper script under the Hermes home: the wrapper resolves the
# bind address at login and then starts the backend, so the job's own ProgramArguments carries no
# ``hermes … serve`` tail at all — only the live PID launchd reports (the wrapper shell) links the
# job to the backend process it supervises (#116536 follow-up report).
WRAPPER_PLIST = (
    '<?xml version="1.0" encoding="UTF-8"?>\n<plist version="1.0">\n<dict>\n'
    "  <key>Label</key>\n  <string>ai.hermes.serve.remote</string>\n"
    "  <key>ProgramArguments</key>\n  <array>\n"
    "    <string>/bin/bash</string>\n    <string>/Users/gabe/.hermes/bin/start-serve-remote.sh</string>\n"
    "  </array>\n</dict>\n</plist>\n"
)


def test_wrapper_script_job_is_collected_and_claims_its_backend_by_ancestor(tmp_path):
    (tmp_path / "ai.hermes.serve.remote.plist").write_text(WRAPPER_PLIST, encoding="utf-8")
    (tmp_path / "com.example.unrelated.plist").write_text(
        WRAPPER_PLIST.replace("ai.hermes.serve.remote", "com.example.unrelated")
        .replace("/Users/gabe/.hermes/bin/start-serve-remote.sh", "/Users/gabe/scripts/sync.sh"),
        encoding="utf-8",
    )
    with (
        mock.patch(
            "hermes_cli.gateway._launchd_print_service_pid", return_value=(True, 4321)
        ) as probe,
        mock.patch.object(
            main_dashboard,
            "_dashboard_cmdline_for_pid",
            return_value=["/usr/bin/rsync", "--server", "--daemon"],
        ),
    ):
        jobs = main_dashboard._loaded_launchd_backend_jobs([("agent", tmp_path)])
    assert jobs == [
        (
            f"gui/{os.getuid()}",
            "ai.hermes.serve.remote",
            ["/bin/bash", "/Users/gabe/.hermes/bin/start-serve-remote.sh"],
            4321,
        )
    ]
    # Both labels were probed — an operator wrapper can live anywhere, so the marker-less sync
    # job now reaches launchctl too — but only the Hermes-referencing wrapper was collected: the
    # sync job's live PID runs an rsync server, which is not holder evidence, so it stays out.
    assert [c.args[1] for c in probe.call_args_list] == [
        "ai.hermes.serve.remote",
        "com.example.unrelated",
    ]
    # And the collected wrapper job claims the backend PID through the ancestor it supervises:
    # the backend's argv never equals the wrapper's, so only the ancestor link can attribute it.
    assert main_dashboard._launchd_job_owning_backend(
        9999, ["hermes", "serve", "--host", "100.64.0.2", "--port", "9119"], jobs, ancestors=[4321, 1]
    ) == (f"gui/{os.getuid()}", "ai.hermes.serve.remote", 4321)
    # Without the ancestor link the wrapper job claims nothing (argv never matches).
    assert main_dashboard._launchd_job_owning_backend(
        9999, ["hermes", "serve", "--host", "100.64.0.2", "--port", "9119"], jobs
    ) is None


# A second wrapper spelling from the field (#116536 follow-up): a single-element argv whose
# path itself lives under the Hermes home. The wrapper injects the dashboard session token,
# guards the port, then exec's the serve — so launchd reports the backend process itself as
# the job's live PID, and attribution goes through the live-PID identity path (no ancestors).
SINGLE_WRAPPER_PLIST = (
    '<?xml version="1.0" encoding="UTF-8"?>\n<plist version="1.0">\n<dict>\n'
    "  <key>Label</key>\n  <string>ai.hermes.mobile-serve</string>\n"
    "  <key>ProgramArguments</key>\n  <array>\n"
    "    <string>/Users/wesker/.hermes/bin/run-hermes-serve-mobile.sh</string>\n"
    "  </array>\n</dict>\n</plist>\n"
)


def test_single_element_wrapper_job_is_collected_and_claims_its_execd_backend_by_live_pid(tmp_path):
    (tmp_path / "ai.hermes.mobile-serve.plist").write_text(SINGLE_WRAPPER_PLIST, encoding="utf-8")
    with mock.patch(
        "hermes_cli.gateway._launchd_print_service_pid", return_value=(True, 36566)
    ) as probe:
        jobs = main_dashboard._loaded_launchd_backend_jobs([("agent", tmp_path)])
    assert jobs == [
        (
            f"gui/{os.getuid()}",
            "ai.hermes.mobile-serve",
            ["/Users/wesker/.hermes/bin/run-hermes-serve-mobile.sh"],
            36566,
        )
    ]
    assert [c.args[1] for c in probe.call_args_list] == ["ai.hermes.mobile-serve"]
    # The wrapper exec'd the backend, so launchd's live PID IS the backend process: the job
    # claims it by identity alone, with no ancestor list and no argv match possible.
    assert main_dashboard._launchd_job_owning_backend(
        36566, ["hermes", "serve", "--host", "100.64.0.2", "--port", "9119"], jobs
    ) == (f"gui/{os.getuid()}", "ai.hermes.mobile-serve", 36566)


# A third wrapper spelling from the field (#116536, 2026-10-05 report): an operator-authored
# LaunchAgent whose ProgramArguments carries no Hermes marker at all — the wrapper script lives
# outside the Hermes home and exec's ``hermes dashboard``, so launchd reports the backend process
# itself as the job's live PID. Before the live-PID evidence path, this job was dropped before the
# launchd-owner matcher ever ran, the healthy launchd backend was classified ``manual-serve``, and
# the update took the doomed respawn-argv path (the bash argv mangled into a python -c one-liner).
MARKERLESS_WRAPPER_PLIST = (
    '<?xml version="1.0" encoding="UTF-8"?>\n<plist version="1.0">\n<dict>\n'
    "  <key>Label</key>\n  <string>com.hermes.dashboard</string>\n"
    "  <key>ProgramArguments</key>\n  <array>\n"
    "    <string>/bin/bash</string>\n    <string>/Users/ops/scripts/start-dashboard.sh</string>\n"
    "  </array>\n</dict>\n</plist>\n"
)


def test_markerless_wrapper_job_survives_on_live_pid_holder_evidence(tmp_path):
    (tmp_path / "com.hermes.dashboard.plist").write_text(
        MARKERLESS_WRAPPER_PLIST, encoding="utf-8"
    )
    with (
        mock.patch(
            "hermes_cli.gateway._launchd_print_service_pid", return_value=(True, 52466)
        ) as probe,
        mock.patch.object(
            main_dashboard,
            "_dashboard_cmdline_for_pid",
            return_value=["/Users/ops/.local/bin/hermes", "dashboard", "--no-open"],
        ),
    ):
        jobs = main_dashboard._loaded_launchd_backend_jobs([("agent", tmp_path)])
    assert jobs == [
        (
            f"gui/{os.getuid()}",
            "com.hermes.dashboard",
            ["/bin/bash", "/Users/ops/scripts/start-dashboard.sh"],
            52466,
        )
    ]
    # The plist argv carries no Hermes marker, so the label was admitted on the live PID's own
    # cmdline being a holder (the wrapper exec'd the backend), never on the plist text.
    assert [c.args[1] for c in probe.call_args_list] == ["com.hermes.dashboard"]
    # launchd's live PID IS the exec'd backend: the job claims it by identity alone — no
    # ancestor list, and the wrapper argv can never match the backend's own cmdline.
    assert main_dashboard._launchd_job_owning_backend(
        52466, ["/Users/ops/.local/bin/hermes", "dashboard", "--no-open"], jobs
    ) == (f"gui/{os.getuid()}", "com.hermes.dashboard", 52466)


def test_markerless_job_whose_live_pid_is_not_a_holder_is_not_collected(tmp_path):
    (tmp_path / "com.hermes.dashboard.plist").write_text(
        MARKERLESS_WRAPPER_PLIST, encoding="utf-8"
    )
    with (
        mock.patch(
            "hermes_cli.gateway._launchd_print_service_pid", return_value=(True, 909)
        ) as probe,
        mock.patch.object(
            main_dashboard,
            "_dashboard_cmdline_for_pid",
            return_value=[
                "/Applications/Updater.app/Contents/MacOS/updater",
                "--serve",
            ],
        ),
    ):
        assert main_dashboard._loaded_launchd_backend_jobs([("agent", tmp_path)]) == []
    # Probed once (the gui domain holds the label), then rejected on evidence: a loaded job
    # whose live PID runs an unrelated ``--serve`` updater claims no Hermes backend — which is
    # also what keeps a manually-started serve out of launchd's restart path.
    probe.assert_called_once()

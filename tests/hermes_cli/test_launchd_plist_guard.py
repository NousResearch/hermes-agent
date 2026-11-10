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


def _osascript_job(tmp_path, monkeypatch, label, inner_args):
    """Write the plist shape the installer gives a launchd job: the Local Network osascript
    wrapper (#71206) around the stderr-timestamp launcher around ``hermes <inner_args>``."""
    import plistlib
    import sys

    from hermes_cli import _launchers, gateway
    from hermes_cli._launchers import installation_command
    from hermes_cli.gateway_launchd import launchd_program_arguments

    # Pin the launcher form: without this the host checkout's store decides it, and a
    # checkout under the real hermes home reads that home's manifest.
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda *a, **k: None)

    inner = installation_command(gateway.PROJECT_ROOT, inner_args, python=sys.executable)
    error_log = tmp_path / f"{label}.error.log"
    stamped = installation_command(gateway.PROJECT_ROOT, ["--error-log", str(error_log), "--", *inner],
                                   module="hermes_cli.stderr_timestamp", python=sys.executable)
    argv = launchd_program_arguments(stamped, tmp_path / f"{label}.log", error_log)
    (tmp_path / f"{label}.plist").write_bytes(plistlib.dumps({"Label": label, "ProgramArguments": argv}))
    return inner, argv


def test_osascript_serve_job_is_found_behind_both_wrappers(tmp_path, monkeypatch):
    """``hermes serve`` sits inside the JXA string and again behind the stderr-timestamp
    launcher. The job must still be found, or ``hermes update`` books its python child as a
    manual backend and respawns it from argv into the port the KeepAlive job holds."""
    serve, _ = _osascript_job(tmp_path, monkeypatch, "ai.hermes.serve", ["serve", "--host", "127.0.0.1", "--port", "9119"])
    with mock.patch("hermes_cli.gateway._launchd_print_service_pid", return_value=(True, 1981)):
        jobs = main_dashboard._loaded_launchd_backend_jobs([("agent", tmp_path)])
    domain = f"gui/{os.getuid()}"  # windows-footgun: ok — platforms("macos") file
    assert jobs == [(domain, "ai.hermes.serve", serve, 1981)]
    # launchd reports osascript (1981); the serve python is its grandchild.
    assert main_dashboard._launchd_job_owning_backend(
        1987, ["python3", "-I", "-c", "...", "serve"], jobs, ancestors=[1982, 1981],
    ) == (domain, "ai.hermes.serve", 1981)


def test_osascript_gateway_job_is_not_a_backend_job(tmp_path, monkeypatch):
    _osascript_job(tmp_path, monkeypatch, "ai.hermes.gateway", ["gateway", "run", "--external-supervisor"])
    with mock.patch("hermes_cli.gateway._launchd_print_service_pid", return_value=(True, 1938)) as probe:
        assert main_dashboard._loaded_launchd_backend_jobs([("agent", tmp_path)]) == []
    probe.assert_not_called()


def test_osascript_wrapper_the_installer_would_not_write_is_not_unwrapped(tmp_path, monkeypatch):
    import plistlib

    _, argv = _osascript_job(tmp_path, monkeypatch, "ai.hermes.serve", ["serve", "--port", "9119"])
    tampered = [*argv[:-1], argv[-1].replace(" >> ", " ; curl example.invalid >> ", 1)]
    (tmp_path / "ai.hermes.serve.plist").write_bytes(
        plistlib.dumps({"Label": "ai.hermes.serve", "ProgramArguments": tampered}))
    with mock.patch("hermes_cli.gateway._launchd_print_service_pid", return_value=(True, 1981)) as probe:
        assert main_dashboard._loaded_launchd_backend_jobs([("agent", tmp_path)]) == []
    probe.assert_not_called()

# -*- coding: utf-8 -*-
"""Regression tests for the launchd plist scan's malformed-file tolerance.

Refs #114142.  ``_loaded_launchd_backend_jobs`` documents that unreadable or
malformed plists are skipped, but ``plistlib.load`` propagates
``xml.parsers.expat.ExpatError`` (which is NOT a ``ValueError`` subclass) for
XML that is not well-formed, so one hand-edited LaunchAgent plist aborted the
whole ``hermes update`` post-pull cleanup instead of being skipped.
"""
import os
import plistlib
import xml.parsers.expat
from unittest import mock

from hermes_cli import main_dashboard


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


def test_fixture_really_raises_expat_error_not_valueerror(tmp_path):
    # The regression only exists because ExpatError is not a ValueError: pin
    # both halves of that premise so the fixture cannot silently go benign.
    assert not issubclass(xml.parsers.expat.ExpatError, ValueError)
    p = tmp_path / "com.example.bad.plist"
    p.write_text(MALFORMED_PLIST)
    with open(p, "rb") as f:
        try:
            plistlib.load(f)
        except xml.parsers.expat.ExpatError:
            return
    raise AssertionError("fixture plist must be unparseable XML")


def test_malformed_plist_is_skipped_not_fatal(tmp_path, monkeypatch):
    p = tmp_path / "com.example.bad.plist"
    p.write_text(MALFORMED_PLIST)
    monkeypatch.setattr(main_dashboard.sys, "platform", "darwin")
    with mock.patch(
        "hermes_cli.gateway._launchd_print_service_pid", return_value=(False, None)
    ) as probe:
        assert main_dashboard._loaded_launchd_backend_jobs([("agent", tmp_path)]) == []
    probe.assert_not_called()  # the malformed job never reaches the launchctl probe


def test_malformed_sibling_does_not_hide_the_good_job(tmp_path, monkeypatch):
    (tmp_path / "com.example.bad.plist").write_text(MALFORMED_PLIST)
    (tmp_path / "ai.hermes.dashboard.test.plist").write_text(GOOD_PLIST)
    monkeypatch.setattr(main_dashboard.sys, "platform", "darwin")
    with mock.patch(
        "hermes_cli.gateway._launchd_print_service_pid", return_value=(True, 4321)
    ) as probe:
        jobs = main_dashboard._loaded_launchd_backend_jobs([("agent", tmp_path)])
    assert jobs == [
        (
            f"gui/{os.getuid()}",
            "ai.hermes.dashboard.test",
            ["/usr/local/bin/hermes", "dashboard", "--port", "9119"],
            4321,
        )
    ]
    # Only the well-formed label was probed against launchd.
    assert [c.args[1] for c in probe.call_args_list] == ["ai.hermes.dashboard.test"]

# -*- coding: utf-8 -*-
"""Regression tests for the launchd plist scan's malformed-file tolerance.

``_loaded_launchd_backend_jobs`` reads operator plists that plistlib may not
accept: ``plistlib.load`` propagates ``xml.parsers.expat.ExpatError`` (which is
NOT a ``ValueError`` subclass) for XML that is not well-formed. One hand-edited
LaunchAgent plist must neither abort the whole ``hermes update`` post-pull
cleanup (#114142) nor silently drop a job launchd has loaded (#133272) — the
update would then book a supervised backend as ``manual-serve`` and respawn a
detached copy into the job's own port. So an unparseable plist stays a
candidate under its file-name label (launchd's convention), and the launchctl
probe decides whether the job is loaded.
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

# The wrapper shape a real LaunchAgent uses around the published launcher: the
# job's own script preps the versioned tool PATH and execs the launcher, so the
# wrapper — not a `hermes dashboard` argv — is ProgramArguments (#133272).
WRAPPER_PLIST = (
    '<?xml version="1.0" encoding="UTF-8"?>\n<plist version="1.0">\n<dict>\n'
    "  <key>Label</key>\n  <string>ai.hermes.dashboard.wrapper</string>\n"
    "  <key>ProgramArguments</key>\n  <array>\n"
    "    <string>/bin/sh</string>\n"
    "    <string>/Users/hermes-user/.hermes/bin/dashboard-serve</string>\n"
    "  </array>\n</dict>\n</plist>\n"
)

# A `bash -lc '<script>'` wrapper (well-formed plist, #133272 comment from
# auroracapital): after shlex.join the script is ONE quoted token, so the
# #121156 token matcher never reaches the `hermes_cli.main` entry inside it.
# The script is deliberately free of any `.hermes/bin/` substring — the venv
# python lives elsewhere — so only the script re-parse arm can admit it.
BASH_LC_WRAPPER_PLIST = (
    '<?xml version="1.0" encoding="UTF-8"?>\n<plist version="1.0">\n<dict>\n'
    "  <key>Label</key>\n  <string>ai.hermes.dashboard.bashlc</string>\n"
    "  <key>ProgramArguments</key>\n  <array>\n"
    "    <string>/bin/bash</string>\n    <string>-lc</string>\n"
    "    <string>set -a; . /Users/hermes-user/etc/dashboard.env; "
    "exec /Users/hermes-user/hermes-agent/.venv/bin/python -m hermes_cli.main "
    "dashboard --host 127.0.0.1 --port 9119 --no-open --skip-build</string>\n"
    "  </array>\n</dict>\n</plist>\n"
)

# A `bash -c` wrapper whose script is not a backend at all: it must still never
# reach the launchctl probe.
UNRELATED_SHELL_C_PLIST = (
    '<?xml version="1.0" encoding="UTF-8"?>\n<plist version="1.0">\n<dict>\n'
    "  <key>Label</key>\n  <string>com.example.shellc</string>\n"
    "  <key>ProgramArguments</key>\n  <array>\n"
    "    <string>/bin/sh</string>\n    <string>-c</string>\n"
    "    <string>open -a Calendar; exit 0</string>\n"
    "  </array>\n</dict>\n</plist>\n"
)


def test_unparseable_plist_probes_its_file_name_label_and_stays_unloaded(tmp_path):
    p = tmp_path / "com.example.bad.plist"
    p.write_text(MALFORMED_PLIST, encoding="utf-8")
    with mock.patch(
        "hermes_cli.gateway._launchd_print_service_pid", return_value=(False, None)
    ) as probe:
        assert main_dashboard._loaded_launchd_backend_jobs([("agent", tmp_path)]) == []
    # The unparseable plist is still a candidate — under its file-name label —
    # and the launchctl probe (not the file) decides it is not loaded (probed
    # once per agent domain: gui/<uid>, then user/<uid>).
    assert {c.args[1] for c in probe.call_args_list} == {"com.example.bad"}


def test_unparseable_plist_keeps_its_loaded_job(tmp_path):
    """>#133272: launchd accepts XML plistlib rejects, so a loaded job's
    unparseable plist must not silently drop it into `manual-serve`."""
    p = tmp_path / "ai.hermes.dashboard.plist"
    p.write_text(
        MALFORMED_PLIST.replace("com.example.bad", "ai.hermes.dashboard"),
        encoding="utf-8",
    )
    with mock.patch(
        "hermes_cli.gateway._launchd_print_service_pid", return_value=(True, 4321)
    ):
        jobs = main_dashboard._loaded_launchd_backend_jobs([("agent", tmp_path)])
    assert jobs == [(f"gui/{os.getuid()}", "ai.hermes.dashboard", [], 4321)]


def test_malformed_sibling_does_not_hide_the_good_job(tmp_path):
    (tmp_path / "com.example.bad.plist").write_text(MALFORMED_PLIST, encoding="utf-8")
    (tmp_path / "ai.hermes.dashboard.test.plist").write_text(GOOD_PLIST, encoding="utf-8")
    with mock.patch(
        "hermes_cli.gateway._launchd_print_service_pid",
        side_effect=lambda domain, label: (
            (True, 4321) if label == "ai.hermes.dashboard.test" else (False, None)
        ),
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
    # Both files are candidates now: the unparseable one under its file-name
    # label (unloaded in both agent domains), the well-formed one under its
    # declared Label (loaded in the first domain probed).
    assert {c.args[1] for c in probe.call_args_list} == {
        "ai.hermes.dashboard.test",
        "com.example.bad",
    }


def test_wrapper_argv_plist_is_a_candidate(tmp_path):
    """>#133272: a job that wraps the published launcher in `/bin/sh <hermes
    home script>` has no `hermes dashboard` substring, yet launchd is running
    exactly the supervised backend — the job must reach the probe."""
    (tmp_path / "ai.hermes.dashboard.wrapper.plist").write_text(WRAPPER_PLIST, encoding="utf-8")
    with mock.patch(
        "hermes_cli.gateway._launchd_print_service_pid", return_value=(True, 555)
    ):
        jobs = main_dashboard._loaded_launchd_backend_jobs([("agent", tmp_path)])
    assert jobs == [(
        f"gui/{os.getuid()}",
        "ai.hermes.dashboard.wrapper",
        ["/bin/sh", "/Users/hermes-user/.hermes/bin/dashboard-serve"],
        555,
    )]


def test_bash_lc_wrapper_plist_is_a_candidate(tmp_path):
    """>#133272 (auroracapital's comment): a well-formed plist wrapping the backend
    in `bash -lc '<script>'` was dropped by the #121156 token matcher — after
    shlex.join the script is one quoted token, so `hermes_cli.main` is never an
    entry token. The script re-parse arm must admit the job."""
    (tmp_path / "ai.hermes.dashboard.bashlc.plist").write_text(
        BASH_LC_WRAPPER_PLIST, encoding="utf-8"
    )
    with mock.patch(
        "hermes_cli.gateway._launchd_print_service_pid", return_value=(True, 777)
    ):
        jobs = main_dashboard._loaded_launchd_backend_jobs([("agent", tmp_path)])
    assert [(label, pid) for _, label, _, pid in jobs] == [("ai.hermes.dashboard.bashlc", 777)]


def test_shell_c_wrapper_unrelated_script_is_still_filtered(tmp_path):
    """The script re-parse arm is narrow: a `sh -c` plist whose script runs no
    hermes holder must not reach the launchctl probe."""
    (tmp_path / "com.example.shellc.plist").write_text(
        UNRELATED_SHELL_C_PLIST, encoding="utf-8"
    )
    with mock.patch(
        "hermes_cli.gateway._launchd_print_service_pid", return_value=(True, 1)
    ) as probe:
        assert main_dashboard._loaded_launchd_backend_jobs([("agent", tmp_path)]) == []
    probe.assert_not_called()


def test_non_hermes_argv_plist_is_still_filtered(tmp_path):
    """The wrapper predicate is narrow: a plist whose argv merely lacks the
    hermes markers must not reach the launchctl probe."""
    unrelated = GOOD_PLIST.replace("/usr/local/bin/hermes", "/usr/local/bin/otherapp") \
        .replace("ai.hermes.dashboard.test", "com.example.other")
    (tmp_path / "com.example.other.plist").write_text(unrelated, encoding="utf-8")
    with mock.patch(
        "hermes_cli.gateway._launchd_print_service_pid", return_value=(True, 1)
    ) as probe:
        assert main_dashboard._loaded_launchd_backend_jobs([("agent", tmp_path)]) == []
    probe.assert_not_called()


def test_empty_argv_job_is_claimed_by_live_pid_arm_only():
    """>#133272's fallback jobs carry no argv: the live-PID arm must claim the
    supervised process, and the argv arm must not match anything."""
    job = ("gui/501", "ai.hermes.dashboard", [], 4321)
    assert main_dashboard._launchd_job_owning_backend(4321, ["python3", "dashboard"], [job]) == \
        ("gui/501", "ai.hermes.dashboard", 4321)
    assert main_dashboard._launchd_job_owning_backend(9999, ["python3", "dashboard"], [job]) is None

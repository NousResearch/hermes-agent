"""Regression tests: debug share captures update.log + desktop-update-handoff.log.

A Desktop-driven update that fails at the Electron rebuild (updater exit 6)
stores its root cause in exactly two files: ``logs/update.log`` (full
stdout/stderr mirror of ``hermes update``) and
``logs/desktop-update-handoff.log`` (hand-off stages incl. the rebuild
retry). The updater's failure message and the Desktop error box both point
users at ``hermes debug share``, yet the share captured neither file, so
update-failure reports arrived with zero root-cause signal (#100874).

These tests fail against the pre-fix capture set (five logs only).
"""

import pytest

# Lines the writers really produce (main.py _UpdateOutputStream mirror and
# scripts/desktop-update/windows.ps1 Write-HandoffLog / posix.sh log()).
UPDATE_LOG_BODY = (
    "=== hermes update started 2026-09-02T03:21:09 ===\n"
    "→ Fetching updates...\n"
    "→ Building desktop packaged app...\n"
    "✗ Desktop GUI build failed\n"
)
HANDOFF_LOG_BODY = (
    "2026-09-02T03:21:08 desktop rebuild exit code: 1\n"
    "2026-09-02T03:21:08 rebuild!| npm ERR! code ELIFECYCLE\n"
)


@pytest.fixture
def home_with_update_logs(tmp_path, monkeypatch):
    """Isolated HERMES_HOME whose logs directory holds the update pair."""
    home = tmp_path / "hermes_test_home"
    logs = home / "logs"
    logs.mkdir(parents=True)
    (logs / "agent.log").write_text(
        "2026-09-02 03:21:00 INFO agent.conversation_loop: API call #1\n"
    )
    (logs / "update.log").write_text(UPDATE_LOG_BODY)
    (logs / "desktop-update-handoff.log").write_text(HANDOFF_LOG_BODY)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def home_without_update_logs(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with only the classic five logs present."""
    home = tmp_path / "hermes_test_home"
    logs = home / "logs"
    logs.mkdir(parents=True)
    for name in ("agent.log", "errors.log", "gateway.log", "gui.log", "desktop.log"):
        (logs / name).write_text("2026-09-02 03:21:00 INFO x: one line\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


class TestCaptureDefaultLogSnapshots:
    """The snapshot collector must include the update/hand-off pair."""

    def test_snapshots_include_update_and_handoff_keys(self, home_with_update_logs):
        from hermes_cli.debug import _capture_default_log_snapshots

        snaps = _capture_default_log_snapshots(100)

        assert "update" in snaps
        assert "handoff" in snaps
        assert "Desktop GUI build failed" in snaps["update"].tail_text
        assert "ELIFECYCLE" in snaps["handoff"].tail_text

    def test_missing_files_report_absence_without_raising(
        self, home_without_update_logs
    ):
        """A fresh install has no update logs — capture degrades, not crashes."""
        from hermes_cli.debug import _capture_default_log_snapshots

        snaps = _capture_default_log_snapshots(100)

        # Keys exist; the snapshots describe the missing files honestly.
        assert "update" in snaps
        assert "handoff" in snaps
        assert "not found" in snaps["update"].tail_text
        assert "not found" in snaps["handoff"].tail_text


class TestCollectDebugReport:
    """The summary report (what users paste) shows the update tails."""

    def test_report_contains_update_and_handoff_sections(self, home_with_update_logs):
        from hermes_cli.debug import collect_debug_report

        report = collect_debug_report(log_lines=50, dump_text="dump")

        assert "--- update.log" in report
        assert "Desktop GUI build failed" in report
        assert "--- desktop-update-handoff.log" in report
        assert "ELIFECYCLE" in report

    def test_report_omits_update_sections_when_logs_absent(
        self, home_without_update_logs
    ):
        from hermes_cli.debug import collect_debug_report

        report = collect_debug_report(log_lines=50, dump_text="dump")

        # Honest absence: the section header appears with a missing-note
        # (snapshot present), never a KeyError. The classic five remain.
        assert "--- agent.log" in report
        assert "--- update.log" in report
        assert "--- desktop-update-handoff.log" in report

    def test_report_tolerates_a_hand_built_snapshots_dict(self, home_with_update_logs):
        """Gateway /debug and older callers pass dicts without the new keys.

        They must keep working (no KeyError) — degraded to the old
        five-section report rather than crashing.
        """
        from hermes_cli.debug import LogSnapshot, collect_debug_report

        legacy_keys = ("agent", "errors", "gateway", "gui", "desktop")
        legacy = {k: LogSnapshot(path=None, tail_text="", full_text="") for k in legacy_keys}

        report = collect_debug_report(
            log_lines=50, dump_text="dump", log_snapshots=legacy
        )

        assert "--- agent.log" in report
        assert "--- update.log" not in report


class TestCollectShareBundle:
    """The upload bundle (paste.rs + --nous) carries the full update logs."""

    def test_bundle_includes_update_logs_when_present(self, home_with_update_logs):
        from hermes_cli.debug import collect_share_bundle

        bundle = collect_share_bundle(log_lines=50, redact=False)

        assert "update.log" in bundle
        assert "Desktop GUI build failed" in bundle["update.log"]
        assert "--- full update.log ---" in bundle["update.log"]
        assert "desktop-update-handoff.log" in bundle
        assert "ELIFECYCLE" in bundle["desktop-update-handoff.log"]

    def test_bundle_omits_update_keys_when_files_absent(
        self, home_without_update_logs
    ):
        from hermes_cli.debug import collect_share_bundle

        bundle = collect_share_bundle(log_lines=50, redact=False)

        assert "update.log" not in bundle
        assert "desktop-update-handoff.log" not in bundle
        # The classic five keep uploading.
        assert "agent.log" in bundle


class TestFullLogUploadLabels:
    """build_debug_share uploads every bundle log — including the new pair."""

    def test_upload_label_list_includes_update_logs(self):
        """The paste-upload loop must not silently skip the new labels.

        Asserts the label tuple (source-derived behavior contract: the
        bundle keys are only reachable for upload if listed here).
        """
        import inspect

        from hermes_cli import debug

        source = inspect.getsource(debug.build_debug_share)
        assert '"update.log"' in source
        assert '"desktop-update-handoff.log"' in source

    def test_local_print_label_list_includes_update_logs(self):
        """run_debug_share --local prints bundle logs by the same contract."""
        import inspect

        from hermes_cli import debug

        source = inspect.getsource(debug.run_debug_share)
        assert '"FULL update.log"' in source
        assert '"FULL desktop-update-handoff.log"' in source


class TestLogsRegistry:
    """LOG_FILES gains readable keys for files `hermes logs list` already shows."""

    def test_update_and_handoff_keys_resolve_real_filenames(self):
        from hermes_cli.logs import LOG_FILES

        assert LOG_FILES.get("update") == "update.log"
        assert LOG_FILES.get("handoff") == "desktop-update-handoff.log"

    def test_tail_log_reads_update_log_from_hermes_home(self, home_with_update_logs):
        """End-to-end: `hermes logs update` resolves and prints the file."""
        from hermes_cli import logs as logs_mod

        logs_mod.tail_log("update", num_lines=10)

    def test_tail_log_names_the_new_logs_when_unknown_requested(
        self, home_with_update_logs, capsys
    ):
        from hermes_cli import logs as logs_mod

        with pytest.raises(SystemExit) as excinfo:
            logs_mod.tail_log("no-such-log")

        assert excinfo.value.code == 1
        # The available-log hint teaches the new keys.
        hint = capsys.readouterr().out
        assert "update" in hint
        assert "handoff" in hint

"""Tests for ``hermes debug`` CLI command and debug utilities."""

import os
import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Set up an isolated HERMES_HOME with minimal logs."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Create log files
    logs_dir = home / "logs"
    logs_dir.mkdir()
    (logs_dir / "agent.log").write_text(
        "2026-04-12 17:00:00 INFO agent: session started\n"
        "2026-04-12 17:00:01 INFO tools.terminal: running ls\n"
        "2026-04-12 17:00:02 WARNING agent: high token usage\n"
    )
    (logs_dir / "errors.log").write_text(
        "2026-04-12 17:00:05 ERROR gateway.run: connection lost\n"
    )
    (logs_dir / "gateway.log").write_text(
        "2026-04-12 17:00:10 INFO gateway.run: started\n"
    )

    return home


# ---------------------------------------------------------------------------
# Unit tests for upload helpers
# ---------------------------------------------------------------------------

class TestUploadPasteRs:
    """Test paste.rs upload path."""

    def test_upload_paste_rs_success(self):
        from hermes_cli.debug import _upload_paste_rs

        mock_resp = MagicMock()
        mock_resp.read.return_value = b"https://paste.rs/abc123\n"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("hermes_cli.debug.urllib.request.urlopen", return_value=mock_resp):
            url = _upload_paste_rs("hello world")

        assert url == "https://paste.rs/abc123"

    def test_upload_paste_rs_bad_response(self):
        from hermes_cli.debug import _upload_paste_rs

        mock_resp = MagicMock()
        mock_resp.read.return_value = b"<html>error</html>"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("hermes_cli.debug.urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(ValueError, match="Unexpected response"):
                _upload_paste_rs("test")

    def test_upload_paste_rs_network_error(self):
        from hermes_cli.debug import _upload_paste_rs

        with patch(
            "hermes_cli.debug.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with pytest.raises(urllib.error.URLError):
                _upload_paste_rs("test")


class TestUploadDpasteCom:
    """Test dpaste.com fallback upload path."""

    def test_upload_dpaste_com_success(self):
        from hermes_cli.debug import _upload_dpaste_com

        mock_resp = MagicMock()
        mock_resp.read.return_value = b"https://dpaste.com/ABCDEFG\n"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("hermes_cli.debug.urllib.request.urlopen", return_value=mock_resp):
            url = _upload_dpaste_com("hello world", expiry_days=7)

        assert url == "https://dpaste.com/ABCDEFG"


class TestUploadToPastebin:
    """Test the combined upload with fallback."""

    def test_tries_paste_rs_first(self):
        from hermes_cli.debug import upload_to_pastebin

        with patch("hermes_cli.debug._upload_paste_rs",
                    return_value="https://paste.rs/test") as prs:
            url = upload_to_pastebin("content")

        assert url == "https://paste.rs/test"
        prs.assert_called_once()

    def test_falls_back_to_dpaste_com(self):
        from hermes_cli.debug import upload_to_pastebin

        with patch("hermes_cli.debug._upload_paste_rs",
                    side_effect=Exception("down")), \
             patch("hermes_cli.debug._upload_dpaste_com",
                    return_value="https://dpaste.com/TEST") as dp:
            url = upload_to_pastebin("content")

        assert url == "https://dpaste.com/TEST"
        dp.assert_called_once()

    def test_raises_when_both_fail(self):
        from hermes_cli.debug import upload_to_pastebin

        with patch("hermes_cli.debug._upload_paste_rs",
                    side_effect=Exception("err1")), \
             patch("hermes_cli.debug._upload_dpaste_com",
                    side_effect=Exception("err2")):
            with pytest.raises(RuntimeError, match="Failed to upload"):
                upload_to_pastebin("content")


# ---------------------------------------------------------------------------
# Log reading
# ---------------------------------------------------------------------------

class TestCaptureLogSnapshot:
    """Test _capture_log_snapshot for log reading and truncation."""

    def test_reads_small_file(self, hermes_home):
        from hermes_cli.debug import _capture_log_snapshot

        snap = _capture_log_snapshot("agent", tail_lines=10)
        assert snap.full_text is not None
        assert "session started" in snap.full_text
        assert "session started" in snap.tail_text

    def test_returns_none_for_missing(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))

        from hermes_cli.debug import _capture_log_snapshot
        snap = _capture_log_snapshot("agent", tail_lines=10)
        assert snap.full_text is None
        assert snap.tail_text == "(file not found)"

    def test_empty_primary_reports_file_empty(self, hermes_home):
        """Empty primary (no .1 fallback) surfaces as '(file empty)', not missing."""
        (hermes_home / "logs" / "agent.log").write_text("")

        from hermes_cli.debug import _capture_log_snapshot
        snap = _capture_log_snapshot("agent", tail_lines=10)
        assert snap.full_text is None
        assert snap.tail_text == "(file empty)"

    def test_race_truncate_after_resolve_reports_empty(self, hermes_home, monkeypatch):
        """If the log is truncated between resolve and stat, say 'empty', not 'missing'."""
        log_path = hermes_home / "logs" / "agent.log"
        from hermes_cli import debug

        monkeypatch.setattr(debug, "_resolve_log_path", lambda _name: log_path)
        log_path.write_text("")

        snap = debug._capture_log_snapshot("agent", tail_lines=10)
        assert snap.path == log_path
        assert snap.full_text is None
        assert snap.tail_text == "(file empty)"

    def test_truncates_large_file(self, hermes_home):
        """Files larger than max_bytes get tail-truncated."""
        from hermes_cli.debug import _capture_log_snapshot

        # Write a file larger than 1KB
        big_content = "x" * 100 + "\n"
        (hermes_home / "logs" / "agent.log").write_text(big_content * 200)

        snap = _capture_log_snapshot("agent", tail_lines=10, max_bytes=1024)
        assert snap.full_text is not None
        assert "truncated" in snap.full_text

    def test_keeps_first_line_when_truncation_on_boundary(self, hermes_home):
        """When truncation lands on a line boundary, keep the first full line."""
        from hermes_cli.debug import _capture_log_snapshot

        # File must exceed the initial chunk_size (8192) used by the
        # backward-reading loop so the truncation path actually fires.
        line = "A" * 99 + "\n"  # 100 bytes per line
        num_lines = 200  # 20000 bytes
        (hermes_home / "logs" / "agent.log").write_text(line * num_lines)

        # max_bytes = 1000 = 100 * 10 → cut at byte 20000 - 1000 = 19000,
        # and byte 19000 - 1 is '\n'.  Boundary hit → keep all 10 lines.
        snap = _capture_log_snapshot("agent", tail_lines=5, max_bytes=1000)
        assert snap.full_text is not None
        assert "truncated" in snap.full_text
        raw = snap.full_text.split("\n", 1)[1]
        kept = [l for l in raw.strip().splitlines() if l.startswith("A")]
        assert len(kept) == 10

    def test_drops_partial_when_truncation_mid_line(self, hermes_home):
        """When truncation lands mid-line, drop the partial fragment."""
        from hermes_cli.debug import _capture_log_snapshot

        line = "A" * 99 + "\n"  # 100 bytes per line
        num_lines = 200  # 20000 bytes
        (hermes_home / "logs" / "agent.log").write_text(line * num_lines)

        # max_bytes = 950 doesn't divide evenly into 100 → mid-line cut.
        snap = _capture_log_snapshot("agent", tail_lines=5, max_bytes=950)
        assert snap.full_text is not None
        assert "truncated" in snap.full_text
        raw = snap.full_text.split("\n", 1)[1]
        kept = [l for l in raw.strip().splitlines() if l.startswith("A")]
        # 950 / 100 = 9.5 → 9 complete lines after dropping partial
        assert len(kept) == 9

    def test_unknown_log_returns_none(self, hermes_home):
        from hermes_cli.debug import _capture_log_snapshot
        snap = _capture_log_snapshot("nonexistent", tail_lines=10)
        assert snap.full_text is None

    def test_falls_back_to_rotated_file(self, hermes_home):
        """When gateway.log doesn't exist, falls back to gateway.log.1."""
        from hermes_cli.debug import _capture_log_snapshot

        logs_dir = hermes_home / "logs"
        # Remove the primary (if any) and create a .1 rotation
        (logs_dir / "gateway.log").unlink(missing_ok=True)
        (logs_dir / "gateway.log.1").write_text(
            "2026-04-12 10:00:00 INFO gateway.run: rotated content\n"
        )

        snap = _capture_log_snapshot("gateway", tail_lines=10)
        assert snap.full_text is not None
        assert "rotated content" in snap.full_text

    def test_prefers_primary_over_rotated(self, hermes_home):
        """Primary log is used when it exists, even if .1 also exists."""
        from hermes_cli.debug import _capture_log_snapshot

        logs_dir = hermes_home / "logs"
        (logs_dir / "gateway.log").write_text("primary content\n")
        (logs_dir / "gateway.log.1").write_text("rotated content\n")

        snap = _capture_log_snapshot("gateway", tail_lines=10)
        assert "primary content" in snap.full_text
        assert "rotated" not in snap.full_text

    def test_falls_back_when_primary_empty(self, hermes_home):
        """Empty primary log falls back to .1 rotation."""
        from hermes_cli.debug import _capture_log_snapshot

        logs_dir = hermes_home / "logs"
        (logs_dir / "agent.log").write_text("")
        (logs_dir / "agent.log.1").write_text("rotated agent data\n")

        snap = _capture_log_snapshot("agent", tail_lines=10)
        assert snap.full_text is not None
        assert "rotated agent data" in snap.full_text


# ---------------------------------------------------------------------------
# Capture log redaction (force=True applies regardless of HERMES_REDACT_SECRETS)
# ---------------------------------------------------------------------------

# A vendor-prefixed token used across redaction tests. Long enough to clear
# the redactor's `floor` parameter so it actually masks rather than fully blanks.
_REDACT_FIXTURE_TOKEN = "sk-proj-A1B2C3D4E5F6G7H8I9J0aA"


class TestCaptureLogSnapshotRedaction:
    """Pin upload-time redaction at the _capture_log_snapshot boundary."""

    @pytest.fixture
    def hermes_home_with_secret(self, tmp_path, monkeypatch):
        """Isolated HERMES_HOME whose agent.log contains a vendor-prefixed token."""
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        # Critical: ensure the user has NOT opted in to redaction. The whole
        # point of this PR is that share-time redaction works for users who
        # never set this env var.
        monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)

        logs_dir = home / "logs"
        logs_dir.mkdir()
        (logs_dir / "agent.log").write_text(
            f"2026-04-12 17:00:00 INFO config: api_key={_REDACT_FIXTURE_TOKEN} loaded\n"
        )
        (logs_dir / "errors.log").write_text("")
        (logs_dir / "gateway.log").write_text("")
        return home

    def test_default_redacts_tail_and_full_text(self, hermes_home_with_secret):
        from hermes_cli.debug import _capture_log_snapshot

        snap = _capture_log_snapshot("agent", tail_lines=10)

        # Both views the upload uses must be sanitized.
        assert _REDACT_FIXTURE_TOKEN not in snap.tail_text
        assert snap.full_text is not None
        assert _REDACT_FIXTURE_TOKEN not in snap.full_text

    def test_redact_false_passes_through(self, hermes_home_with_secret):
        from hermes_cli.debug import _capture_log_snapshot

        snap = _capture_log_snapshot("agent", tail_lines=10, redact=False)

        # Original token survives when the caller opts out.
        assert _REDACT_FIXTURE_TOKEN in snap.tail_text
        assert _REDACT_FIXTURE_TOKEN in (snap.full_text or "")

    def test_force_true_overrides_unset_env_var(self, hermes_home_with_secret):
        """Regression test: redact_sensitive_text short-circuits without force=True.

        If a future refactor drops `force=True` from `_redact_log_text`, this
        test fails immediately. Without `force=True`, the redactor returns the
        input unchanged when HERMES_REDACT_SECRETS is unset, and the feature
        ships silently broken for its target audience.
        """
        import os

        from hermes_cli.debug import _capture_log_snapshot

        # Belt-and-suspenders: confirm the env var is genuinely unset for this
        # test so we know we're exercising the force=True path.
        assert os.environ.get("HERMES_REDACT_SECRETS", "") == ""

        snap = _capture_log_snapshot("agent", tail_lines=10)

        assert _REDACT_FIXTURE_TOKEN not in snap.tail_text
        assert snap.full_text is not None
        assert _REDACT_FIXTURE_TOKEN not in snap.full_text

    def test_capture_default_log_snapshots_threads_redact(
        self, hermes_home_with_secret
    ):
        from hermes_cli.debug import _capture_default_log_snapshots

        snaps = _capture_default_log_snapshots(50)

        # Default threads redact=True to all three captured logs.
        assert _REDACT_FIXTURE_TOKEN not in snaps["agent"].tail_text
        assert _REDACT_FIXTURE_TOKEN not in (snaps["agent"].full_text or "")

    def test_capture_default_log_snapshots_no_redact_passes_through(
        self, hermes_home_with_secret
    ):
        from hermes_cli.debug import _capture_default_log_snapshots

        snaps = _capture_default_log_snapshots(50, redact=False)

        assert _REDACT_FIXTURE_TOKEN in snaps["agent"].tail_text
        assert _REDACT_FIXTURE_TOKEN in (snaps["agent"].full_text or "")


# ---------------------------------------------------------------------------
# Debug report collection
# ---------------------------------------------------------------------------

class TestCollectDebugReport:
    """Test the debug report builder."""

    def test_report_includes_dump_output(self, hermes_home):
        from hermes_cli.debug import collect_debug_report

        with patch("hermes_cli.dump.run_dump") as mock_dump:
            mock_dump.side_effect = lambda args: print(
                "--- hermes dump ---\nversion: 0.8.0\n--- end dump ---"
            )
            report = collect_debug_report(log_lines=50)

        assert "--- hermes dump ---" in report
        assert "version: 0.8.0" in report

    def test_report_includes_agent_log(self, hermes_home):
        from hermes_cli.debug import collect_debug_report

        with patch("hermes_cli.dump.run_dump"):
            report = collect_debug_report(log_lines=50)

        assert "--- agent.log" in report
        assert "session started" in report

    def test_report_includes_errors_log(self, hermes_home):
        from hermes_cli.debug import collect_debug_report

        with patch("hermes_cli.dump.run_dump"):
            report = collect_debug_report(log_lines=50)

        assert "--- errors.log" in report
        assert "connection lost" in report

    def test_report_includes_gateway_log(self, hermes_home):
        from hermes_cli.debug import collect_debug_report

        with patch("hermes_cli.dump.run_dump"):
            report = collect_debug_report(log_lines=50)

        assert "--- gateway.log" in report

    def test_missing_logs_handled(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))

        from hermes_cli.debug import collect_debug_report

        with patch("hermes_cli.dump.run_dump"):
            report = collect_debug_report(log_lines=50)

        assert "(file not found)" in report


# ---------------------------------------------------------------------------
# CLI entry point — run_debug_share
# ---------------------------------------------------------------------------

class TestRunDebugShare:
    """Test the run_debug_share CLI handler."""

    def test_share_sweeps_expired_pastes(self, hermes_home, capsys):
        """Slash-command path should sweep old pending deletes before uploading."""
        from hermes_cli.debug import run_debug_share

        args = MagicMock()
        args.lines = 50
        args.expire = 7
        args.local = False

        with patch("hermes_cli.dump.run_dump"), \
             patch("hermes_cli.debug._sweep_expired_pastes", return_value=(0, 0)) as mock_sweep, \
             patch("hermes_cli.debug.upload_to_pastebin",
                    return_value="https://paste.rs/test"):
            run_debug_share(args)

        mock_sweep.assert_called_once()
        assert "Debug report uploaded" in capsys.readouterr().out

    def test_share_survives_sweep_failure(self, hermes_home, capsys):
        """Expired-paste cleanup is best-effort and must not block sharing."""
        from hermes_cli.debug import run_debug_share

        args = MagicMock()
        args.lines = 50
        args.expire = 7
        args.local = False

        with patch("hermes_cli.dump.run_dump"), \
             patch(
                 "hermes_cli.debug._sweep_expired_pastes",
                 side_effect=RuntimeError("offline"),
             ), \
             patch("hermes_cli.debug.upload_to_pastebin",
                    return_value="https://paste.rs/test"):
            run_debug_share(args)

        assert "https://paste.rs/test" in capsys.readouterr().out

    def test_local_flag_prints_full_logs(self, hermes_home, capsys):
        """--local prints the report plus full log contents."""
        from hermes_cli.debug import run_debug_share

        args = MagicMock()
        args.lines = 50
        args.expire = 7
        args.local = True

        with patch("hermes_cli.dump.run_dump"):
            run_debug_share(args)

        out = capsys.readouterr().out
        assert "--- agent.log" in out
        assert "FULL agent.log" in out
        assert "FULL gateway.log" in out

    def test_share_uploads_three_pastes(self, hermes_home, capsys):
        """Successful share uploads report + agent.log + gateway.log."""
        from hermes_cli.debug import run_debug_share

        args = MagicMock()
        args.lines = 50
        args.expire = 7
        args.local = False

        call_count = [0]
        uploaded_content = []
        def _mock_upload(content, expiry_days=7):
            call_count[0] += 1
            uploaded_content.append(content)
            return f"https://paste.rs/paste{call_count[0]}"

        with patch("hermes_cli.dump.run_dump") as mock_dump, \
             patch("hermes_cli.debug.upload_to_pastebin",
                    side_effect=_mock_upload):
            mock_dump.side_effect = lambda a: print("--- hermes dump ---\nversion: test\n--- end dump ---")
            run_debug_share(args)

        out = capsys.readouterr().out
        # Should have 3 uploads: report, agent.log, gateway.log
        assert call_count[0] == 3
        assert "paste.rs/paste1" in out  # Report
        assert "paste.rs/paste2" in out  # agent.log
        assert "paste.rs/paste3" in out  # gateway.log
        assert "Report" in out
        assert "agent.log" in out
        assert "gateway.log" in out

        # Each log paste should start with the dump header
        agent_paste = uploaded_content[1]
        assert "--- hermes dump ---" in agent_paste
        assert "--- full agent.log ---" in agent_paste
        gateway_paste = uploaded_content[2]
        assert "--- hermes dump ---" in gateway_paste
        assert "--- full gateway.log ---" in gateway_paste

    def test_share_keeps_report_and_full_log_on_same_snapshot(self, hermes_home, capsys):
        """A mid-run rotation must not make full agent.log older than the report."""
        from hermes_cli.debug import run_debug_share, collect_debug_report as real_collect_debug_report

        logs_dir = hermes_home / "logs"
        (logs_dir / "agent.log").write_text(
            "2026-04-22 12:00:00 INFO agent: newest line\n"
        )
        (logs_dir / "agent.log.1").write_text(
            "2026-04-10 12:00:00 INFO agent: old rotated line\n"
        )

        args = MagicMock()
        args.lines = 50
        args.expire = 7
        args.local = False

        uploaded_content = []

        def _mock_upload(content, expiry_days=7):
            uploaded_content.append(content)
            return f"https://paste.rs/paste{len(uploaded_content)}"

        def _wrapped_collect_debug_report(*, log_lines=200, dump_text="", log_snapshots=None):
            report = real_collect_debug_report(
                log_lines=log_lines,
                dump_text=dump_text,
                log_snapshots=log_snapshots,
            )
            # Simulate the live log rotating after the report is built but
            # before the old implementation would have re-read agent.log for
            # standalone upload.
            (logs_dir / "agent.log").write_text("")
            (logs_dir / "agent.log.1").write_text(
                "2026-04-10 12:00:00 INFO agent: old rotated line\n"
            )
            return report

        with patch("hermes_cli.dump.run_dump"), \
             patch("hermes_cli.debug.collect_debug_report", side_effect=_wrapped_collect_debug_report), \
             patch("hermes_cli.debug.upload_to_pastebin", side_effect=_mock_upload):
            run_debug_share(args)

        report_paste = uploaded_content[0]
        agent_paste = uploaded_content[1]
        assert "2026-04-22 12:00:00 INFO agent: newest line" in report_paste
        assert "2026-04-22 12:00:00 INFO agent: newest line" in agent_paste
        assert "old rotated line" not in agent_paste

    def test_share_skips_missing_logs(self, tmp_path, monkeypatch, capsys):
        """Only uploads logs that exist."""
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))

        from hermes_cli.debug import run_debug_share

        args = MagicMock()
        args.lines = 50
        args.expire = 7
        args.local = False

        call_count = [0]
        def _mock_upload(content, expiry_days=7):
            call_count[0] += 1
            return f"https://paste.rs/paste{call_count[0]}"

        with patch("hermes_cli.dump.run_dump"), \
             patch("hermes_cli.debug.upload_to_pastebin",
                    side_effect=_mock_upload):
            run_debug_share(args)

        out = capsys.readouterr().out
        # Only the report should be uploaded (no log files exist)
        assert call_count[0] == 1
        assert "Report" in out

    def test_share_continues_on_log_upload_failure(self, hermes_home, capsys):
        """Log upload failure doesn't stop the report from being shared."""
        from hermes_cli.debug import run_debug_share

        args = MagicMock()
        args.lines = 50
        args.expire = 7
        args.local = False

        call_count = [0]
        def _mock_upload(content, expiry_days=7):
            call_count[0] += 1
            if call_count[0] > 1:
                raise RuntimeError("upload failed")
            return "https://paste.rs/report"

        with patch("hermes_cli.dump.run_dump"), \
             patch("hermes_cli.debug.upload_to_pastebin",
                    side_effect=_mock_upload):
            run_debug_share(args)

        out = capsys.readouterr().out
        assert "Report" in out
        assert "paste.rs/report" in out
        assert "failed to upload" in out

    def test_share_exits_on_report_upload_failure(self, hermes_home, capsys):
        """If the main report fails to upload, exit with code 1."""
        from hermes_cli.debug import run_debug_share

        args = MagicMock()
        args.lines = 50
        args.expire = 7
        args.local = False

        with patch("hermes_cli.dump.run_dump"), \
             patch("hermes_cli.debug.upload_to_pastebin",
                    side_effect=RuntimeError("all failed")):
            with pytest.raises(SystemExit) as exc_info:
                run_debug_share(args)

        assert exc_info.value.code == 1
        out = capsys.readouterr()
        assert "all failed" in out.err


# ---------------------------------------------------------------------------
# Share-time redaction wiring + visible banner
# ---------------------------------------------------------------------------

class TestRunDebugShareRedaction:
    """End-to-end: --no-redact flag, banner injection, default behavior."""

    @pytest.fixture
    def hermes_home_with_secret(self, tmp_path, monkeypatch):
        """Isolated HERMES_HOME whose agent.log contains a vendor-prefixed token."""
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)

        logs_dir = home / "logs"
        logs_dir.mkdir()
        (logs_dir / "agent.log").write_text(
            f"2026-04-12 17:00:00 INFO config: api_key={_REDACT_FIXTURE_TOKEN} loaded\n"
        )
        (logs_dir / "errors.log").write_text("")
        (logs_dir / "gateway.log").write_text(
            f"2026-04-12 17:00:01 INFO gateway.run: token {_REDACT_FIXTURE_TOKEN}\n"
        )
        return home

    def test_default_share_redacts_uploaded_content(
        self, hermes_home_with_secret, capsys
    ):
        """The uploaded report and full-log pastes do not contain the raw token."""
        from hermes_cli.debug import run_debug_share

        args = MagicMock()
        args.lines = 50
        args.expire = 7
        args.local = False
        args.no_redact = False

        captured: list[str] = []

        def fake_upload(content, expiry_days=7):
            captured.append(content)
            return f"https://paste.rs/{len(captured)}"

        with patch("hermes_cli.dump.run_dump"), \
             patch("hermes_cli.debug._sweep_expired_pastes", return_value=(0, 0)), \
             patch("hermes_cli.debug.upload_to_pastebin", side_effect=fake_upload):
            run_debug_share(args)

        # At least the report plus one full log paste reached the upload path.
        assert len(captured) >= 2
        for content in captured:
            assert _REDACT_FIXTURE_TOKEN not in content, (
                "raw token leaked into upload-bound content"
            )

    def test_default_share_includes_redaction_banner(
        self, hermes_home_with_secret, capsys
    ):
        """Each upload-bound paste carries the visible redaction banner."""
        from hermes_cli.debug import run_debug_share

        args = MagicMock()
        args.lines = 50
        args.expire = 7
        args.local = False
        args.no_redact = False

        captured: list[str] = []

        def fake_upload(content, expiry_days=7):
            captured.append(content)
            return f"https://paste.rs/{len(captured)}"

        with patch("hermes_cli.dump.run_dump"), \
             patch("hermes_cli.debug._sweep_expired_pastes", return_value=(0, 0)), \
             patch("hermes_cli.debug.upload_to_pastebin", side_effect=fake_upload):
            run_debug_share(args)

        for content in captured:
            assert "redacted at upload time" in content, (
                "redaction banner missing from upload-bound content"
            )

    def test_no_redact_flag_disables_redaction_and_banner(
        self, hermes_home_with_secret, capsys
    ):
        """--no-redact preserves original log content and omits the banner."""
        from hermes_cli.debug import run_debug_share

        args = MagicMock()
        args.lines = 50
        args.expire = 7
        args.local = False
        args.no_redact = True

        captured: list[str] = []

        def fake_upload(content, expiry_days=7):
            captured.append(content)
            return f"https://paste.rs/{len(captured)}"

        with patch("hermes_cli.dump.run_dump"), \
             patch("hermes_cli.debug._sweep_expired_pastes", return_value=(0, 0)), \
             patch("hermes_cli.debug.upload_to_pastebin", side_effect=fake_upload):
            run_debug_share(args)

        # The agent.log paste should now contain the raw token.
        assert any(_REDACT_FIXTURE_TOKEN in c for c in captured), (
            "expected raw token in --no-redact upload"
        )
        # No banner anywhere when redaction is disabled.
        for content in captured:
            assert "redacted at upload time" not in content, (
                "banner present with --no-redact"
            )


# ---------------------------------------------------------------------------
# run_debug router
# ---------------------------------------------------------------------------

class TestRunDebug:
    def test_no_subcommand_shows_usage(self, capsys):
        from hermes_cli.debug import run_debug

        args = MagicMock()
        args.debug_command = None

        run_debug(args)

        out = capsys.readouterr().out
        assert "hermes debug" in out
        assert "share" in out
        assert "delete" in out

    def test_share_subcommand_routes(self, hermes_home):
        from hermes_cli.debug import run_debug

        args = MagicMock()
        args.debug_command = "share"
        args.lines = 200
        args.expire = 7
        args.local = True

        with patch("hermes_cli.dump.run_dump"):
            run_debug(args)


# ---------------------------------------------------------------------------
# Argparse integration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Delete / auto-delete
# ---------------------------------------------------------------------------

class TestExtractPasteId:
    def test_paste_rs_url(self):
        from hermes_cli.debug import _extract_paste_id
        assert _extract_paste_id("https://paste.rs/abc123") == "abc123"

    def test_paste_rs_trailing_slash(self):
        from hermes_cli.debug import _extract_paste_id
        assert _extract_paste_id("https://paste.rs/abc123/") == "abc123"

    def test_http_variant(self):
        from hermes_cli.debug import _extract_paste_id
        assert _extract_paste_id("http://paste.rs/xyz") == "xyz"

    def test_non_paste_rs_returns_none(self):
        from hermes_cli.debug import _extract_paste_id
        assert _extract_paste_id("https://dpaste.com/ABCDEF") is None

    def test_empty_returns_none(self):
        from hermes_cli.debug import _extract_paste_id
        assert _extract_paste_id("") is None


class TestDeletePaste:
    def test_delete_sends_delete_request(self):
        from hermes_cli.debug import delete_paste

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("hermes_cli.debug.urllib.request.urlopen",
                    return_value=mock_resp) as mock_open:
            result = delete_paste("https://paste.rs/abc123")

        assert result is True
        req = mock_open.call_args[0][0]
        assert req.method == "DELETE"
        assert "paste.rs/abc123" in req.full_url

    def test_delete_rejects_non_paste_rs(self):
        from hermes_cli.debug import delete_paste

        with pytest.raises(ValueError, match="only paste.rs"):
            delete_paste("https://dpaste.com/something")


class TestScheduleAutoDelete:
    """``_schedule_auto_delete`` used to spawn a detached Python subprocess
    per call (one per paste URL batch).  Those subprocesses slept 6 hours
    and accumulated forever under repeated use — 15+ orphaned interpreters
    were observed in production.

    The new implementation is stateless: it records pending deletions to
    ``~/.hermes/pastes/pending.json`` and lets ``_sweep_expired_pastes``
    handle the DELETE requests synchronously on the next ``hermes debug``
    invocation.
    """

    def test_does_not_spawn_subprocess(self, hermes_home):
        """Regression guard: _schedule_auto_delete must NEVER spawn subprocesses.

        We assert this structurally rather than by mocking Popen: the new
        implementation doesn't even import ``subprocess`` at module scope,
        so a mock patch wouldn't find it.
        """
        import ast
        import inspect
        from hermes_cli.debug import _schedule_auto_delete

        # Strip the docstring before scanning so the regression-rationale
        # prose inside it doesn't trigger our banned-word checks.
        source = inspect.getsource(_schedule_auto_delete)
        tree = ast.parse(source)
        func_node = tree.body[0]
        if (
            func_node.body
            and isinstance(func_node.body[0], ast.Expr)
            and isinstance(func_node.body[0].value, ast.Constant)
            and isinstance(func_node.body[0].value.value, str)
        ):
            func_node.body = func_node.body[1:]
        code_only = ast.unparse(func_node)

        assert "Popen" not in code_only, (
            "_schedule_auto_delete must not spawn subprocesses — "
            "use pending.json + _sweep_expired_pastes instead"
        )
        assert "subprocess" not in code_only, (
            "_schedule_auto_delete must not reference subprocess at all"
        )
        assert "time.sleep" not in code_only, (
            "Regression: sleeping in _schedule_auto_delete is the bug being fixed"
        )

        # And verify that calling it doesn't produce any orphaned children
        # (it should just write pending.json synchronously).
        import os as _os
        before = set(_os.listdir("/proc")) if _os.path.exists("/proc") else None
        _schedule_auto_delete(
            ["https://paste.rs/abc", "https://paste.rs/def"],
            delay_seconds=10,
        )
        if before is not None:
            after = set(_os.listdir("/proc"))
            new = after - before
            # Filter to only integer-named entries (process PIDs)
            new_pids = [p for p in new if p.isdigit()]
            # It's fine if unrelated processes appeared — we just need to make
            # sure we didn't spawn a long-sleeping one.  The old bug spawned
            # a python interpreter whose cmdline contained "time.sleep".
            for pid in new_pids:
                try:
                    with open(f"/proc/{pid}/cmdline", "rb") as f:
                        cmdline = f.read().decode("utf-8", errors="replace")
                    assert "time.sleep" not in cmdline, (
                        f"Leaked sleeper subprocess PID {pid}: {cmdline}"
                    )
                except OSError:
                    pass  # process exited already

    def test_records_pending_to_json(self, hermes_home):
        """Scheduled URLs are persisted to pending.json with expiration."""
        from hermes_cli.debug import _schedule_auto_delete, _pending_file
        import json

        _schedule_auto_delete(
            ["https://paste.rs/abc", "https://paste.rs/def"],
            delay_seconds=10,
        )

        pending_path = _pending_file()
        assert pending_path.exists()

        entries = json.loads(pending_path.read_text())
        assert len(entries) == 2
        urls = {e["url"] for e in entries}
        assert urls == {"https://paste.rs/abc", "https://paste.rs/def"}

        # expire_at is ~now + delay_seconds
        import time
        for e in entries:
            assert e["expire_at"] > time.time()
            assert e["expire_at"] <= time.time() + 15

    def test_skips_non_paste_rs_urls(self, hermes_home):
        """dpaste.com URLs auto-expire — don't track them."""
        from hermes_cli.debug import _schedule_auto_delete, _pending_file

        _schedule_auto_delete(["https://dpaste.com/something"])

        # pending.json should not be created for non-paste.rs URLs
        assert not _pending_file().exists()

    def test_merges_with_existing_pending(self, hermes_home):
        """Subsequent calls merge into existing pending.json."""
        from hermes_cli.debug import _schedule_auto_delete, _load_pending

        _schedule_auto_delete(["https://paste.rs/first"], delay_seconds=10)
        _schedule_auto_delete(["https://paste.rs/second"], delay_seconds=10)

        entries = _load_pending()
        urls = {e["url"] for e in entries}
        assert urls == {"https://paste.rs/first", "https://paste.rs/second"}

    def test_dedupes_same_url(self, hermes_home):
        """Same URL recorded twice → one entry with the later expire_at."""
        from hermes_cli.debug import _schedule_auto_delete, _load_pending

        _schedule_auto_delete(["https://paste.rs/dup"], delay_seconds=10)
        _schedule_auto_delete(["https://paste.rs/dup"], delay_seconds=100)

        entries = _load_pending()
        assert len(entries) == 1
        assert entries[0]["url"] == "https://paste.rs/dup"


class TestSweepExpiredPastes:
    """Test the opportunistic sweep that replaces the sleeping subprocess."""

    def test_sweep_empty_is_noop(self, hermes_home):
        from hermes_cli.debug import _sweep_expired_pastes

        deleted, remaining = _sweep_expired_pastes()
        assert deleted == 0
        assert remaining == 0

    def test_sweep_deletes_expired_entries(self, hermes_home):
        from hermes_cli.debug import (
            _sweep_expired_pastes,
            _save_pending,
            _load_pending,
        )
        import time

        # Seed pending.json with one expired + one future entry
        _save_pending([
            {"url": "https://paste.rs/expired", "expire_at": time.time() - 100},
            {"url": "https://paste.rs/future", "expire_at": time.time() + 3600},
        ])

        delete_calls = []

        def fake_delete(url):
            delete_calls.append(url)
            return True

        with patch("hermes_cli.debug.delete_paste", side_effect=fake_delete):
            deleted, remaining = _sweep_expired_pastes()

        assert delete_calls == ["https://paste.rs/expired"]
        assert deleted == 1
        assert remaining == 1

        entries = _load_pending()
        urls = {e["url"] for e in entries}
        assert urls == {"https://paste.rs/future"}

    def test_sweep_leaves_future_entries_alone(self, hermes_home):
        from hermes_cli.debug import _sweep_expired_pastes, _save_pending
        import time

        _save_pending([
            {"url": "https://paste.rs/future1", "expire_at": time.time() + 3600},
            {"url": "https://paste.rs/future2", "expire_at": time.time() + 7200},
        ])

        with patch("hermes_cli.debug.delete_paste") as mock_delete:
            deleted, remaining = _sweep_expired_pastes()

        mock_delete.assert_not_called()
        assert deleted == 0
        assert remaining == 2

    def test_sweep_survives_network_failure(self, hermes_home):
        """Failed DELETEs stay in pending.json until the 24h grace window."""
        from hermes_cli.debug import (
            _sweep_expired_pastes,
            _save_pending,
            _load_pending,
        )
        import time

        _save_pending([
            {"url": "https://paste.rs/flaky", "expire_at": time.time() - 100},
        ])

        with patch(
            "hermes_cli.debug.delete_paste",
            side_effect=Exception("network down"),
        ):
            deleted, remaining = _sweep_expired_pastes()

        # Failure within 24h grace → kept for retry
        assert deleted == 0
        assert remaining == 1
        assert len(_load_pending()) == 1

    def test_sweep_drops_entries_past_grace_window(self, hermes_home):
        """After 24h past expiration, give up even on network failures."""
        from hermes_cli.debug import (
            _sweep_expired_pastes,
            _save_pending,
            _load_pending,
        )
        import time

        # Expired 25 hours ago → past the 24h grace window
        very_old = time.time() - (25 * 3600)
        _save_pending([
            {"url": "https://paste.rs/ancient", "expire_at": very_old},
        ])

        with patch(
            "hermes_cli.debug.delete_paste",
            side_effect=Exception("network down"),
        ):
            deleted, remaining = _sweep_expired_pastes()

        assert deleted == 1
        assert remaining == 0
        assert _load_pending() == []


class TestRunDebugSweepsOnInvocation:
    """``run_debug`` must sweep expired pastes on every invocation."""

    def test_run_debug_calls_sweep(self, hermes_home):
        from hermes_cli.debug import run_debug

        args = MagicMock()
        args.debug_command = None  # default → prints help

        with patch("hermes_cli.debug._sweep_expired_pastes") as mock_sweep:
            run_debug(args)

        mock_sweep.assert_called_once()

    def test_run_debug_survives_sweep_failure(self, hermes_home, capsys):
        """If the sweep throws, the subcommand still runs."""
        from hermes_cli.debug import run_debug

        args = MagicMock()
        args.debug_command = None

        with patch(
            "hermes_cli.debug._sweep_expired_pastes",
            side_effect=RuntimeError("boom"),
        ):
            run_debug(args)  # must not raise

        # Default subcommand still printed help
        out = capsys.readouterr().out
        assert "Usage: hermes debug" in out


class TestRunDebugDelete:
    def test_deletes_valid_url(self, capsys):
        from hermes_cli.debug import run_debug_delete

        args = MagicMock()
        args.urls = ["https://paste.rs/abc"]

        with patch("hermes_cli.debug.delete_paste", return_value=True):
            run_debug_delete(args)

        out = capsys.readouterr().out
        assert "Deleted" in out
        assert "paste.rs/abc" in out

    def test_handles_delete_failure(self, capsys):
        from hermes_cli.debug import run_debug_delete

        args = MagicMock()
        args.urls = ["https://paste.rs/abc"]

        with patch("hermes_cli.debug.delete_paste",
                    side_effect=Exception("network error")):
            run_debug_delete(args)

        out = capsys.readouterr().out
        assert "Could not delete" in out

    def test_no_urls_shows_usage(self, capsys):
        from hermes_cli.debug import run_debug_delete

        args = MagicMock()
        args.urls = []

        run_debug_delete(args)

        out = capsys.readouterr().out
        assert "Usage" in out


class TestShareIncludesAutoDelete:
    """Verify that run_debug_share schedules auto-deletion and prints TTL."""

    def test_share_schedules_auto_delete(self, hermes_home, capsys):
        from hermes_cli.debug import run_debug_share

        args = MagicMock()
        args.lines = 50
        args.expire = 7
        args.local = False

        with patch("hermes_cli.dump.run_dump"), \
             patch("hermes_cli.debug.upload_to_pastebin",
                    return_value="https://paste.rs/test1"), \
             patch("hermes_cli.debug._schedule_auto_delete") as mock_sched:
            run_debug_share(args)

        # auto-delete was scheduled with the uploaded URLs
        mock_sched.assert_called_once()
        urls_arg = mock_sched.call_args[0][0]
        assert "https://paste.rs/test1" in urls_arg

        out = capsys.readouterr().out
        assert "auto-delete" in out

    def test_share_shows_privacy_notice(self, hermes_home, capsys):
        from hermes_cli.debug import run_debug_share

        args = MagicMock()
        args.lines = 50
        args.expire = 7
        args.local = False

        with patch("hermes_cli.dump.run_dump"), \
             patch("hermes_cli.debug.upload_to_pastebin",
                    return_value="https://paste.rs/test"), \
             patch("hermes_cli.debug._schedule_auto_delete"):
            run_debug_share(args)

        out = capsys.readouterr().out
        assert "public paste service" in out

    def test_local_no_privacy_notice(self, hermes_home, capsys):
        from hermes_cli.debug import run_debug_share

        args = MagicMock()
        args.lines = 50
        args.expire = 7
        args.local = True

        with patch("hermes_cli.dump.run_dump"):
            run_debug_share(args)

        out = capsys.readouterr().out
        assert "public paste service" not in out

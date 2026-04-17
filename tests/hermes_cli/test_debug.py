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

class TestReadFullLog:
    """Test _read_full_log for standalone log uploads."""

    def test_reads_small_file(self, hermes_home):
        from hermes_cli.debug import _read_full_log

        content = _read_full_log("agent")
        assert content is not None
        assert "session started" in content

    def test_returns_none_for_missing(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))

        from hermes_cli.debug import _read_full_log
        assert _read_full_log("agent") is None

    def test_returns_none_for_empty(self, hermes_home):
        # Truncate agent.log to empty
        (hermes_home / "logs" / "agent.log").write_text("")

        from hermes_cli.debug import _read_full_log
        assert _read_full_log("agent") is None

    def test_truncates_large_file(self, hermes_home):
        """Files larger than max_bytes get tail-truncated."""
        from hermes_cli.debug import _read_full_log

        # Write a file larger than 1KB
        big_content = "x" * 100 + "\n"
        (hermes_home / "logs" / "agent.log").write_text(big_content * 200)

        content = _read_full_log("agent", max_bytes=1024)
        assert content is not None
        assert "truncated" in content

    def test_unknown_log_returns_none(self, hermes_home):
        from hermes_cli.debug import _read_full_log
        assert _read_full_log("nonexistent") is None

    def test_falls_back_to_rotated_file(self, hermes_home):
        """When gateway.log doesn't exist, falls back to gateway.log.1."""
        from hermes_cli.debug import _read_full_log

        logs_dir = hermes_home / "logs"
        # Remove the primary (if any) and create a .1 rotation
        (logs_dir / "gateway.log").unlink(missing_ok=True)
        (logs_dir / "gateway.log.1").write_text(
            "2026-04-12 10:00:00 INFO gateway.run: rotated content\n"
        )

        content = _read_full_log("gateway")
        assert content is not None
        assert "rotated content" in content

    def test_prefers_primary_over_rotated(self, hermes_home):
        """Primary log is used when it exists, even if .1 also exists."""
        from hermes_cli.debug import _read_full_log

        logs_dir = hermes_home / "logs"
        (logs_dir / "gateway.log").write_text("primary content\n")
        (logs_dir / "gateway.log.1").write_text("rotated content\n")

        content = _read_full_log("gateway")
        assert "primary content" in content
        assert "rotated" not in content

    def test_falls_back_when_primary_empty(self, hermes_home):
        """Empty primary log falls back to .1 rotation."""
        from hermes_cli.debug import _read_full_log

        logs_dir = hermes_home / "logs"
        (logs_dir / "agent.log").write_text("")
        (logs_dir / "agent.log.1").write_text("rotated agent data\n")

        content = _read_full_log("agent")
        assert content is not None
        assert "rotated agent data" in content


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
    def test_spawns_detached_process(self):
        from hermes_cli.debug import _schedule_auto_delete

        with patch("subprocess.Popen") as mock_popen:
            _schedule_auto_delete(
                ["https://paste.rs/abc", "https://paste.rs/def"],
                delay_seconds=10,
            )

        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        # Verify detached
        assert call_args[1]["start_new_session"] is True
        # Verify the script references both URLs
        script = call_args[0][0][2]  # [python, -c, script]
        assert "paste.rs/abc" in script
        assert "paste.rs/def" in script
        assert "time.sleep(10)" in script

    def test_skips_non_paste_rs_urls(self):
        from hermes_cli.debug import _schedule_auto_delete

        with patch("subprocess.Popen") as mock_popen:
            _schedule_auto_delete(["https://dpaste.com/something"])

        mock_popen.assert_not_called()

    def test_handles_popen_failure_gracefully(self):
        from hermes_cli.debug import _schedule_auto_delete

        with patch("subprocess.Popen",
                    side_effect=OSError("no such file")):
            # Should not raise
            _schedule_auto_delete(["https://paste.rs/abc"])


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

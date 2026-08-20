"""Tests for ``hermes debug`` CLI command and debug utilities."""

import os
import urllib.error
from unittest.mock import MagicMock, patch

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
    (logs_dir / "gui.log").write_text(
        "2026-04-12 17:00:12 INFO hermes_cli.web_server: dashboard request\n"
    )
    (logs_dir / "desktop.log").write_text(
        "2026-04-12 17:00:15 INFO desktop: backend spawned\n"
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


    def test_upload_paste_rs_network_error(self):
        from hermes_cli.debug import _upload_paste_rs

        with patch(
            "hermes_cli.debug.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with pytest.raises(urllib.error.URLError):
                _upload_paste_rs("test")




class TestUploadToPastebin:
    """Test the combined upload with fallback."""


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


class TestMissingLogNote:
    """A missing log explains itself when the writer isn't this backend.

    `hermes debug share` runs on the backend, so a desktop connected to a
    remote/docker/SSH backend can never contribute desktop.log. Reporting a
    bare absence sends triage after a client-side bug it cannot see.
    """

    def test_backend_written_log_reports_plain_absence(self, hermes_home):
        from hermes_cli.debug import _capture_log_snapshot

        (hermes_home / "logs" / "agent.log").unlink()

        snap = _capture_log_snapshot("agent", tail_lines=10)
        assert snap.full_text is None
        assert snap.tail_text == "(file not found)"

    def test_client_written_log_names_its_writer_and_path(self, hermes_home):
        from hermes_cli.debug import _capture_log_snapshot

        (hermes_home / "logs" / "desktop.log").unlink()

        snap = _capture_log_snapshot("desktop", tail_lines=10)
        assert snap.full_text is None
        assert "not on this host" in snap.tail_text
        assert "Hermes Desktop" in snap.tail_text
        # The reader needs the path to collect by hand on the client machine.
        assert str(hermes_home / "logs" / "desktop.log") in snap.tail_text

    def test_present_client_log_is_captured_normally(self, hermes_home):
        """A local backend still reads desktop.log — the note is only for a miss."""
        from hermes_cli.debug import _capture_log_snapshot

        snap = _capture_log_snapshot("desktop", tail_lines=10)
        assert "backend spawned" in snap.tail_text
        assert "not on this host" not in snap.tail_text

    def test_empty_client_log_is_empty_not_absent(self, hermes_home):
        """An empty file means the app ran and logged nothing — a different fact."""
        from hermes_cli.debug import _capture_log_snapshot

        (hermes_home / "logs" / "desktop.log").write_text("")

        snap = _capture_log_snapshot("desktop", tail_lines=10)
        assert snap.tail_text == "(file empty)"

    def test_report_carries_the_note_for_a_remote_backend(self, hermes_home):
        """The uploaded report — what people paste into support — must explain it."""
        from hermes_cli.debug import collect_debug_report

        (hermes_home / "logs" / "desktop.log").unlink()

        report = collect_debug_report(log_lines=10, dump_text="dump\n")
        assert "--- desktop.log" in report
        assert "not on this host" in report




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
        # Baseline fixture: no explicit env-var opinion. With the post-#17691
        # default of ON, the default-path tests below exercise the
        # secure-default behaviour. The `force=True` regression test
        # setenvs to "false" inline to prove force=True works even when
        # the runtime flag is disabled.
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

    def test_force_true_works_when_redaction_disabled(
        self, hermes_home_with_secret, monkeypatch
    ):
        """Regression test: redact_sensitive_text short-circuits without force=True.

        If a future refactor drops `force=True` from `_redact_log_text`, this
        test fails immediately. Without `force=True`, the redactor returns the
        input unchanged when HERMES_REDACT_SECRETS=false, and the share-time
        redaction feature ships silently broken for users who opted out of
        runtime redaction (e.g. developers working on the redactor itself).
        """

        # Force the runtime flag off so we're exercising the force=True path,
        # not the default-on path.
        monkeypatch.setenv("HERMES_REDACT_SECRETS", "false")

        from hermes_cli.debug import _capture_log_snapshot

        assert os.environ.get("HERMES_REDACT_SECRETS", "") == "false"

        snap = _capture_log_snapshot("agent", tail_lines=10)

        assert _REDACT_FIXTURE_TOKEN not in snap.tail_text
        assert snap.full_text is not None
        assert _REDACT_FIXTURE_TOKEN not in snap.full_text

    def test_default_redacts_email_addresses_for_public_share(
        self, hermes_home_with_secret
    ):
        from hermes_cli.debug import _capture_log_snapshot

        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(
            "2026-04-12 17:00:00 INFO gateway.run: "
            "inbound message: platform=bluebubbles "
            "user=person@example.com chat=iMessage;-;person@example.com msg='hello'\n"
        )

        snap = _capture_log_snapshot("agent", tail_lines=10)

        assert "person@example.com" not in snap.tail_text
        assert "[REDACTED_EMAIL]" in snap.tail_text
        assert snap.full_text is not None
        assert "person@example.com" not in snap.full_text

    @pytest.mark.parametrize("platform", ["whatsapp", "whatsapp_cloud"])
    def test_default_redacts_bare_wa_id_for_public_share(
        self, hermes_home_with_secret, platform
    ):
        from hermes_cli.debug import _capture_log_snapshot

        wa_id = "15551234567"
        preview_phone = "447700900123"
        unix_timestamp = "1786032123456"
        diagnostic_id = "9876543210"
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(
            "2026-08-06 12:00:00 INFO gateway.run: "
            f"inbound message: platform={platform} user={wa_id} chat={wa_id} "
            f"msg='call {preview_phone}'\n"
            f"2026-08-06 12:00:01 INFO worker: unix_ts={unix_timestamp} "
            f"diagnostic_id={diagnostic_id}\n"
        )

        snap = _capture_log_snapshot("agent", tail_lines=10)

        assert wa_id not in snap.tail_text
        assert preview_phone not in snap.tail_text
        assert unix_timestamp in snap.tail_text
        assert diagnostic_id in snap.tail_text
        assert snap.full_text is not None
        assert wa_id not in snap.full_text
        assert preview_phone not in snap.full_text
        assert unix_timestamp in snap.full_text
        assert diagnostic_id in snap.full_text

    def test_no_redact_preserves_bare_wa_id(self, hermes_home_with_secret):
        from hermes_cli.debug import _capture_log_snapshot

        wa_id = "15551234567"
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(
            "2026-08-06 12:00:00 INFO gateway.run: "
            f"inbound message: platform=whatsapp_cloud user={wa_id} chat={wa_id}\n"
        )

        snap = _capture_log_snapshot("agent", tail_lines=10, redact=False)

        assert wa_id in snap.tail_text
        assert wa_id in (snap.full_text or "")

    def test_redaction_expansion_marks_post_redaction_truncation(
        self, hermes_home_with_secret
    ):
        """Redaction expansion must not silently omit the earliest record."""
        from hermes_cli.debug import _capture_log_snapshot

        wa_id = "1234567"
        line = (
            "2026-08-06 12:00:00 INFO gateway.run: inbound message: "
            f"platform=whatsapp user={wa_id} chat={wa_id}\n"
        )
        raw = line * 3
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(raw)

        redacted = _capture_log_snapshot(
            "agent", tail_lines=10, max_bytes=len(raw)
        )

        assert redacted.full_text is not None
        assert "[... truncated" in redacted.full_text
        redacted_body = redacted.full_text.split("\n", 1)[1]
        assert len(redacted_body.encode("utf-8")) <= len(raw)
        assert wa_id not in redacted_body

        # The opt-out path remains an exact-content control when the raw view
        # itself fits within the byte cap.
        unredacted = _capture_log_snapshot(
            "agent", tail_lines=10, max_bytes=len(raw), redact=False
        )
        assert unredacted.full_text == raw

    @pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
    @pytest.mark.parametrize("view", ["tail", "full", "bundle"])
    def test_default_redacts_historical_whatsapp_exception_traceback(
        self, hermes_home_with_secret, line_ending, view
    ):
        """Traceback continuations cannot bypass a safe WhatsApp header."""
        from hermes_cli import debug

        media_id = "wamid.SECRET-987654321012345"
        private_name = "Private Contact"
        record = line_ending.join(
            [
                "2026-08-06 12:00:00 INFO gateway.platforms.whatsapp_cloud: "
                "[whatsapp_cloud] media metadata fetch raised "
                "(id=present(len=28), error_type=RuntimeError)",
                "Traceback (most recent call last):",
                "  File '/tmp/adapter.py', line 1, in get",
                f"RuntimeError: GET https://graph.facebook.com/v20.0/{media_id} "
                f"failed for {private_name}",
                "2026-08-06 12:00:01 INFO worker: diagnostic_id=9876543210",
            ]
        ) + line_ending
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_bytes(record.encode())

        direct = debug._redact_log_text(record)
        snapshot = debug._capture_log_snapshot(
            "agent", tail_lines=20, max_bytes=10_000
        )
        if view == "tail":
            selected = snapshot.tail_text
        elif view == "full":
            selected = snapshot.full_text or ""
        else:
            selected = "\n".join(
                debug.collect_share_bundle(log_lines=20).values()
            )

        for safe in (direct, selected):
            assert media_id not in safe
            assert private_name not in safe
            assert "graph.facebook.com" not in safe
            assert "RuntimeError" in safe
            assert "[REDACTED_EXCEPTION_TRACEBACK]" in safe
            # Historical exception records have no unforgeable traceback
            # terminator, so later diagnostics are conservatively redacted.
            assert "diagnostic_id=9876543210" not in safe

    @pytest.mark.parametrize(
        "prefix",
        [
            "Processing queued message after agent completion",
            "Processing pending message",
            "Delivering leftover /steer as next turn",
        ],
    )
    @pytest.mark.parametrize(
        ("opening", "closing", "control_visible"),
        [("'", "'", True), ("", "", False), ("'", "", False)],
        ids=["quoted", "unquoted", "unterminated"],
    )
    @pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
    def test_default_redacts_multiline_legacy_previews(
        self,
        hermes_home_with_secret,
        prefix,
        opening,
        closing,
        control_visible,
        line_ending,
    ):
        """All historical queued/pending/leftover previews fail closed."""
        from hermes_cli import debug

        wa_id = "15551234567"
        private_marker = "private-medical-detail"
        record = (
            f"2026-08-06 12:00:00 DEBUG gateway.run: {prefix}: "
            f"{opening}first {private_marker}{line_ending}"
            f"second {private_marker} call {wa_id}"
            f"{'...' if control_visible else ''}{closing}{line_ending}"
            "2026-08-06 12:00:01 INFO worker: "
            f"diagnostic_id=9876543210{line_ending}"
        )
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_bytes(record.encode())

        direct = debug._redact_log_text(record)
        snapshot = debug._capture_log_snapshot(
            "agent", tail_lines=20, max_bytes=10_000
        )

        for safe in (direct, snapshot.tail_text, snapshot.full_text or ""):
            assert wa_id not in safe
            assert private_marker not in safe
            assert "[REDACTED_MESSAGE_PREVIEW]" in safe
            # The historical formatter's quote/ellipsis suffix is forgeable
            # by message text; remain redacted through EOF in every variant.
            assert "diagnostic_id=9876543210" not in safe

    def test_split_watch_pattern_whatsapp_record_fails_closed(
        self, hermes_home_with_secret
    ):
        """The generic ``for whatsapp_cloud`` selector survives cap splits."""
        from hermes_cli import debug

        wa_id = "15551234567"
        raw = (
            "2026-08-06 12:00:00 INFO gateway.run: "
            "Watch pattern notification "
            + ("P" * 9_000)
            + f" for whatsapp_cloud chat={wa_id} thread={wa_id}"
        )
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(raw)

        snapshot = debug._capture_log_snapshot(
            "agent", tail_lines=1, max_bytes=240
        )

        for safe in (snapshot.tail_text, snapshot.full_text or ""):
            assert wa_id not in safe
            assert "[REDACTED_LOG_FRAGMENT]" in safe

    @pytest.mark.parametrize(
        "prefix",
        [
            "Processing queued message after agent completion",
            "Processing pending message",
            "Delivering leftover /steer as next turn",
        ],
    )
    @pytest.mark.parametrize("variant", ["quoted", "unquoted", "unterminated"])
    @pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
    @pytest.mark.parametrize("view", ["tail", "full", "bundle"])
    def test_legacy_preview_apostrophe_cannot_close_redaction_state(
        self,
        hermes_home_with_secret,
        prefix,
        variant,
        line_ending,
        view,
    ):
        """Legacy preview apostrophes do not expose later continuation text."""
        from hermes_cli import debug

        wa_id = "15551234567"
        private_marker = "private-legacy-apostrophe-marker"
        diagnostic_id = "9876543210"
        opening = "'" if variant != "unquoted" else ""
        continuation = (
            f"first {private_marker}{line_ending}"
            f"attacker's private line call {wa_id}{line_ending}"
        )
        if variant == "quoted":
            continuation += f"last {private_marker} call {wa_id}...'"
        elif variant == "unterminated":
            continuation += f"last {private_marker} call {wa_id}"
        else:
            continuation += f"last {private_marker} call {wa_id}"
        post_closure = (
            f"post-closure diagnostic {diagnostic_id}{line_ending}"
            if variant == "quoted"
            else ""
        )
        record = (
            f"2026-08-06 12:00:00 DEBUG gateway.run: {prefix}: "
            f"{opening}{continuation}{line_ending}"
            f"{post_closure}"
            "2026-08-06 12:00:01 INFO worker: "
            f"diagnostic_id={diagnostic_id}{line_ending}"
        )
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_bytes(record.encode())

        snapshot = debug._capture_log_snapshot(
            "agent", tail_lines=20, max_bytes=10_000
        )
        if view == "tail":
            selected = snapshot.tail_text
        elif view == "full":
            selected = snapshot.full_text or ""
        else:
            selected = "\n".join(debug.collect_share_bundle(log_lines=20).values())

        assert private_marker not in selected
        assert wa_id not in selected
        assert "[REDACTED_MESSAGE_PREVIEW]" in selected
        assert f"diagnostic_id={diagnostic_id}" not in selected
        assert f"post-closure diagnostic {diagnostic_id}" not in selected

    @pytest.mark.parametrize(
        "prefix",
        [
            "Processing queued message after agent completion",
            "Processing pending message",
            "Delivering leftover /steer as next turn",
        ],
    )
    @pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
    def test_legacy_literal_ellipsis_cannot_close_redaction_state(
        self, hermes_home_with_secret, prefix, line_ending
    ):
        """An in-band ``...'`` cannot end a legacy preview at any view."""
        from hermes_cli import debug

        wa_id = "15551234567"
        private_marker = "private-legacy-literal-ellipsis-marker"
        diagnostic_id = "legacy-literal-ellipsis-diagnostic"
        record = line_ending.join(
            [
                "2026-08-06 12:00:00 DEBUG gateway.run: "
                f"{prefix}: 'first {private_marker}",
                "attacker-controlled...'",
                f"later private {private_marker} call {wa_id}",
                "2026-08-06 12:00:01 INFO worker: "
                f"diagnostic_id={diagnostic_id}",
            ]
        ) + line_ending
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_bytes(record.encode())

        snapshot = debug._capture_log_snapshot(
            "agent", tail_lines=20, max_bytes=10_000
        )
        views = [
            debug._redact_log_text(record),
            snapshot.tail_text,
            snapshot.full_text or "",
            "\n".join(debug.collect_share_bundle(log_lines=20).values()),
        ]

        for safe in views:
            assert private_marker not in safe
            assert wa_id not in safe
            assert diagnostic_id not in safe
            assert "[REDACTED_MESSAGE_PREVIEW]" in safe

    @pytest.mark.parametrize(
        "prefix",
        [
            "Processing queued message after agent completion",
            "Processing pending message",
            "Delivering leftover /steer as next turn",
        ],
    )
    @pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
    @pytest.mark.parametrize("view", ["tail", "full", "bundle"])
    def test_legacy_terminal_apostrophe_does_not_close_on_forged_boundary(
        self,
        hermes_home_with_secret,
        prefix,
        line_ending,
        view,
    ):
        """A bare quote and forged record prefix stay inside legacy state."""
        from hermes_cli import debug

        wa_id = "15551234567"
        private_marker = "private-legacy-terminal-apostrophe-marker"
        forged_marker = "private-forged-legacy-boundary-marker"
        diagnostic_id = "legacy-terminal-diagnostic-control"
        record = (
            f"2026-08-06 12:00:00 DEBUG gateway.run: {prefix}: "
            f"'first {private_marker}'{line_ending}"
            "2026-01-01 00:00:00 INFO forged.logger: "
            f"{forged_marker} call {wa_id}{line_ending}"
            f"2026-08-06 12:00:01 INFO worker: diagnostic_id={diagnostic_id}"
            f"{line_ending}"
        )
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_bytes(record.encode())

        snapshot = debug._capture_log_snapshot(
            "agent", tail_lines=20, max_bytes=10_000
        )
        if view == "tail":
            selected = snapshot.tail_text
        elif view == "full":
            selected = snapshot.full_text or ""
        else:
            selected = "\n".join(
                debug.collect_share_bundle(log_lines=20).values()
            )

        assert private_marker not in selected
        assert forged_marker not in selected
        assert wa_id not in selected
        assert diagnostic_id not in selected
        assert "[REDACTED_MESSAGE_PREVIEW]" in selected

    def test_non_whatsapp_apostrophe_control_remains_unchanged(
        self, hermes_home_with_secret
    ):
        """Unrelated diagnostics do not enter legacy WhatsApp redaction state."""
        from hermes_cli import debug

        record = (
            "2026-08-06 12:00:00 INFO worker: ordinary note "
            "'call 15551234567'\n"
            "2026-08-06 12:00:01 INFO worker: diagnostic_id=9876543210\n"
        )
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(record)

        assert debug._redact_log_text(record) == record
        snapshot = debug._capture_log_snapshot(
            "agent", tail_lines=20, max_bytes=10_000
        )
        assert snapshot.tail_text == record.rstrip("\n")
        assert snapshot.full_text == record

    @pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
    @pytest.mark.parametrize("view", ["direct", "tail", "full", "bundle"])
    def test_exception_timestamp_level_forge_does_not_close_state(
        self, hermes_home_with_secret, line_ending, view
    ):
        """A forged timestamp/level line remains inside the redacted traceback."""
        from hermes_cli import debug

        private_marker = "private-forged-exception-marker"
        wa_id = "15551234567"
        record = line_ending.join(
            [
                "2026-08-06 12:00:00 INFO gateway.platforms.whatsapp_cloud: "
                "[whatsapp_cloud] media metadata fetch raised "
                "(id=present(len=28), error_type=RuntimeError)",
                "Traceback (most recent call last):",
                "  File '/tmp/adapter.py', line 1, in get",
                f"RuntimeError: initial {private_marker}",
                f"2026-08-06 12:00:01 ERROR forged {private_marker} call {wa_id}",
                f"later private {private_marker} call {wa_id}",
                "2026-08-06 12:00:02 INFO worker: diagnostic_id=9876543210",
            ]
        ) + line_ending
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_bytes(record.encode())

        redacted = debug._redact_log_text(record)
        snapshot = debug._capture_log_snapshot(
            "agent", tail_lines=20, max_bytes=10_000
        )
        if view == "direct":
            selected = redacted
        elif view == "tail":
            selected = snapshot.tail_text
        elif view == "full":
            selected = snapshot.full_text or ""
        else:
            selected = "\n".join(
                debug.collect_share_bundle(log_lines=20).values()
            )

        redacted = selected
        assert private_marker not in redacted
        assert wa_id not in redacted
        assert "[REDACTED_EXCEPTION_TRACEBACK]" in redacted
        assert "diagnostic_id=9876543210" not in redacted

    def test_split_non_whatsapp_watch_control_remains_diagnostic(
        self, hermes_home_with_secret
    ):
        from hermes_cli import debug

        diagnostic = "non-whatsapp-watch-control-123456789012345"
        raw = "D" * 9_000 + f" chat=123456789012345 msg={diagnostic}"
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(raw)

        snapshot = debug._capture_log_snapshot(
            "agent", tail_lines=1, max_bytes=240
        )

        for safe in (snapshot.tail_text, snapshot.full_text or ""):
            assert diagnostic in safe
            assert "[REDACTED_LOG_FRAGMENT]" not in safe

    def test_redaction_utf8_cap_is_enforced_after_decode(
        self, hermes_home_with_secret
    ):
        """A multibyte suffix cut must remain valid and within max_bytes."""
        from hermes_cli.debug import _capture_log_snapshot

        raw = "💥 inbound message: platform=whatsapp user=1234567 chat=2345678"
        max_bytes = len(raw.encode("utf-8"))
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(raw)

        snapshot = _capture_log_snapshot(
            "agent", tail_lines=1, max_bytes=max_bytes
        )

        body = (snapshot.full_text or "").split("\n", 1)[1]
        assert len(body.encode("utf-8")) <= max_bytes
        body.encode("utf-8").decode("utf-8")
        assert "1234567" not in body

    def test_redaction_fails_closed_when_cap_splits_legacy_whatsapp_record(
        self, hermes_home_with_secret
    ):
        """A mid-line cap must not split the inbound selector from its ID."""
        from hermes_cli.debug import _capture_log_snapshot

        wa_id = "15551234567"
        private_marker = "private-medical-detail"
        prefix = "2026-08-06 INFO gateway.run: inbound message: "
        suffix = (
            "platform=whatsapp_cloud user=Alice "
            + ("P" * 8100)
            + f" chat={wa_id} msg={private_marker} call {wa_id}"
        )
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(prefix + suffix)

        redacted = _capture_log_snapshot(
            "agent", tail_lines=1, max_bytes=240, redact=True
        )
        for selected in (redacted.tail_text, redacted.full_text or ""):
            assert wa_id not in selected
            assert private_marker not in selected
            assert "[REDACTED_LOG_FRAGMENT]" in selected

        # The explicit operator opt-out retains the exact raw selected view.
        unredacted = _capture_log_snapshot(
            "agent", tail_lines=1, max_bytes=240, redact=False
        )
        assert wa_id in unredacted.tail_text
        assert private_marker in unredacted.tail_text
        assert wa_id in (unredacted.full_text or "")
        assert private_marker in (unredacted.full_text or "")

    def test_split_non_whatsapp_record_retains_safe_diagnostics(
        self, hermes_home_with_secret
    ):
        """A large split non-WhatsApp record is not replaced wholesale."""
        from hermes_cli.debug import _capture_log_snapshot

        diagnostic = "non-whatsapp-diagnostic-control-123456789012345"
        raw = "D" * 10_000 + f" chat=123456789012345 msg={diagnostic}"
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(raw)

        redacted = _capture_log_snapshot(
            "agent", tail_lines=1, max_bytes=240, redact=True
        )
        unredacted = _capture_log_snapshot(
            "agent", tail_lines=1, max_bytes=240, redact=False
        )

        for selected in (redacted.tail_text, redacted.full_text or ""):
            assert diagnostic in selected
            assert "[REDACTED_LOG_FRAGMENT]" not in selected
        assert diagnostic in unredacted.tail_text

    @pytest.mark.parametrize(
        "record",
        [
            (
                "2026-08-06 12:00:00 INFO agent.turn_context: "
                "conversation turn: session=s1 model=m provider=p "
                "platform=whatsapp_cloud history=0 "
                'msg="call 15551234567 about health"\n'
            ),
            (
                "2026-08-06 12:00:00 DEBUG gateway.run: "
                "Processing queued message after agent completion: "
                "'call 15551234567 about health...'\n"
            ),
            (
                "2026-08-06 12:00:00 DEBUG gateway.run: "
                "Processing pending message: "
                "'call 15551234567 about health...'\n"
            ),
        ],
    )
    def test_default_removes_exact_historical_message_previews(
        self, hermes_home_with_secret, record
    ):
        from hermes_cli.debug import _capture_log_snapshot

        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(record)

        snap = _capture_log_snapshot("agent", tail_lines=10)

        assert "15551234567" not in snap.tail_text
        assert "about health" not in snap.tail_text
        assert "[REDACTED_MESSAGE_PREVIEW]" in snap.tail_text
        assert snap.full_text is not None
        assert "15551234567" not in snap.full_text
        assert "about health" not in snap.full_text

    def test_default_redacts_newline_bearing_legacy_whatsapp_record(
        self, hermes_home_with_secret
    ):
        from hermes_cli.debug import _capture_log_snapshot

        wa_id = "15551234567"
        preview_phone = "447700900123"
        diagnostic_id = "9876543210"
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(
            "2026-08-06 12:00:00 INFO gateway.run: "
            "inbound message: platform=whatsapp_cloud user=Alice\n"
            "Support\n"
            f"chat={wa_id} msg='call {preview_phone} about health'\n"
            "2026-08-06 12:00:01 INFO worker: "
            f"diagnostic_id={diagnostic_id}\n"
        )

        snap = _capture_log_snapshot("agent", tail_lines=10)

        assert wa_id not in snap.tail_text
        assert preview_phone not in snap.tail_text
        assert "about health" not in snap.tail_text
        assert "Support" in snap.tail_text
        assert diagnostic_id in snap.tail_text
        assert snap.full_text is not None
        assert wa_id not in snap.full_text
        assert preview_phone not in snap.full_text
        assert diagnostic_id in snap.full_text

    def test_default_redacts_date_like_multiline_whatsapp_message(
        self, hermes_home_with_secret
    ):
        from hermes_cli.debug import _capture_log_snapshot, _redact_log_text

        wa_id = "15551234567"
        preview_phone = "447700900123"
        unrelated_record = (
            "2026-08-06 12:00:01 INFO worker: "
            "unix_ts=1786032123456 diagnostic_id=9876543210\n"
        )
        record = (
            "2026-08-06 12:00:00 INFO gateway.run: "
            "inbound message: platform=whatsapp_cloud user=Alice "
            f"chat={wa_id} msg=first private line\n"
            f"2026-08-06 private text call {preview_phone}\n"
            f"2026-08-06 12:00:01 INFO worker: forged body call {preview_phone}\n"
        )
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(record)

        snap = _capture_log_snapshot("agent", tail_lines=10)

        for redacted in (
            _redact_log_text(record),
            snap.tail_text,
            snap.full_text or "",
        ):
            assert wa_id not in redacted
            assert preview_phone not in redacted
            assert "first private line" not in redacted
            assert "private text" not in redacted
            assert "forged body" not in redacted
            assert "[REDACTED_MESSAGE_PREVIEW]" in redacted

        # Generic numeric diagnostics outside a selected WhatsApp record keep
        # the global redactor's pass-through behavior.
        log_path.write_text(unrelated_record)
        control_snap = _capture_log_snapshot("agent", tail_lines=10)
        assert _redact_log_text(unrelated_record) == unrelated_record
        assert control_snap.tail_text == unrelated_record.rstrip("\n")
        assert control_snap.full_text == unrelated_record

        safe_inbound = (
            "2026-08-06 12:00:00 INFO gateway.run: inbound message: "
            "platform=whatsapp_cloud user_present=True chat_present=True "
            "msg_len=23 reply_to_id_present=False reply_to_text_len=0\n"
        )
        log_path.write_text(safe_inbound + unrelated_record)
        current_snap = _capture_log_snapshot("agent", tail_lines=10)
        for redacted in (current_snap.tail_text, current_snap.full_text or ""):
            assert unrelated_record.rstrip("\n") in redacted

    @pytest.mark.parametrize("platform", ["whatsapp", "whatsapp_cloud"])
    @pytest.mark.parametrize("quoted", [False, True])
    @pytest.mark.parametrize("view", ["tail", "full"])
    def test_truncated_multiline_whatsapp_snapshot_preserves_redaction_state(
        self,
        hermes_home_with_secret,
        platform,
        quoted,
        view,
    ):
        """A view beginning inside a legacy preview must remain upload-safe."""
        from hermes_cli.debug import _capture_log_snapshot

        wa_id = "15551234567"
        private_marker = "private-medical-detail"
        generic_diagnostic = "diagnostic_id=9876543210"
        opening = "'" if quoted else ""
        closing = "'" if quoted else ""
        continuation = "".join(
            f"continuation-{index} {private_marker} call {wa_id}\n"
            for index in range(40)
        )
        record = (
            "2026-08-06 12:00:00 INFO agent.turn_context: "
            "conversation turn: session=s1 model=m provider=p "
            f"platform={platform} history=0 msg={opening}first {private_marker}\n"
            f"{continuation}last {private_marker} {wa_id}{closing}\n"
        )
        if quoted:
            # The apparent closing quote is message-controlled text.  It must
            # not reopen the retained suffix as ordinary diagnostics.
            record += (
                "2026-08-06 12:00:01 INFO worker: "
                f"{generic_diagnostic}\n"
            )

        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(record)
        snap = _capture_log_snapshot(
            "agent",
            tail_lines=3,
            max_bytes=160 if view == "full" else 10_000,
        )
        selected = snap.tail_text if view == "tail" else (snap.full_text or "")

        assert "conversation turn:" not in selected
        assert wa_id not in selected
        assert private_marker not in selected
        assert "[REDACTED_MESSAGE_PREVIEW]" in selected
        assert generic_diagnostic not in selected

    @pytest.mark.parametrize("quoted", [False, True])
    def test_rotated_path_cannot_change_truncated_whatsapp_state(
        self,
        hermes_home_with_secret,
        monkeypatch,
        quoted,
    ):
        """Prefix state and retained bytes must use one open log snapshot."""
        from hermes_cli import debug

        wa_id = "15551234567"
        private_marker = "private-rotation-race-marker"
        generic_diagnostic = "diagnostic_id=9876543210"
        opening = "'" if quoted else ""
        closing = "'" if quoted else ""
        record = (
            "2026-08-06 12:00:00 INFO agent.turn_context: "
            "conversation turn: session=s1 model=m provider=p "
            f"platform=whatsapp_cloud history=0 msg={opening}first private line\n"
            + "".join(
                f"continuation-{index:04d} {private_marker} call {wa_id}\n"
                for index in range(1_000)
            )
            + f"last {private_marker} call {wa_id}{closing}\n"
        )
        if quoted:
            record += (
                "2026-08-06 12:00:01 INFO worker: "
                f"{generic_diagnostic}\n"
            )

        log_path = hermes_home_with_secret / "logs" / "agent.log"
        replacement_path = hermes_home_with_secret / "logs" / "replacement.log"
        log_path.write_text(record)
        original_state_at = debug._whatsapp_log_state_at

        def rotate_before_state_reconstruction(
            log_source,
            byte_offset,
            *,
            content_hash=None,
        ):
            replacement_path.write_text(
                "2026-08-06 12:00:02 INFO worker: replacement log\n"
            )
            os.replace(replacement_path, log_path)
            return original_state_at(
                log_source,
                byte_offset,
                content_hash=content_hash,
            )

        monkeypatch.setattr(
            debug,
            "_whatsapp_log_state_at",
            rotate_before_state_reconstruction,
        )

        snap = debug._capture_log_snapshot(
            "agent",
            tail_lines=20,
            max_bytes=240,
        )

        for upload_view in (snap.tail_text, snap.full_text or ""):
            assert private_marker not in upload_view
            assert wa_id not in upload_view
            assert "[REDACTED_MESSAGE_PREVIEW]" in upload_view
        assert generic_diagnostic not in (snap.full_text or "")

    @pytest.mark.parametrize("redact", [True, False])
    def test_append_during_snapshot_returns_initial_snapshot(
        self,
        hermes_home_with_secret,
        monkeypatch,
        redact,
    ):
        """Ordinary growth keeps the coherent bytes selected at open time."""
        from hermes_cli import debug

        initial_line = "2026-08-06 12:00:00 INFO worker: capture-start\n"
        appended_line = "2026-08-06 12:00:01 INFO worker: appended-later\n"
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(initial_line)
        original_fstat = debug.os.fstat
        fstat_calls = 0

        def append_before_final_fstat(fd):
            nonlocal fstat_calls
            fstat_calls += 1
            if fstat_calls == 2:
                with log_path.open("a") as mutable_log:
                    mutable_log.write(appended_line)
            return original_fstat(fd)

        monkeypatch.setattr(debug.os, "fstat", append_before_final_fstat)

        snap = debug._capture_log_snapshot(
            "agent",
            tail_lines=20,
            max_bytes=10_000,
            redact=redact,
        )

        assert snap.full_text is not None
        assert "capture-start" in snap.tail_text
        assert "capture-start" in snap.full_text
        assert "appended-later" not in snap.tail_text
        assert "appended-later" not in snap.full_text

    def test_in_place_overwrite_during_snapshot_fails_closed(
        self,
        hermes_home_with_secret,
        monkeypatch,
    ):
        """A mutable inode cannot yield a mixed upload snapshot."""
        from hermes_cli import debug

        private_marker = "private-changing-snapshot-marker"
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(
            "2026-08-06 12:00:00 INFO agent.turn_context: "
            "conversation turn: session=s1 model=m provider=p "
            "platform=whatsapp_cloud history=0 msg=first private line\n"
            + "".join(
                f"continuation-{index:04d} {private_marker}\n"
                for index in range(1_000)
            )
        )
        original_state_at = debug._whatsapp_log_state_at

        def overwrite_before_state_reconstruction(
            log_source,
            byte_offset,
            *,
            content_hash=None,
        ):
            original = log_path.read_bytes()
            marker_offset = original.rfind(private_marker.encode())
            assert marker_offset >= byte_offset
            with log_path.open("r+b") as mutable_log:
                mutable_log.seek(marker_offset)
                mutable_log.write(b"X" * len(private_marker))
            return original_state_at(
                log_source,
                byte_offset,
                content_hash=content_hash,
            )

        monkeypatch.setattr(
            debug,
            "_whatsapp_log_state_at",
            overwrite_before_state_reconstruction,
        )

        snap = debug._capture_log_snapshot(
            "agent",
            tail_lines=20,
            max_bytes=240,
        )

        assert private_marker not in snap.tail_text
        assert "log changed during snapshot capture" in snap.tail_text
        assert snap.full_text is None

    def test_truncate_during_snapshot_fails_closed(
        self,
        hermes_home_with_secret,
        monkeypatch,
    ):
        """Shrinking the selected initial range cannot yield a partial upload."""
        from hermes_cli import debug

        private_marker = "private-truncated-snapshot-marker"
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(
            "2026-08-06 12:00:00 INFO agent.turn_context: "
            "conversation turn: session=s1 model=m provider=p "
            "platform=whatsapp_cloud history=0 msg=first private line\n"
            + "".join(
                f"continuation-{index:04d} {private_marker}\n"
                for index in range(1_000)
            )
        )
        original_state_at = debug._whatsapp_log_state_at

        def truncate_before_state_reconstruction(
            log_source,
            byte_offset,
            *,
            content_hash=None,
        ):
            with log_path.open("r+b") as mutable_log:
                mutable_log.truncate(byte_offset // 2)
            return original_state_at(
                log_source,
                byte_offset,
                content_hash=content_hash,
            )

        monkeypatch.setattr(
            debug,
            "_whatsapp_log_state_at",
            truncate_before_state_reconstruction,
        )

        snap = debug._capture_log_snapshot(
            "agent",
            tail_lines=20,
            max_bytes=240,
        )

        assert private_marker not in snap.tail_text
        assert "log changed during snapshot capture" in snap.tail_text
        assert snap.full_text is None

    @pytest.mark.parametrize("platform", ["whatsapp", "whatsapp_cloud"])
    def test_default_redacts_multiline_message_after_legacy_whatsapp_chat(
        self, hermes_home_with_secret, platform
    ):
        from hermes_cli.debug import _capture_log_snapshot

        wa_id = "15551234567"
        preview_phone = "447700900123"
        unrelated_record = (
            "2026-08-06 12:00:01 INFO worker: "
            "unix_ts=1786032123456 diagnostic_id=9876543210\n"
        )
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(
            "2026-08-06 12:00:00 INFO gateway.run: "
            f"inbound message: platform={platform} user=Alice "
            f"chat={wa_id} msg='first private line\n"
            f"call {preview_phone} about health\n"
            "last private line'\n"
            f"{unrelated_record}"
        )

        snap = _capture_log_snapshot("agent", tail_lines=10)

        for redacted in (snap.tail_text, snap.full_text or ""):
            assert wa_id not in redacted
            assert preview_phone not in redacted
            assert "first private line" not in redacted
            assert "about health" not in redacted
            assert "last private line" not in redacted
            assert "[REDACTED_MESSAGE_PREVIEW]" in redacted
            assert unrelated_record.rstrip("\n") not in redacted

    @pytest.mark.parametrize("platform", ["whatsapp", "whatsapp_cloud"])
    def test_default_redacts_multiline_legacy_whatsapp_conversation(
        self, hermes_home_with_secret, platform
    ):
        from hermes_cli.debug import _capture_log_snapshot, _redact_log_text

        preview_phone = "15551234567"
        diagnostic_id = "9876543210"
        safe_inbound = (
            "2026-08-06 12:00:01 INFO gateway.run: inbound message: "
            f"platform={platform} user_present=True chat_present=True "
            "msg_len=23 reply_to_id_present=False reply_to_text_len=0\n"
        )
        unrelated_record = (
            "2026-08-06 12:00:02 INFO worker: "
            f"diagnostic_id={diagnostic_id}\n"
        )
        record = (
            "2026-08-06 12:00:00 INFO agent.turn_context: "
            "conversation turn: session=s1 model=m provider=p "
            f"platform={platform} history=0 msg='first private line\n"
            f"call {preview_phone} about health\n"
            "last private line'\n"
            f"{safe_inbound}{unrelated_record}"
        )
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(record)

        snap = _capture_log_snapshot("agent", tail_lines=10)

        for redacted in (
            _redact_log_text(record),
            snap.tail_text,
            snap.full_text or "",
        ):
            assert preview_phone not in redacted
            assert "first private line" not in redacted
            assert "about health" not in redacted
            assert "last private line" not in redacted
            assert "[REDACTED_MESSAGE_PREVIEW]" in redacted
            assert "[WHATSAPP_INBOUND_METADATA] msg_len=23" in redacted
            assert diagnostic_id not in redacted

    @pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
    @pytest.mark.parametrize("view", ["direct", "tail", "full", "bundle"])
    def test_nonlegacy_whatsapp_apostrophe_cannot_close_state(
        self, hermes_home_with_secret, line_ending, view
    ):
        """Message apostrophes cannot expose later multiline bytes."""
        from hermes_cli import debug

        wa_id = "15551234567"
        private_marker = "private-apostrophe-conversation-marker"
        diagnostic_id = "nonwa-diagnostic-9876543210"
        record = (
            "2026-08-06 12:00:00 INFO agent.turn_context: "
            "conversation turn: session=s1 model=m provider=p "
            "platform=whatsapp_cloud history=0 "
            f"msg='first {private_marker}{line_ending}"
            f"attacker's contraction call {wa_id}{line_ending}"
            f"later private {private_marker} call {wa_id}{line_ending}"
            "2026-08-06 12:00:01 INFO gateway.run: inbound message: "
            "platform=whatsapp_cloud user_present=True chat_present=True "
            f"msg_len=23 reply_to_id_present=False reply_to_text_len=0{line_ending}"
            f"2026-08-06 12:00:02 INFO worker: {diagnostic_id}{line_ending}"
        )
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_bytes(record.encode())

        snapshot = debug._capture_log_snapshot(
            "agent", tail_lines=20, max_bytes=10_000
        )
        if view == "direct":
            selected = debug._redact_log_text(record)
        elif view == "tail":
            selected = snapshot.tail_text
        elif view == "full":
            selected = snapshot.full_text or ""
        else:
            selected = "\n".join(
                debug.collect_share_bundle(log_lines=20).values()
            )

        assert wa_id not in selected
        assert private_marker not in selected
        assert "[REDACTED_MESSAGE_PREVIEW]" in selected
        assert "[WHATSAPP_INBOUND_METADATA] msg_len=23" in selected
        assert diagnostic_id not in selected

        # The explicit opt-out remains a raw diagnostic control.
        raw = debug._capture_log_snapshot(
            "agent", tail_lines=20, max_bytes=10_000, redact=False
        )
        assert wa_id in raw.tail_text
        assert private_marker in raw.tail_text

    def test_bounded_prefix_state_recovery_returns_safe_fragment(
        self, hermes_home_with_secret
    ):
        """A selected record older than the scan window cannot leak suffixes."""
        from hermes_cli import debug

        private_marker = "private-prefix-outside-window-marker"
        line = f"continuation {private_marker} call 15551234567\n"
        record = (
            "2026-08-06 12:00:00 INFO agent.turn_context: "
            "conversation turn: session=s1 model=m provider=p "
            "platform=whatsapp_cloud history=0 msg='first private line\n"
            + line * ((debug._WHATSAPP_STATE_SCAN_BYTES // len(line)) + 5_000)
        )
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(record)

        snapshot = debug._capture_log_snapshot(
            "agent", tail_lines=10, max_bytes=240
        )

        assert private_marker not in snapshot.tail_text
        assert private_marker not in (snapshot.full_text or "")
        assert "[REDACTED_LOG_FRAGMENT]" in snapshot.tail_text
        assert "[REDACTED_LOG_FRAGMENT]" in (snapshot.full_text or "")

    def test_prefix_state_recovery_replays_only_bounded_window(
        self, hermes_home_with_secret, monkeypatch
    ):
        """Large ordinary prefixes do not trigger an unbounded line replay."""
        from hermes_cli import debug

        line = "2026-08-06 12:00:00 INFO worker: ordinary diagnostic line\n"
        raw = line * ((16 * 1024 * 1024) // len(line) + 1)
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(raw)

        calls = 0
        original = debug._redact_log_text_with_state

        def count_state_replays(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(
            debug, "_redact_log_text_with_state", count_state_replays
        )
        with log_path.open("rb") as log_file:
            state = debug._whatsapp_log_state_at(
                log_file, len(raw) - 1024
            )

        assert state.prefix_unresolved is False
        assert calls == 0

    def test_selector_free_large_snapshot_preserves_diagnostics(
        self, hermes_home_with_secret
    ):
        """The bounded privacy fallback does not erase ordinary log tails."""
        from hermes_cli import debug

        diagnostic = "ordinary-large-log-diagnostic-marker"
        line = f"2026-08-06 12:00:00 INFO worker: {diagnostic}\n"
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(line * ((4 * 1024 * 1024) // len(line) + 1))

        snapshot = debug._capture_log_snapshot(
            "agent", tail_lines=3, max_bytes=240
        )

        assert diagnostic in snapshot.tail_text
        assert diagnostic in (snapshot.full_text or "")
        assert "[REDACTED_LOG_FRAGMENT]" not in snapshot.tail_text
        assert "[REDACTED_LOG_FRAGMENT]" not in (snapshot.full_text or "")

    def test_large_log_with_safe_inbound_metadata_preserves_diagnostics(
        self, hermes_home_with_secret
    ):
        """Current body-free inbound metadata is not treated as an opener."""
        from hermes_cli import debug

        diagnostic = "keep-current-whatsapp-tail-diagnostic"
        metadata = (
            "2026-08-06 12:00:00 INFO gateway.run: inbound message: "
            "platform=whatsapp_cloud user_present=True chat_present=True "
            "msg_len=5 reply_to_id_present=False reply_to_text_len=0\n"
        )
        line = f"2026-08-06 12:00:01 INFO worker: {diagnostic}\n"
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(
            metadata + line * ((4 * 1024 * 1024) // len(line) + 1)
        )

        snapshot = debug._capture_log_snapshot(
            "agent", tail_lines=3, max_bytes=240
        )

        assert diagnostic in snapshot.tail_text
        assert diagnostic in (snapshot.full_text or "")
        assert "[REDACTED_LOG_FRAGMENT]" not in snapshot.tail_text
        assert "[REDACTED_LOG_FRAGMENT]" not in (snapshot.full_text or "")

    @pytest.mark.parametrize(
        "safe_error",
        [
            "[whatsapp_cloud] media metadata fetch raised "
            "(id=present(len=28), error_type=RuntimeError)",
            "[WhatsApp] WhatsApp read receipt failed "
            "(error_type=TimeoutError)",
            "[whatsapp_cloud] Error handling message (error_type=RuntimeError)",
            "[whatsapp] Poll error (error_type=RuntimeError)",
            "[whatsapp] Failed to cache image (error_type=RuntimeError)",
            "[whatsapp] Failed to cache audio (error_type=RuntimeError)",
            "[whatsapp] Error building event (error_type=RuntimeError)",
            "[whatsapp] Failed to send image (error_detail=present)",
            "[whatsapp] Error sending local file present "
            "(error_type=RuntimeError)",
            "[whatsapp] send_typing error "
            "(non-fatal; error_type=RuntimeError)",
            "[whatsapp] Failed to resolve live adapter for final delivery",
            "[whatsapp] send_private_notice failed, falling back to public "
            "(error_detail=present)",
            "[whatsapp] Post-stream image batch delivery failed: present",
            "[whatsapp] Post-stream media delivery failed: present",
        ],
    )
    @pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
    def test_current_type_only_whatsapp_error_does_not_open_traceback(
        self, hermes_home_with_secret, safe_error, line_ending
    ):
        from hermes_cli import debug

        keep = "KEEP_AFTER_SAFE_ERROR"
        text = (
            f"2026-08-06 12:00:00 WARNING gateway: {safe_error}{line_ending}"
            f"2026-08-06 12:00:01 INFO worker: {keep}{line_ending}"
        )

        redacted = debug._redact_log_text(text)

        assert keep in redacted
        assert "[REDACTED_EXCEPTION_TRACEBACK]" not in redacted
        assert "present(len=28))" not in redacted

    def test_old_selected_state_cannot_be_cleared_by_recent_selector(
        self, hermes_home_with_secret
    ):
        """A later metadata-shaped line cannot close an older message state."""
        from hermes_cli import debug

        private_marker = "private-old-continuation-marker"
        continuation = f"old continuation {private_marker} call 15551234567\n"
        record = (
            "2026-08-06 12:00:00 INFO gateway.run: "
            "Processing pending message: 'first private line\n"
            + continuation
            * ((debug._WHATSAPP_STATE_SCAN_BYTES // len(continuation)) + 5_000)
            + (
                "2026-08-06 12:00:01 INFO gateway.run: inbound message: "
                "platform=whatsapp_cloud user_present=True chat_present=True "
                "msg_len=23 reply_to_id_present=False reply_to_text_len=0\n"
            )
        )
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(record)

        snapshot = debug._capture_log_snapshot(
            "agent", tail_lines=10, max_bytes=240
        )

        for selected in (snapshot.tail_text, snapshot.full_text or ""):
            assert private_marker not in selected
            assert "15551234567" not in selected
            assert "[REDACTED_LOG_FRAGMENT]" in selected

    def test_selector_before_replay_window_boundary_stays_unresolved(
        self, hermes_home_with_secret
    ):
        """A selector split across the hash/replay boundary remains unsafe."""
        from hermes_cli import debug

        scan_bytes = debug._WHATSAPP_STATE_SCAN_BYTES
        scan_start = 65_536
        byte_offset = scan_start + scan_bytes
        prefix = bytearray(b"D" * byte_offset)
        dangerous = (
            b"Processing pending message: 'first private line\n"
            b"PRIVATE_BOUNDARY_CONTINUATION call 15551234567\n"
        )
        dangerous_start = scan_start - 6
        dangerous = b" " + dangerous
        prefix[dangerous_start : dangerous_start + len(dangerous)] = dangerous
        safe = (
            b"inbound message: platform=whatsapp_cloud user_present=True "
            b"chat_present=True msg_len=23 reply_to_id_present=False "
            b"reply_to_text_len=0\n"
        )
        prefix[byte_offset - len(safe) : byte_offset] = safe
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_bytes(prefix + b"PRIVATE_BOUNDARY_CONTINUATION call 15551234567\n")

        with log_path.open("rb") as log_file:
            state = debug._whatsapp_log_state_at(log_file, byte_offset)

        assert state.prefix_unresolved is True

    @pytest.mark.parametrize(
        "message_field",
        ["msg=first private line", "msg='first private line"],
        ids=["unquoted", "unterminated-quote"],
    )
    def test_default_fails_closed_for_ambiguous_whatsapp_conversation(
        self, hermes_home_with_secret, message_field
    ):
        from hermes_cli.debug import _capture_log_snapshot, _redact_log_text

        preview_phone = "15551234567"
        diagnostic_id = "9876543210"
        record = (
            "2026-08-06 12:00:00 INFO agent.turn_context: "
            "conversation turn: session=s1 model=m provider=p "
            f"platform=whatsapp_cloud history=0 {message_field}\n"
            f"2026-08-06 forged body call {preview_phone} about health\n"
            "2026-08-06 12:00:01 INFO worker: "
            f"diagnostic_id={diagnostic_id}\n"
        )
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(record)

        snap = _capture_log_snapshot("agent", tail_lines=10)

        for redacted in (
            _redact_log_text(record),
            snap.tail_text,
            snap.full_text or "",
        ):
            assert preview_phone not in redacted
            assert "first private line" not in redacted
            assert "forged body" not in redacted
            assert "about health" not in redacted
            assert diagnostic_id not in redacted
            assert redacted.count("[REDACTED_MESSAGE_PREVIEW]") >= 3

    def test_default_redacts_whatsapp_session_key_only(
        self, hermes_home_with_secret
    ):
        from hermes_cli.debug import _capture_log_snapshot

        wa_id = "15551234567"
        discord_id = "123456789012345"
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(
            "2026-08-06 12:00:00 DEBUG gateway.session: "
            f"session=agent:main:whatsapp_cloud:dm:{wa_id}\n"
            "2026-08-06 12:00:01 DEBUG gateway.session: "
            f"session=agent:main:discord:dm:{discord_id}\n"
        )

        snap = _capture_log_snapshot("agent", tail_lines=10)

        assert wa_id not in snap.tail_text
        assert discord_id in snap.tail_text
        assert snap.full_text is not None
        assert wa_id not in snap.full_text
        assert discord_id in snap.full_text

    @pytest.mark.parametrize("platform", ["whatsapp", "whatsapp_cloud"])
    def test_default_redacts_generic_whatsapp_gateway_identity_records(
        self, hermes_home_with_secret, platform
    ):
        from hermes_cli.debug import _capture_log_snapshot

        wa_id = "15551234567"
        discord_id = "123456789012345"
        records = [
            f"Unauthorized user: {wa_id} (Alice) on {platform}",
            f"pre_gateway_dispatch skip: reason=test platform={platform} chat={wa_id}",
            f"Sent shutdown notification to active chat {platform}:{wa_id}",
            f"Failed to send shutdown notification to home channel {platform}:{wa_id}: failed",
            f"Sent post-update notification to {platform}:{wa_id} (exit=0)",
            f"Restart notification to {platform}:{wa_id} was not delivered: failed",
            f"Sent restart notification to {platform}:{wa_id}",
            f"Home-channel startup notification failed for {platform}:{wa_id}: failed",
            f"Sent home-channel startup notification to {platform}:{wa_id}",
            f"No profile route matched: platform={platform} chat_id={wa_id} thread_id=None parent_chat_id=None",
            f"Profile route matching failed for {platform}/{wa_id}, falling back to default",
            f"Profile 'missing' does not exist for source {platform}/{wa_id} (guild_id=None), falling back",
            f"Failed to resolve profile directory for source {platform}/{wa_id} (guild_id=None), falling back",
        ]
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(
            "".join(
                f"2026-08-06 12:00:{index:02d} INFO gateway.run: {record}\n"
                for index, record in enumerate(records)
            )
            + (
                "2026-08-06 12:01:00 INFO gateway.run: "
                f"Sent restart notification to discord:{discord_id}\n"
            )
        )

        snap = _capture_log_snapshot("agent", tail_lines=100)

        for redacted in (snap.tail_text, snap.full_text or ""):
            assert wa_id not in redacted
            assert discord_id in redacted

    def test_no_redact_preserves_email_addresses(self, hermes_home_with_secret):
        from hermes_cli.debug import _capture_log_snapshot

        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(
            "2026-04-12 17:00:00 INFO gateway.run: "
            "inbound message: platform=bluebubbles "
            "user=person@example.com chat=iMessage;-;person@example.com msg='hello'\n"
        )

        snap = _capture_log_snapshot("agent", tail_lines=10, redact=False)

        assert "person@example.com" in snap.tail_text
        assert "person@example.com" in (snap.full_text or "")

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
        args.nous = False

        with patch("hermes_cli.dump.run_dump"), \
             patch("hermes_cli.debug._sweep_expired_pastes", return_value=(0, 0)) as mock_sweep, \
             patch("hermes_cli.debug.upload_to_pastebin",
                    return_value="https://paste.rs/test"):
            run_debug_share(args)

        mock_sweep.assert_called_once()
        assert "Debug report uploaded" in capsys.readouterr().out



    def test_share_uploads_five_pastes(self, hermes_home, capsys):
        """Successful share uploads report + agent.log + gateway.log + gui.log + desktop.log."""
        from hermes_cli.debug import run_debug_share

        args = MagicMock()
        args.lines = 50
        args.expire = 7
        args.local = False
        args.nous = False

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
        # Should have 5 uploads: report, agent.log, gateway.log, gui.log, desktop.log
        assert call_count[0] == 5
        assert "paste.rs/paste1" in out  # Report
        assert "paste.rs/paste2" in out  # agent.log
        assert "paste.rs/paste3" in out  # gateway.log
        assert "paste.rs/paste4" in out  # gui.log
        assert "paste.rs/paste5" in out  # desktop.log
        assert "Report" in out
        assert "agent.log" in out
        assert "gateway.log" in out
        assert "gui.log" in out
        assert "desktop.log" in out

        # Each log paste should start with the dump header
        agent_paste = uploaded_content[1]
        assert "--- hermes dump ---" in agent_paste
        assert "--- full agent.log ---" in agent_paste
        gateway_paste = uploaded_content[2]
        assert "--- hermes dump ---" in gateway_paste
        assert "--- full gateway.log ---" in gateway_paste
        gui_paste = uploaded_content[3]
        assert "--- hermes dump ---" in gui_paste
        assert "--- full gui.log ---" in gui_paste
        desktop_paste = uploaded_content[4]
        assert "--- hermes dump ---" in desktop_paste
        assert "--- full desktop.log ---" in desktop_paste




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
        args.nous = False
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
        args.nous = False
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
        args.nous = False
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
        args.nous = False

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


class TestRunDebugSweepsOnInvocation:
    """``run_debug`` must sweep expired pastes on every invocation."""

    def test_run_debug_calls_sweep(self, hermes_home):
        from hermes_cli.debug import run_debug

        args = MagicMock()
        args.debug_command = None  # default → prints help

        with patch("hermes_cli.debug._sweep_expired_pastes") as mock_sweep:
            run_debug(args)

        mock_sweep.assert_called_once()


class TestRunDebugDelete:

    def test_handles_delete_failure(self, capsys):
        from hermes_cli.debug import run_debug_delete

        args = MagicMock()
        args.urls = ["https://paste.rs/abc"]

        with patch("hermes_cli.debug.delete_paste",
                    side_effect=Exception("network error")):
            run_debug_delete(args)

        out = capsys.readouterr().out
        assert "Could not delete" in out


class TestShareIncludesAutoDelete:
    """Verify that run_debug_share schedules auto-deletion and prints TTL."""


    def test_share_shows_privacy_notice(self, hermes_home, capsys):
        from hermes_cli.debug import run_debug_share

        args = MagicMock()
        args.lines = 50
        args.expire = 7
        args.local = False
        args.nous = False

        with patch("hermes_cli.dump.run_dump"), \
             patch("hermes_cli.debug.upload_to_pastebin",
                    return_value="https://paste.rs/test"), \
             patch("hermes_cli.debug._schedule_auto_delete"):
            run_debug_share(args)

        out = capsys.readouterr().out
        assert "PUBLIC paste service" in out
        assert "NOT redacted" in out


# ---------------------------------------------------------------------------
# build_debug_share — structured core used by the dashboard endpoint
# ---------------------------------------------------------------------------


class TestBuildDebugShare:
    """The shared core that returns structured paste URLs (not printed text).

    Backs both ``hermes debug share`` (CLI) and ``POST /api/ops/debug-share``
    (dashboard). The dashboard renders ``urls`` as real, copyable links, so the
    contract here is the return value, not stdout.
    """



    def test_redaction_keeps_secrets_out_of_payload(self, hermes_home):
        from hermes_cli.debug import build_debug_share

        secret = "sk-proj-SUPERSECRETtoken1234567890"
        (hermes_home / "logs" / "agent.log").write_text(
            f"line one\nauthorization token={secret}\nline three\n"
        )

        uploaded = []

        def _upload(content, expiry_days=7):
            uploaded.append(content)
            return "https://paste.rs/x"

        with patch("hermes_cli.dump.run_dump"), patch(
            "hermes_cli.debug.upload_to_pastebin", side_effect=_upload
        ), patch("hermes_cli.debug._schedule_auto_delete"):
            result = build_debug_share(log_lines=50, redact=True)

        assert result.redacted is True
        joined = "\n".join(uploaded)
        assert secret not in joined, "secret leaked into upload payload"

    def test_optional_log_failure_is_collected_not_raised(self, hermes_home):
        from hermes_cli.debug import build_debug_share

        count = [0]

        def _upload(content, expiry_days=7):
            count[0] += 1
            # First call (the required Report) succeeds; a later one fails.
            if count[0] == 2:
                raise RuntimeError("paste service hiccup")
            return f"https://paste.rs/p{count[0]}"

        with patch("hermes_cli.dump.run_dump"), patch(
            "hermes_cli.debug.upload_to_pastebin", side_effect=_upload
        ), patch("hermes_cli.debug._schedule_auto_delete"):
            result = build_debug_share(log_lines=50, redact=True)

        assert "Report" in result.urls
        assert len(result.failures) == 1
        assert "paste service hiccup" in result.failures[0]



# ---------------------------------------------------------------------------
# Shared bundle collection + Nous-S3 path
# ---------------------------------------------------------------------------

class TestCollectShareBundle:

    def test_no_redact_omits_banner(self, hermes_home):
        from hermes_cli.debug import collect_share_bundle

        with patch("hermes_cli.dump.run_dump"):
            bundle = collect_share_bundle(log_lines=50, redact=False)

        assert "redacted at upload time" not in bundle["report"]

    def test_redaction_keeps_secrets_out(self, hermes_home):
        from hermes_cli.debug import collect_share_bundle

        secret = "sk-proj-abcdefghijklmnopqrstuvwxyz1234567890"
        (hermes_home / "logs" / "agent.log").write_text(
            f"line one\nOPENAI_API_KEY={secret}\nline three\n"
        )
        with patch("hermes_cli.dump.run_dump"):
            redacted = collect_share_bundle(log_lines=50, redact=True)
            unredacted = collect_share_bundle(log_lines=50, redact=False)

        # Sanity: without redaction the secret is present in the bundle.
        assert secret in "\n".join(unredacted.values())
        # With redaction it must be scrubbed everywhere.
        assert secret not in "\n".join(redacted.values())




class TestBuildNousBundle:
    def test_envelope_shape_and_gzip(self, hermes_home):
        import gzip
        import json as _json

        from hermes_cli.debug import build_nous_bundle

        files = {"report": "hello", "agent.log": "log line"}
        blob = build_nous_bundle(files, redact=True)

        # It's gzip — magic bytes.
        assert blob[:2] == b"\x1f\x8b"
        envelope = _json.loads(gzip.decompress(blob).decode())
        assert envelope["format"] == "hermes-debug-share/1"
        assert envelope["redacted"] is True
        assert envelope["files"] == files
        assert "created" in envelope

    def test_redacted_false_recorded(self):
        import gzip
        import json as _json

        from hermes_cli.debug import build_nous_bundle

        blob = build_nous_bundle({"report": "x"}, redact=False)
        envelope = _json.loads(gzip.decompress(blob).decode())
        assert envelope["redacted"] is False


class TestRunDebugShareNous:
    def _args(self, **over):
        class _A:
            lines = 50
            expire = 7
            local = False
            nous = True
            no_redact = False
            yes = True

        a = _A()
        for k, v in over.items():
            setattr(a, k, v)
        return a

    def test_nous_success_prints_view_url(self, hermes_home, capsys):
        from hermes_cli.debug import run_debug_share

        res = {
            "id": "id-1",
            "viewUrl": "https://support.example.com/diagnostics/id-1",
            "expiresAt": "2026-06-20T00:00:00Z",
        }
        with patch("hermes_cli.dump.run_dump"), patch(
            "hermes_cli.diagnostics_upload.share_to_nous", return_value=res
        ) as share:
            run_debug_share(self._args())

        out = capsys.readouterr().out
        assert "Nous-INTERNAL" in out
        assert "https://support.example.com/diagnostics/id-1" in out
        assert "2026-06-20T00:00:00Z" in out
        # The blob passed to share_to_nous must be gzip bytes.
        blob = share.call_args[0][0]
        assert isinstance(blob, (bytes, bytearray)) and blob[:2] == b"\x1f\x8b"

    def test_nous_failure_suggests_local(self, hermes_home, capsys):
        from hermes_cli.debug import run_debug_share

        with patch("hermes_cli.dump.run_dump"), patch(
            "hermes_cli.diagnostics_upload.share_to_nous",
            side_effect=RuntimeError("service down"),
        ):
            with pytest.raises(SystemExit) as exc:
                run_debug_share(self._args())
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "Nous upload failed" in err
        assert "--local" in err

    def test_nous_does_not_touch_pastebin(self, hermes_home):
        from hermes_cli.debug import run_debug_share

        res = {"id": "id-1", "viewUrl": "https://v"}
        with patch("hermes_cli.dump.run_dump"), patch(
            "hermes_cli.diagnostics_upload.share_to_nous", return_value=res
        ), patch("hermes_cli.debug.upload_to_pastebin") as paste:
            run_debug_share(self._args())
        paste.assert_not_called()


class TestDebugSlashCommand:
    """`/debug [nous|local]` parsing in the CLI/TUI handler.

    The classic CLI and the TUI slash worker both dispatch through
    ``HermesCLI.process_command`` → ``_handle_debug_command(cmd_original)``,
    which parses an optional destination word and builds the args namespace
    handed to ``run_debug_share``.
    """

    def _handler(self):
        from hermes_cli.cli_commands_mixin import CLICommandsMixin

        class _Stub(CLICommandsMixin):
            pass

        return _Stub()._handle_debug_command

    def _captured(self, cmd_original):
        captured = {}

        def _fake_run(args):
            captured.update(vars(args))

        with patch("hermes_cli.debug.run_debug_share", _fake_run):
            self._handler()(cmd_original)
        return captured

    def test_bare_debug_defaults_to_paste(self):
        c = self._captured("/debug")
        assert c["nous"] is False and c["local"] is False
        assert c["lines"] == 200 and c["expire"] == 7
        # The slash command IS the consent action → skip the [y/N] prompt
        # (input() would hang inside prompt_toolkit's event loop).
        assert c["yes"] is True


    def test_word_parsing_is_case_insensitive(self):
        c = self._captured("/debug NOUS")
        assert c["nous"] is True


    def test_no_arg_default_keyword(self):
        # Calling with no cmd_original (legacy callers) must still work.
        c = self._captured("")
        assert c["nous"] is False and c["local"] is False


class TestShareConsentGate:
    """`hermes debug share` requires explicit consent before uploading.

    Uses SimpleNamespace rather than MagicMock so ``args.yes`` is a real
    ``False`` — a MagicMock auto-provides a truthy ``.yes`` and would silently
    bypass the very gate under test.
    """

    def _args(self, **over):
        from types import SimpleNamespace

        base = dict(lines=50, expire=7, local=False, nous=False,
                    no_redact=False, yes=False)
        base.update(over)
        return SimpleNamespace(**base)




    def test_non_interactive_requires_yes(self, hermes_home, capsys, monkeypatch):
        """No TTY + no --yes → exit(1), never upload silently."""
        from hermes_cli.debug import run_debug_share

        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        with patch("hermes_cli.dump.run_dump"), \
             patch("hermes_cli.debug.upload_to_pastebin") as mock_upload:
            with pytest.raises(SystemExit) as exc:
                run_debug_share(self._args())

        assert exc.value.code == 1
        mock_upload.assert_not_called()
        err = capsys.readouterr().err
        assert "Non-interactive mode requires --yes" in err
        assert "personal data" in err


    def test_local_never_prompts(self, hermes_home, capsys, monkeypatch):
        """--local renders to stdout and must not prompt or upload."""
        from hermes_cli.debug import run_debug_share

        def _boom(_):
            raise AssertionError("input() must not be called for --local")

        monkeypatch.setattr("builtins.input", _boom)

        with patch("hermes_cli.dump.run_dump"), \
             patch("hermes_cli.debug.upload_to_pastebin") as mock_upload:
            run_debug_share(self._args(local=True))

        mock_upload.assert_not_called()
        assert "Aborted" not in capsys.readouterr().out

"""Historical log redaction and descriptor snapshot regressions."""

import os


import pytest

from hermes_cli import debug_log_snapshot as snapshot_module


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


_REDACT_FIXTURE_TOKEN = "sk-proj-A1B2C3D4E5F6G7H8I9J0aA"


class TestCaptureLogSnapshotRedaction:
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

        redacted = _capture_log_snapshot("agent", tail_lines=10, max_bytes=len(raw))

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
        record = (
            line_ending.join([
                "2026-08-06 12:00:00 INFO gateway.platforms.whatsapp_cloud: "
                "[whatsapp_cloud] media metadata fetch raised "
                "(id=present(len=28), error_type=RuntimeError)",
                "Traceback (most recent call last):",
                "  File '/tmp/adapter.py', line 1, in get",
                f"RuntimeError: GET https://graph.facebook.com/v20.0/{media_id} "
                f"failed for {private_name}",
                "2026-08-06 12:00:01 INFO worker: diagnostic_id=9876543210",
            ])
            + line_ending
        )
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_bytes(record.encode())

        direct = debug._redact_log_text(record)
        snapshot = debug._capture_log_snapshot("agent", tail_lines=20, max_bytes=10_000)
        if view == "tail":
            selected = snapshot.tail_text
        elif view == "full":
            selected = snapshot.full_text or ""
        else:
            selected = "\n".join(debug.collect_share_bundle(log_lines=20).values())

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
        snapshot = debug._capture_log_snapshot("agent", tail_lines=20, max_bytes=10_000)

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

        snapshot = debug._capture_log_snapshot("agent", tail_lines=1, max_bytes=240)

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

        snapshot = debug._capture_log_snapshot("agent", tail_lines=20, max_bytes=10_000)
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
        record = (
            line_ending.join([
                "2026-08-06 12:00:00 DEBUG gateway.run: "
                f"{prefix}: 'first {private_marker}",
                "attacker-controlled...'",
                f"later private {private_marker} call {wa_id}",
                f"2026-08-06 12:00:01 INFO worker: diagnostic_id={diagnostic_id}",
            ])
            + line_ending
        )
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_bytes(record.encode())

        snapshot = debug._capture_log_snapshot("agent", tail_lines=20, max_bytes=10_000)
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

        snapshot = debug._capture_log_snapshot("agent", tail_lines=20, max_bytes=10_000)
        if view == "tail":
            selected = snapshot.tail_text
        elif view == "full":
            selected = snapshot.full_text or ""
        else:
            selected = "\n".join(debug.collect_share_bundle(log_lines=20).values())

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
        snapshot = debug._capture_log_snapshot("agent", tail_lines=20, max_bytes=10_000)
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
        record = (
            line_ending.join([
                "2026-08-06 12:00:00 INFO gateway.platforms.whatsapp_cloud: "
                "[whatsapp_cloud] media metadata fetch raised "
                "(id=present(len=28), error_type=RuntimeError)",
                "Traceback (most recent call last):",
                "  File '/tmp/adapter.py', line 1, in get",
                f"RuntimeError: initial {private_marker}",
                f"2026-08-06 12:00:01 ERROR forged {private_marker} call {wa_id}",
                f"later private {private_marker} call {wa_id}",
                "2026-08-06 12:00:02 INFO worker: diagnostic_id=9876543210",
            ])
            + line_ending
        )
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_bytes(record.encode())

        redacted = debug._redact_log_text(record)
        snapshot = debug._capture_log_snapshot("agent", tail_lines=20, max_bytes=10_000)
        if view == "direct":
            selected = redacted
        elif view == "tail":
            selected = snapshot.tail_text
        elif view == "full":
            selected = snapshot.full_text or ""
        else:
            selected = "\n".join(debug.collect_share_bundle(log_lines=20).values())

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
        raw = "D" * 9_000 + f" diagnostic_id=123456789012345 status={diagnostic}"
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(raw)

        snapshot = debug._capture_log_snapshot("agent", tail_lines=1, max_bytes=240)

        for safe in (snapshot.tail_text, snapshot.full_text or ""):
            assert diagnostic in safe
            assert "[REDACTED_LOG_FRAGMENT]" not in safe

    def test_redaction_utf8_cap_is_enforced_after_decode(self, hermes_home_with_secret):
        """A multibyte suffix cut must remain valid and within max_bytes."""
        from hermes_cli.debug import _capture_log_snapshot

        raw = "💥 inbound message: platform=whatsapp user=1234567 chat=2345678"
        max_bytes = len(raw.encode("utf-8"))
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(raw)

        snapshot = _capture_log_snapshot("agent", tail_lines=1, max_bytes=max_bytes)

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
        raw = "D" * 10_000 + f" diagnostic_id=123456789012345 status={diagnostic}"
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
            record += f"2026-08-06 12:00:01 INFO worker: {generic_diagnostic}\n"

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
            record += f"2026-08-06 12:00:01 INFO worker: {generic_diagnostic}\n"

        log_path = hermes_home_with_secret / "logs" / "agent.log"
        replacement_path = hermes_home_with_secret / "logs" / "replacement.log"
        log_path.write_text(record)
        original_state_at = snapshot_module._whatsapp_log_state_at

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
            snapshot_module,
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
        original_fstat = snapshot_module.os.fstat
        fstat_calls = 0

        def append_before_final_fstat(fd):
            nonlocal fstat_calls
            fstat_calls += 1
            if fstat_calls == 2:
                with log_path.open("a") as mutable_log:
                    mutable_log.write(appended_line)
            return original_fstat(fd)

        monkeypatch.setattr(snapshot_module.os, "fstat", append_before_final_fstat)

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
                f"continuation-{index:04d} {private_marker}\n" for index in range(1_000)
            )
        )
        original_state_at = snapshot_module._whatsapp_log_state_at

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
            snapshot_module,
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
                f"continuation-{index:04d} {private_marker}\n" for index in range(1_000)
            )
        )
        original_state_at = snapshot_module._whatsapp_log_state_at

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
            snapshot_module,
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
            f"2026-08-06 12:00:02 INFO worker: diagnostic_id={diagnostic_id}\n"
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

        snapshot = debug._capture_log_snapshot("agent", tail_lines=20, max_bytes=10_000)
        if view == "direct":
            selected = debug._redact_log_text(record)
        elif view == "tail":
            selected = snapshot.tail_text
        elif view == "full":
            selected = snapshot.full_text or ""
        else:
            selected = "\n".join(debug.collect_share_bundle(log_lines=20).values())

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
            + line
            * ((snapshot_module._WHATSAPP_STATE_SCAN_BYTES // len(line)) + 5_000)
        )
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(record)

        snapshot = debug._capture_log_snapshot("agent", tail_lines=10, max_bytes=240)

        assert private_marker not in snapshot.tail_text
        assert private_marker not in (snapshot.full_text or "")
        assert "[REDACTED_LOG_FRAGMENT]" in snapshot.tail_text
        assert "[REDACTED_LOG_FRAGMENT]" in (snapshot.full_text or "")

    def test_prefix_state_recovery_replays_only_bounded_window(
        self, hermes_home_with_secret, monkeypatch
    ):
        """Large ordinary prefixes do not trigger an unbounded line replay."""

        line = "2026-08-06 12:00:00 INFO worker: ordinary diagnostic line\n"
        raw = line * ((16 * 1024 * 1024) // len(line) + 1)
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(raw)

        calls = 0
        original = snapshot_module._redact_log_text_with_state

        def count_state_replays(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(
            snapshot_module, "_redact_log_text_with_state", count_state_replays
        )
        with log_path.open("rb") as log_file:
            state = snapshot_module._whatsapp_log_state_at(log_file, len(raw) - 1024)

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

        snapshot = debug._capture_log_snapshot("agent", tail_lines=3, max_bytes=240)

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
        log_path.write_text(metadata + line * ((4 * 1024 * 1024) // len(line) + 1))

        snapshot = debug._capture_log_snapshot("agent", tail_lines=3, max_bytes=240)

        assert diagnostic in snapshot.tail_text
        assert diagnostic in (snapshot.full_text or "")
        assert "[REDACTED_LOG_FRAGMENT]" not in snapshot.tail_text
        assert "[REDACTED_LOG_FRAGMENT]" not in (snapshot.full_text or "")

    @pytest.mark.parametrize(
        "safe_error",
        [
            "[whatsapp_cloud] media metadata fetch raised "
            "(id=present(len=28), error_type=RuntimeError)",
            "[WhatsApp] WhatsApp read receipt failed (error_type=TimeoutError)",
            "[whatsapp_cloud] Error handling message (error_type=RuntimeError)",
            "[whatsapp] Poll error (error_type=RuntimeError)",
            "[whatsapp] Failed to cache image (error_type=RuntimeError)",
            "[whatsapp] Failed to cache audio (error_type=RuntimeError)",
            "[whatsapp] Error building event (error_type=RuntimeError)",
            "[whatsapp] Failed to send image (error_detail=present)",
            "[whatsapp] Error sending local file present (error_type=RuntimeError)",
            "[whatsapp] send_typing error (non-fatal; error_type=RuntimeError)",
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
            * (
                (snapshot_module._WHATSAPP_STATE_SCAN_BYTES // len(continuation))
                + 5_000
            )
            + (
                "2026-08-06 12:00:01 INFO gateway.run: inbound message: "
                "platform=whatsapp_cloud user_present=True chat_present=True "
                "msg_len=23 reply_to_id_present=False reply_to_text_len=0\n"
            )
        )
        log_path = hermes_home_with_secret / "logs" / "agent.log"
        log_path.write_text(record)

        snapshot = debug._capture_log_snapshot("agent", tail_lines=10, max_bytes=240)

        for selected in (snapshot.tail_text, snapshot.full_text or ""):
            assert private_marker not in selected
            assert "15551234567" not in selected
            assert "[REDACTED_LOG_FRAGMENT]" in selected

    def test_selector_before_replay_window_boundary_stays_unresolved(
        self, hermes_home_with_secret
    ):
        """A selector split across the hash/replay boundary remains unsafe."""

        scan_bytes = snapshot_module._WHATSAPP_STATE_SCAN_BYTES
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
        log_path.write_bytes(
            prefix + b"PRIVATE_BOUNDARY_CONTINUATION call 15551234567\n"
        )

        with log_path.open("rb") as log_file:
            state = snapshot_module._whatsapp_log_state_at(log_file, byte_offset)

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

    def test_collector_redacts_session_fields_for_all_platforms(
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
        assert discord_id not in snap.tail_text
        assert snap.full_text is not None
        assert wa_id not in snap.full_text
        assert discord_id not in snap.full_text

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

"""Extended historical diagnostic privacy cases preserved from #69561."""

from unittest.mock import patch


import pytest


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

    def test_default_redacts_positional_log_formats_and_ip_addresses(
        self, hermes_home_with_secret
    ):
        from hermes_cli.debug import _capture_log_snapshot

        log_path = hermes_home_with_secret / "logs" / "agent.log"
        records = (
            "2026-04-12 WARNING gateway.run: "
            "Unauthorized user: 123456 (Alice Example) on telegram\n"
            "2026-04-12 INFO tools.vision_tools: "
            "User prompt: private medical question\n"
            "2026-04-12 INFO gateway.platforms.webhook: "
            "[webhook] Response for github:owner/repo: internal answer\n"
            "2026-04-12 INFO gateway.platforms.webhook: "
            "[webhook] direct-deliver log-only: private fallback\n"
            "2026-04-12 INFO service: backend=10.0.0.7:8080 "
            "endpoint=[2001:db8::1]:443 scoped=fe80::1%eth0 version=1.2.3.4\n"
            "2026-04-12 INFO service: user=alice\n"
            "2026-04-12 INFO service: next=record\n"
        )

        # Each fixture is an independent record. Once a raw message opener
        # occurs in a file, later apparent records may be its forged payload.
        snapshots = []
        for record in records.splitlines(keepends=True):
            log_path.write_bytes(record.encode())
            snapshots.append(_capture_log_snapshot("agent", tail_lines=20))
            assert log_path.read_bytes() == record.encode()

        for text in (
            "\n".join(snap.tail_text for snap in snapshots),
            "\n".join(snap.full_text or "" for snap in snapshots),
        ):
            for sensitive in (
                "123456",
                "Alice Example",
                "private medical question",
                "github:owner/repo",
                "internal answer",
                "private fallback",
                "10.0.0.7",
                "2001:db8::1",
                "fe80::1%eth0",
                "user=alice",
            ):
                assert sensitive not in text
            assert "Unauthorized user: [REDACTED_ID] ([REDACTED_NAME])" in text
            assert "User prompt: [REDACTED_MESSAGE]" in text
            assert "[webhook] Response for [REDACTED_ID]: [REDACTED_MESSAGE]" in text
            assert "[webhook] direct-deliver log-only: [REDACTED_MESSAGE]" in text
            assert "backend=[REDACTED_IP]:8080" in text
            assert "version=1.2.3.4" in text
            assert "endpoint=[[REDACTED_IP]]:443" in text
            assert "scoped=[REDACTED_IP]" in text
            assert "user=[REDACTED_ID]" in text

    @pytest.mark.parametrize(
        ("line", "sensitive"),
        [
            (
                "[Telegram] Blocked unauthorized user 123 in chat -456",
                ("123", "-456"),
            ),
            (
                "Voice input from user 123: medical transcript",
                ("123", "medical transcript"),
            ),
            (
                "[Email] New message from alice@example.test: payroll subject",
                ("alice@example.test", "payroll subject"),
            ),
            (
                "Generating image with flux (flux) — prompt: private portrait",
                ("private portrait",),
            ),
            (
                "Forwarded update prompt to discord:123: private question",
                ("discord:123", "private question"),
            ),
            (
                "Suppressing duplicate voice transcript for guild=12 user=34: secret words",
                ("guild=12", "user=34", "secret words"),
            ),
            (
                "api_calls=2 response=review confidential status=done",
                ("review confidential",),
            ),
            (
                "chat_id=oc_1 sender=user:ou_2 text='private feishu message' media=0",
                ("oc_1", "user:ou_2", "private feishu message"),
            ),
            (
                "session=abc answer=private choice user=Alice Example",
                ("abc", "private choice", "Alice Example"),
            ),
        ],
    )
    def test_redacts_historical_sensitive_log_formats(self, line, sensitive):
        from hermes_cli.debug import _redact_log_text

        redacted = _redact_log_text(line)

        assert all(value not in redacted for value in sensitive)

    def test_redacts_nested_display_name_and_multiline_message_continuation(self):
        from hermes_cli.debug import _redact_log_text

        text = (
            "2026-04-12 17:00:00 INFO gateway.run: "
            "Unauthorized user: 123 (Alice (Admin)) on telegram\n"
            "2026-04-12 17:00:01 INFO tools.vision_tools: User prompt: first line\n"
            "private continuation\r\n"
            "2026-04-12 17:00:02 INFO service: next record\n"
        )

        redacted = _redact_log_text(text)

        for sensitive in ("123", "Alice (Admin)", "first line", "private continuation"):
            assert sensitive not in redacted
        # A forged timestamp cannot terminate an unframed message record.
        assert "2026-04-12 17:00:02 INFO service: next record" not in redacted
        assert "[REDACTED_MESSAGE_PREVIEW]\r\n" in redacted

    def test_default_redacts_sensitive_snapshot_read_error_path(
        self, hermes_home_with_secret
    ):
        from hermes_cli.debug import _capture_log_snapshot

        log_path = hermes_home_with_secret / "logs" / "agent.log"
        with patch(
            "builtins.open",
            side_effect=PermissionError(f"cannot read {log_path}"),
        ):
            snap = _capture_log_snapshot("agent", tail_lines=10)

        assert str(hermes_home_with_secret) not in snap.tail_text
        assert "PermissionError" in snap.tail_text
        assert snap.full_text is None

    def test_malformed_quoted_field_distrusts_next_record_prefix(self):
        from hermes_cli.debug import _redact_log_text

        text = "msg='unterminated\r\n2026-04-12 17:00:01 INFO service: healthy\r\n"

        redacted = _redact_log_text(text)

        assert "unterminated" not in redacted
        assert "2026-04-12 17:00:01 INFO service: healthy" not in redacted


class TestCollectShareBundle:
    def test_returns_report_and_logs(self, hermes_home):
        from hermes_cli.debug import collect_share_bundle

        with patch("hermes_cli.dump.run_dump"):
            bundle = collect_share_bundle(log_lines=50, redact=True)

        assert "report" in bundle
        assert "agent.log" in bundle
        assert "gateway.log" in bundle
        assert "desktop.log" in bundle
        # Banner is prepended under redact=True.
        assert "best-effort secret and privacy redaction" in bundle["report"]
        assert "session started" in bundle["agent.log"]

    def test_redaction_covers_dump_text_in_every_bundle_payload(self, hermes_home):
        from hermes_cli.debug import collect_share_bundle

        dump_text = (
            "terminal.docker_image: 10.0.0.7/private\n"
            "operator_email: alice@example.test\n"
        )
        (hermes_home / "logs" / "agent.log").write_text("agent ready\n")

        with (
            patch("hermes_cli.debug._capture_dump", return_value=dump_text),
            patch("hermes_cli.dump.run_dump"),
        ):
            redacted = collect_share_bundle(log_lines=50, redact=True)
            unredacted = collect_share_bundle(log_lines=50, redact=False)

        assert all("10.0.0.7" not in value for value in redacted.values())
        assert all("alice@example.test" not in value for value in redacted.values())
        assert any("10.0.0.7/private" in value for value in unredacted.values())
        assert any("alice@example.test" in value for value in unredacted.values())

    def test_redaction_sanitizes_positional_pii_across_full_bundle(self, hermes_home):
        from hermes_cli.debug import collect_share_bundle

        sensitive = {
            "123456",
            "Alice Example",
            "private medical question",
            "room-987",
            "internal answer",
            "888",
            "-999",
            "777",
            "private transcript",
            "payroll@example.test",
            "salary review",
            "private portrait",
            "discord:555",
            "private decision",
            "10.0.0.7",
            "2001:db8::1",
            "doc-secret",
            "comment-secret",
            "user-secret",
            "private request body",
            "private raw response",
            "Confidential roadmap",
            "https://tenant.example.test/private",
            "private quote",
            "private reply",
            "private whole comment",
            "private model response",
            "private full prompt",
            "private continuation",
            "private agent response",
            "comment-doc:docx:session-secret",
            "alice@example.test",
            "private subject",
        }
        fixture = (
            "2026-04-12 17:00:00 INFO service: "
            "Unauthorized user: 123456 (Alice Example) on telegram\n"
            "2026-04-12 17:00:01 INFO service: "
            "User prompt: private medical question\n"
            "2026-04-12 17:00:02 INFO service: "
            "[webhook] Response for room-987: internal answer\n"
            "2026-04-12 17:00:03 INFO service: "
            "[Telegram] Blocked unauthorized user 888 in chat -999\n"
            "2026-04-12 17:00:04 INFO service: "
            "Voice input from user 777: private transcript\n"
            "2026-04-12 17:00:05 INFO service: "
            "[Email] New message from payroll@example.test: salary review\n"
            "2026-04-12 17:00:06 INFO service: "
            "Generating image with flux (flux) — prompt: private portrait\n"
            "2026-04-12 17:00:07 INFO service: "
            "Forwarded update prompt to discord:555: private decision\n"
            "2026-04-12 17:00:08 INFO service: "
            "backend=10.0.0.7 peer=2001:db8::1\n"
            "2026-04-12 17:00:09 INFO service: "
            "[Feishu-Comment] API >>> POST /comments "
            "paths={'file_token': 'doc-secret'} body=private request body\n"
            "2026-04-12 17:00:10 WARNING service: "
            "[Feishu-Comment] API FAIL raw response: private raw response\n"
            "2026-04-12 17:00:11 DEBUG service: "
            "[Feishu-Comment] query_document_meta: raw metas type=list "
            "value=Confidential roadmap\n"
            "2026-04-12 17:00:12 INFO service: "
            "[Feishu-Comment] query_document_meta: title=Confidential roadmap "
            "url=https://tenant.example.test/private\n"
            "2026-04-12 17:00:13 INFO service: "
            "[Feishu-Comment] batch_query_comment: is_whole=False "
            "quote=private quote reply_count=1\n"
            "2026-04-12 17:00:14 INFO service: "
            "[Feishu-Comment] reply_to_comment: comment_id=comment-secret "
            "text=private reply\n"
            "2026-04-12 17:00:15 INFO service: "
            "[Feishu-Comment] add_whole_comment: file_token=doc-secret "
            "text=private whole comment\n"
            "2026-04-12 17:00:16 INFO service: "
            "[Feishu-Comment] _run_comment_agent: done api_calls=2 "
            "response_len=22 response=private model response\n"
            "2026-04-12 17:00:17 INFO service: "
            "[Feishu-Comment] Event: notice=reply file=docx:doc-secret "
            "comment=comment-secret from=user-secret\n"
            "2026-04-12 17:00:18 INFO service: "
            "[Feishu-Comment] Local timeline: 2 entries target_idx=1 "
            "quote=private quote root=private reply target=private whole comment\n"
            "2026-04-12 17:00:19 DEBUG service: "
            "[Feishu-Comment] Full prompt: private full prompt\n"
            "private continuation\n"
            "2026-04-12 17:00:20 INFO service: "
            "[Feishu-Comment] Agent response (22 chars): private agent response\n"
            "2026-04-12 17:00:21 INFO service: "
            "[Feishu-Comment] Session saved: comment-doc:docx:session-secret "
            "(2 messages)\n"
            "2026-04-12 17:00:22 INFO service: "
            "Telegram update prompt answered 'y' by user user-secret\n"
            "2026-04-12 17:00:23 INFO service: "
            "[Email] Sent reply to alice@example.test (subject: private subject)\n"
        )
        for name in ("agent.log", "gateway.log", "gui.log", "desktop.log"):
            (hermes_home / "logs" / name).write_text(fixture)

        with patch("hermes_cli.dump.run_dump"):
            redacted = collect_share_bundle(log_lines=50, redact=True)
            unredacted = collect_share_bundle(log_lines=50, redact=False)

        expected_outputs = {
            "report",
            "agent.log",
            "gateway.log",
            "gui.log",
            "desktop.log",
        }
        assert set(redacted) == expected_outputs
        assert set(unredacted) == expected_outputs
        for output in expected_outputs:
            assert all(value in unredacted[output] for value in sensitive), output
            assert all(value not in redacted[output] for value in sensitive), output
            assert "[REDACTED_MESSAGE]" in redacted[output]


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    logs = home / "logs"
    logs.mkdir(parents=True)
    for name in ("agent", "errors", "gateway", "gui", "desktop"):
        (logs / f"{name}.log").write_text("session started\n")
    return home

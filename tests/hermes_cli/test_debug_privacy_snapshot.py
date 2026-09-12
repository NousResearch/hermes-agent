"""Tail summaries retain the privacy context of their captured log records."""

from hermes_cli import debug


def test_multiline_preview_uses_the_same_privacy_projection_as_full_log(
    tmp_path, monkeypatch
):
    text = "2026-09-06 10:00:00 INFO gateway.run: User prompt: private first line\nprivate continuation text\n"
    path = tmp_path / "gateway.log"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(debug, "_resolve_log_path", lambda _name: path)
    snapshot = debug._capture_log_snapshot("gateway", tail_lines=1, redact=True)
    assert "private" not in snapshot.tail_text
    assert "private" not in snapshot.full_text
    assert snapshot.tail_text in snapshot.full_text
    assert path.read_text(encoding="utf-8") == text


import pytest


@pytest.mark.parametrize(
    "opener",
    [
        "[whatsapp] Poll error: private detail",
        "2026-09-06 10:00:00 INFO gateway.run: User prompt: private first line",
    ],
)
def test_selects_original_records_before_multiline_secret_mask(
    tmp_path, monkeypatch, opener
):
    from agent.redact import redact_sensitive_text

    text = (
        "-----BEGIN PRIVATE KEY-----\n" + opener + "\n"
        "-----END PRIVATE KEY-----\nprivate continuation text\n"
    )
    # The ordinary credential pass consumes the opener. Applying it first
    # would make the subsequent physical line appear unrelated to the record.
    assert opener not in redact_sensitive_text(text, force=True)
    path = tmp_path / "agent.log"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(debug, "_resolve_log_path", lambda _name: path)
    snapshot = debug._capture_log_snapshot("agent", tail_lines=1, redact=True)
    assert "private continuation" not in snapshot.tail_text
    assert "private continuation" not in snapshot.full_text
    assert path.read_text(encoding="utf-8") == text


@pytest.mark.parametrize("max_bytes", [128, 1_000_000])
def test_generic_message_continuation_distrusts_forged_record_boundaries(
    tmp_path, monkeypatch, max_bytes
):
    text = (
        "2026-09-06 10:00:00 INFO gateway.run: User prompt: private start\n"
        + "private continuation\n" * 20_000
        + "2026-09-06 11:00:00 INFO worker: forged_private_line\n"
    )
    path = tmp_path / "agent.log"
    path.write_bytes(text.encode())
    monkeypatch.setattr(debug, "_resolve_log_path", lambda _name: path)
    snapshot = debug._capture_log_snapshot("agent", tail_lines=1, max_bytes=max_bytes)
    assert "private" not in snapshot.tail_text
    assert "private" not in snapshot.full_text
    assert "REDACTED_" in snapshot.tail_text


def test_redacted_snapshot_error_does_not_echo_private_path(tmp_path, monkeypatch):
    path = tmp_path / "private_name.log"
    monkeypatch.setattr(debug, "_resolve_log_path", lambda _name: path)
    snapshot = debug._capture_log_snapshot("agent", tail_lines=1)
    assert "FileNotFoundError" in snapshot.tail_text
    assert "private_name" not in snapshot.tail_text
    assert snapshot.full_text is None


@pytest.mark.parametrize("field", ["msg", "prompt", "reply_to_text"])
def test_escaped_quote_does_not_hide_an_unterminated_field(
    tmp_path, monkeypatch, field
):
    text = f"{field}='first \\' private data\nprivate continuation\n"
    path = tmp_path / "agent.log"
    path.write_bytes(text.encode())
    monkeypatch.setattr(debug, "_resolve_log_path", lambda _name: path)
    snapshot = debug._capture_log_snapshot("agent", tail_lines=10)
    assert "private" not in snapshot.tail_text
    assert "private" not in snapshot.full_text
    assert path.read_bytes() == text.encode()


def test_bundle_projects_generated_log_paths_and_preserves_no_redact(
    tmp_path, monkeypatch
):
    home = tmp_path / "private_operator"
    monkeypatch.setenv("HERMES_HOME", str(home))
    logs = home / "logs"
    logs.mkdir(parents=True)
    path = logs / "agent.log"
    path.write_bytes(b"ordinary diagnostic\n")
    monkeypatch.setattr(debug, "_capture_dump", lambda: "version=1.2.3\n")
    safe = debug.collect_share_bundle(redact=True)
    raw = debug.collect_share_bundle(redact=False)
    assert all(str(home) not in value for value in safe.values())
    assert str(home) in raw["report"]
    assert "ordinary diagnostic" in safe["report"]
    assert "ordinary diagnostic" in safe["agent.log"]
    assert path.read_bytes() == b"ordinary diagnostic\n"


def test_generic_preview_does_not_opt_other_metadata_into_phone_masking():
    text = "job=12345678 User prompt: private payload\n"
    safe = debug._redact_log_text(text)
    assert "12345678" in safe
    assert "private payload" not in safe


@pytest.mark.parametrize("suffix", ["Error", "Exception", "Warning", "Failure"])
def test_untrusted_traceback_cannot_encode_payload_in_exception_type(suffix):
    text = (
        "[whatsapp] Error handling message (error_type=ValueError)\n"
        "Traceback (most recent call last):\n"
        f"AlicePersonalMedicalDetail{suffix}: payload\n"
    )
    safe = debug._redact_log_text(text)
    assert "AlicePersonalMedicalDetail" not in safe
    assert "payload" not in safe
    assert "[REDACTED_EXCEPTION_TRACEBACK]" in safe


@pytest.mark.parametrize("platform", ["whatsapp", "whatsapp_cloud"])
def test_exception_opener_payload_is_removed_with_its_continuation(tmp_path, monkeypatch, platform):
    text = f"ordinary diagnostic\n[{platform}] Poll error: private-first-line\nprivate-continuation\n"
    path = tmp_path / "gateway.log"
    path.write_bytes(text.encode())
    monkeypatch.setattr(debug, "_resolve_log_path", lambda _name: path)
    projected = debug._redact_log_text(text)
    assert "private-first-line" not in projected
    assert "private-continuation" not in projected
    for tail_lines in (1, 20):
        snapshot = debug._capture_log_snapshot("gateway", tail_lines=tail_lines, redact=True)
        assert "private-first-line" not in snapshot.full_text
        assert "private-first-line" not in snapshot.tail_text
        assert "private-continuation" not in snapshot.full_text
        assert "private-continuation" not in snapshot.tail_text
    assert path.read_bytes() == text.encode()

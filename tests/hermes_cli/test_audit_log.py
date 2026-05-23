import json
import os
import stat
from datetime import datetime, timezone

from hermes_cli.audit_log import append_audit_event


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def test_append_audit_event_creates_private_jsonl_under_permissive_umask(tmp_path):
    audit_dir = tmp_path / "audit"
    fake_key = "sk-" + "test1234567890abcdef"
    old_umask = os.umask(0)
    try:
        path = append_audit_event(
            {
                "event_type": "approval.decision",
                "decision": "approved",
                "session_key": "private-session-key",
                "command": {
                    "preview": f"OPENAI_API_KEY={fake_key} rm -rf /tmp/x",
                },
                "request_body": "private prompt body",
            },
            audit_dir=audit_dir,
            now=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
        )
    finally:
        os.umask(old_umask)

    assert path == audit_dir / "2026-05-20.jsonl"
    assert _mode(audit_dir) == 0o700
    assert _mode(path) == 0o600

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["redaction"]["status"] == "redacted"
    assert payload["session_key"]["present"] is True
    assert payload["session_key"]["sha256_12"]
    text = path.read_text(encoding="utf-8")
    assert "private-session-key" not in text
    assert fake_key not in text
    assert "request_body" in text
    assert "private prompt body" not in text


def test_append_audit_event_tightens_existing_jsonl_mode(tmp_path):
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    path = audit_dir / "2026-05-20.jsonl"
    path.write_text("", encoding="utf-8")
    path.chmod(0o666)

    append_audit_event(
        {"event_type": "approval.decision", "decision": "denied"},
        audit_dir=audit_dir,
        now=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
    )

    assert _mode(path) == 0o600
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1

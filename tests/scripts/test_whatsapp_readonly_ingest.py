import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "whatsapp_readonly_ingest.py"
spec = importlib.util.spec_from_file_location("whatsapp_readonly_ingest", SCRIPT)
assert spec is not None
assert spec.loader is not None
readonly = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = readonly
spec.loader.exec_module(readonly)


def _args(**overrides):
    defaults = dict(
        config=None,
        bridge_url=None,
        output=None,
        group=None,
        group_policy=None,
        dm_policy=None,
        poll_interval=None,
        max_body_chars=None,
        include_raw=False,
        once=True,
        print_sample_config=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_config_requires_group_allowlist_by_default(tmp_path):
    cfg = tmp_path / "ingest.yaml"
    cfg.write_text("bridge_url: http://127.0.0.1:3000\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        readonly.build_config(_args(config=cfg))

    assert "group_allowlist" in str(exc.value)


def test_group_allowlist_filters_before_persistence(tmp_path):
    config = readonly.IngestConfig(
        output_path=tmp_path / "messages.jsonl",
        group_policy="allowlist",
        group_allowlist=frozenset({"allowed@g.us"}),
        dm_policy="disabled",
    )

    allowed = {"chatId": "allowed@g.us", "isGroup": True, "body": "keep"}
    blocked_group = {"chatId": "blocked@g.us", "isGroup": True, "body": "drop"}
    dm = {"chatId": "15551234567@s.whatsapp.net", "isGroup": False, "body": "drop dm"}

    assert readonly.message_allowed(allowed, config) is True
    assert readonly.message_allowed(blocked_group, config) is False
    assert readonly.message_allowed(dm, config) is False


def test_append_messages_writes_jsonl_with_private_permissions(tmp_path):
    out = tmp_path / "messages.jsonl"
    message = readonly.StoredMessage(
        ingested_at="2026-01-01T00:00:00+00:00",
        message_id="m1",
        chat_id="allowed@g.us",
        chat_name="Allowed Group",
        sender_id="user@s.whatsapp.net",
        sender_name="User",
        body="hello",
        timestamp=123,
        has_media=False,
        media_type=None,
        media_urls=[],
    )

    assert readonly.append_messages(out, [message]) == 1
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["chat_id"] == "allowed@g.us"
    assert rows[0]["body"] == "hello"
    assert oct(out.stat().st_mode & 0o777) == "0o600"


def test_fetch_bridge_messages_only_uses_get_messages(monkeypatch):
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'[{"chatId":"allowed@g.us","body":"hi"}]'

    def fake_urlopen(req, timeout):
        calls.append((req.full_url, req.get_method(), timeout))
        return FakeResponse()

    monkeypatch.setattr(readonly.urllib.request, "urlopen", fake_urlopen)

    messages = readonly.fetch_bridge_messages("http://127.0.0.1:3000", timeout=7)

    assert messages == [{"chatId": "allowed@g.us", "body": "hi"}]
    assert calls == [("http://127.0.0.1:3000/messages", "GET", 7)]


def test_sanitize_truncates_body_and_strips_sensitive_raw_keys():
    config = readonly.IngestConfig(
        group_policy="open",
        include_raw=True,
        max_body_chars=5,
    )
    stored = readonly.sanitize_message(
        {
            "messageId": "m1",
            "chatId": "allowed@g.us",
            "senderId": "u@s.whatsapp.net",
            "body": "0123456789",
            "botIds": ["secret"],
            "mentionedIds": ["secret"],
            "mediaUrls": ["/tmp/a.png"],
            "hasMedia": True,
            "mediaType": "image",
        },
        config,
    )

    assert stored.body == "01234…[truncated]"
    assert stored.raw is not None
    assert "botIds" not in stored.raw
    assert "mentionedIds" not in stored.raw

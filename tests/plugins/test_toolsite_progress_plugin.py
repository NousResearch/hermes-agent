from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4


PLUGIN_PATH = Path(__file__).resolve().parents[2] / "plugins" / "toolsite-progress" / "__init__.py"


def load_plugin(monkeypatch, tmp_path):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TOOL_SITE_REMOTE_STATE_DIR", str(state_dir))
    module_name = f"toolsite_progress_test_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, state_dir


def make_event(*, text="", message_id="42", media_urls=None, media_types=None, raw_message=None):
    return SimpleNamespace(
        text=text,
        source=SimpleNamespace(platform="telegram", chat_id="123"),
        message_id=message_id,
        media_urls=media_urls or [],
        media_types=media_types or [],
        raw_message=raw_message,
    )


def read_inbox(state_dir):
    inbox = state_dir / "toolsite-inbox.jsonl"
    return [json.loads(line) for line in inbox.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_photo_message_writes_attachment_metadata(monkeypatch, tmp_path):
    plugin, state_dir = load_plugin(monkeypatch, tmp_path)
    source_image = tmp_path / "source.jpg"
    source_image.write_bytes(b"\xff\xd8\xff\xe0" + (b"photo-bytes" * 100))
    raw_message = SimpleNamespace(
        caption="开始正式建站\n关键词：demo",
        photo=[SimpleNamespace(file_id="telegram-large-photo", width=1600, height=1200)],
        document=None,
    )
    event = make_event(
        text="开始正式建站\n关键词：demo",
        media_urls=[str(source_image)],
        media_types=["image/jpeg"],
        raw_message=raw_message,
    )

    result = plugin._rewrite_status(event)

    assert result["action"] == "rewrite"
    [record] = read_inbox(state_dir)
    assert record["text"].startswith("开始正式建站")
    assert record["attachments"][0]["kind"] == "image"
    assert record["attachments"][0]["telegram_file_id"] == "telegram-large-photo"
    assert record["attachments"][0]["mime_type"] == "image/jpeg"
    assert record["attachments"][0]["width"] == 1600
    assert record["attachments"][0]["height"] == 1200
    local_path = Path(record["attachments"][0]["local_path"])
    assert local_path.is_file()
    assert local_path.stat().st_size > 0


def test_text_and_photo_preserves_caption(monkeypatch, tmp_path):
    plugin, state_dir = load_plugin(monkeypatch, tmp_path)
    source_image = tmp_path / "captioned.webp"
    source_image.write_bytes(b"RIFFxxxxWEBP" + (b"photo-bytes" * 100))
    raw_message = SimpleNamespace(
        caption="参考我发的插画",
        photo=[SimpleNamespace(file_id="captioned-photo", width=800, height=600)],
        document=None,
    )
    event = make_event(
        text="参考我发的插画",
        message_id="43",
        media_urls=[str(source_image)],
        media_types=["image/webp"],
        raw_message=raw_message,
    )

    plugin._rewrite_status(event)

    [record] = read_inbox(state_dir)
    assert record["text"] == "参考我发的插画"
    assert record["attachments"][0]["telegram_file_id"] == "captioned-photo"
    assert record["attachments"][0]["mime_type"] == "image/webp"


def test_plain_text_message_keeps_existing_inbox_shape(monkeypatch, tmp_path):
    plugin, state_dir = load_plugin(monkeypatch, tmp_path)
    event = make_event(text="普通消息", message_id="44")

    plugin._rewrite_status(event)

    [record] = read_inbox(state_dir)
    assert record["text"] == "普通消息"
    assert "attachments" not in record

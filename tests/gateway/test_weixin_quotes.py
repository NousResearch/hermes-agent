"""ID-only quote recovery and bounded media ownership across profiles, accounts and peers."""

import hashlib
import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms import weixin
from gateway.platforms import weixin_quotes as quotes


@pytest.mark.asyncio
async def test_id_only_and_partial_quotes_survive_adapter_restart_and_stay_in_peer(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = PlatformConfig(enabled=True, extra={"account_id": "bot", "dm_policy": "allowlist", "allow_from": ["speaker", "other"]})
    adapter = weixin.WeixinAdapter(config)
    adapter._poll_session, adapter._token = Mock(), ""
    adapter._enqueue_text_event = Mock()
    await adapter._process_message({"from_user_id": "speaker", "item_list": [{
        "type": weixin.ITEM_TEXT, "msg_id": "18446744073709551615", "text_item": {"text": "甲[中文]乙[中文]丙"},
    }]})
    restored = weixin.WeixinAdapter(config)
    restored._poll_session, restored._token = Mock(), ""
    restored._enqueue_text_event = Mock()
    reference = {"svr_id": "18446744073709551615", "partial_text": {
        "start": "[", "end": "]", "startindex": 1, "endindex": 1,
        "quotemd5": hashlib.md5("[中文]".encode()).hexdigest(),
    }}
    message = {"from_user_id": "speaker", "message_id": "new", "item_list": [{
        "type": weixin.ITEM_TEXT, "text_item": {"text": "解释这一段"}, "ref_msg": reference,
    }]}
    await restored._process_message(message)
    event = restored._enqueue_text_event.call_args.args[0]
    assert event.reply_to_message_id == "18446744073709551615"
    assert event.text == "[引用: [中文]]\n解释这一段"
    await restored._process_message({**message, "from_user_id": "other", "message_id": "foreign"})
    assert "未缓存" in restored._enqueue_text_event.call_args.args[0].text
    assert quotes.WeixinQuoteStore(str(tmp_path), "other-bot").find("speaker", reference["svr_id"]) is None
    assert quotes.WeixinQuoteStore(str(tmp_path / "other-profile"), "bot").find("speaker", reference["svr_id"]) is None


def test_cache_owns_media_and_applies_text_media_and_size_limits(tmp_path, monkeypatch):
    clock = {"now": 1_000_000.0}
    monkeypatch.setattr(quotes.time, "time", lambda: clock["now"])
    source = tmp_path / "file.txt"
    source.write_bytes(b"1234")
    store = quotes.WeixinQuoteStore(str(tmp_path), "bot", {"max_messages_per_account": 2, "retention_days": 30,
                                  "media_retention_days": 1, "max_single_media_bytes": 5, "max_media_bytes_per_account": 5})
    store.put("peer", "1", "original", str(source), "text/plain", "file.txt")
    retained = Path(store.find("peer", "1")["media_path"])
    assert retained != source and retained.read_bytes() == b"1234"
    orphan = store.media_root / "orphan.txt"
    orphan.write_text("old copy", encoding="utf-8")
    os.utime(orphan, (clock["now"] - 301, clock["now"] - 301))
    store.sweep()
    assert not orphan.exists() and retained.exists()
    source.unlink()
    clock["now"] += 86401
    assert store.find("peer", "1")["media_path"] is None
    assert not retained.exists()
    source.write_bytes(b"123456")
    store.put("peer", "2", "oversized", str(source), "text/plain", "file.txt")
    assert store.find("peer", "2")["media_path"] is None
    clock["now"] += 1
    store.put("peer", "3", "newest")
    assert store.find("peer", "1") is None
    clock["now"] += 31 * 86400
    assert store.find("peer", "3") is None


@pytest.mark.asyncio
async def test_outbound_server_id_restores_quoted_media_after_source_is_removed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = weixin.WeixinAdapter(PlatformConfig(enabled=True, token="token", extra={"account_id": "bot"}))
    adapter._send_session = Mock()
    source = tmp_path / "report.txt"
    source.write_text("report", encoding="utf-8")
    monkeypatch.setattr(weixin, "_get_upload_url", AsyncMock(return_value={"upload_param": "test"}))
    monkeypatch.setattr(weixin, "_upload_ciphertext", AsyncMock(return_value="download-query"))
    monkeypatch.setattr(weixin, "_send_items", AsyncMock(return_value={"ret": 0, "message_id": 18446744073709551615}))
    await adapter._send_file("peer", str(source), "")
    source.unlink()
    paths, types = [], []
    text, identifier = await quotes.resolve_quote(adapter, [{"ref_msg": {"svr_id": "18446744073709551615"}}], "peer", "分析附件", paths, types)
    assert identifier == "18446744073709551615" and "report.txt" in text
    assert Path(paths[0]).read_text(encoding="utf-8") == "report"
    assert types == ["text/plain"]


def test_disabled_and_broken_caches_do_not_block_delivery(tmp_path):
    store = quotes.WeixinQuoteStore(str(tmp_path), "bot", {"enabled": False})
    store.put("peer", "id", "text")
    assert not store.db_path.exists()
    store.enabled = True
    store.root.mkdir()
    store.db_path.write_bytes(b"not a database")
    store.put("peer", "id", "text")
    assert not store.enabled and store.find("peer", "id") is None

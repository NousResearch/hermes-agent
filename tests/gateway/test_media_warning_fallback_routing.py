"""Salt R3 B4/S1/S3: refactor regressions reproduced as behavior contracts."""
import json
from unittest.mock import AsyncMock, MagicMock
import pytest
from gateway.config import Platform


def _cfg(tmp_path, monkeypatch, setting):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path)); monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    cfg = {} if setting is None else {"display": {"suppress_warning_notifications": setting}}
    (tmp_path / "config.yaml").write_text(json.dumps(cfg))


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", [None, False])
async def test_matrix_shown_fallback_routing_is_legacy_bytes(tmp_path, monkeypatch, setting):
    """S1: legacy shown fallback passed NO metadata; the shared helper must not add thread routing."""
    _cfg(tmp_path, monkeypatch, setting)
    from gateway.platforms.base import BasePlatformAdapter, SendResult
    calls = []
    class A(BasePlatformAdapter):
        name = "matrix"; platform = Platform.MATRIX
        def __init__(self): pass
        async def connect(self): pass
        async def disconnect(self): pass
        async def get_chat_info(self, chat_id): return {}
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            calls.append((content, reply_to, metadata)); return SendResult(success=True)
    a = A()
    await a.emit_media_warning("!r", "⚠️ Couldn't deliver the attachment.", caption="cap", reply_to="$e", metadata={"thread_id": "$t"})
    assert calls == [("cap\n⚠️ Couldn't deliver the attachment.", "$e", None)]

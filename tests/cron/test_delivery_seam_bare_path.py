"""RED→GREEN regression for the scheduler delivery-seam tuple-type defect.

The merge loop in ``_deliver_result`` (cron/scheduler.py ~line 1638) appends
bare local file paths (strings, from ``extract_local_files``) into a list of
``(path, is_voice)`` tuples (from ``extract_media``). Downstream
``_send_media_via_adapter`` → ``filter_media_delivery_paths`` unpacks each
entry as ``for media_path, is_voice in media_files:`` →
``ValueError: too many values to unpack (expected 2)`` on ANY bare local file
path (e.g. cron HTML reports emitted as plain paths).

This test exercises the GENUINE delivery seam end-to-end by calling the real
``_deliver_result`` (the production function that owns the merge loop) with a
live stub adapter + running fake event loop — the exact path that reaches
``_send_media_via_adapter``. The crash fires inside
``filter_media_delivery_paths`` (the unpack loop) during the media-attachment
send, so the test surfaces the real ``ValueError`` against the real merged
list (not a hand-rolled replica).

We test ``_deliver_result`` (not ``_send_media_via_adapter`` directly) so the
PRODUCTION merge loop at line 1638 is the code under test — the fix lands
there and the test verifies the fix end-to-end through the real call path.

Harness reuses the same pattern as the existing
``TestDeliverResultWrapping.test_live_adapter_sends_media_as_attachments``:
- ``MEDIA_DELIVERY_SAFE_ROOTS`` monkey-patched to the tmp_path so the real
  path validators accept the test files.
- A fake event loop via ``fake_run_coro`` so ``safe_schedule_threadsafe``
  dispatches the stub-adapter coroutines synchronously.
"""
import asyncio
import sys
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cron.scheduler import _deliver_result  # noqa: E402


def _safe_media_path(tmp_path, monkeypatch, name, data=b"<html></html>"):
    """Create a real on-disk file under a safe-root tmp dir and register that
    dir as the sole MEDIA_DELIVERY_SAFE_ROOT so the production path validators
    accept it. Mirrors TestDeliverResultWrapping._safe_media_path."""
    root = tmp_path / "media-cache"
    media_file = root / name
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(data)
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
        (root,),
    )
    return media_file.resolve()


def _fake_run_coro(coro, _loop):
    """Run the routed coroutine synchronously and wrap the result in a
    completed Future (matches asyncio.run_coroutine_threadsafe's contract)."""
    future = Future()
    try:
        future.set_result(asyncio.run(coro))
    except BaseException as e:  # noqa: BLE001
        future.set_exception(e)
    return future


def _deliver_with_live_adapter(tmp_path, monkeypatch, content, adapter):
    """Call the real ``_deliver_result`` with a live adapter + running loop so
    the production merge loop runs and media reaches ``_send_media_via_adapter``.

    Returns the stub adapter so the caller can inspect which send methods were
    called (and confirm no ValueError leaked).
    """
    from gateway.config import Platform

    pconfig = MagicMock()
    pconfig.enabled = True
    mock_cfg = MagicMock()
    mock_cfg.platforms = {Platform.DISCORD: pconfig}

    loop = MagicMock()
    loop.is_running.return_value = True

    job = {
        "id": "seam-job",
        "deliver": "origin",
        "origin": {"platform": "discord", "chat_id": "9876"},
    }

    with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("asyncio.run_coroutine_threadsafe", side_effect=_fake_run_coro):
        _deliver_result(
            job,
            content,
            adapters={Platform.DISCORD: adapter},
            loop=loop,
        )
    return adapter


# ---------------------------------------------------------------------------
# (a) bare local path only (no MEDIA: tag) — the reported crash path
# ---------------------------------------------------------------------------
def test_bare_local_path_only(tmp_path, monkeypatch):
    html = _safe_media_path(tmp_path, monkeypatch, "report.html", b"<html><body>report</body></html>")
    adapter = AsyncMock()
    adapter.send.return_value = MagicMock(success=True)
    adapter.send_document.return_value = MagicMock(success=True)
    # Must not raise ValueError from the unpack loop.
    _deliver_with_live_adapter(tmp_path, monkeypatch, f"Daily review ready.\nFull report: {html}", adapter)
    adapter.send.assert_called_once()
    # Document now routes through the generic batch method (send_multiple_documents).
    adapter.send_multiple_documents.assert_called_once()


# ---------------------------------------------------------------------------
# (b) explicit MEDIA:/path only — the already-working path
# ---------------------------------------------------------------------------
def test_explicit_media_tag_only(tmp_path, monkeypatch):
    html = _safe_media_path(tmp_path, monkeypatch, "explicit.html", b"<html><body>explicit</body></html>")
    adapter = AsyncMock()
    adapter.send.return_value = MagicMock(success=True)
    adapter.send_document.return_value = MagicMock(success=True)
    _deliver_with_live_adapter(tmp_path, monkeypatch, f"Summary.\nMEDIA:{html}", adapter)
    adapter.send.assert_called_once()
    adapter.send_multiple_documents.assert_called_once()


# ---------------------------------------------------------------------------
# (c) mixed explicit MEDIA:/path + a bare /path in the same content
# ---------------------------------------------------------------------------
def test_mixed_media_tag_and_bare_path(tmp_path, monkeypatch):
    html_a = _safe_media_path(tmp_path, monkeypatch, "tagged.html", b"<html><body>tagged</body></html>")
    html_b = _safe_media_path(tmp_path, monkeypatch, "bare.html", b"<html><body>bare</body></html>")
    adapter = AsyncMock()
    adapter.send.return_value = MagicMock(success=True)
    adapter.send_document.return_value = MagicMock(success=True)
    _deliver_with_live_adapter(
        tmp_path, monkeypatch,
        f"Mixed delivery.\nMEDIA:{html_a}\nAlso see: {html_b}",
        adapter,
    )
    adapter.send.assert_called_once()
    # Both files must be delivered in one batch (no crash, no drop).
    adapter.send_multiple_documents.assert_called_once()

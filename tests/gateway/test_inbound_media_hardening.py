"""Hardening for the inbound-media cache (#142498 review: B1, S1–S6).

Each test pins one finding from the review of ``75014c948a``:

* B1 — the WhatsApp skip line printed a live capability URL (``oe``/``oh`` signature).
* S3 — a remote 300-char name reached the filesystem and failed with ENAMETOOLONG.
* S4 — one ``downloadCode`` appearing twice was downloaded and written twice.
* S5 — a relative entry was appended unchecked and later read as a file.
* S6 — a hostile ``ext`` escaped the cache dir through ``_write_cache_file``.
* S2 — an over-budget message was rejected wholesale by ingress.
"""

import sys
from types import SimpleNamespace

import pytest

import gateway.platforms.base as base
from gateway.platforms.event import MessageType


# --- B1 / S5: WhatsApp bridge entries ------------------------------------------------


def _whatsapp_adapter():
    from gateway.platforms.base import Platform
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    return adapter


@pytest.mark.asyncio
async def test_the_skip_line_does_not_print_the_capability_url(capsys):
    """A WhatsApp CDN URL is a credential: oe (expiry) + oh (HMAC) must not reach stdout."""
    adapter = _whatsapp_adapter()
    url = "https://mmg.whatsapp.net/v/t62/report.pdf?oe=SECRETEXP&oh=SECRETHMAC"

    urls, _ = await adapter._collect_bridge_media({"mediaUrls": [url]}, MessageType.DOCUMENT)

    printed = capsys.readouterr().out
    assert urls == []
    assert "SECRETHMAC" not in printed and "SECRETEXP" not in printed


@pytest.mark.asyncio
async def test_a_relative_entry_is_never_accepted(monkeypatch, tmp_path):
    """Only an absolute path inside a Hermes cache dir may reach media_urls."""
    adapter = _whatsapp_adapter()
    monkeypatch.setattr("plugins.platforms.whatsapp.adapter._is_allowed_bridge_path", lambda p: False)

    urls, _ = await adapter._collect_bridge_media(
        {"mediaUrls": ["../../etc/passwd", "report.pdf"]}, MessageType.DOCUMENT
    )

    assert urls == []


# --- S6: cache file extension ---------------------------------------------------------


def test_a_hostile_cache_extension_cannot_escape_the_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(base, "get_image_cache_dir", lambda: tmp_path)
    written = base._write_cache_file(tmp_path, "img", "/../../../evil.jpg", b"x")
    path = base.Path(written)

    assert path.parent.resolve() == tmp_path.resolve(), written
    assert path.name.count(".") == 1 and path.suffix == ".jpg", written


def test_a_plain_extension_still_writes(monkeypatch, tmp_path):
    written = base._write_cache_file(tmp_path, "img", ".jpg", b"x")
    assert written.endswith(".jpg")
    assert (tmp_path / written.split("\\")[-1].split("/")[-1]).exists()


# --- S3: long / Windows-hostile document names ---------------------------------------


def test_a_very_long_document_name_is_capped(monkeypatch, tmp_path):
    monkeypatch.setattr(base, "get_document_cache_dir", lambda: tmp_path)
    written = base.cache_document_from_bytes(b"x", "a" * 300 + ".pdf")

    name = written.replace("\\", "/").rsplit("/", 1)[-1]
    assert len(name) <= 96, name
    assert name.endswith(".pdf")
    assert (tmp_path / name).exists()


@pytest.mark.platforms("windows")
def test_a_colon_in_the_name_is_replaced_on_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(base, "get_document_cache_dir", lambda: tmp_path)
    written = base.cache_document_from_bytes(b"x", "ads.txt:evil.exe")

    name = written.replace("\\", "/").rsplit("/", 1)[-1]
    assert ":" not in name
    assert (tmp_path / name).exists()


# --- S4: one code, one download -------------------------------------------------------


def test_one_download_code_is_collected_once():
    from plugins.platforms.dingtalk.inbound import collect_download_codes

    message = SimpleNamespace(
        rich_text=[{"type": "picture", "downloadCode": "CODE1", "pictureDownloadCode": "CODE1"}],
        message_type="file",
        extensions={"content": {"downloadCode": "CODE1", "fileName": "a.pdf"}},
    )

    pairs = collect_download_codes(message)

    assert [key for _obj, key, _kind in pairs] == ["downloadCode"]


# --- S2: per-message media budget -----------------------------------------------------


def test_the_media_budget_sheds_attachments_instead_of_the_message(monkeypatch, tmp_path):
    from plugins.platforms.dingtalk.adapter import DingTalkAdapter

    monkeypatch.setattr(base, "get_inbound_media_max_bytes", lambda: 10)
    first, second = tmp_path / "a.bin", tmp_path / "b.bin"
    first.write_bytes(b"0123456789")
    second.write_bytes(b"0123456789")
    items = [{"downloadCode": str(first)}, {"downloadCode": str(second)}]

    class _Adapter:
        name = "DingTalk"
        _write_media_ref = staticmethod(DingTalkAdapter._write_media_ref)

    DingTalkAdapter._enforce_media_budget(_Adapter(), [(i, "downloadCode", "file") for i in items])

    assert items[0]["downloadCode"] == str(first)
    assert items[1]["downloadCode"] == ""

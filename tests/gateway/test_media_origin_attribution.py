"""Tests for sender-attribution injection into vision / transcription enrichment.

When the gateway auto-describes a user image or transcribes a voice message,
the resulting enrichment text now includes a "from <sender> at HH:MM" tag
read from the JSON sidecar that lives next to the cached media. This lets
the agent know *who* sent the image/voice in shared-session groups (and
even in DMs, gives an unambiguous timestamp).

The tag is read from the sidecar written by gateway/platforms/telegram.py's
_record_media_sender at cache time.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gateway.run import _describe_media_origin


def _write_image_with_sidecar(tmp_path: Path, *, basename: str, **sidecar_fields):
    img_path = tmp_path / basename
    img_path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    sidecar_path = tmp_path / f"{basename}.json"
    sidecar_path.write_text(json.dumps(sidecar_fields, ensure_ascii=False))
    return str(img_path)


def test_describe_media_origin_returns_name_and_time(tmp_path):
    """Reading a sidecar with from_name and date yields a "from X at HH:MM" tag."""
    img = _write_image_with_sidecar(
        tmp_path,
        basename="img_abc.jpg",
        from_name="子家",
        from_id=8682281996,
        date="2026-04-28T06:04:02+00:00",
        platform="telegram",
        chat_type="supergroup",
    )

    out = _describe_media_origin(img)
    assert "子家" in out
    # date should render as local/explicit HH:MM, not the raw ISO blob.
    assert "T" not in out
    # The original ISO must not leak verbatim.
    assert "2026-04-28T06:04:02" not in out


def test_describe_media_origin_handles_missing_sidecar(tmp_path):
    """An image with no sidecar yields an empty string (not an exception)."""
    img = tmp_path / "img_naked.jpg"
    img.write_bytes(b"\xff\xd8\xffno-sidecar")
    assert _describe_media_origin(str(img)) == ""


def test_describe_media_origin_handles_only_name(tmp_path):
    """If the sidecar only has from_name (no date), still attribute by name."""
    img = _write_image_with_sidecar(
        tmp_path,
        basename="img_name_only.jpg",
        from_name="玉青",
    )
    out = _describe_media_origin(img)
    assert "玉青" in out


def test_describe_media_origin_handles_only_date(tmp_path):
    """If the sidecar only has date (no name), still attribute by time."""
    img = _write_image_with_sidecar(
        tmp_path,
        basename="img_date_only.jpg",
        date="2026-04-28T06:04:02+00:00",
    )
    out = _describe_media_origin(img)
    # Some HH:MM-ish marker should appear.
    assert ":" in out


def test_describe_media_origin_handles_unix_timestamp_date(tmp_path):
    """Sidecars sometimes store date as a unix timestamp (int/float)."""
    img = _write_image_with_sidecar(
        tmp_path,
        basename="img_unix.jpg",
        from_name="玉青",
        date=1714286642,
    )
    out = _describe_media_origin(img)
    assert "玉青" in out
    assert ":" in out


def test_describe_media_origin_handles_corrupt_sidecar(tmp_path):
    """A corrupt JSON sidecar must not crash — return empty."""
    img = tmp_path / "img_corrupt.jpg"
    img.write_bytes(b"\xff\xd8\xfffake")
    (tmp_path / "img_corrupt.jpg.json").write_text("{not valid json")
    assert _describe_media_origin(str(img)) == ""


def test_describe_media_origin_marks_quoted_reply(tmp_path):
    """A sidecar with is_quoted_reply=True hints the image was the *original*
    media being replied to, so the description should make that clear."""
    img = _write_image_with_sidecar(
        tmp_path,
        basename="img_reply.jpg",
        from_name="子家",
        date="2026-04-28T06:04:02+00:00",
        is_quoted_reply=True,
    )
    out = _describe_media_origin(img)
    assert "子家" in out
    # Originality hint: "originally" is the most natural English marker.
    assert "originally" in out.lower() or "原" in out


# ---------------------------------------------------------------------------
# End-to-end: enrichment functions surface origin tag into prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_message_with_vision_includes_origin_tag(tmp_path, monkeypatch):
    """The text the agent ultimately sees must mention the sender from the
    sidecar, so it can answer "who sent this image?" without an extra tool
    call."""
    from gateway.run import GatewayRunner

    img = _write_image_with_sidecar(
        tmp_path,
        basename="img_e2e.jpg",
        from_name="玉青",
        date="2026-04-28T06:04:02+00:00",
        platform="telegram",
    )

    async def _fake_vision(image_url, user_prompt):
        return json.dumps({"success": True, "analysis": "A bowl of rice noodles."})

    monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _fake_vision)

    runner = object.__new__(GatewayRunner)
    out = await runner._enrich_message_with_vision("", [img])
    assert "玉青" in out
    assert "rice noodles" in out
    # And the sender attribution should appear in the leading bracket, not
    # buried somewhere inside the description.
    head = out.split("\n", 1)[0]
    assert "玉青" in head


@pytest.mark.asyncio
async def test_enrich_message_with_vision_works_without_sidecar(tmp_path, monkeypatch):
    """Images without a sidecar (e.g. legacy cached images) must still go
    through the vision pipeline, just without the origin tag."""
    from gateway.run import GatewayRunner

    img = tmp_path / "img_no_sidecar.jpg"
    img.write_bytes(b"\xff\xd8\xfffake")

    async def _fake_vision(image_url, user_prompt):
        return json.dumps({"success": True, "analysis": "Some image."})

    monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _fake_vision)

    runner = object.__new__(GatewayRunner)
    out = await runner._enrich_message_with_vision("", [str(img)])
    assert "Some image" in out
    # No "(from ...)" tag because there's no sidecar.
    assert "(from " not in out

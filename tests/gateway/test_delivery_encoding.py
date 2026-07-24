"""Regression: cron delivery output is written as UTF-8, not the platform default.

``DeliveryRouter._save_full_output`` and ``_save_to_file`` persist cron job
output (and the human-readable .md mirror) to disk. Both used
``Path.write_text(content)`` with no ``encoding=``. ``Path.write_text``'s
default encoding is ``locale.getpreferredencoding(False)`` — on Windows that
is the ANSI code page (cp1252), which cannot represent emoji / CJK / many
accented characters. A cron job whose output contains "✅ café 你好 🚀" therefore
raised ``UnicodeEncodeError`` (delivery lost) or silently mojibake'd on Windows,
while working fine on macOS/Linux where the locale default is UTF-8.

These tests assert the invariant directly: whatever the process locale, the
bytes on disk must be the UTF-8 encoding of the content and must round-trip.
They fail on the pre-fix code under a non-UTF-8 locale and pass after adding
``encoding="utf-8"`` to both write sites.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import patch

import pytest

import gateway.delivery as delivery


# A payload that is unrepresentable in cp1252 (Windows ANSI default):
#   ✅ 🚀 → emoji (no cp1252 mapping)
#   你好  → CJK (no cp1252 mapping)
#   café résumé naïve → Latin-1-ish but exercises non-ASCII
#   Москва → Cyrillic (no cp1252 mapping)
NON_ASCII = "Daily brief ✅ — café résumé 你好 🚀 naïve Москва"


def _router() -> delivery.DeliveryRouter:
    # DeliveryRouter is constructed by the gateway with a config object; for
    # these unit tests we only exercise the two pure disk-write helpers, which
    # do not touch config. Build the instance without running __init__ side
    # effects we don't need.
    return delivery.DeliveryRouter.__new__(delivery.DeliveryRouter)


def test_save_full_output_writes_utf8(tmp_path: Path) -> None:
    router = _router()
    fake_home = tmp_path / "hermes_home"
    with patch.object(delivery, "get_hermes_home", return_value=fake_home):
        out_path = router._save_full_output(NON_ASCII, job_id="job_emoji")

    # The bytes on disk must be exactly the UTF-8 encoding of the content,
    # regardless of the process locale (this is the contract the fix enforces).
    assert out_path.read_bytes() == NON_ASCII.encode("utf-8")
    # And it must round-trip when read back as UTF-8.
    assert out_path.read_text(encoding="utf-8") == NON_ASCII


def test_deliver_local_writes_utf8(tmp_path: Path) -> None:
    router = _router()
    router.output_dir = tmp_path / "out"
    result = router._deliver_local(
        content=NON_ASCII,
        job_id="job_emoji",
        job_name="emoji job",
        metadata={"note": "café ☕"},
    )
    saved = Path(result["path"])
    # The content line and the non-ASCII metadata must survive as UTF-8.
    data = saved.read_bytes().decode("utf-8")
    assert NON_ASCII in data
    assert "café ☕" in data


def test_writes_pass_explicit_utf8_not_locale_default(tmp_path: Path, monkeypatch) -> None:
    """Delivery must pass an explicit ``encoding="utf-8"`` to every write.

    The Windows corruption bug comes from ``Path.write_text(content)`` falling
    back to ``locale.getpreferredencoding(False)`` (cp1252 on Windows ANSI).
    CPython's C-level ``open()`` reads the *interpreter* locale, not the Python
    ``locale`` module, so simulating Windows by monkeypatching ``locale`` is a
    false test (it silently passes on a UTF-8 host). We instead assert the
    actual contract the fix establishes: the production code opens its output
    files with an explicit ``encoding="utf-8"``, so the on-disk bytes never
    depend on the host locale. This is genuinely red on the pre-fix code
    (captured encoding is ``None``) and green after.
    """
    captured: list[str | None] = []
    original = Path.write_text

    def _recording_write_text(self, data, encoding=None, *args, **kwargs):
        captured.append(encoding)
        return original(self, data, encoding=encoding, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _recording_write_text)

    router = _router()
    router.output_dir = tmp_path / "out"
    fake_home = tmp_path / "hermes_home"
    with patch.object(delivery, "get_hermes_home", return_value=fake_home):
        router._save_full_output(NON_ASCII, job_id="job_locale")
    router._deliver_local(
        content=NON_ASCII, job_id="job_locale", job_name="j", metadata=None
    )

    assert captured, "expected delivery to write at least one output file"
    # Every write must name utf-8 explicitly — never rely on the locale default.
    assert all(enc == "utf-8" for enc in captured), (
        f"delivery wrote with non-utf-8 encoding(s): {captured!r}"
    )

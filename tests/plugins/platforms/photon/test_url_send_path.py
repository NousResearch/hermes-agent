"""Regression tests for Photon raw-URL message delivery.

The iMessage markdown builder enables data detection inside spectrum-ts. On
some IMAgentKit sends, that path returns a 500 when the message contains a raw
URL. The sidecar should keep markdown rendering for URL-free messages, but use
plain text for messages containing URLs so iMessage can auto-link them without
hitting the data-detection failure path.
"""

from __future__ import annotations

import re
from pathlib import Path


SIDECAR = Path("plugins/platforms/photon/sidecar/index.mjs")


def _source() -> str:
    return SIDECAR.read_text(encoding="utf-8")


def test_photon_sidecar_detects_raw_urls_before_builder_selection() -> None:
    src = _source()

    assert "hasUrl" in src
    assert "https?:\\/\\/" in src
    assert re.search(r"const\s+hasUrl\s*=\s*/https\?:", src)


def test_markdown_builder_is_used_only_for_url_free_markdown() -> None:
    src = _source()

    assert re.search(
        r'format\s*===\s*["\']markdown["\']\s*&&\s*!hasUrl',
        src,
    )
    assert "spectrumMarkdown(text)" in src


def test_text_builder_is_fallback_for_markdown_messages_with_urls() -> None:
    src = _source()

    # Builder selection must still have spectrumText as the fallback branch so
    # markdown+URL messages avoid spectrumMarkdown's data-detection path.
    assert "spectrumText(text)" in src
    assert re.search(
        r'format\s*===\s*["\']markdown["\']\s*&&\s*!hasUrl[\s\S]*\?[\s\S]*spectrumMarkdown\(text\)[\s\S]*:[\s\S]*spectrumText\(text\)',
        src,
    )

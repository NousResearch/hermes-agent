"""Regression coverage for BasePlatformAdapter.extract_media() — the PRODUCTION
extractor (not the local reimplementation in test_media_extraction.py).

Guards the curated _DELIVERY_MEDIA_EXTS contract: every supported extension is
extracted; markdown-bold/quoted/uppercase/spaced paths are handled; tags are
stripped from cleaned text; excluded source/config extensions are NOT matched;
the MEDIA: literal stays case-sensitive; results are deduplicated.
"""

import pytest

from gateway.platforms.base import (
    BasePlatformAdapter,
    _DELIVERY_MEDIA_EXTS,
)


def _paths(content):
    media, _cleaned = BasePlatformAdapter.extract_media(content)
    return [p for p, _voice in media]


@pytest.mark.parametrize("ext", sorted(_DELIVERY_MEDIA_EXTS))
def test_every_delivery_extension_is_extracted(ext):
    content = f"Here you go: MEDIA:/tmp/output{ext}"
    assert _paths(content) == [f"/tmp/output{ext}"]


def test_markdown_extensions_extracted():
    # The core regression: .md and .html were absent from the old allowlist.
    assert _paths("MEDIA:/tmp/notes.md") == ["/tmp/notes.md"]
    assert _paths("MEDIA:/tmp/report.html") == ["/tmp/report.html"]


def test_uppercase_extension_extracted():
    assert _paths("MEDIA:/tmp/IMG.PNG") == ["/tmp/IMG.PNG"]
    assert _paths("MEDIA:/tmp/Mixed.JpG") == ["/tmp/Mixed.JpG"]


def test_markdown_bold_wrapped_path_extracted_and_cleaned():
    media, cleaned = BasePlatformAdapter.extract_media("**MEDIA:/tmp/x.png**")
    assert [p for p, _ in media] == ["/tmp/x.png"]
    assert "*" not in cleaned
    assert "MEDIA:" not in cleaned


def test_quoted_path_with_space_extracted():
    media, _ = BasePlatformAdapter.extract_media('MEDIA:"/tmp/My Report.pdf"')
    assert [p for p, _ in media] == ["/tmp/My Report.pdf"]


def test_bare_spaced_path_extracted():
    media, _ = BasePlatformAdapter.extract_media("MEDIA:/tmp/My Report.pdf done")
    assert [p for p, _ in media] == ["/tmp/My Report.pdf"]


def test_trailing_prose_not_consumed():
    media, _ = BasePlatformAdapter.extract_media("MEDIA:/tmp/a.png and then I did stuff")
    assert [p for p, _ in media] == ["/tmp/a.png"]


def test_duplicate_paths_deduplicated():
    media, _ = BasePlatformAdapter.extract_media("MEDIA:/tmp/a.png MEDIA:/tmp/a.png")
    assert [p for p, _ in media] == ["/tmp/a.png"]


def test_tag_stripped_from_cleaned_text():
    _media, cleaned = BasePlatformAdapter.extract_media("Done. MEDIA:/tmp/a.png")
    assert "MEDIA:" not in cleaned
    assert cleaned.strip() == "Done."


@pytest.mark.parametrize("ext", [".py", ".sh", ".ts", ".toml", ".ini", ".cfg", ".log"])
def test_source_and_config_extensions_not_extracted(ext):
    assert _paths(f"MEDIA:/tmp/script{ext}") == []


def test_media_literal_is_case_sensitive():
    # Lowercase `media:` must NOT trigger extraction (avoids prose false-positives).
    assert _paths("see the media:/tmp/a.png reference") == []


def test_extract_local_files_uses_shared_extension_set(tmp_path):
    # Bare-path delivery must honor the SAME extensions as MEDIA: tags.
    f = tmp_path / "report.md"
    f.write_text("hi")
    paths, _cleaned = BasePlatformAdapter.extract_local_files(f"see {f}")
    assert paths == [str(f)]

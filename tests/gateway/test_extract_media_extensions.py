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


def test_tool_media_re_matches_broadened_extensions():
    from gateway.platforms.base import _TOOL_MEDIA_RE
    content = '{"success": true, "media_tag": "MEDIA:/tmp/notes.md"}'
    paths = [m.group(1).rstrip('",}') for m in _TOOL_MEDIA_RE.finditer(content)]
    assert paths == ["/tmp/notes.md"]

    upper = '{"media_tag": "MEDIA:/tmp/IMG.PNG"}'
    assert [m.group(1).rstrip('",}') for m in _TOOL_MEDIA_RE.finditer(upper)] == ["/tmp/IMG.PNG"]


@pytest.mark.parametrize("ext", [".kml", ".kmz", ".geojson", ".gpx"])
def test_gis_extensions_extracted(ext):
    # GIS/geospatial deliverables requested in issue #24032 (route to
    # send_document). Includes the spaced-path repro from that issue.
    assert _paths(f"MEDIA:/tmp/track{ext}") == [f"/tmp/track{ext}"]
    media, _ = BasePlatformAdapter.extract_media(f"MEDIA:/home/user/My Folder/coords{ext}")
    assert [p for p, _v in media] == [f"/home/user/My Folder/coords{ext}"]


# --- Windows path recognition (#28989, #24032) -----------------------------
# Recognition only: Windows drive-letter / UNC MEDIA: tags are parsed so they
# are STRIPPED from user-visible text (no raw C:\ leak) and routed through the
# delivery gate. Safe Windows *delivery* is deferred to the L0 path-validation
# security PR; validate_media_delivery_path rejects them (see test_platform_base).

@pytest.mark.parametrize("path", [
    r"C:\Users\foo\report.pdf",
    "C:/Users/foo/report.pdf",
    r"C:\Users\Foo\My Folder\report.pdf",  # spaced Windows path (OneDrive repro)
    r"D:\data\track.gpx",
])
def test_windows_media_tag_recognized_and_stripped(path):
    media, cleaned = BasePlatformAdapter.extract_media(f"Here you go: MEDIA:{path}")
    assert [p for p, _v in media] == [path]
    assert "MEDIA:" not in cleaned
    assert "C:" not in cleaned and "D:" not in cleaned  # raw path no longer leaks


def test_windows_bold_media_tag_stripped():
    media, cleaned = BasePlatformAdapter.extract_media(r"**MEDIA:C:\out\img.png**")
    assert [p for p, _v in media] == [r"C:\out\img.png"]
    assert "MEDIA:" not in cleaned
    assert "*" not in cleaned


@pytest.mark.parametrize("path", [
    r"\\server\share\report.pdf",
    r"\\host\My Share\doc.pdf",  # spaced UNC
])
def test_windows_unc_media_tag_recognized_and_stripped(path):
    # UNC paths must be recognized (and stripped) too — the validate gate
    # fail-closes their delivery, but leaving them unrecognized would leak the
    # raw path into the user's message (the #28989/#24032 symptom).
    media, cleaned = BasePlatformAdapter.extract_media(f"Here: MEDIA:{path}")
    assert [p for p, _v in media] == [path]
    assert "MEDIA:" not in cleaned
    assert "\\" not in cleaned  # no raw backslash path remains


def test_tool_media_re_recognizes_windows_paths():
    from gateway.platforms.base import _TOOL_MEDIA_RE
    for content, expected in [
        (r'{"media_tag": "MEDIA:C:\out\img.png"}', r"C:\out\img.png"),
        (r'{"media_tag": "MEDIA:\\srv\share\f.pdf"}', r"\\srv\share\f.pdf"),
    ]:
        got = [m.group(1).rstrip('",}') for m in _TOOL_MEDIA_RE.finditer(content)]
        assert got == [expected]

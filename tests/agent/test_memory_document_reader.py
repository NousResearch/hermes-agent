"""Real local-file evidence and authorization boundaries for the document handler."""

import hashlib
from pathlib import Path

import pytest

from agent.memory_reasoning.document_reader import LocalDocumentGrant, ScopedDocumentEvidenceReader
from agent.memory_reasoning.runner import ToolBudgetExceeded


def reader(root: Path, path: Path, **grant_options) -> ScopedDocumentEvidenceReader:
    return ScopedDocumentEvidenceReader(
        grants={"doc:approved": LocalDocumentGrant(root=root, path=path, **grant_options)},
        max_bytes=32, max_lines=3,
    )


def test_utf8_full_read_and_new_version_changes_fingerprint(tmp_path):
    path = tmp_path / "note.md"
    original = "第一行\n第二行\n"
    path.write_bytes(original.encode("utf-8"))
    calls = []
    approved = reader(tmp_path, path)
    first = approved.read("doc:approved", lambda: calls.append("meter"))
    assert calls == ["meter"]
    assert first.status == "complete"
    assert first.content["text"] == original
    assert first.digest == "sha256:" + hashlib.sha256(original.encode()).hexdigest()
    assert first.content["fingerprint_scope"] == "full-file"
    assert first.content["byte_end"] == len(original.encode())
    assert first.content["file_id"] and first.content["version"]
    assert "lines:1-2" in first.location
    assert first.read_at.endswith("+00:00")

    path.write_bytes("新的一行\n第二行\n".encode("utf-8"))
    second = approved.read("doc:approved", lambda: calls.append("meter"))
    assert second.status == "complete"
    assert second.digest != first.digest
    assert second.content["version"] != first.content["version"]
    assert calls == ["meter", "meter"]

    pinned = reader(tmp_path, path, expected_version=first.content["version"])
    mismatch = pinned.read("doc:approved", lambda: None)
    assert mismatch.status == "error"
    assert mismatch.content["error_type"] == "VersionChanged"


def test_bounded_window_and_long_utf8_line_are_partial(tmp_path):
    path = tmp_path / "long.txt"
    path.write_bytes(("skip\n中文窗口\n" + "x" * 100000).encode("utf-8"))
    window = reader(tmp_path, path, start_line=2, end_line=2).read("doc:approved", lambda: None)
    assert window.status == "truncated"
    assert window.content["text"] == "中文窗口\n"
    assert window.content["fingerprint_scope"] == "window"
    assert window.content["byte_start"] == len("skip\n".encode())
    assert window.content["file_bytes"] > 100000
    assert "window" in window.coverage
    assert window.digest != "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    # The bound falls inside a Chinese code point; expose only valid UTF-8.
    path.write_text("中" * 100, encoding="utf-8")
    tiny = ScopedDocumentEvidenceReader(
        grants={"r": LocalDocumentGrant(tmp_path, path)}, max_bytes=4, max_lines=1,
    ).read("r", lambda: None)
    assert tiny.status == "truncated"
    assert tiny.content["text"] == "中"
    assert tiny.content["byte_end"] == 3
    assert tiny.content["fingerprint_scope"] == "window"


def test_missing_unsupported_directory_and_invalid_utf8_are_distinct(tmp_path):
    missing = reader(tmp_path, tmp_path / "gone.txt").read("doc:approved", lambda: None)
    assert missing.status == "missing"
    assert missing.content["error_type"] == "FileNotFoundError"

    pdf = tmp_path / "file.pdf"
    pdf.write_bytes(b"%PDF")
    unsupported = reader(tmp_path, pdf).read("doc:approved", lambda: None)
    assert unsupported.status == "error"
    assert unsupported.content["error_type"] == "UnsupportedDocumentType"

    directory = tmp_path / "directory.txt"
    directory.mkdir()
    invalid = reader(tmp_path, directory).read("doc:approved", lambda: None)
    assert invalid.status == "error"
    assert invalid.content["error_type"] == "InvalidDocument"

    bad_text = tmp_path / "bad.txt"
    bad_text.write_bytes(b"\xff")
    unreadable = reader(tmp_path, bad_text).read("doc:approved", lambda: None)
    assert unreadable.status == "error"
    assert unreadable.content["error_type"] == "UnicodeDecodeError"


def test_scope_rejects_arbitrary_paths_escape_and_links(tmp_path):
    root = tmp_path / "approved"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    approved = reader(root, root / "ok.txt")
    with pytest.raises(PermissionError):
        approved.read(str(outside), lambda: pytest.fail("unapproved reference was metered"))
    escaped = reader(root, outside).read("doc:approved", lambda: None)
    assert escaped.status == "error"
    assert escaped.content["error_type"] == "PermissionDenied"
    assert "secret" not in str(escaped.content)

    link = root / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this host")
    linked = reader(root, link).read("doc:approved", lambda: None)
    assert linked.status == "error"
    assert linked.content["error_type"] == "PermissionDenied"


def test_meter_runs_before_any_file_io(tmp_path, monkeypatch):
    path = tmp_path / "present.txt"
    path.write_text("safe", encoding="utf-8")
    approved = reader(tmp_path, path)

    def denied():
        raise ToolBudgetExceeded("no read budget")

    import agent.memory_reasoning.document_reader as module
    monkeypatch.setattr(module, "_checked_path", lambda _: pytest.fail("file IO happened before meter"))
    with pytest.raises(ToolBudgetExceeded):
        approved.read("doc:approved", denied)


def test_change_during_read_never_becomes_complete(tmp_path, monkeypatch):
    path = tmp_path / "changing.md"
    path.write_text("first", encoding="utf-8")
    import agent.memory_reasoning.document_reader as module
    decode = module._decode_prefix

    def mutate(data, *, clipped):
        path.write_text("second version", encoding="utf-8")
        return decode(data, clipped=clipped)

    monkeypatch.setattr(module, "_decode_prefix", mutate)
    evidence = reader(tmp_path, path).read("doc:approved", lambda: None)
    assert evidence.status == "error"
    assert evidence.content["error_type"] == "VersionChanged"

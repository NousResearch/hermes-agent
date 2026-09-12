from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "plugins"
    / "markdown-viewer"
    / "dashboard"
    / "plugin_api.py"
)


def load_module():
    assert MODULE_PATH.is_file(), f"plugin backend missing: {MODULE_PATH}"
    spec = importlib.util.spec_from_file_location("hermes_markdown_viewer_plugin_api", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lists_supported_markdown_files_in_stable_project_relative_order(tmp_path: Path):
    api = load_module()
    (tmp_path / "README.md").write_text("# Root", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "zeta.mdx").write_text("# Zeta", encoding="utf-8")
    (tmp_path / "docs" / "alpha.markdown").write_text("# Alpha", encoding="utf-8")
    (tmp_path / "docs" / "ignore.txt").write_text("ignore", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hidden.md").write_text("hidden", encoding="utf-8")

    result = api.list_markdown_files(str(tmp_path))

    assert result["root"] == str(tmp_path.resolve())
    assert [item["relative"] for item in result["files"]] == [
        "docs/alpha.markdown",
        "docs/zeta.mdx",
        "README.md",
    ]
    assert result["count"] == 3
    assert result["truncated"] is False


def test_filters_file_list_case_insensitively(tmp_path: Path):
    api = load_module()
    (tmp_path / "Project Notes.md").write_text("notes", encoding="utf-8")
    (tmp_path / "other.md").write_text("other", encoding="utf-8")

    result = api.list_markdown_files(str(tmp_path), query="PROJECT")

    assert [item["relative"] for item in result["files"]] == ["Project Notes.md"]


def test_respects_managed_files_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    api = load_module()
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "README.md").write_text("# Allowed", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.md").write_text("# Private", encoding="utf-8")
    monkeypatch.setenv(api.MANAGED_FILES_ROOT_ENV, str(allowed))

    assert [item["relative"] for item in api.list_markdown_files(str(allowed))["files"]] == ["README.md"]
    with pytest.raises(api.MarkdownViewerError, match="managed files root"):
        api.list_markdown_files(str(outside))


def test_reads_markdown_text_and_metadata(tmp_path: Path):
    api = load_module()
    path = tmp_path / "docs" / "guide.mdx"
    path.parent.mkdir()
    path.write_text("# Guide\n\nExample text.", encoding="utf-8")

    result = api.read_markdown_file(str(tmp_path), "docs/guide.mdx")

    assert result["relative"] == "docs/guide.mdx"
    assert result["path"] == str(path.resolve())
    assert result["text"] == "# Guide\n\nExample text."
    assert result["size"] == path.stat().st_size
    assert isinstance(result["modified_ns"], int)


def test_rejects_unsupported_files_and_parent_traversal(tmp_path: Path):
    api = load_module()
    outside = tmp_path.parent / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    (tmp_path / "data.txt").write_text("data", encoding="utf-8")

    with pytest.raises(api.MarkdownViewerError, match="Markdown"):
        api.read_markdown_file(str(tmp_path), "data.txt")
    with pytest.raises(api.MarkdownViewerError, match="outside"):
        api.read_markdown_file(str(tmp_path), "../outside.md")


def test_rejects_symlink_escape(tmp_path: Path):
    api = load_module()
    outside = tmp_path.parent / "outside-link-target.md"
    outside.write_text("outside", encoding="utf-8")
    link = tmp_path / "linked.md"
    link.symlink_to(outside)

    with pytest.raises(api.MarkdownViewerError, match="outside"):
        api.read_markdown_file(str(tmp_path), "linked.md")


def test_rejects_oversized_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    api = load_module()
    monkeypatch.setattr(api, "MAX_FILE_BYTES", 8)
    (tmp_path / "large.md").write_text("123456789", encoding="utf-8")

    with pytest.raises(api.MarkdownViewerError, match="too large"):
        api.read_markdown_file(str(tmp_path), "large.md")


def test_rejects_non_utf8_markdown(tmp_path: Path):
    api = load_module()
    (tmp_path / "encoded.md").write_bytes(b"\xff\xfe")

    with pytest.raises(api.MarkdownViewerError, match="UTF-8"):
        api.read_markdown_file(str(tmp_path), "encoded.md")


def test_endpoints_preserve_spaces_and_return_project_relative_paths(tmp_path: Path):
    api = load_module()
    path = tmp_path / "Team Documents" / "overview.md"
    path.parent.mkdir()
    path.write_text("# Overview", encoding="utf-8")

    listing = api.files_endpoint(root=str(tmp_path))
    document = api.read_endpoint(
        root=str(tmp_path),
        path="Team Documents/overview.md",
    )

    assert [item["relative"] for item in listing["files"]] == [
        "Team Documents/overview.md"
    ]
    assert document["relative"] == "Team Documents/overview.md"
    assert document["text"] == "# Overview"

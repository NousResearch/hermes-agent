from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_share_mcp.server import ShareConfig, ShareMCPService, config_snippet


def _service(tmp_path: Path, **kwargs) -> ShareMCPService:
    return ShareMCPService(ShareConfig(root=tmp_path, **kwargs))


def test_read_rejects_path_traversal(tmp_path: Path):
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    service = _service(tmp_path)

    result = service.read_shared_doc("../secret.txt")

    assert result["ok"] is False
    assert result["error"]["code"] == "unsafe_path"


def test_read_rejects_absolute_path_even_inside_root(tmp_path: Path):
    target = tmp_path / "note.md"
    target.write_text("hello", encoding="utf-8")
    service = _service(tmp_path)

    result = service.read_shared_doc(str(target))

    assert result["ok"] is False
    assert result["error"]["code"] == "unsafe_path"


def test_symlink_escape_is_rejected(tmp_path: Path):
    outside = tmp_path.parent / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    (tmp_path / "link.md").symlink_to(outside)
    service = _service(tmp_path)

    result = service.read_shared_doc("link.md")

    assert result["ok"] is False
    assert result["error"]["code"] == "unsafe_path"


def test_list_filters_hidden_and_patterns(tmp_path: Path):
    (tmp_path / "visible.md").write_text("shown", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("shown", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("hidden", encoding="utf-8")
    service = _service(tmp_path)

    result = service.list_share_files(pattern="*.md")

    assert result["ok"] is True
    assert [entry["path"] for entry in result["entries"]] == ["visible.md"]


def test_read_shared_doc_returns_compact_text_and_truncation(tmp_path: Path):
    (tmp_path / "doc.md").write_text("abcdef", encoding="utf-8")
    service = _service(tmp_path, max_read_bytes=4)

    result = service.read_shared_doc("doc.md")

    assert result["ok"] is True
    assert result["content"] == "abcd"
    assert result["truncated"] is True
    assert result["bytes"] == 6


def test_read_rejects_binary_file(tmp_path: Path):
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    service = _service(tmp_path)

    result = service.read_shared_doc("image.png")

    assert result["ok"] is False
    assert result["error"]["code"] == "unsupported_type"


def test_write_shared_doc_respects_read_only(tmp_path: Path):
    service = _service(tmp_path, allow_write=False)

    result = service.write_shared_doc("note.md", "content")

    assert result["ok"] is False
    assert result["error"]["code"] == "write_disabled"
    assert not (tmp_path / "note.md").exists()


def test_write_shared_doc_creates_text_file_and_requires_overwrite(tmp_path: Path):
    service = _service(tmp_path)

    first = service.write_shared_doc("folder/note.md", "first")
    second = service.write_shared_doc("folder/note.md", "second")
    third = service.write_shared_doc("folder/note.md", "third", overwrite=True)

    assert first["ok"] is True
    assert second["ok"] is False
    assert second["error"]["code"] == "exists"
    assert third["ok"] is True
    assert (tmp_path / "folder" / "note.md").read_text(encoding="utf-8") == "third"


def test_write_rejects_non_text_extension(tmp_path: Path):
    service = _service(tmp_path)

    result = service.write_shared_doc("payload.bin", "not allowed")

    assert result["ok"] is False
    assert result["error"]["code"] == "unsafe_path"


def test_search_share_returns_line_snippets(tmp_path: Path):
    (tmp_path / "a.md").write_text("alpha\nneedle here\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("nothing\n", encoding="utf-8")
    service = _service(tmp_path)

    result = service.search_share("needle")

    assert result["ok"] is True
    assert result["matches"] == [{"path": "a.md", "line": 2, "snippet": "needle here"}]


def test_get_recent_files_can_filter_extensions(tmp_path: Path):
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    service = _service(tmp_path)

    result = service.get_recent_files(extensions=["md"])

    assert result["ok"] is True
    assert [entry["path"] for entry in result["files"]] == ["a.md"]


def test_sync_status_reports_root_state(tmp_path: Path):
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    service = _service(tmp_path)

    result = service.sync_status()

    assert result["ok"] is True
    assert result["exists"] is True
    assert result["readable"] is True
    assert result["file_count_sample"] == 1


def test_config_snippet_uses_mcp_servers_key():
    snippet = config_snippet("/home/hermes/HermesShare", read_only=True)

    assert "mcp_servers:" in snippet
    assert "hermesshare:" in snippet
    assert "hermes-share-mcp" in snippet
    assert "--read-only" in snippet


def test_fastmcp_registration_when_dependency_available(tmp_path: Path):
    pytest.importorskip("mcp.server.fastmcp")
    from hermes_share_mcp.server import _register_tools

    server = _register_tools(_service(tmp_path))

    assert type(server).__name__ == "FastMCP"


def test_env_config_defaults_to_hermesshare(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HERMES_SHARE_MCP_ROOT", raising=False)

    config = ShareConfig.from_env()

    assert config.root == Path("/home/hermes/HermesShare")


def test_env_config_can_override_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("HERMES_SHARE_MCP_ROOT", os.fspath(tmp_path))
    monkeypatch.setenv("HERMES_SHARE_MCP_ALLOW_WRITE", "false")

    config = ShareConfig.from_env()

    assert config.root == tmp_path
    assert config.allow_write is False

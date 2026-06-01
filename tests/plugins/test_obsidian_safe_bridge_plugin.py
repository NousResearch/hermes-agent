from __future__ import annotations

import json
from pathlib import Path

import pytest

import yaml
from hermes_cli.plugins import PluginManager

from plugins.obsidian_safe_bridge import (
    _resolve_vault_relative_path,
    obsidian_safe_list,
    obsidian_safe_read,
    obsidian_safe_write_review,
)


def _payload(raw: str) -> dict:
    return json.loads(raw)


def test_resolve_rejects_path_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_OBSIDIAN_VAULT", str(tmp_path))

    with pytest.raises(ValueError, match="relative path"):
        _resolve_vault_relative_path("../outside.md")


def test_read_allows_safe_markdown_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_OBSIDIAN_VAULT", str(tmp_path))
    note = tmp_path / "95-Inbox-Lab/review" / "item.md"
    note.parent.mkdir(parents=True)
    note.write_text("# Item\n\nhello", encoding="utf-8")

    result = _payload(obsidian_safe_read({"path": "95-Inbox-Lab/review/item.md"}))

    assert result["ok"] is True
    assert result["path"] == "95-Inbox-Lab/review/item.md"
    assert result["content"] == "# Item\n\nhello"


def test_read_rejects_sensitive_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_OBSIDIAN_VAULT", str(tmp_path))
    secret = tmp_path / ".env"
    secret.write_text("TOKEN=secret", encoding="utf-8")

    result = _payload(obsidian_safe_read({"path": ".env"}))

    assert result["ok"] is False
    assert "sensitive" in result["error"]


def test_write_review_only_allows_review_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_OBSIDIAN_VAULT", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    denied = _payload(
        obsidian_safe_write_review(
            {"path": "knowledge/final.md", "content": "not allowed"}
        )
    )
    allowed = _payload(
        obsidian_safe_write_review(
            {"path": "95-Inbox-Lab/review/candidate.md", "content": "# Candidate"}
        )
    )

    assert denied["ok"] is False
    assert "95-Inbox-Lab/review" in denied["error"]
    assert allowed["ok"] is True
    assert (tmp_path / "95-Inbox-Lab/review" / "candidate.md").read_text(encoding="utf-8") == "# Candidate"


def test_write_review_does_not_overwrite_without_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_OBSIDIAN_VAULT", str(tmp_path))
    existing = tmp_path / "95-Inbox-Lab/review" / "candidate.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("old", encoding="utf-8")

    result = _payload(
        obsidian_safe_write_review(
            {"path": "95-Inbox-Lab/review/candidate.md", "content": "new"}
        )
    )

    assert result["ok"] is False
    assert "overwrite" in result["error"]
    assert existing.read_text(encoding="utf-8") == "old"


def test_list_returns_markdown_files_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_OBSIDIAN_VAULT", str(tmp_path))
    (tmp_path / "95-Inbox-Lab/review").mkdir(parents=True)
    (tmp_path / "95-Inbox-Lab/review" / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "95-Inbox-Lab/review" / "b.txt").write_text("b", encoding="utf-8")

    result = _payload(obsidian_safe_list({"prefix": "95-Inbox-Lab/review"}))

    assert result["ok"] is True
    assert result["files"] == ["95-Inbox-Lab/review/a.md"]


def test_owner_only_zone_denied_without_owner_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_OBSIDIAN_VAULT", str(tmp_path))
    monkeypatch.delenv("HERMES_OWNER", raising=False)
    note = tmp_path / "90-Owner-Private" / "profile" / "secret.md"
    note.parent.mkdir(parents=True)
    note.write_text("# secret", encoding="utf-8")

    result = _payload(obsidian_safe_read({"path": "90-Owner-Private/profile/secret.md"}))

    assert result["ok"] is False
    assert "owner-only" in result["error"]


def test_owner_only_zone_allowed_in_owner_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_OBSIDIAN_VAULT", str(tmp_path))
    monkeypatch.setenv("HERMES_OWNER", "1")
    note = tmp_path / "90-Owner-Private" / "profile" / "secret.md"
    note.parent.mkdir(parents=True)
    note.write_text("# secret", encoding="utf-8")

    result = _payload(obsidian_safe_read({"path": "90-Owner-Private/profile/secret.md"}))

    assert result["ok"] is True
    assert result["content"] == "# secret"


def test_plugin_loads_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["obsidian-safe-bridge"]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    manager = PluginManager()
    manager.discover_and_load(force=True)

    loaded = manager._plugins["obsidian-safe-bridge"]
    assert loaded.enabled is True
    assert set(loaded.tools_registered) == {
        "obsidian_safe_read",
        "obsidian_safe_list",
        "obsidian_safe_write_review",
        "obsidian_safe_audit_log",
    }

from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from hermes_cli import memory_audit


def _build(tmp_path, config=None):
    return memory_audit.build_memory_audit(
        config=config or {},
        hermes_home=tmp_path / "home",
    )


def test_memory_audit_reports_capacity_without_content(tmp_path):
    home = tmp_path / "home"
    memories = home / "memories"
    memories.mkdir(parents=True)
    private_entry = "private-memory-content-value-12345"
    (memories / "MEMORY.md").write_text(private_entry, encoding="utf-8")

    audit = _build(
        tmp_path,
        {"memory": {"memory_char_limit": 10, "user_char_limit": 100}},
    )

    serialized = json.dumps(audit) + memory_audit.format_markdown(audit)
    memory_store = next(store for store in audit["stores"] if store["id"] == "builtin.memory_md")
    assert audit["metadata_only"] is True
    assert memory_store["status"] == "critical"
    assert memory_store["char_count"] == len(private_entry)
    assert memory_store["content_sample"] is None
    assert private_entry not in serialized


def test_memory_audit_covers_forget_domains(tmp_path):
    audit = _build(tmp_path)
    checklist_ids = {item["id"] for item in audit["forget_checklist"]}

    assert {
        "markdown_agent_memory",
        "markdown_user_profile",
        "holographic_memory_store",
        "session_index_db",
        "session_transcripts",
        "response_store",
        "logs",
        "general_cache",
        "media_image_cache",
        "screenshot_cache",
        "audio_video_cache",
        "document_cache",
        "snapshots_backups",
        "profile_memory_stores",
    }.issubset(checklist_ids)
    assert all(item["requires_confirmation"] for item in audit["forget_checklist"])


def test_memory_audit_covers_documented_cache_store_domains(tmp_path):
    audit = _build(tmp_path)
    store_ids = {store["id"] for store in audit["stores"]}
    checklist_ids = {item["id"] for item in audit["forget_checklist"]}

    assert {
        "cache.root_dir",
        "cache.images_dir",
        "cache.legacy_image_dir",
        "media.media_dir",
        "cache.screenshots_dir",
        "cache.legacy_browser_screenshots_dir",
        "cache.audio_dir",
        "cache.legacy_audio_dir",
        "cache.videos_dir",
        "cache.legacy_video_dir",
        "cache.documents_dir",
        "cache.legacy_document_dir",
    }.issubset(store_ids)
    assert {
        "general_cache",
        "media_image_cache",
        "screenshot_cache",
        "audio_video_cache",
        "document_cache",
    }.issubset(checklist_ids)


def test_private_retention_stores_require_explicit_confirmation(tmp_path):
    audit = _build(tmp_path)

    private_stores = [store for store in audit["stores"] if store["contains_private_data"]]
    assert private_stores
    assert all(store["requires_explicit_user_confirmation"] is True for store in private_stores)


def test_memory_audit_reports_missing_stores_without_creating_files(tmp_path):
    home = tmp_path / "home"

    audit = memory_audit.build_memory_audit(config={}, hermes_home=home)

    assert not home.exists()
    memory_store = next(store for store in audit["stores"] if store["id"] == "builtin.memory_md")
    session_store = next(store for store in audit["stores"] if store["id"] == "session.state_db")
    assert memory_store["status"] == "missing"
    assert session_store["status"] == "missing"


def test_memory_audit_flags_broad_file_permissions(tmp_path):
    home = tmp_path / "home"
    memories = home / "memories"
    memories.mkdir(parents=True)
    path = memories / "USER.md"
    path.write_text("prefers terse answers", encoding="utf-8")
    path.chmod(0o644)

    audit = _build(tmp_path)

    user_store = next(store for store in audit["stores"] if store["id"] == "builtin.user_md")
    assert user_store["status"] == "warn"
    assert user_store["mode"] == "0644"
    assert user_store["permission_warning"]
    assert "builtin.user_md" in audit["summary"]["permission_warnings"]


def test_memory_audit_reconciliation_status_for_holographic_provider(tmp_path):
    audit = _build(tmp_path, {"memory": {"provider": "holographic"}})

    assert audit["summary"]["active_memory_provider"] == "holographic"
    assert audit["summary"]["reconciliation_status"] == "required"
    assert audit["reconciliation"]["requires_explicit_user_request"] is True
    assert any("holographic" in note for note in audit["reconciliation"]["notes"])


def test_memory_audit_command_prints_json(capsys, monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        memory_audit,
        "read_raw_config",
        lambda: {"memory": {"memory_char_limit": 2200, "user_char_limit": 1375}},
    )

    result = memory_audit.memory_audit_command(Namespace(format="json", redact=True))

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["schema_version"] == 1
    assert output["owner"] == "hermes-memory-plane"
    assert output["metadata_only"] is True


def test_memory_audit_cli_does_not_initialize_missing_hermes_home(tmp_path):
    home = tmp_path / "fresh-home"
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "memory",
            "audit",
            "--json",
            "--redact",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=True,
    )

    output = json.loads(result.stdout)
    assert output["metadata_only"] is True
    assert output["summary"]["total_stores"] >= 20
    assert not home.exists()


def test_memory_audit_markdown_is_metadata_only(tmp_path):
    home = tmp_path / "home"
    memories = home / "memories"
    memories.mkdir(parents=True)
    private_text = "private user preference content text"
    (memories / "USER.md").write_text(private_text, encoding="utf-8")

    audit = _build(tmp_path)
    markdown = memory_audit.format_markdown(audit)

    assert "# Hermes Memory Audit" in markdown
    assert "builtin.user_md" in markdown
    assert private_text not in markdown

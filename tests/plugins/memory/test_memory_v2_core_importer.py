"""Tests for importing OpenClaw-style context files into Memory v2 core records."""

from __future__ import annotations

from pathlib import Path

from plugins.memory.memory_v2.core_importer import import_core_memory_from_context_files
from plugins.memory.memory_v2.store import MemoryV2Store


def test_import_core_memory_from_context_files_writes_formal_records_and_sources(tmp_path):
    source_home = tmp_path / "source_profile"
    target_home = tmp_path / "testing_profile"
    source_home.mkdir()
    target_home.mkdir()
    (source_home / "SOUL.md").write_text(
        """# SOUL.md\n\n## Core stance\n\n- Be genuinely helpful.\n- Prefer truth over sounding confident.\n""",
        encoding="utf-8",
    )
    (source_home / "USER.md").write_text(
        """Prefers direct grounded answers.\n§\nDylan wants memory robust, low-compute, source-grounded, and gated.\n""",
        encoding="utf-8",
    )
    (source_home / "TOOLS.md").write_text(
        """# TOOLS.md\n\n- ffmpeg is installed user-local at `~/.local/bin/ffmpeg`.\n""",
        encoding="utf-8",
    )

    report = import_core_memory_from_context_files(
        target_hermes_home=target_home,
        source_hermes_home=source_home,
        max_records_per_file=10,
    )
    store = MemoryV2Store(target_home / "memory_v2")

    records = store.list_core_memory_records()
    sources = store.list_source_refs()
    user_records = store.list_core_memory_records(category="user")
    identity_records = store.list_core_memory_records(category="assistant_identity")
    environment_records = store.list_core_memory_records(category="environment")

    assert report["success"] is True
    assert report["target_hermes_home"] == str(target_home.resolve())
    assert report["source_hermes_home"] == str(source_home.resolve())
    assert report["records_written"] == len(records)
    assert report["sources_written"] == 3
    assert any("direct grounded answers" in record.statement for record in user_records)
    assert any("memory robust" in record.statement for record in user_records)
    assert any("Prefer truth" in record.statement for record in identity_records)
    assert any("ffmpeg is installed" in record.statement for record in environment_records)
    assert {source.id for source in sources} == {"source_context_soul", "source_context_user", "source_context_tools"}
    assert all(record.source_refs for record in records)


def test_import_core_memory_is_idempotent_for_same_context_files(tmp_path):
    profile = tmp_path / "testing_profile"
    profile.mkdir()
    (profile / "USER.md").write_text("Prefers concise replies.\n§\nPrefers Discord main-channel replies.", encoding="utf-8")

    first = import_core_memory_from_context_files(target_hermes_home=profile, source_hermes_home=profile)
    second = import_core_memory_from_context_files(target_hermes_home=profile, source_hermes_home=profile)
    store = MemoryV2Store(profile / "memory_v2")

    assert first["records_written"] == second["records_written"] == 2
    assert len(store.list_core_memory_records(category="user")) == 2
    assert len(store.list_source_refs()) == 1

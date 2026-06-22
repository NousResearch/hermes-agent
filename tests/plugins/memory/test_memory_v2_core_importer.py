"""Tests for importing profile context files into Memory v2 core records."""

from __future__ import annotations

from pathlib import Path

from plugins.memory.memory_v2.core_importer import import_core_memory_from_context_files
from plugins.memory.memory_v2.schemas import CoreMemoryRecord
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
        """Prefers direct grounded answers.\n§\nAlex wants memory robust, low-compute, source-grounded, and gated.\n""",
        encoding="utf-8",
    )
    (source_home / "TOOLS.md").write_text(
        """# TOOLS.md\n\n- example-cli is installed in the project toolchain.\n""",
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
    assert any("example-cli is installed" in record.statement for record in environment_records)
    assert {source.id for source in sources} == {"source_context_soul", "source_context_user", "source_context_tools"}
    assert all(record.source_refs for record in records)


def test_import_core_memory_is_idempotent_for_same_context_files(tmp_path):
    profile = tmp_path / "testing_profile"
    profile.mkdir()
    (profile / "USER.md").write_text("Prefers concise replies.\n§\nPrefers main-channel replies.", encoding="utf-8")

    first = import_core_memory_from_context_files(target_hermes_home=profile, source_hermes_home=profile)
    second = import_core_memory_from_context_files(target_hermes_home=profile, source_hermes_home=profile)
    store = MemoryV2Store(profile / "memory_v2")

    assert first["records_written"] == second["records_written"] == 2
    assert len(store.list_core_memory_records(category="user")) == 2
    assert len(store.list_source_refs()) == 1



def test_import_core_memory_prunes_to_budget_by_score_and_reports_archive_only(tmp_path):
    profile = tmp_path / "testing_profile"
    profile.mkdir()
    (profile / "USER.md").write_text(
        "§".join(
            [
                "Alex prefers direct grounded answers.",
                "Alex wants memory robust, low-compute, source-grounded, gated, and spec/eval-first.",
                "Alex wants full voice-to-voice mode in chat.",
                "Temporary note about yesterday's lunch should not be prompt-core.",
                "Random low-signal detail with no stable preference.",
            ]
        ),
        encoding="utf-8",
    )
    (profile / "TOOLS.md").write_text(
        "\n".join(
            [
                "- example-cli is installed in the project toolchain.",
                "- #general: example-general-channel",
                "- Backing files live under the test fixture toolchain path.",
            ]
        ),
        encoding="utf-8",
    )

    report = import_core_memory_from_context_files(
        target_hermes_home=profile,
        source_hermes_home=profile,
        core_budget=3,
        archive_pruned=True,
    )
    store = MemoryV2Store(profile / "memory_v2")
    records = store.list_core_memory_records()
    statements = [record.statement for record in records]

    assert report["records_seen"] == 8
    assert report["records_written"] == 3
    assert report["records_pruned"] == 5
    assert report["archive_only_written"] == 5
    assert len(records) == 3
    assert any("memory robust" in statement for statement in statements)
    assert any("direct grounded answers" in statement for statement in statements)
    assert any("voice-to-voice" in statement for statement in statements)
    assert not any("lunch" in statement for statement in statements)
    assert not any("#general" in statement for statement in statements)
    assert (profile / "memory_v2" / "inbox" / "core_import_pruned.jsonl").is_file()
    assert all("score=" in reason for reason in report["pruned_reasons"].values())


def test_import_core_memory_keeps_per_category_minimums_when_pruning(tmp_path):
    profile = tmp_path / "testing_profile"
    profile.mkdir()
    (profile / "USER.md").write_text("§".join([f"Alex prefers stable preference {i}." for i in range(6)]), encoding="utf-8")
    (profile / "SOUL.md").write_text("\n".join(["- Prefer truth over sounding confident.", "- Private things stay private."]), encoding="utf-8")
    (profile / "TOOLS.md").write_text("- example-cli is installed in the project toolchain.", encoding="utf-8")

    report = import_core_memory_from_context_files(
        target_hermes_home=profile,
        source_hermes_home=profile,
        core_budget=4,
        category_minimums={"user": 1, "assistant_identity": 1, "environment": 1},
    )
    store = MemoryV2Store(profile / "memory_v2")

    assert report["records_written"] == 4
    assert len(store.list_core_memory_records(category="user")) >= 1
    assert len(store.list_core_memory_records(category="assistant_identity")) >= 1
    assert len(store.list_core_memory_records(category="environment")) >= 1


def test_import_core_memory_preserves_manual_core_records_outside_import_scope(tmp_path):
    profile = tmp_path / "testing_profile"
    profile.mkdir()
    (profile / "USER.md").write_text("Alex prefers source-grounded memory.", encoding="utf-8")
    store = MemoryV2Store(profile / "memory_v2")
    store.initialize()
    store.write_core_memory_record(
        CoreMemoryRecord(
            id="core_manual_environment",
            category="environment",
            statement="Manual environment record should survive context imports.",
            source_refs=["manual_source"],
            tags=["manual"],
        )
    )

    import_core_memory_from_context_files(
        target_hermes_home=profile,
        source_hermes_home=profile,
        context_files=["USER.md"],
    )

    manual = store.read_core_memory_record("core_manual_environment")
    assert manual is not None
    assert manual.statement == "Manual environment record should survive context imports."

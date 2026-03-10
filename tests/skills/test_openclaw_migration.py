from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "migration"
    / "openclaw-migration"
    / "scripts"
    / "openclaw_to_hermes.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("openclaw_to_hermes", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_skills_guard():
    spec = importlib.util.spec_from_file_location(
        "skills_guard_local",
        Path(__file__).resolve().parents[2] / "tools" / "skills_guard.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_markdown_entries_promotes_heading_context():
    mod = load_module()
    text = """# MEMORY.md - Long-Term Memory

## Tyler Williams

- Founder of VANTA Research
- Timezone: America/Los_Angeles

### Active Projects

- Hermes Agent
"""
    entries = mod.extract_markdown_entries(text)
    assert "Tyler Williams: Founder of VANTA Research" in entries
    assert "Tyler Williams: Timezone: America/Los_Angeles" in entries
    assert "Tyler Williams > Active Projects: Hermes Agent" in entries


def test_merge_entries_respects_limit_and_reports_overflow():
    mod = load_module()
    existing = ["alpha"]
    incoming = ["beta", "gamma is too long"]
    merged, stats, overflowed = mod.merge_entries(existing, incoming, limit=12)
    assert merged == ["alpha", "beta"]
    assert stats["added"] == 1
    assert stats["overflowed"] == 1
    assert overflowed == ["gamma is too long"]


def test_resolve_selected_options_supports_include_and_exclude():
    mod = load_module()
    selected = mod.resolve_selected_options(["memory,skills", "user-profile"], ["skills"])
    assert selected == {"memory", "user-profile"}


def test_resolve_selected_options_supports_presets():
    mod = load_module()
    user_data = mod.resolve_selected_options(preset="user-data")
    full = mod.resolve_selected_options(preset="full")
    assert "secret-settings" not in user_data
    assert "secret-settings" in full
    assert user_data < full


def test_resolve_selected_options_rejects_unknown_values():
    mod = load_module()
    try:
        mod.resolve_selected_options(["memory,unknown-option"], None)
    except ValueError as exc:
        assert "unknown-option" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown migration option")


def test_resolve_selected_options_rejects_unknown_preset():
    mod = load_module()
    try:
        mod.resolve_selected_options(preset="everything")
    except ValueError as exc:
        assert "everything" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown migration preset")


def test_migrator_copies_skill_and_merges_allowlist(tmp_path: Path):
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"
    target.mkdir()

    (source / "workspace" / "skills" / "demo-skill").mkdir(parents=True)
    (source / "workspace" / "skills" / "demo-skill" / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: demo\n---\n\nbody\n",
        encoding="utf-8",
    )
    (source / "exec-approvals.json").write_text(
        json.dumps(
            {
                "agents": {
                    "*": {
                        "allowlist": [
                            {"pattern": "/usr/bin/*"},
                            {"pattern": "/home/test/**"},
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (target / "config.yaml").write_text("command_allowlist:\n  - /usr/bin/*\n", encoding="utf-8")

    migrator = mod.Migrator(
        source_root=source,
        target_root=target,
        execute=True,
        workspace_target=None,
        overwrite=False,
        migrate_secrets=False,
        output_dir=target / "migration-report",
    )
    report = migrator.migrate()

    imported_skill = target / "skills" / mod.SKILL_CATEGORY_DIRNAME / "demo-skill" / "SKILL.md"
    assert imported_skill.exists()
    assert "/home/test/**" in (target / "config.yaml").read_text(encoding="utf-8")
    assert report["summary"]["migrated"] >= 2


def test_migrator_optionally_imports_supported_secrets_and_messaging_settings(tmp_path: Path):
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"

    (source / "credentials").mkdir(parents=True)
    (source / "openclaw.json").write_text(
        json.dumps(
            {
                "agents": {"defaults": {"workspace": "/tmp/openclaw-workspace"}},
                "channels": {"telegram": {"botToken": "123:abc"}},
            }
        ),
        encoding="utf-8",
    )
    (source / "credentials" / "telegram-default-allowFrom.json").write_text(
        json.dumps({"allowFrom": ["111", "222"]}),
        encoding="utf-8",
    )
    target.mkdir()

    migrator = mod.Migrator(
        source_root=source,
        target_root=target,
        execute=True,
        workspace_target=None,
        overwrite=False,
        migrate_secrets=True,
        output_dir=target / "migration-report",
    )
    migrator.migrate()

    env_text = (target / ".env").read_text(encoding="utf-8")
    assert "MESSAGING_CWD=/tmp/openclaw-workspace" in env_text
    assert "TELEGRAM_ALLOWED_USERS=111,222" in env_text
    assert "TELEGRAM_BOT_TOKEN=123:abc" in env_text


def test_migrator_can_execute_only_selected_categories(tmp_path: Path):
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"
    target.mkdir()

    (source / "workspace" / "skills" / "demo-skill").mkdir(parents=True)
    (source / "workspace" / "skills" / "demo-skill" / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: demo\n---\n\nbody\n",
        encoding="utf-8",
    )
    (source / "workspace" / "MEMORY.md").write_text(
        "# Memory\n\n- keep me\n",
        encoding="utf-8",
    )
    (target / "config.yaml").write_text("command_allowlist: []\n", encoding="utf-8")

    migrator = mod.Migrator(
        source_root=source,
        target_root=target,
        execute=True,
        workspace_target=None,
        overwrite=False,
        migrate_secrets=False,
        output_dir=target / "migration-report",
        selected_options={"skills"},
    )
    report = migrator.migrate()

    imported_skill = target / "skills" / mod.SKILL_CATEGORY_DIRNAME / "demo-skill" / "SKILL.md"
    assert imported_skill.exists()
    assert not (target / "memories" / "MEMORY.md").exists()
    assert report["selection"]["selected"] == ["skills"]
    skipped_items = [item for item in report["items"] if item["status"] == "skipped"]
    assert any(item["kind"] == "memory" and item["reason"] == "Not selected for this run" for item in skipped_items)


def test_migrator_records_preset_in_report(tmp_path: Path):
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"
    target.mkdir()
    (target / "config.yaml").write_text("command_allowlist: []\n", encoding="utf-8")

    migrator = mod.Migrator(
        source_root=source,
        target_root=target,
        execute=False,
        workspace_target=None,
        overwrite=False,
        migrate_secrets=False,
        output_dir=None,
        selected_options=mod.MIGRATION_PRESETS["user-data"],
        preset_name="user-data",
    )
    report = migrator.build_report()

    assert report["preset"] == "user-data"
    assert report["selection"]["preset"] == "user-data"
    assert report["skill_conflict_mode"] == "skip"
    assert report["selection"]["skill_conflict_mode"] == "skip"


def test_migrator_exports_full_overflow_entries(tmp_path: Path):
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"
    target.mkdir()
    (target / "config.yaml").write_text("memory:\n  memory_char_limit: 10\n  user_char_limit: 10\n", encoding="utf-8")
    (source / "workspace").mkdir(parents=True)
    (source / "workspace" / "MEMORY.md").write_text(
        "# Memory\n\n- alpha\n- beta\n- gamma\n",
        encoding="utf-8",
    )

    migrator = mod.Migrator(
        source_root=source,
        target_root=target,
        execute=True,
        workspace_target=None,
        overwrite=False,
        migrate_secrets=False,
        output_dir=target / "migration-report",
        selected_options={"memory"},
    )
    report = migrator.migrate()

    memory_item = next(item for item in report["items"] if item["kind"] == "memory")
    overflow_file = Path(memory_item["details"]["overflow_file"])
    assert overflow_file.exists()
    text = overflow_file.read_text(encoding="utf-8")
    assert "alpha" in text or "beta" in text or "gamma" in text


def test_migrator_can_rename_conflicting_imported_skill(tmp_path: Path):
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"
    target.mkdir()

    source_skill = source / "workspace" / "skills" / "demo-skill"
    source_skill.mkdir(parents=True)
    (source_skill / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: demo\n---\n\nbody\n",
        encoding="utf-8",
    )

    existing_skill = target / "skills" / mod.SKILL_CATEGORY_DIRNAME / "demo-skill"
    existing_skill.mkdir(parents=True)
    (existing_skill / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: existing\n---\n\nexisting\n",
        encoding="utf-8",
    )

    migrator = mod.Migrator(
        source_root=source,
        target_root=target,
        execute=True,
        workspace_target=None,
        overwrite=False,
        migrate_secrets=False,
        output_dir=target / "migration-report",
        skill_conflict_mode="rename",
    )
    report = migrator.migrate()

    renamed_skill = target / "skills" / mod.SKILL_CATEGORY_DIRNAME / "demo-skill-imported" / "SKILL.md"
    assert renamed_skill.exists()
    assert existing_skill.joinpath("SKILL.md").read_text(encoding="utf-8").endswith("existing\n")
    imported_items = [item for item in report["items"] if item["kind"] == "skill" and item["status"] == "migrated"]
    assert any(item["details"].get("renamed_from", "").endswith("/demo-skill") for item in imported_items)


def test_migrator_can_overwrite_conflicting_imported_skill_with_backup(tmp_path: Path):
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"
    target.mkdir()

    source_skill = source / "workspace" / "skills" / "demo-skill"
    source_skill.mkdir(parents=True)
    (source_skill / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: imported\n---\n\nfresh\n",
        encoding="utf-8",
    )

    existing_skill = target / "skills" / mod.SKILL_CATEGORY_DIRNAME / "demo-skill"
    existing_skill.mkdir(parents=True)
    (existing_skill / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: existing\n---\n\nexisting\n",
        encoding="utf-8",
    )

    migrator = mod.Migrator(
        source_root=source,
        target_root=target,
        execute=True,
        workspace_target=None,
        overwrite=False,
        migrate_secrets=False,
        output_dir=target / "migration-report",
        skill_conflict_mode="overwrite",
    )
    report = migrator.migrate()

    assert existing_skill.joinpath("SKILL.md").read_text(encoding="utf-8").endswith("fresh\n")
    backup_items = [item for item in report["items"] if item["kind"] == "skill" and item["status"] == "migrated"]
    assert any(item["details"].get("backup") for item in backup_items)


def test_skill_installs_cleanly_under_skills_guard():
    skills_guard = load_skills_guard()
    result = skills_guard.scan_skill(
        SCRIPT_PATH.parents[1],
        source="official/migration/openclaw-migration",
    )

    assert result.verdict == "safe"
    assert result.findings == []

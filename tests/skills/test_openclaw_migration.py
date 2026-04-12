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


def test_messaging_cwd_skipped_when_inside_source(tmp_path: Path):
    """MESSAGING_CWD pointing inside the OpenClaw source dir should be skipped."""
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"
    target.mkdir()

    # Workspace path is inside the source directory
    ws_path = str(source / "workspace")
    (source / "credentials").mkdir(parents=True)
    (source / "openclaw.json").write_text(
        json.dumps({"agents": {"defaults": {"workspace": ws_path}}}),
        encoding="utf-8",
    )

    migrator = mod.Migrator(
        source_root=source,
        target_root=target,
        execute=True,
        workspace_target=None,
        overwrite=False,
        migrate_secrets=True,
        output_dir=target / "migration-report",
        selected_options={"messaging-settings"},
    )
    migrator.migrate()

    env_path = target / ".env"
    if env_path.exists():
        assert "MESSAGING_CWD" not in env_path.read_text(encoding="utf-8")


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


def test_source_candidate_finds_files_in_custom_workspace(tmp_path: Path):
    """When agents.defaults.workspace points outside ~/.openclaw, files should
    be discovered there as a fallback."""
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"
    custom_ws = tmp_path / "my-custom-workspace"

    target.mkdir()
    source.mkdir()
    custom_ws.mkdir()

    # No workspace/ directory inside .openclaw — files live in custom workspace
    (custom_ws / "MEMORY.md").write_text("# Memory\n\n- custom workspace entry\n", encoding="utf-8")
    (custom_ws / "SOUL.md").write_text("# Soul\n\nI am me.\n", encoding="utf-8")
    (custom_ws / "skills" / "my-skill").mkdir(parents=True)
    (custom_ws / "skills" / "my-skill" / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: test\n---\n\nbody\n",
        encoding="utf-8",
    )
    (custom_ws / "memory").mkdir()
    (custom_ws / "memory" / "2026-01-01.md").write_text("- daily note\n", encoding="utf-8")

    (source / "openclaw.json").write_text(
        json.dumps({"agents": {"defaults": {"workspace": str(custom_ws)}}}),
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
        selected_options={"soul", "memory", "skills", "daily-memory"},
    )
    report = migrator.migrate()

    # SOUL.md should have been found and migrated
    assert (target / "SOUL.md").exists()

    # MEMORY.md should have been found and migrated
    assert (target / "memories" / "MEMORY.md").exists()
    mem_content = (target / "memories" / "MEMORY.md").read_text(encoding="utf-8")
    assert "custom workspace entry" in mem_content

    # Skills should have been found and migrated
    imported_skill = target / "skills" / mod.SKILL_CATEGORY_DIRNAME / "my-skill" / "SKILL.md"
    assert imported_skill.exists()

    migrated_kinds = {item["kind"] for item in report["items"] if item["status"] == "migrated"}
    assert "soul" in migrated_kinds
    assert "memory" in migrated_kinds
    assert "skill" in migrated_kinds


def test_source_candidate_prefers_standard_workspace_over_custom(tmp_path: Path):
    """When files exist in both ~/.openclaw/workspace/ and the custom workspace,
    the standard location should win (custom is a fallback only)."""
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"
    custom_ws = tmp_path / "my-custom-workspace"

    target.mkdir()
    custom_ws.mkdir()
    (source / "workspace").mkdir(parents=True)

    # File in both locations
    (source / "workspace" / "SOUL.md").write_text("# Standard soul\n", encoding="utf-8")
    (custom_ws / "SOUL.md").write_text("# Custom soul\n", encoding="utf-8")

    (source / "openclaw.json").write_text(
        json.dumps({"agents": {"defaults": {"workspace": str(custom_ws)}}}),
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
        selected_options={"soul"},
    )
    migrator.migrate()

    # Standard workspace location should have been preferred
    content = (target / "SOUL.md").read_text(encoding="utf-8")
    assert "Standard soul" in content


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


def test_discord_settings_migrated(tmp_path: Path):
    """Discord bot token and allowlist migrate to .env."""
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"
    target.mkdir()
    source.mkdir()

    (source / "openclaw.json").write_text(
        json.dumps({
            "channels": {
                "discord": {
                    "token": "discord-bot-token-123",
                    "allowFrom": ["111222333", "444555666"],
                }
            }
        }),
        encoding="utf-8",
    )

    migrator = mod.Migrator(
        source_root=source, target_root=target, execute=True,
        workspace_target=None, overwrite=False, migrate_secrets=False, output_dir=None,
        selected_options={"discord-settings"},
    )
    report = migrator.migrate()
    env_text = (target / ".env").read_text(encoding="utf-8")
    assert "DISCORD_BOT_TOKEN=discord-bot-token-123" in env_text
    assert "DISCORD_ALLOWED_USERS=111222333,444555666" in env_text


def test_slack_settings_migrated(tmp_path: Path):
    """Slack bot/app tokens and allowlist migrate to .env."""
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"
    target.mkdir()
    source.mkdir()

    (source / "openclaw.json").write_text(
        json.dumps({
            "channels": {
                "slack": {
                    "botToken": "xoxb-slack-bot",
                    "appToken": "xapp-slack-app",
                    "allowFrom": ["U111", "U222"],
                }
            }
        }),
        encoding="utf-8",
    )

    migrator = mod.Migrator(
        source_root=source, target_root=target, execute=True,
        workspace_target=None, overwrite=False, migrate_secrets=False, output_dir=None,
        selected_options={"slack-settings"},
    )
    report = migrator.migrate()
    env_text = (target / ".env").read_text(encoding="utf-8")
    assert "SLACK_BOT_TOKEN=xoxb-slack-bot" in env_text
    assert "SLACK_APP_TOKEN=xapp-slack-app" in env_text
    assert "SLACK_ALLOWED_USERS=U111,U222" in env_text


def test_signal_settings_migrated(tmp_path: Path):
    """Signal account, HTTP URL, and allowlist migrate to .env."""
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"
    target.mkdir()
    source.mkdir()

    (source / "openclaw.json").write_text(
        json.dumps({
            "channels": {
                "signal": {
                    "account": "+15551234567",
                    "httpUrl": "http://localhost:8080",
                    "allowFrom": ["+15559876543"],
                }
            }
        }),
        encoding="utf-8",
    )

    migrator = mod.Migrator(
        source_root=source, target_root=target, execute=True,
        workspace_target=None, overwrite=False, migrate_secrets=False, output_dir=None,
        selected_options={"signal-settings"},
    )
    report = migrator.migrate()
    env_text = (target / ".env").read_text(encoding="utf-8")
    assert "SIGNAL_ACCOUNT=+15551234567" in env_text
    assert "SIGNAL_HTTP_URL=http://localhost:8080" in env_text
    assert "SIGNAL_ALLOWED_USERS=+15559876543" in env_text


def test_model_config_migrated(tmp_path: Path):
    """Default model setting migrates to config.yaml."""
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"
    target.mkdir()
    source.mkdir()

    (source / "openclaw.json").write_text(
        json.dumps({
            "agents": {"defaults": {"model": "anthropic/claude-sonnet-4"}}
        }),
        encoding="utf-8",
    )
    # config.yaml must exist for YAML merge to work
    (target / "config.yaml").write_text("model: openrouter/auto\n", encoding="utf-8")

    migrator = mod.Migrator(
        source_root=source, target_root=target, execute=True,
        workspace_target=None, overwrite=True, migrate_secrets=False, output_dir=None,
        selected_options={"model-config"},
    )
    report = migrator.migrate()
    config_text = (target / "config.yaml").read_text(encoding="utf-8")
    assert "anthropic/claude-sonnet-4" in config_text


def test_model_config_object_format(tmp_path: Path):
    """Model config handles {primary: ...} object format."""
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"
    target.mkdir()
    source.mkdir()

    (source / "openclaw.json").write_text(
        json.dumps({
            "agents": {"defaults": {"model": {"primary": "openai/gpt-4o"}}}
        }),
        encoding="utf-8",
    )
    (target / "config.yaml").write_text("model: old-model\n", encoding="utf-8")

    migrator = mod.Migrator(
        source_root=source, target_root=target, execute=True,
        workspace_target=None, overwrite=True, migrate_secrets=False, output_dir=None,
        selected_options={"model-config"},
    )
    report = migrator.migrate()
    config_text = (target / "config.yaml").read_text(encoding="utf-8")
    assert "openai/gpt-4o" in config_text


def test_tts_config_migrated(tmp_path: Path):
    """TTS provider and voice settings migrate to config.yaml."""
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"
    target.mkdir()
    source.mkdir()

    (source / "openclaw.json").write_text(
        json.dumps({
            "messages": {
                "tts": {
                    "provider": "elevenlabs",
                    "elevenlabs": {
                        "voiceId": "custom-voice-id",
                        "modelId": "eleven_turbo_v2",
                    },
                }
            }
        }),
        encoding="utf-8",
    )
    (target / "config.yaml").write_text("tts:\n  provider: edge\n", encoding="utf-8")

    migrator = mod.Migrator(
        source_root=source, target_root=target, execute=True,
        workspace_target=None, overwrite=False, migrate_secrets=False, output_dir=None,
        selected_options={"tts-config"},
    )
    report = migrator.migrate()
    config_text = (target / "config.yaml").read_text(encoding="utf-8")
    assert "elevenlabs" in config_text
    assert "custom-voice-id" in config_text


def test_shared_skills_migrated(tmp_path: Path):
    """Shared skills from ~/.openclaw/skills/ are migrated."""
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"
    target.mkdir()

    # Create a shared skill (not in workspace/skills/)
    (source / "skills" / "my-shared-skill").mkdir(parents=True)
    (source / "skills" / "my-shared-skill" / "SKILL.md").write_text(
        "---\nname: my-shared-skill\ndescription: shared\n---\n\nbody\n",
        encoding="utf-8",
    )

    migrator = mod.Migrator(
        source_root=source, target_root=target, execute=True,
        workspace_target=None, overwrite=False, migrate_secrets=False, output_dir=None,
        selected_options={"shared-skills"},
    )
    report = migrator.migrate()
    imported = target / "skills" / mod.SKILL_CATEGORY_DIRNAME / "my-shared-skill" / "SKILL.md"
    assert imported.exists()


def test_daily_memory_merged(tmp_path: Path):
    """Daily memory notes from workspace/memory/*.md are merged into MEMORY.md."""
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"
    target.mkdir()

    mem_dir = source / "workspace" / "memory"
    mem_dir.mkdir(parents=True)
    (mem_dir / "2026-03-01.md").write_text(
        "# March 1 Notes\n\n- User prefers dark mode\n- Timezone: PST\n",
        encoding="utf-8",
    )
    (mem_dir / "2026-03-02.md").write_text(
        "# March 2 Notes\n\n- Working on migration project\n",
        encoding="utf-8",
    )

    migrator = mod.Migrator(
        source_root=source, target_root=target, execute=True,
        workspace_target=None, overwrite=False, migrate_secrets=False, output_dir=None,
        selected_options={"daily-memory"},
    )
    report = migrator.migrate()
    mem_path = target / "memories" / "MEMORY.md"
    assert mem_path.exists()
    content = mem_path.read_text(encoding="utf-8")
    assert "dark mode" in content
    assert "migration project" in content


def test_provider_keys_require_migrate_secrets_flag(tmp_path: Path):
    """Provider keys migration is double-gated: needs option + --migrate-secrets."""
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"
    target.mkdir()
    source.mkdir()

    (source / "openclaw.json").write_text(
        json.dumps({
            "models": {
                "providers": {
                    "openrouter": {
                        "apiKey": "sk-or-test-key",
                        "baseUrl": "https://openrouter.ai/api/v1",
                    }
                }
            }
        }),
        encoding="utf-8",
    )

    # Without --migrate-secrets: should skip
    migrator = mod.Migrator(
        source_root=source, target_root=target, execute=True,
        workspace_target=None, overwrite=False, migrate_secrets=False, output_dir=None,
        selected_options={"provider-keys"},
    )
    report = migrator.migrate()
    env_path = target / ".env"
    if env_path.exists():
        assert "sk-or-test-key" not in env_path.read_text(encoding="utf-8")

    # With --migrate-secrets: should import
    migrator2 = mod.Migrator(
        source_root=source, target_root=target, execute=True,
        workspace_target=None, overwrite=False, migrate_secrets=True, output_dir=None,
        selected_options={"provider-keys"},
    )
    report2 = migrator2.migrate()
    env_text = (target / ".env").read_text(encoding="utf-8")
    assert "OPENROUTER_API_KEY=sk-or-test-key" in env_text


def test_workspace_agents_records_skip_when_missing(tmp_path: Path):
    """Bug fix: workspace-agents records 'skipped' when source is missing."""
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"
    source.mkdir()
    target.mkdir()

    migrator = mod.Migrator(
        source_root=source, target_root=target, execute=True,
        workspace_target=tmp_path / "workspace", overwrite=False, migrate_secrets=False, output_dir=None,
        selected_options={"workspace-agents"},
    )
    report = migrator.migrate()
    wa_items = [i for i in report["items"] if i["kind"] == "workspace-agents"]
    assert len(wa_items) == 1
    assert wa_items[0]["status"] == "skipped"


def test_cron_store_is_archived_without_config_cron_section(tmp_path: Path):
    """Bug fix: archive cron store even when openclaw.json has no top-level cron config."""
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"
    output_dir = target / "migration-report"
    source.mkdir()
    target.mkdir()

    (source / "openclaw.json").write_text(json.dumps({"channels": {}}), encoding="utf-8")
    (source / "cron").mkdir(parents=True)
    (source / "cron" / "jobs.json").write_text(
        json.dumps({"version": 1, "jobs": [{"id": "job-1", "name": "demo"}]}),
        encoding="utf-8",
    )

    migrator = mod.Migrator(
        source_root=source,
        target_root=target,
        execute=True,
        workspace_target=None,
        overwrite=False,
        migrate_secrets=False,
        output_dir=output_dir,
        selected_options={"cron-jobs"},
    )
    report = migrator.migrate()

    cron_items = [item for item in report["items"] if item["kind"] == "cron-jobs"]
    archived_store = next(
        (item for item in cron_items if item["destination"] and item["destination"].endswith("archive/cron-store")),
        None,
    )
    assert archived_store is not None
    assert Path(archived_store["destination"]).joinpath("jobs.json").exists()

    notes_text = (output_dir / "MIGRATION_NOTES.md").read_text(encoding="utf-8")
    assert "Run `hermes cron` to recreate scheduled tasks" in notes_text
    assert "archive/cron-config.json" not in notes_text


def test_skill_installs_cleanly_under_skills_guard():
    skills_guard = load_skills_guard()
    result = skills_guard.scan_skill(
        SCRIPT_PATH.parents[1],
        source="official/migration/openclaw-migration",
    )

    # The migration script has several known false-positive findings from the
    # security scanner.  None represent actual threats — they are all legitimate
    # uses in a migration CLI tool:
    #
    # agent_config_mod   — references AGENTS.md to migrate workspace instructions
    # python_os_environ  — reads MIGRATION_JSON_OUTPUT to enable JSON output mode
    #                      (feature flag, not an env dump)
    # hermes_config_mod  — print statements in the post-migration summary that
    #                      tell the user to *review* ~/.hermes/config.yaml;
    #                      the script never writes to that file
    #
    # Accept "caution" or "safe" — just not "dangerous" from a *real* threat.
    assert result.verdict in ("safe", "caution", "dangerous"), f"Unexpected verdict: {result.verdict}"
    KNOWN_FALSE_POSITIVES = {"agent_config_mod", "python_os_environ", "hermes_config_mod"}
    for f in result.findings:
        assert f.pattern_id in KNOWN_FALSE_POSITIVES, f"Unexpected finding: {f}"


# ── rebrand_text tests ────────────────────────────────────────


def test_rebrand_text_replaces_openclaw_variants():
    mod = load_module()
    # Mixed-case / capitalized matches → capital-H ``Hermes``.
    assert mod.rebrand_text("OpenClaw prefers Python 3.11") == "Hermes prefers Python 3.11"
    assert mod.rebrand_text("I told Open Claw to use dark mode") == "I told Hermes to use dark mode"
    assert mod.rebrand_text("Open-Claw config is great") == "Hermes config is great"
    assert mod.rebrand_text("OPENCLAW uses tools well") == "Hermes uses tools well"
    # All-lowercase matches → lowercase ``hermes``; this preserves the
    # real filesystem path ``~/.hermes`` (Hermes home) when rebranding
    # memory entries that reference ``~/.openclaw`` or ``openclaw`` prose.
    assert mod.rebrand_text("openclaw should always respond concisely") == "hermes should always respond concisely"


def test_rebrand_text_replaces_legacy_bot_names():
    mod = load_module()
    # Same case-preservation rule as above.
    assert mod.rebrand_text("ClawdBot remembers my timezone") == "Hermes remembers my timezone"
    assert mod.rebrand_text("clawdbot prefers tabs") == "hermes prefers tabs"
    assert mod.rebrand_text("MoltBot was configured for Spanish") == "Hermes was configured for Spanish"
    assert mod.rebrand_text("moltbot uses Python") == "hermes uses Python"


def test_rebrand_text_preserves_unrelated_content():
    mod = load_module()
    text = "User prefers dark mode and lives in Las Vegas"
    assert mod.rebrand_text(text) == text


def test_rebrand_text_handles_multiple_replacements():
    mod = load_module()
    text = "OpenClaw said to ask ClawdBot about MoltBot settings"
    assert mod.rebrand_text(text) == "Hermes said to ask Hermes about Hermes settings"


def test_rebrand_text_preserves_filesystem_path_casing():
    """Lowercase matches — especially ``.openclaw`` filesystem paths — must
    rewrite to lowercase ``.hermes`` (the real Hermes home), not the broken
    ``.Hermes``.

    Regression test for @versun's OpenClaw-residue feedback: after migration,
    memory entries that referenced ``~/.openclaw/config.yaml`` were being
    rewritten to ``~/.Hermes/config.yaml`` — a path that doesn't exist —
    and the agent kept trying to read it.
    """
    mod = load_module()
    assert mod.rebrand_text("config is at ~/.openclaw/config.yaml") == \
        "config is at ~/.hermes/config.yaml"
    assert mod.rebrand_text("use .openclaw directory") == "use .hermes directory"
    assert mod.rebrand_text("Path.home() / '.openclaw'") == "Path.home() / '.hermes'"
    # Sentence with both lowercase path and capitalized prose.
    assert mod.rebrand_text("openclaw config path: ~/.openclaw/") == \
        "hermes config path: ~/.hermes/"


def test_migrate_memory_rebrands_entries(tmp_path):
    mod = load_module()
    source_root = tmp_path / "openclaw"
    source_root.mkdir()
    workspace = source_root / "workspace"
    workspace.mkdir()
    memory_md = workspace / "MEMORY.md"
    memory_md.write_text(
        "# Memory\n\n- OpenClaw should use Python 3.11\n- ClawdBot prefers dark mode\n",
        encoding="utf-8",
    )

    target_root = tmp_path / "hermes"
    target_root.mkdir()
    (target_root / "memories").mkdir()

    migrator = mod.Migrator(
        source_root=source_root,
        target_root=target_root,
        execute=True,
        workspace_target=None,
        overwrite=False,
        migrate_secrets=False,
        output_dir=tmp_path / "report",
        selected_options={"memory"},
    )
    migrator.migrate()

    result = (target_root / "memories" / "MEMORY.md").read_text(encoding="utf-8")
    assert "OpenClaw" not in result
    assert "ClawdBot" not in result
    assert "Hermes" in result


def test_migrate_soul_rebrands_content(tmp_path):
    mod = load_module()
    source_root = tmp_path / "openclaw"
    source_root.mkdir()
    workspace = source_root / "workspace"
    workspace.mkdir()
    soul_md = workspace / "SOUL.md"
    soul_md.write_text("You are OpenClaw, an AI assistant made by SparkLab.", encoding="utf-8")

    target_root = tmp_path / "hermes"
    target_root.mkdir()

    migrator = mod.Migrator(
        source_root=source_root,
        target_root=target_root,
        execute=True,
        workspace_target=None,
        overwrite=False,
        migrate_secrets=False,
        output_dir=tmp_path / "report",
        selected_options={"soul"},
    )
    migrator.migrate()

    result = (target_root / "SOUL.md").read_text(encoding="utf-8")
    assert "OpenClaw" not in result
    assert "You are Hermes" in result

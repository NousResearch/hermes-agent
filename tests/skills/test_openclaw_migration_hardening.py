"""Tests for the OpenClaw→Hermes migration hardening features.

Covers the changes in the "claw migrate hardening" PR:
  - secret redaction (engine-level, applied to report JSON)
  - warnings[] / next_steps[] on the report
  - blocked-by-earlier-conflict sequencing for config.yaml mutations
  - --json output mode on the migration script
  - enum-like constants and ItemResult.sensitive field
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
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


def _load():
    spec = importlib.util.spec_from_file_location("openclaw_to_hermes_hard", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ───────────────────────────────────────────────────────────────────────
# Redaction
# ───────────────────────────────────────────────────────────────────────
def test_redact_replaces_secret_by_key_name():
    mod = _load()
    out = mod.redact_migration_value({"OPENROUTER_API_KEY": "sk-or-v1-abcdef12345678"})
    assert out["OPENROUTER_API_KEY"] == mod.REDACTED_MIGRATION_VALUE


def test_redact_replaces_secret_by_value_pattern():
    mod = _load()
    # Even under a non-secret-looking key, the sk-... pattern should be replaced inline.
    out = mod.redact_migration_value({"note": "use sk-or-v1-9Xs7fF2JkLmNpQrT to authenticate"})
    assert "sk-or-" not in out["note"]
    assert mod.REDACTED_MIGRATION_VALUE in out["note"]


def test_redact_handles_github_token_pattern():
    mod = _load()
    out = mod.redact_migration_value({"detail": "token: ghp_1234567890abcdef1234"})
    assert "ghp_" not in out["detail"]
    assert mod.REDACTED_MIGRATION_VALUE in out["detail"]


def test_redact_handles_slack_token_pattern():
    mod = _load()
    out = mod.redact_migration_value("xoxb-1234567890-abcdef")
    assert out == mod.REDACTED_MIGRATION_VALUE


def test_redact_handles_google_api_key_pattern():
    mod = _load()
    out = mod.redact_migration_value("AIzaSyA-abc123def456ghi")
    # Google key is a prefix — whole value is scrubbed
    assert "AIza" not in out


def test_redact_handles_bearer_header():
    mod = _load()
    out = mod.redact_migration_value({"hint": "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abc"})
    # Key "hint" is not a secret marker — only the Bearer <token> substring
    # gets scrubbed inline by the value pattern.
    assert "Bearer eyJ" not in out["hint"]
    assert mod.REDACTED_MIGRATION_VALUE in out["hint"]


def test_redact_is_recursive():
    mod = _load()
    nested = {
        "outer": {
            "items": [
                {"password": "hunter2"},
                {"details": {"apiKey": "my-key"}},
            ],
        },
    }
    out = mod.redact_migration_value(nested)
    assert out["outer"]["items"][0]["password"] == mod.REDACTED_MIGRATION_VALUE
    assert out["outer"]["items"][1]["details"]["apiKey"] == mod.REDACTED_MIGRATION_VALUE


def test_redact_preserves_non_secret_keys_and_values():
    mod = _load()
    input_data = {"name": "hermes", "count": 42, "tags": ["a", "b"]}
    out = mod.redact_migration_value(input_data)
    assert out == input_data


def test_redact_normalizes_key_case_and_punctuation():
    mod = _load()
    # "Api Key", "api-key", "API_KEY" all normalize the same way.
    for key in ("Api Key", "api-key", "API_KEY", "apikey"):
        out = mod.redact_migration_value({key: "secret"})
        assert out[key] == mod.REDACTED_MIGRATION_VALUE, f"failed to redact: {key}"


def test_redact_leaves_env_secretref_alone():
    """SecretRef-like shapes ({source: env, id: ...}) are pointers, not secrets."""
    mod = _load()
    ref = {"source": "env", "id": "OPENAI_API_KEY"}
    out = mod.redact_migration_value({"apiKey": ref})
    # The key "apiKey" itself triggers redaction today — this test locks that in.
    # If we later want to exempt SecretRef values the way OpenClaw does, update
    # both this test and _redact_internal together.
    assert out["apiKey"] == mod.REDACTED_MIGRATION_VALUE


def test_write_report_redacts_api_keys_on_disk(tmp_path):
    mod = _load()
    report = {
        "timestamp": "20260427T120000",
        "mode": "execute",
        "source_root": "/src",
        "target_root": "/tgt",
        "summary": {"migrated": 1, "conflict": 0, "error": 0, "skipped": 0, "archived": 0},
        "items": [
            {
                "kind": "provider-keys",
                "source": "openclaw.json",
                "destination": "/tgt/.env",
                "status": "migrated",
                "reason": "",
                "details": {"OPENROUTER_API_KEY": "sk-or-v1-1234567890abcdef"},
            },
        ],
    }
    mod.write_report(tmp_path, report)
    persisted = json.loads((tmp_path / "report.json").read_text())
    # The raw secret must not appear anywhere in the persisted JSON.
    assert "sk-or-v1-1234567890abcdef" not in (tmp_path / "report.json").read_text()
    assert persisted["items"][0]["details"]["OPENROUTER_API_KEY"] == mod.REDACTED_MIGRATION_VALUE


# ───────────────────────────────────────────────────────────────────────
# Warnings and next-steps
# ───────────────────────────────────────────────────────────────────────
def _make_minimal_migrator(mod, tmp_path, **overrides):
    source = tmp_path / "openclaw"
    source.mkdir()
    # Minimal valid OpenClaw layout so the Migrator constructor doesn't choke.
    (source / "openclaw.json").write_text("{}", encoding="utf-8")
    target = tmp_path / "hermes"
    target.mkdir()
    defaults = dict(
        source_root=source,
        target_root=target,
        execute=False,
        workspace_target=None,
        overwrite=False,
        migrate_secrets=False,
        output_dir=None,
        selected_options=set(),
    )
    defaults.update(overrides)
    return mod.Migrator(**defaults)


def test_dry_run_report_includes_rerun_next_step(tmp_path):
    mod = _load()
    migrator = _make_minimal_migrator(mod, tmp_path)
    report = migrator.migrate()
    steps = report["next_steps"]
    assert any("dry-run" in step.lower() or "re-run" in step.lower() for step in steps)


def test_conflict_produces_overwrite_warning(tmp_path):
    mod = _load()
    migrator = _make_minimal_migrator(mod, tmp_path, execute=True)
    # Inject a conflict on a config.yaml target to exercise the warning pathway.
    migrator.record(
        "tts-config",
        source=None,
        destination=migrator.target_root / "config.yaml",
        status=mod.STATUS_CONFLICT,
        reason="TTS already configured",
    )
    report = migrator.build_report()
    assert any("--overwrite" in w for w in report["warnings"])
    # The conflict on config.yaml should have flipped the block flag too.
    assert migrator._config_apply_blocked is True


def test_error_produces_inspect_warning(tmp_path):
    mod = _load()
    migrator = _make_minimal_migrator(mod, tmp_path, execute=True)
    migrator.record("mcp-servers", None, None, mod.STATUS_ERROR, "Bad YAML")
    report = migrator.build_report()
    assert any("failed" in w.lower() for w in report["warnings"])


def test_provider_keys_skipped_warning_when_secrets_disabled(tmp_path):
    mod = _load()
    migrator = _make_minimal_migrator(mod, tmp_path, execute=True, migrate_secrets=False)
    migrator.record(
        "provider-keys",
        source=None,
        destination=None,
        status=mod.STATUS_SKIPPED,
        reason="--migrate-secrets not set",
    )
    report = migrator.build_report()
    assert any("--migrate-secrets" in w for w in report["warnings"])


# ───────────────────────────────────────────────────────────────────────
# Blocked-by-earlier-conflict sequencing
# ───────────────────────────────────────────────────────────────────────
def test_config_apply_block_flips_on_config_yaml_conflict(tmp_path):
    mod = _load()
    migrator = _make_minimal_migrator(mod, tmp_path, execute=True)
    assert migrator._config_apply_blocked is False
    migrator.record(
        "model-config",
        source=None,
        destination=migrator.target_root / "config.yaml",
        status=mod.STATUS_CONFLICT,
    )
    assert migrator._config_apply_blocked is True


def test_config_apply_block_flips_on_config_yaml_error(tmp_path):
    mod = _load()
    migrator = _make_minimal_migrator(mod, tmp_path, execute=True)
    migrator.record(
        "tts-config",
        source=None,
        destination=migrator.target_root / "config.yaml",
        status=mod.STATUS_ERROR,
        reason="YAML write failed",
    )
    assert migrator._config_apply_blocked is True


def test_config_apply_block_does_not_flip_on_non_config_conflict(tmp_path):
    mod = _load()
    migrator = _make_minimal_migrator(mod, tmp_path, execute=True)
    migrator.record(
        "skill",
        source=None,
        destination=migrator.target_root / "skills" / "foo" / "SKILL.md",
        status=mod.STATUS_CONFLICT,
    )
    assert migrator._config_apply_blocked is False


def test_run_if_selected_skips_config_ops_after_block(tmp_path):
    mod = _load()
    migrator = _make_minimal_migrator(
        mod, tmp_path, execute=True, selected_options={"model-config", "tts-config"}
    )
    migrator._config_apply_blocked = True
    called = []
    migrator.run_if_selected("tts-config", lambda: called.append(True))
    assert called == []
    # The skipped record uses the blocked reason.
    blocked = [i for i in migrator.items if i.kind == "tts-config"]
    assert len(blocked) == 1
    assert blocked[0].status == mod.STATUS_SKIPPED
    assert blocked[0].reason == mod.REASON_BLOCKED_BY_APPLY_CONFLICT


def test_run_if_selected_runs_non_config_ops_even_after_block(tmp_path):
    mod = _load()
    migrator = _make_minimal_migrator(
        mod, tmp_path, execute=True, selected_options={"soul"}
    )
    migrator._config_apply_blocked = True
    called = []
    migrator.run_if_selected("soul", lambda: called.append(True))
    assert called == [True]


def test_dry_run_never_blocks_even_after_conflict(tmp_path):
    """Dry runs must preview the full plan — blocking mid-preview would hide
    conflicts and mislead the user about what would actually happen."""
    mod = _load()
    migrator = _make_minimal_migrator(
        mod, tmp_path, execute=False, selected_options={"tts-config"}
    )
    migrator._config_apply_blocked = True
    called = []
    migrator.run_if_selected("tts-config", lambda: called.append(True))
    assert called == [True]


# ───────────────────────────────────────────────────────────────────────
# --json output mode
# ───────────────────────────────────────────────────────────────────────
def test_json_mode_emits_structured_report(tmp_path):
    """End-to-end: run the CLI with --json and no --execute, parse stdout."""
    source = tmp_path / "openclaw"
    source.mkdir()
    (source / "openclaw.json").write_text(
        json.dumps({"agents": {"defaults": {"model": "openrouter/anthropic/claude-sonnet-4"}}}),
        encoding="utf-8",
    )
    target = tmp_path / "hermes"
    target.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--source", str(source),
            "--target", str(target),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "summary" in payload
    assert "warnings" in payload
    assert "next_steps" in payload
    assert payload["mode"] == "dry-run"


def test_json_mode_redacts_secrets_in_output(tmp_path):
    """Even plan-only JSON output goes through the redactor — the stdout
    capture path is what gets piped into CI / support tickets."""
    source = tmp_path / "openclaw"
    source.mkdir()
    (source / "openclaw.json").write_text("{}", encoding="utf-8")
    # Plant a fake OpenClaw .env with a recognizably-shaped key.
    (source / ".env").write_text(
        "OPENROUTER_API_KEY=sk-or-v1-abcdef1234567890abcdef\n", encoding="utf-8"
    )
    target = tmp_path / "hermes"
    target.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--source", str(source),
            "--target", str(target),
            "--migrate-secrets",  # so provider-keys surface in the plan
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    # The raw key value must never appear in the JSON output.
    assert "sk-or-v1-abcdef1234567890abcdef" not in result.stdout


# ───────────────────────────────────────────────────────────────────────
# ItemResult schema additions
# ───────────────────────────────────────────────────────────────────────
def test_item_result_has_sensitive_field():
    mod = _load()
    item = mod.ItemResult(kind="x", source=None, destination=None, status="migrated")
    assert item.sensitive is False


def test_record_honors_sensitive_flag(tmp_path):
    mod = _load()
    migrator = _make_minimal_migrator(mod, tmp_path)
    migrator.record("x", None, None, "migrated", sensitive=True)
    assert migrator.items[0].sensitive is True


def test_status_constants_match_historical_strings():
    """Downstream consumers (claw.py, tests, docs) depend on these string values."""
    mod = _load()
    assert mod.STATUS_MIGRATED == "migrated"
    assert mod.STATUS_SKIPPED == "skipped"
    assert mod.STATUS_CONFLICT == "conflict"
    assert mod.STATUS_ERROR == "error"
    assert mod.STATUS_ARCHIVED == "archived"

"""Tests for portable Hermes profile sync."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hermes_cli.sync import (
    SyncResult,
    _queue_tree_import,
    doctor_sync_repo,
    export_profile_sync,
    import_profile_sync,
    run_migrate,
)


def _symlink_or_skip(target: Path, link: Path) -> None:
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):  # pragma: no cover
        pytest.skip("platform does not support symlinks")


def _write_sync_manifest(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "manifest.yaml").write_text(
        "format: hermes-profile-sync\n"
        "version: 1\n"
        "created_at: '2026-04-29T00:00:00+00:00'\n"
        "source_device: test-device\n",
        encoding="utf-8",
    )


def _memory_entry(target: str, content: str) -> dict:
    import hashlib

    digest = hashlib.sha256(f"{target}\0{content}".encode("utf-8")).hexdigest()[:24]
    return {
        "id": f"mem_{digest}",
        "target": target,
        "content": content,
        "source_device": "remote",
        "created_at": "1970-01-01T00:00:00Z",
        "updated_at": "",
        "deleted": False,
    }


def _public_ops(result):
    return [
        {key: value for key, value in op.items() if not key.startswith("_")}
        for op in result.plan
    ]


def test_export_writes_structured_repo_and_excludes_runtime_state(tmp_path):
    home = tmp_path / "home"
    repo = tmp_path / "sync-repo"
    (home / "memories").mkdir(parents=True)
    (home / "skills" / "my-skill").mkdir(parents=True)
    (home / "skins").mkdir()
    (home / "logs").mkdir()

    (home / "config.yaml").write_text(
        "model: openrouter/test-model\n"
        "display:\n"
        "  skin: slate\n"
        "terminal:\n"
        "  cwd: /home/amos/project\n"
        "skills:\n"
        "  template_vars: false\n"
        "  external_dirs:\n"
        "    - /shared/private-skills\n"
        "providers:\n"
        "  custom:\n"
        "    base_url: https://models.example.test/v1\n"
        "    api_key: sk-test-secret-value-1234567890\n"
        "unknown_feature:\n"
        "  enabled: true\n",
        encoding="utf-8",
    )
    (home / ".env").write_text("OPENAI_API_KEY=sk-should-not-export\n", encoding="utf-8")
    (home / "auth.json").write_text('{"access_token": "secret"}', encoding="utf-8")
    (home / "state.db").write_bytes(b"sqlite")
    (home / "logs" / "agent.log").write_text("log\n", encoding="utf-8")
    (home / "SOUL.md").write_text("I prefer concise answers.\n", encoding="utf-8")
    (home / "memories" / "MEMORY.md").write_text(
        "# Memory\n- likes tea\nsk-secret-memory-value-123456789\n",
        encoding="utf-8",
    )
    (home / "memories" / "USER.md").write_text("# User\n- uses vim\n", encoding="utf-8")
    (home / "skills" / "my-skill" / "SKILL.md").write_text("# My Skill\n", encoding="utf-8")
    (home / "skins" / "cyber.yaml").write_text("name: cyber\n", encoding="utf-8")

    result = export_profile_sync(repo, hermes_home=home, device_id="Laptop 1")

    assert result.ok
    assert (repo / "manifest.yaml").exists()
    assert (repo / "config" / "global.yaml").exists()
    assert (repo / "config" / "devices" / "laptop-1.yaml").exists()
    assert (repo / "soul" / "SOUL.md").read_text(encoding="utf-8") == "I prefer concise answers.\n"
    assert (repo / "skills" / "files" / "my-skill" / "SKILL.md").exists()
    assert (repo / "skins" / "cyber.yaml").exists()

    global_cfg = yaml.safe_load((repo / "config" / "global.yaml").read_text(encoding="utf-8"))
    device_cfg = yaml.safe_load(
        (repo / "config" / "devices" / "laptop-1.yaml").read_text(encoding="utf-8")
    )
    assert global_cfg["model"] == "openrouter/test-model"
    assert global_cfg["display"]["skin"] == "slate"
    assert global_cfg["skills"]["template_vars"] is False
    assert "external_dirs" not in global_cfg["skills"]
    assert "terminal" not in global_cfg
    assert "unknown_feature" not in global_cfg
    assert "api_key" not in global_cfg["providers"]["custom"]
    assert device_cfg["terminal"]["cwd"] == "/home/amos/project"
    assert device_cfg["skills"]["external_dirs"] == ["/shared/private-skills"]

    exported_paths = [p.relative_to(repo).as_posix() for p in repo.rglob("*") if p.is_file()]
    assert ".env" not in exported_paths
    assert "auth.json" not in exported_paths
    assert "state.db" not in exported_paths
    assert "logs/agent.log" not in exported_paths
    exported_text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in repo.rglob("*") if p.is_file())
    assert "sk-test-secret" not in exported_text
    assert "sk-secret-memory" not in exported_text
    assert any("unknown sync ownership" in warning for warning in result.warnings)
    assert doctor_sync_repo(repo).ok


def test_import_dry_run_does_not_write_and_actual_uses_same_plan(tmp_path):
    home = tmp_path / "home"
    repo = tmp_path / "sync-repo"
    home.mkdir()
    _write_sync_manifest(repo)
    (repo / "config").mkdir()
    (repo / "config" / "global.yaml").write_text(
        "model: remote-model\n"
        "display:\n"
        "  skin: mono\n",
        encoding="utf-8",
    )
    (repo / "soul").mkdir()
    (repo / "soul" / "SOUL.md").write_text("remote soul\n", encoding="utf-8")

    dry_run = import_profile_sync(repo, hermes_home=home, device_id="laptop", dry_run=True)

    assert dry_run.ok
    assert not (home / "config.yaml").exists()
    assert not (home / "SOUL.md").exists()
    assert {op["target"] for op in dry_run.plan} == {"config.yaml", "SOUL.md"}

    actual = import_profile_sync(repo, hermes_home=home, device_id="laptop", dry_run=False)

    assert actual.ok
    assert _public_ops(actual) == _public_ops(dry_run)
    assert yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["model"] == "remote-model"
    assert (home / "SOUL.md").read_text(encoding="utf-8") == "remote soul\n"


def test_import_merges_remote_and_local_memory_entries(tmp_path):
    home = tmp_path / "home"
    repo = tmp_path / "sync-repo"
    (home / "memories").mkdir(parents=True)
    _write_sync_manifest(repo)
    (repo / "memory").mkdir()
    (repo / "memory" / "entries.jsonl").write_text(
        json.dumps(_memory_entry("memory", "- remote memory"), sort_keys=True) + "\n"
        + json.dumps(_memory_entry("user", "- remote user"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (home / "memories" / "MEMORY.md").write_text("# Memory\n- local memory\n", encoding="utf-8")

    result = import_profile_sync(repo, hermes_home=home, device_id="laptop", dry_run=False)

    assert result.ok
    memory_text = (home / "memories" / "MEMORY.md").read_text(encoding="utf-8")
    user_text = (home / "memories" / "USER.md").read_text(encoding="utf-8")
    assert "- local memory" in memory_text
    assert "- remote memory" in memory_text
    assert "- remote user" in user_text


def test_import_keeps_local_soul_and_writes_conflict_file(tmp_path):
    home = tmp_path / "home"
    repo = tmp_path / "sync-repo"
    home.mkdir()
    _write_sync_manifest(repo)
    (repo / "soul").mkdir()
    (repo / "soul" / "SOUL.md").write_text("remote soul\n", encoding="utf-8")
    (home / "SOUL.md").write_text("local soul\n", encoding="utf-8")

    result = import_profile_sync(repo, hermes_home=home, device_id="laptop", dry_run=False)

    assert result.ok
    assert (home / "SOUL.md").read_text(encoding="utf-8") == "local soul\n"
    conflicts = list(home.glob("SOUL.md.sync-conflict-*"))
    assert len(conflicts) == 1
    assert conflicts[0].read_text(encoding="utf-8") == "remote soul\n"
    assert any("Conflict for SOUL.md" in warning for warning in result.warnings)


def test_import_applies_matching_device_config_without_overwriting_other_device(tmp_path):
    home = tmp_path / "home"
    repo = tmp_path / "sync-repo"
    home.mkdir()
    _write_sync_manifest(repo)
    (repo / "config" / "devices").mkdir(parents=True)
    (repo / "config" / "global.yaml").write_text(
        "display:\n"
        "  skin: slate\n",
        encoding="utf-8",
    )
    (repo / "config" / "devices" / "other.yaml").write_text(
        "terminal:\n"
        "  cwd: /other/device\n",
        encoding="utf-8",
    )
    (repo / "config" / "devices" / "laptop.yaml").write_text(
        "terminal:\n"
        "  cwd: /this/device\n",
        encoding="utf-8",
    )

    result = import_profile_sync(repo, hermes_home=home, device_id="laptop", dry_run=False)

    assert result.ok
    config = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert config["display"]["skin"] == "slate"
    assert config["terminal"]["cwd"] == "/this/device"


def test_doctor_rejects_malformed_manifest_forbidden_paths_and_plaintext_secrets(tmp_path):
    repo = tmp_path / "sync-repo"
    repo.mkdir()
    (repo / "manifest.yaml").write_text("format: wrong\nversion: 1\n", encoding="utf-8")

    result = doctor_sync_repo(repo)

    assert not result.ok
    assert any("invalid or missing format" in error for error in result.errors)

    (repo / "manifest.yaml").write_text("format: hermes-profile-sync\nversion: 1\n", encoding="utf-8")
    (repo / "state.db").write_bytes(b"sqlite")
    (repo / "skills").mkdir()
    (repo / "skills" / "leak.txt").write_text("sk-live-secret-value-1234567890\n", encoding="utf-8")

    result = doctor_sync_repo(repo)

    assert not result.ok
    assert any("forbidden path" in error and "state.db" in error for error in result.errors)
    assert any("plaintext secret-like" in error and "skills/leak.txt" in error for error in result.errors)


def test_migrate_verify_and_doctor_share_sync_validation(tmp_path, capsys):
    repo = tmp_path / "migration-bundle"
    _write_sync_manifest(repo)

    run_migrate(SimpleNamespace(migrate_type="verify", repo=str(repo)))
    verify_output = capsys.readouterr().out

    run_migrate(SimpleNamespace(migrate_type="doctor", repo=str(repo)))
    doctor_output = capsys.readouterr().out

    assert "Sync repo OK" in verify_output
    assert "Sync repo OK" in doctor_output


def test_migrate_export_import_round_trip_uses_portable_bundle(tmp_path, monkeypatch):
    source_home = tmp_path / "source-home"
    target_home = tmp_path / "target-home"
    bundle = tmp_path / "migration-bundle"
    source_home.mkdir()
    target_home.mkdir()

    (source_home / "config.yaml").write_text(
        "model: openrouter/test-model\n"
        "terminal:\n"
        "  cwd: /source/local/path\n",
        encoding="utf-8",
    )
    (source_home / "SOUL.md").write_text("portable soul\n", encoding="utf-8")

    monkeypatch.setattr("hermes_cli.sync.get_hermes_home", lambda: source_home)
    run_migrate(
        SimpleNamespace(
            migrate_type="export",
            out=str(bundle),
            device_id="laptop",
        )
    )
    assert (bundle / "manifest.yaml").exists()

    monkeypatch.setattr("hermes_cli.sync.get_hermes_home", lambda: target_home)
    run_migrate(
        SimpleNamespace(
            migrate_type="import",
            source_dir=str(bundle),
            dry_run=False,
            device_id="laptop",
        )
    )

    target_config = yaml.safe_load(
        (target_home / "config.yaml").read_text(encoding="utf-8")
    )
    assert target_config["model"] == "openrouter/test-model"
    assert target_config["terminal"]["cwd"] == "/source/local/path"
    assert (target_home / "SOUL.md").read_text(encoding="utf-8") == "portable soul\n"


def test_doctor_rejects_symlinked_bundle_entries(tmp_path):
    repo = tmp_path / "sync-repo"
    _write_sync_manifest(repo)
    secret = tmp_path / "outside-secret.txt"
    secret.write_text("not a bundle file\n", encoding="utf-8")
    (repo / "skills" / "files").mkdir(parents=True)
    _symlink_or_skip(secret, repo / "skills" / "files" / "evil.md")

    result = doctor_sync_repo(repo)

    assert not result.ok
    assert any(
        "symlink" in error and "skills/files/evil.md" in error
        for error in result.errors
    )


def test_import_rejects_symlinked_skill_and_skin_files(tmp_path):
    home = tmp_path / "home"
    repo = tmp_path / "sync-repo"
    home.mkdir()
    _write_sync_manifest(repo)
    secret = tmp_path / "outside-secret.txt"
    secret.write_text("sensitive local content\n", encoding="utf-8")
    (repo / "skills" / "files" / "my-skill").mkdir(parents=True)
    (repo / "skins").mkdir()
    _symlink_or_skip(secret, repo / "skills" / "files" / "my-skill" / "SKILL.md")
    _symlink_or_skip(secret, repo / "skins" / "cyber.yaml")

    result = import_profile_sync(repo, hermes_home=home, device_id="laptop", dry_run=False)

    assert not result.ok
    assert any("symlink" in error for error in result.errors)
    # Nothing may be written when the bundle contains symlinks.
    assert not (home / "skills").exists()
    assert not (home / "skins").exists()
    assert "sensitive local content" not in "".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in home.rglob("*")
        if p.is_file()
    )


def test_queue_tree_import_rejects_symlinks_even_without_doctor(tmp_path):
    """Regression: _queue_tree_import must be safe in isolation (pre-read pass)."""
    home = tmp_path / "home"
    src_root = tmp_path / "bundle-skills"
    (src_root / "nested").mkdir(parents=True)
    secret = tmp_path / "outside-secret.txt"
    secret.write_text("sensitive local content\n", encoding="utf-8")
    _symlink_or_skip(secret, src_root / "nested" / "linked.md")

    result = SyncResult(repo=src_root)
    _queue_tree_import(result, src_root, home / "skills", hermes_home=home)

    assert not result.ok
    assert result.plan == []
    assert any(
        "symlink" in error and "nested/linked.md" in error for error in result.errors
    )

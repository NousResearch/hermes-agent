from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

from hermes_cli import profile_sync


def _write_skill(root: Path, rel: str, body: str, *, extra_name: str | None = None) -> None:
    skill_dir = root / "skills" / rel
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")
    if extra_name:
        (skill_dir / extra_name).write_text("extra", encoding="utf-8")


def _patch_profile_dirs(monkeypatch, source: Path, target: Path) -> None:
    monkeypatch.setattr(profile_sync, "profile_exists", lambda name: True)
    monkeypatch.setattr(
        profile_sync,
        "get_profile_dir",
        lambda name: source if name == "source" else target,
    )


class _BaseArgs:
    source = "source"
    target = "target"
    json = False
    apply = False
    with_backup = False


def test_profile_sync_skills_dry_run_reports_missing_extra_and_changed(tmp_path, monkeypatch):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    _write_skill(source, "cat/alpha", "---\nname: alpha\n---\nsource\n")
    _write_skill(source, "cat/shared", "---\nname: shared\n---\nsource\n")
    _write_skill(target, "cat/shared", "---\nname: shared\n---\ntarget\n")
    _write_skill(target, "cat/extra", "---\nname: extra\n---\ntarget\n")
    _patch_profile_dirs(monkeypatch, source, target)

    class Args(_BaseArgs):
        profile_sync_action = "skills"

    report = profile_sync._build_report(Args())
    skills = report["skills"]

    assert report["mode"] == "dry-run"
    assert skills["missing_in_target"] == ["cat/alpha"]
    assert skills["extra_in_target"] == ["cat/extra"]
    assert skills["changed"] == ["cat/shared"]
    assert skills["would_copy"] == ["cat/alpha", "cat/shared"]
    assert skills["would_delete"] == []
    assert report["safety"]["read_only"] is True
    assert ".env" in report["safety"]["excluded_state"]
    assert "auth.json" in report["safety"]["excluded_state"]


def test_profile_sync_memories_dry_run_reports_entry_counts(tmp_path, monkeypatch):
    source = tmp_path / "source"
    target = tmp_path / "target"
    (source / "memories").mkdir(parents=True)
    (target / "memories").mkdir(parents=True)
    (source / "memories" / "MEMORY.md").write_text("one\n§\ntwo\n", encoding="utf-8")
    (target / "memories" / "MEMORY.md").write_text("one\n", encoding="utf-8")
    (source / "memories" / "USER.md").write_text("same", encoding="utf-8")
    (target / "memories" / "USER.md").write_text("same", encoding="utf-8")
    _patch_profile_dirs(monkeypatch, source, target)

    class Args(_BaseArgs):
        profile_sync_action = "memories"

    report = profile_sync._build_report(Args())
    files = {item["path"]: item for item in report["memories"]["files"]}

    assert files["memories/MEMORY.md"]["status"] == "changed"
    assert files["memories/MEMORY.md"]["source_entries"] == 2
    assert files["memories/MEMORY.md"]["target_entries"] == 1
    assert files["memories/USER.md"]["status"] == "same"
    assert report["memories"]["apply_supported"] is False


def test_profile_sync_memories_apply_is_refused(tmp_path, monkeypatch):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    _patch_profile_dirs(monkeypatch, source, target)

    class Args(_BaseArgs):
        profile_sync_action = "memories"
        apply = True

    try:
        profile_sync._build_report(Args())
    except ValueError as exc:
        assert "memory apply is disabled" in str(exc)
    else:
        raise AssertionError("memory apply should be refused")


def test_profile_sync_skills_apply_requires_backup(tmp_path, monkeypatch):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    _write_skill(source, "cat/alpha", "---\nname: alpha\n---\nsource\n")
    _patch_profile_dirs(monkeypatch, source, target)

    class Args(_BaseArgs):
        profile_sync_action = "skills"
        apply = True
        with_backup = False

    try:
        profile_sync._build_report(Args())
    except ValueError as exc:
        assert "requires --with-backup" in str(exc)
    else:
        raise AssertionError("skills apply without backup should be refused")


def test_profile_sync_skills_apply_copies_missing_and_changed_keeps_extra_and_backs_up(tmp_path, monkeypatch):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    _write_skill(source, "cat/alpha", "---\nname: alpha\n---\nsource\n")
    _write_skill(source, "cat/shared", "---\nname: shared\n---\nsource\n", extra_name="note.md")
    _write_skill(target, "cat/shared", "---\nname: shared\n---\ntarget\n")
    _write_skill(target, "cat/extra", "---\nname: extra\n---\ntarget\n")
    _patch_profile_dirs(monkeypatch, source, target)

    class Args(_BaseArgs):
        profile_sync_action = "skills"
        apply = True
        with_backup = True

    report = profile_sync._build_report(Args())

    assert report["mode"] == "apply"
    assert report["apply"]["copied"] == ["cat/alpha", "cat/shared"]
    assert report["apply"]["deleted"] == []
    assert Path(report["apply"]["backup_path"]).exists()
    assert (target / "skills" / "cat" / "alpha" / "SKILL.md").read_text(encoding="utf-8").endswith("source\n")
    assert (target / "skills" / "cat" / "shared" / "SKILL.md").read_text(encoding="utf-8").endswith("source\n")
    assert (target / "skills" / "cat" / "shared" / "note.md").exists()
    assert (target / "skills" / "cat" / "extra" / "SKILL.md").exists()
    assert report["post_apply_skills"]["missing_in_target"] == []
    assert report["post_apply_skills"]["changed"] == []


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_profile_sync_memories_apply_exits_nonzero_through_cli_entrypoint(tmp_path):
    hermes_home = tmp_path / "hermes-home"
    (hermes_home / "profiles" / "engineer" / "skills").mkdir(parents=True)
    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    env["PYTHONPATH"] = os.getcwd()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "profile-sync",
            "memories",
            "--source",
            "engineer",
            "--target",
            "default",
            "--apply",
        ],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert "memory apply is disabled until merge rules exist" in result.stderr


def test_profile_sync_skills_apply_leaves_non_skill_profile_state_untouched(tmp_path, monkeypatch):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    _write_skill(source, "cat/alpha", "---\nname: alpha\n---\nsource\n")
    _write_skill(target, "cat/alpha", "---\nname: alpha\n---\ntarget\n")

    sentinel_files = {
        ".env": "ENV_SENTINEL=value\n",
        "auth.json": '{"auth":"keep"}\n',
        "config.yaml": "model:\n  default: keep\n",
        "state.db": "sqlite-ish sentinel\n",
        "memories/MEMORY.md": "target memory\n§\nkeep\n",
        "memories/USER.md": "target user\n§\nkeep\n",
        "sessions/session.jsonl": "session sentinel\n",
        "cron/jobs.json": "cron sentinel\n",
    }
    for rel, content in sentinel_files.items():
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    before_hashes = _hash_tree(target)
    _patch_profile_dirs(monkeypatch, source, target)

    class Args(_BaseArgs):
        profile_sync_action = "skills"
        apply = True
        with_backup = True

    report = profile_sync._build_report(Args())

    assert report["apply"]["copied"] == ["cat/alpha"]
    after_hashes = _hash_tree(target)
    for rel in sentinel_files:
        assert rel in after_hashes
        assert after_hashes[rel] == before_hashes[rel]
    assert (target / "sessions").is_dir()
    assert (target / "cron").is_dir()

"""E2E tests for ``hermes cloud export`` (migration bundles).

Runs the real export path against a synthetic HERMES_HOME (temp dir via
monkeypatched HERMES_HOME env), then verifies the archive contract: secret
files absent by default, re-includable by flag, a manifest at the root, a
secret scan that reports (but never drops) planted values, and round-trip
restorability through the real ``hermes import`` path.
"""

from __future__ import annotations

import json
import os
import sqlite3
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.cloud_migrate import (
    MIGRATION_SCHEMA_VERSION,
    run_cloud_export,
)


@pytest.fixture()
def synth_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A minimal but real HERMES_HOME with every asset class the bundle covers."""
    home = tmp_path / "hermes"
    (home / "skills" / "custom-skill").mkdir(parents=True)
    (home / "memory").mkdir()
    (home / "cron").mkdir()
    (home / "profiles" / "other").mkdir(parents=True)

    (home / "config.yaml").write_text(
        "_config_version: 4\nmodel:\n  provider: nous\n", encoding="utf-8"
    )
    (home / "skills" / "custom-skill" / "SKILL.md").write_text(
        "---\nname: custom-skill\ndescription: test\n---\n# Body\n",
        encoding="utf-8",
    )
    (home / "memory" / "memory.md").write_text("remember this", encoding="utf-8")
    (home / "SOUL.md").write_text("be concise", encoding="utf-8")
    (home / "cron" / "jobs.json").write_text(
        json.dumps({"jobs": [{"id": "a"}, {"id": "b"}]}), encoding="utf-8"
    )
    (home / ".env").write_text(
        "# comment\nANTHROPIC_API_KEY=sk-ant-real-value\nOPENROUTER_API_KEY=or-key\n",
        encoding="utf-8",
    )
    (home / "auth.json").write_text('{"token": "x"}', encoding="utf-8")
    # A per-profile secret file: basename exclusion must reach it too.
    (home / "profiles" / "other" / ".env").write_text(
        "PROFILE_KEY=value\n", encoding="utf-8"
    )

    conn = sqlite3.connect(home / "state.db")
    conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, body TEXT)")
    conn.execute("INSERT INTO sessions (body) VALUES ('hello')")
    conn.commit()
    conn.close()

    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _run_export(capsys, out: Path, **flags):
    args = SimpleNamespace(output=str(out), force=True, **flags)
    code = run_cloud_export(args)
    captured = capsys.readouterr()
    assert code == 0, captured.out
    return captured


def _namelist(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return zf.namelist()


def _manifest(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read("migration-manifest.json"))


def _contains_basename(names: list[str], basename: str) -> bool:
    return any(Path(n).name == basename for n in names)


def test_export_defaults_exclude_secrets_and_history(synth_home, tmp_path, capsys):
    out = tmp_path / "bundle.zip"
    _run_export(capsys, out)

    names = _namelist(out)
    assert "config.yaml" in names
    assert "skills/custom-skill/SKILL.md" in names
    assert "memory/memory.md" in names
    assert "SOUL.md" in names
    assert "cron/jobs.json" in names

    # Secret files excluded at any depth, state.db excluded by default.
    for basename in (".env", "auth.json", "state.db"):
        assert not _contains_basename(names, basename), basename

    manifest = _manifest(out)
    assert manifest["schema_version"] == MIGRATION_SCHEMA_VERSION
    assert manifest["secrets_included"] is False
    assert sorted(manifest["secrets_excluded"]) == [
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
    ]
    assert manifest["assets"]["state_db"] == {"included": False, "bytes": 0}
    assert manifest["assets"]["skills_count"] == 1
    assert manifest["assets"]["cron_jobs"] == 2
    assert manifest["config_version"] == 4


def test_include_history_adds_state_db(synth_home, tmp_path, capsys):
    out = tmp_path / "bundle.zip"
    _run_export(capsys, out, include_history=True)

    names = _namelist(out)
    assert _contains_basename(names, "state.db")
    assert not _contains_basename(names, ".env")

    manifest = _manifest(out)
    assert manifest["assets"]["state_db"]["included"] is True
    assert manifest["assets"]["state_db"]["bytes"] > 0


def test_include_secrets_adds_env_and_auth(synth_home, tmp_path, capsys):
    out = tmp_path / "bundle.zip"
    _run_export(capsys, out, include_secrets=True)

    names = _namelist(out)
    assert _contains_basename(names, ".env")
    assert _contains_basename(names, "auth.json")
    assert not _contains_basename(names, "state.db")

    manifest = _manifest(out)
    assert manifest["secrets_included"] is True
    assert manifest["secrets_excluded"] == []


def test_secret_scan_reports_but_does_not_drop(synth_home, tmp_path, capsys):
    # Plant a recognizable secret shape inside an included skill file.
    skill = synth_home / "skills" / "custom-skill" / "SKILL.md"
    skill.write_text(
        "---\nname: custom-skill\n---\n# Body\nexample: ghp_" + "A" * 36 + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "bundle.zip"
    captured = _run_export(capsys, out)

    names = _namelist(out)
    assert "skills/custom-skill/SKILL.md" in names  # never dropped
    assert "github token" in captured.out  # reported


def test_refuses_overwrite_without_force(synth_home, tmp_path, capsys):
    out = tmp_path / "bundle.zip"
    _run_export(capsys, out)

    args = SimpleNamespace(output=str(out), force=False)
    assert run_cloud_export(args) == 1
    assert "already exists" in capsys.readouterr().out


def test_bundle_roundtrips_through_import(synth_home, tmp_path, capsys, monkeypatch):
    out = tmp_path / "bundle.zip"
    _run_export(capsys, out, include_history=True)

    # Restore into a fresh home through the real import path.
    fresh = tmp_path / "restored"
    fresh.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(fresh))

    from hermes_cli.backup import run_import

    run_import(SimpleNamespace(zipfile=str(out), force=True))

    for rel in (
        "config.yaml",
        "SOUL.md",
        "memory/memory.md",
        "skills/custom-skill/SKILL.md",
        "cron/jobs.json",
        "state.db",
    ):
        assert (fresh / rel).exists(), rel
    assert not (fresh / ".env").exists()
    assert not (fresh / "auth.json").exists()
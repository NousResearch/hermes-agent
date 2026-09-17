"""Public curator consolidation contract against a disposable Hermes home."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest


def _package_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_skill(skills: Path, name: str, *, support: bool = False) -> Path:
    root = skills / name
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: synthetic skill\n---\n\n# {name}\n",
        encoding="utf-8",
    )
    if support:
        refs = root / "references"
        refs.mkdir()
        (refs / "inventory.md").write_text("synthetic source support file\n", encoding="utf-8")
    return root


@pytest.fixture
def consolidation_env(tmp_path, monkeypatch):
    """A complete disposable home with only synthetic skills and cron state."""
    home = tmp_path / ".hermes"
    skills = home / "skills"
    skills.mkdir(parents=True)
    source = _write_skill(skills, "source-skill", support=True)
    destination = _write_skill(skills, "destination-skill")
    (skills / ".usage.json").write_text(json.dumps({
        "source-skill": {"created_by": "agent", "state": "active", "pinned": False},
        "destination-skill": {"created_by": "agent", "state": "active", "pinned": False},
    }), encoding="utf-8")
    cron = home / "cron"
    cron.mkdir()
    jobs_file = cron / "jobs.json"
    original_jobs = {
        "jobs": [
            {"id": "skills-job", "name": "skills form", "skills": ["source-skill", "destination-skill"]},
            {"id": "skill-job", "name": "legacy form", "skill": "source-skill"},
        ],
        "updated_at": "synthetic-before",
    }
    jobs_file.write_text(json.dumps(original_jobs, indent=2), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_constants
    import cron.jobs as cron_jobs
    import tools.skill_usage as skill_usage
    import tools.skill_ledger as skill_ledger
    import agent.curator_backup as curator_backup
    importlib.reload(hermes_constants)
    importlib.reload(cron_jobs)
    importlib.reload(skill_usage)
    importlib.reload(skill_ledger)
    importlib.reload(curator_backup)
    return {
        "home": home,
        "skills": skills,
        "source": source,
        "destination": destination,
        "jobs_file": jobs_file,
        "original_jobs": original_jobs,
    }


def test_public_consolidate_archives_forwards_restores_and_rolls_back(consolidation_env, capsys):
    """The public command is one recoverable source-to-destination transaction."""
    from agent import curator_backup
    from hermes_cli import curator as curator_cli
    from tools import skill_ledger, skill_usage

    source_manifest = _package_manifest(consolidation_env["source"])
    destination_manifest = _package_manifest(consolidation_env["destination"])

    assert curator_cli.cli_main(["consolidate", "source-skill", "destination-skill"]) == 0
    receipt = json.loads(capsys.readouterr().out.removeprefix("curator: "))
    assert receipt["success"] is True
    assert receipt["source"] == "source-skill"
    assert receipt["destination"] == "destination-skill"
    assert receipt["forwarding"]["readback"] is True
    assert receipt["rollback_handle"]

    assert not consolidation_env["source"].exists()
    archived = consolidation_env["skills"] / ".archive" / "source-skill"
    assert _package_manifest(archived) == source_manifest
    assert skill_usage.list_archived_skill_names() == ["source-skill"]
    assert _package_manifest(consolidation_env["destination"]) == destination_manifest

    ledger = skill_ledger.list_entries(skill="source-skill", limit=10)
    consolidation_entry = next(row for row in ledger if row["action"] == "consolidate")
    assert consolidation_entry["evidence"]["absorbed_into"] == "destination-skill"
    assert consolidation_entry["evidence"]["archived"] is True
    assert consolidation_entry["evidence"]["rollback_handle"] == receipt["rollback_handle"]

    rewritten = json.loads(consolidation_env["jobs_file"].read_text(encoding="utf-8"))["jobs"]
    assert rewritten[0]["skills"] == ["destination-skill"]
    assert rewritten[0]["skill"] == "destination-skill"
    assert rewritten[1]["skills"] == ["destination-skill"]
    assert rewritten[1]["skill"] == "destination-skill"

    assert curator_cli.cli_main(["restore", "source-skill"]) == 0
    capsys.readouterr()
    assert _package_manifest(consolidation_env["source"]) == source_manifest

    ok, message, _ = curator_backup.rollback(receipt["rollback_handle"])
    assert ok, message
    assert _package_manifest(consolidation_env["source"]) == source_manifest
    restored_jobs = json.loads(consolidation_env["jobs_file"].read_text(encoding="utf-8"))["jobs"]
    assert restored_jobs[0].get("skills") == ["source-skill", "destination-skill"]
    assert "skill" not in restored_jobs[0]
    assert restored_jobs[1].get("skill") == "source-skill"
    assert "skills" not in restored_jobs[1]


@pytest.mark.parametrize("failure", ["save", "readback"])
def test_consolidate_recovers_source_and_cron_after_post_archive_failure(
    consolidation_env, monkeypatch, capsys, failure,
):
    """Cron persistence/readback failure after archival never loses the active source."""
    from hermes_cli import curator as curator_cli
    import cron.jobs as cron_jobs

    source_manifest = _package_manifest(consolidation_env["source"])
    original_cron = consolidation_env["jobs_file"].read_bytes()
    if failure == "save":
        monkeypatch.setattr(cron_jobs, "save_jobs", lambda jobs: (_ for _ in ()).throw(OSError("injected save failure")))
    else:
        real_load = cron_jobs.load_jobs
        calls = {"count": 0}

        def fail_second_load():
            calls["count"] += 1
            if calls["count"] > 1:
                raise RuntimeError("injected readback failure")
            return real_load()
        monkeypatch.setattr(cron_jobs, "load_jobs", fail_second_load)

    assert curator_cli.cli_main(["consolidate", "source-skill", "destination-skill"]) == 1
    receipt = json.loads(capsys.readouterr().out.removeprefix("curator: "))
    assert receipt["success"] is False
    assert receipt["recovery"]["attempted"] is True
    assert receipt["recovery"]["source_restored"] is True
    assert receipt["recovery"]["cron_restored"] is True
    assert _package_manifest(consolidation_env["source"]) == source_manifest
    assert consolidation_env["jobs_file"].read_bytes() == original_cron
    assert not (consolidation_env["skills"] / ".archive" / "source-skill").exists()


def test_background_consolidation_routes_through_the_public_transaction(consolidation_env, monkeypatch):
    """The legacy background declaration delegates to the shared recoverable primitive."""
    import tools.skill_manager_tool as skill_manager_tool

    monkeypatch.setattr(skill_manager_tool, "_is_background_review", lambda: True)
    result = skill_manager_tool._delete_skill("source-skill", absorbed_into="destination-skill")

    assert result["success"] is True
    assert result["_archived"] is True
    assert result["receipt"]["success"] is True
    assert result["receipt"]["forwarding"]["readback"] is True


def test_consolidate_refuses_preflight_without_mutating_fixture(consolidation_env):
    """Invalid destination/provenance input fails before snapshot or archival."""
    from agent.curator_consolidation import consolidate_skills

    source_manifest = _package_manifest(consolidation_env["source"])
    before_cron = consolidation_env["jobs_file"].read_bytes()
    receipt = consolidate_skills("source-skill", "source-skill")

    assert receipt["success"] is False
    assert "distinct" in receipt["error"]
    assert receipt["recovery"]["attempted"] is False
    assert _package_manifest(consolidation_env["source"]) == source_manifest
    assert consolidation_env["jobs_file"].read_bytes() == before_cron

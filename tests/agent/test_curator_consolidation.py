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


def _file_bytes_or_none(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


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
    from tools import skill_ledger

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
    from hermes_cli.main import _build_cli_parser
    parser, _subparsers = _build_cli_parser()
    list_archived = parser.parse_args(["curator", "list-archived"])
    assert list_archived.func(list_archived) == 0
    assert capsys.readouterr().out.strip() == "source-skill"
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


def test_public_ledger_rollback_restores_exact_preconsolidation_usage_sidecar(
    consolidation_env, capsys,
):
    """The consolidation ledger carries lifecycle state, not just package and cron files."""
    from hermes_cli import curator as curator_cli
    from tools import skill_ledger

    usage_file = consolidation_env["skills"] / ".usage.json"
    original_usage = b'''{
  "source-skill": {
    "created_by": "agent",
    "state": "active",
    "pinned": false,
    "use_count": 17,
    "view_count": 5,
    "patch_count": 3,
    "last_used_at": "2026-09-16T23:58:01+00:00",
    "last_viewed_at": "2026-09-16T23:57:01+00:00",
    "last_patched_at": "2026-09-16T23:56:01+00:00",
    "archived_at": null
  },
  "destination-skill": {"created_by": "agent", "state": "active", "pinned": false}
}
'''
    usage_file.write_bytes(original_usage)
    source_manifest = _package_manifest(consolidation_env["source"])
    original_cron = consolidation_env["jobs_file"].read_bytes()

    assert curator_cli.cli_main(["consolidate", "source-skill", "destination-skill"]) == 0
    receipt = json.loads(capsys.readouterr().out.removeprefix("curator: "))
    entry = skill_ledger.get_entry(receipt["ledger_entry"])
    assert entry is not None
    assert str(usage_file) in {item["path"] for item in entry["before"]}
    assert str(usage_file) in {item["path"] for item in entry["after"]}
    assert json.loads(usage_file.read_text(encoding="utf-8"))["source-skill"]["state"] == "archived"

    assert curator_cli.cli_main(["rollback", receipt["ledger_entry"], "-y"]) == 0
    assert _package_manifest(consolidation_env["source"]) == source_manifest
    assert consolidation_env["jobs_file"].read_bytes() == original_cron
    assert usage_file.read_bytes() == original_usage
    assert not (consolidation_env["skills"] / ".archive" / "source-skill").exists()


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


def test_consolidate_receipt_failure_recovers_and_compensates_success_ledger(
    consolidation_env, monkeypatch, capsys,
):
    """A failed post-ledger receipt records recovery instead of a false archived success."""
    from agent import curator_consolidation
    from hermes_cli import curator as curator_cli
    from tools import skill_ledger

    source_manifest = _package_manifest(consolidation_env["source"])
    original_cron = consolidation_env["jobs_file"].read_bytes()

    def fail_after_success_ledger(receipt):
        assert any(
            row["action"] == "consolidate"
            for row in skill_ledger.list_entries(skill="source-skill")
        )
        raise OSError("injected receipt write failure")

    monkeypatch.setattr(curator_consolidation, "_write_receipt", fail_after_success_ledger)

    assert curator_cli.cli_main(["consolidate", "source-skill", "destination-skill"]) == 1
    receipt = json.loads(capsys.readouterr().out.removeprefix("curator: "))
    assert receipt["success"] is False
    assert receipt["recovery"]["attempted"] is True
    assert receipt["recovery"]["source_restored"] is True
    assert receipt["recovery"]["cron_restored"] is True
    assert _package_manifest(consolidation_env["source"]) == source_manifest
    assert consolidation_env["jobs_file"].read_bytes() == original_cron
    assert not (consolidation_env["skills"] / ".archive" / "source-skill").exists()

    entries = skill_ledger.list_entries(skill="source-skill", limit=10)
    success_entry = next(row for row in entries if row["action"] == "consolidate")
    recovery_entry = next(row for row in entries if row["action"] == "consolidate-recovery")
    assert recovery_entry["evidence"]["operation_id"] == receipt["operation_id"]
    assert recovery_entry["evidence"]["recovered_consolidation_entry"] == success_entry["id"]
    assert recovery_entry["evidence"]["source_restored"] is True
    assert recovery_entry["evidence"]["cron_restored"] is True

    assert curator_cli.cli_main(["ledger", "--skill", "source-skill"]) == 0
    ledger_output = capsys.readouterr().out
    assert "consolidate-recovery" in ledger_output
    assert f"recovered consolidation {success_entry['id']}" in ledger_output


def test_background_consolidation_routes_through_the_public_transaction(consolidation_env, monkeypatch):
    """The legacy background declaration delegates to the shared recoverable primitive."""
    import tools.skill_manager_tool as skill_manager_tool

    monkeypatch.setattr(skill_manager_tool, "_is_background_review", lambda: True)
    result = skill_manager_tool._delete_skill("source-skill", absorbed_into="destination-skill")

    assert result["success"] is True
    assert result["_archived"] is True
    assert result["receipt"]["success"] is True
    assert result["receipt"]["forwarding"]["readback"] is True


def test_public_background_skill_manage_consolidation_has_one_ledger_entry_and_rolls_back(
    consolidation_env,
):
    """The public background dispatcher must retain the transaction's complete ledger entry.

    A second dispatcher ``delete`` entry has only the source package and cannot
    undo the archived package plus cron forwarding.  Exercise the real public
    ``skill_manage`` route rather than its private delete helper.
    """
    from tools import skill_ledger
    from tools.skill_manager_tool import skill_manage
    from tools.skill_provenance import (
        BACKGROUND_REVIEW,
        reset_current_write_origin,
        set_current_write_origin,
    )

    source_manifest = _package_manifest(consolidation_env["source"])
    original_cron = consolidation_env["jobs_file"].read_bytes()
    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        result = json.loads(skill_manage(
            action="delete",
            name="source-skill",
            absorbed_into="destination-skill",
        ))
    finally:
        reset_current_write_origin(token)

    assert result["success"] is True, result
    receipt = result["receipt"]
    entries = skill_ledger.list_entries(skill="source-skill", limit=10)
    assert [entry["action"] for entry in entries] == ["consolidate"]
    assert entries[0]["id"] == receipt["ledger_entry"]
    assert not consolidation_env["source"].exists()
    assert (consolidation_env["skills"] / ".archive" / "source-skill").is_dir()

    ok, message = skill_ledger.rollback_entry(receipt["ledger_entry"])
    assert ok is True, message
    assert _package_manifest(consolidation_env["source"]) == source_manifest
    assert not (consolidation_env["skills"] / ".archive" / "source-skill").exists()
    assert consolidation_env["jobs_file"].read_bytes() == original_cron


def test_consolidation_rollback_keeps_preexisting_empty_archive_parent(consolidation_env, capsys):
    """Rollback removes only the transaction-created archive package, not its pre-existing parent."""
    from hermes_cli import curator as curator_cli
    from tools import skill_ledger

    archive_root = consolidation_env["skills"] / ".archive"
    archive_root.mkdir()

    assert curator_cli.cli_main(["consolidate", "source-skill", "destination-skill"]) == 0
    receipt = json.loads(capsys.readouterr().out.removeprefix("curator: "))
    archived_package = archive_root / "source-skill"
    assert archived_package.is_dir()

    ok, message = skill_ledger.rollback_entry(receipt["ledger_entry"])

    assert ok is True, message
    assert not archived_package.exists()
    assert archive_root.is_dir(), "rollback must not prune a pre-existing empty archive parent"


def test_consolidate_recovers_when_archive_usage_persistence_fails(
    consolidation_env, monkeypatch, capsys,
):
    """Archival is unsuccessful unless the archived usage state durably lands."""
    from hermes_cli import curator as curator_cli
    from tools import skill_ledger, skill_usage

    source_manifest = _package_manifest(consolidation_env["source"])
    original_cron = consolidation_env["jobs_file"].read_bytes()
    usage_file = consolidation_env["skills"] / ".usage.json"
    original_usage = usage_file.read_bytes()
    monkeypatch.setattr(skill_usage, "set_state", lambda *_args, **_kwargs: False)

    assert curator_cli.cli_main(["consolidate", "source-skill", "destination-skill"]) == 1
    receipt = json.loads(capsys.readouterr().out.removeprefix("curator: "))

    assert receipt["success"] is False
    assert "lifecycle" in receipt["error"]
    assert receipt["recovery"]["attempted"] is True
    assert receipt["recovery"]["source_restored"] is True
    assert receipt["recovery"]["cron_restored"] is True
    assert _package_manifest(consolidation_env["source"]) == source_manifest
    assert consolidation_env["jobs_file"].read_bytes() == original_cron
    assert usage_file.read_bytes() == original_usage
    assert not (consolidation_env["skills"] / ".archive" / "source-skill").exists()

    entries = skill_ledger.list_entries(skill="source-skill", limit=10)
    assert [entry["action"] for entry in entries] == ["consolidate-recovery"]
    assert entries[0]["evidence"]["operation_id"] == receipt["operation_id"]
    assert entries[0]["evidence"]["source_restored"] is True
    assert entries[0]["evidence"]["cron_restored"] is True


@pytest.mark.parametrize(
    ("refusal", "error_fragment"),
    [
        ("same-name", "distinct"),
        ("missing-destination", "destination skill 'missing-destination' is not an active local skill"),
        ("pinned", "pinned"),
        ("bundled", "bundled"),
        ("hub", "hub-installed"),
        ("protected", "protected built-in"),
        ("external", "not an active local skill"),
        ("wrong-profile", "not an active local skill"),
        ("unmanaged", "not curator-managed"),
    ],
)
def test_public_consolidate_refuses_invalid_source_or_destination_before_mutation(
    consolidation_env, monkeypatch, capsys, refusal, error_fragment,
):
    """Every public refusal leaves the candidate package, cron, snapshots, and ledger untouched.

    External and wrong-profile sources deliberately reach the active-local lookup
    refusal: the registered command parser represents both names, but the
    transaction must never resolve a package from an external directory or a
    sibling profile into the active profile's mutation domain.
    """
    from agent import curator_backup
    from hermes_cli import curator as curator_cli
    from tools import skill_ledger, skill_usage

    source_name = "source-skill"
    source_dir = consolidation_env["source"]
    destination_name = "destination-skill"

    if refusal == "same-name":
        destination_name = source_name
    elif refusal == "missing-destination":
        destination_name = "missing-destination"
    elif refusal == "pinned":
        usage = json.loads((consolidation_env["skills"] / ".usage.json").read_text(encoding="utf-8"))
        usage[source_name]["pinned"] = True
        (consolidation_env["skills"] / ".usage.json").write_text(json.dumps(usage), encoding="utf-8")
    elif refusal == "bundled":
        (consolidation_env["skills"] / ".bundled_manifest").write_text(
            f"{source_name}:synthetic-hash\n", encoding="utf-8")
    elif refusal == "hub":
        hub = consolidation_env["skills"] / ".hub"
        hub.mkdir()
        (hub / "lock.json").write_text(json.dumps({
            "version": 1,
            "installed": {source_name: {"install_path": source_name}},
        }), encoding="utf-8")
    elif refusal == "protected":
        monkeypatch.setattr(skill_usage, "PROTECTED_BUILTIN_SKILLS", {source_name})
    elif refusal == "external":
        external_root = consolidation_env["home"] / "external-skills"
        source_name = "external-source"
        source_dir = _write_skill(external_root, source_name, support=True)
        (consolidation_env["home"] / "config.yaml").write_text(
            "skills:\n  external_dirs:\n    - external-skills\n", encoding="utf-8")
        import agent.skill_utils as skill_utils
        skill_utils._external_dirs_cache_clear()
    elif refusal == "wrong-profile":
        source_name = "other-profile-source"
        source_dir = _write_skill(
            consolidation_env["home"] / "profiles" / "other" / "skills", source_name, support=True,
        )
        (consolidation_env["home"] / "profiles" / "other" / "config.yaml").write_text(
            "{}\n", encoding="utf-8")
    elif refusal == "unmanaged":
        source_name = "unmanaged-source"
        source_dir = _write_skill(consolidation_env["skills"], source_name, support=True)

    source_manifest = _package_manifest(source_dir)
    before_cron = consolidation_env["jobs_file"].read_bytes()
    ledger_path = skill_ledger.ledger_path()
    before_ledger = _file_bytes_or_none(ledger_path)
    assert curator_backup.list_backups() == []

    assert curator_cli.cli_main(["consolidate", source_name, destination_name]) == 1
    receipt = json.loads(capsys.readouterr().out.removeprefix("curator: "))

    assert receipt["success"] is False
    assert error_fragment in receipt["error"]
    assert receipt["archive_location"] is None
    assert receipt["rollback_handle"] is None
    assert receipt["recovery"]["attempted"] is False
    assert _package_manifest(source_dir) == source_manifest
    assert consolidation_env["jobs_file"].read_bytes() == before_cron
    assert not (consolidation_env["skills"] / ".archive" / source_dir.name).exists()
    assert curator_backup.list_backups() == []
    assert _file_bytes_or_none(ledger_path) == before_ledger

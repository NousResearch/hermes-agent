"""Run accounting must observe disk mutations without usage telemetry."""
import importlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("consolidate", [False, True])
def test_run_counts_archives_without_usage_rows(tmp_path, monkeypatch, consolidate):
    home = tmp_path / ".hermes"
    skills = home / "skills"
    skills.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from agent import curator, curator_backup
    from tools import skill_usage
    importlib.reload(skill_usage)
    importlib.reload(curator)
    monkeypatch.setattr(curator_backup, "snapshot_skills", lambda **kw: None)

    def seed(root, name):
        directory = root / "group" / name
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            f"---\nname: {name}\ncreated_by: agent\n---\n# {name}\n"
        )

    for name in ("archive-one", "archive-two", "keeper"):
        seed(skills, name)
    shared = home / "skills-shared"
    (home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {shared}\n"
    )
    seed(shared, "shared-keeper")
    seed(shared, "keeper")  # Same logical name is counted once.
    seed(shared / ".archive", "already-archived")
    assert skill_usage.curated_report() == []

    def archive():
        for name in ("archive-one", "archive-two"):
            ok, message = skill_usage.archive_skill(name)
            assert ok, message
            assert (skills / ".archive" / name / "SKILL.md").is_file()
        return {"checked": 3, "archived": 2, "marked_stale": 0, "reactivated": 0}

    def automatic(**kw):
        if not consolidate:
            return archive()
        return {"checked": 3, "archived": 0, "marked_stale": 0, "reactivated": 0}

    def review(prompt):
        archive()
        seed(shared, "new-umbrella")
        return {"summary": "consolidated", "final": "", "tool_calls": [
            {"name": "skill_manage", "arguments": json.dumps({
                "action": "delete", "name": "archive-one", "absorbed_into": "new-umbrella"
            })}
        ]}

    monkeypatch.setattr(curator, "apply_automatic_transitions", automatic)
    monkeypatch.setattr(curator, "_run_llm_review", review)
    curator.run_curator_review(synchronous=True, consolidate=consolidate)
    report_dir = Path(curator.load_state()["last_report_path"])
    report = json.loads((report_dir / "run.json").read_text())
    assert report["counts"]["archived_this_run"] == 2
    assert report["counts"]["before"] == 4
    assert report["counts"]["after"] == 2 + int(consolidate)
    assert report["counts"]["delta"] == -2 + int(consolidate)
    assert report["counts"]["added_this_run"] == int(consolidate)
    assert report["counts"]["consolidated_this_run"] == int(consolidate)
    assert report["archived"] == ["archive-one", "archive-two"]
    if consolidate:
        assert "archive-one → new-umbrella" in curator.load_state()["last_run_summary"]

# /// script
# requires-python = ">=3.10"
# dependencies = ["pytest>=8.0", "ruamel.yaml>=0.18"]
# ///
"""Tests for sprint_plan.py — deterministic sprint-status generation.

Run: uv run scripts/tests/test_sprint_plan.py
 or: uv run --with pytest --with ruamel.yaml -m pytest scripts/tests/test_sprint_plan.py
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from ruamel.yaml import YAML

SCRIPT = Path(__file__).resolve().parents[1] / "sprint_plan.py"
TEMPLATE = Path(__file__).resolve().parents[2] / "sprint-status-template.yaml"

spec = importlib.util.spec_from_file_location("sprint_plan", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

EPICS_FIXTURE = """\
# Project Epics

## Epic 1: Foundation
Some prose.

### Story 1.1: User Authentication
Acceptance criteria...

### Story 1.2: Account Management

## Epic 2: Chat
### Story 2.1: Personality System
### Story 2.6a: Split Story, With Punctuation!
"""

DATE = "08-01-2026 14:30"


def run_generate(tmp_path, epics_text=EPICS_FIXTURE, existing=None, stories=(), extra=()):
    epic_file = tmp_path / "epics.md"
    epic_file.write_text(epics_text, encoding="utf-8")
    status_file = tmp_path / "impl" / "sprint-status.yaml"
    if existing is not None:
        status_file.parent.mkdir(parents=True, exist_ok=True)
        status_file.write_text(existing, encoding="utf-8")
    stories_dir = tmp_path / "impl"
    stories_dir.mkdir(parents=True, exist_ok=True)
    for name in stories:
        (stories_dir / f"{name}.md").write_text("story", encoding="utf-8")
    argv = [
        "generate", "--epic-file", str(epic_file), "--status-file", str(status_file),
        "--stories-dir", str(stories_dir), "--project", "My Project", "--date", DATE,
        *extra,
    ]
    mod.main(argv)
    return status_file


def load(status_file):
    yaml = YAML()
    with open(status_file, encoding="utf-8") as fh:
        return yaml.load(fh)


def out_json(capsys):
    return json.loads(capsys.readouterr().out)


def test_fresh_generate_orders_and_defaults(tmp_path, capsys):
    status_file = run_generate(tmp_path)
    result = out_json(capsys)
    data = load(status_file)
    keys = list(data["development_status"].keys())
    assert keys == [
        "epic-1", "1-1-user-authentication", "1-2-account-management", "epic-1-retrospective",
        "epic-2", "2-1-personality-system", "2-6a-split-story-with-punctuation", "epic-2-retrospective",
    ]
    assert data["development_status"]["epic-1"] == "backlog"
    assert data["development_status"]["1-1-user-authentication"] == "backlog"
    assert data["development_status"]["epic-1-retrospective"] == "optional"
    assert data["project"] == "My Project"
    assert data["generated"] == DATE and data["last_updated"] == DATE
    assert result["ok"] and result["epics"] == 2 and result["stories"] == 4
    text = status_file.read_text(encoding="utf-8")
    assert "STATUS DEFINITIONS" in text
    assert "\n\n  epic-2:" in text  # blank line between epic groups


def test_header_comment_matches_template():
    """The template's STATUS DEFINITIONS block and the script's HEADER_COMMENT
    are two copies of one contract; this pins them together."""
    lines = TEMPLATE.read_text(encoding="utf-8").splitlines()
    start = lines.index("# STATUS DEFINITIONS:")
    block = []
    for line in lines[start:]:
        if not line.startswith("#"):
            break
        block.append(line[2:] if line.startswith("# ") else line[1:])
    assert "\n".join(block) + "\n" == mod.HEADER_COMMENT


EXISTING = """\
generated: 01-01-2026 09:00
last_updated: 01-01-2026 09:00
project: My Project
project_key: NOKEY
tracking_system: file-system
story_location: impl

development_status:
  epic-1: in-progress
  1-1-user-authentication: done
  1-2-account-management: backlog
  epic-1-retrospective: optional
  9-9-ghost-story: done

action_items:
  - epic: 1
    action: "Add error-handling review; watch: quotes, commas"
    owner: "Charlie"
    status: open
"""


def test_merge_preserves_and_never_downgrades(tmp_path, capsys):
    status_file = run_generate(tmp_path, existing=EXISTING)
    result = out_json(capsys)
    data = load(status_file)
    assert data["development_status"]["epic-1"] == "in-progress"
    assert data["development_status"]["1-1-user-authentication"] == "done"
    assert data["generated"] == "01-01-2026 09:00"
    assert data["last_updated"] == DATE
    assert result["dropped_orphans"] == [{"key": "9-9-ghost-story", "status": "done"}]
    assert "9-9-ghost-story" not in data["development_status"]


def test_legacy_statuses_merge_by_meaning_not_reset(tmp_path, capsys):
    existing = (EXISTING
                .replace("1-2-account-management: backlog", "1-2-account-management: drafted")
                .replace("epic-1: in-progress", "epic-1: contexted"))
    status_file = run_generate(tmp_path, existing=existing)
    result = out_json(capsys)
    data = load(status_file)
    assert data["development_status"]["1-2-account-management"] == "ready-for-dev"
    assert data["development_status"]["epic-1"] == "in-progress"
    assert {"key": "1-2-account-management", "from": "drafted", "to": "ready-for-dev"} in result["legacy_mapped"]
    assert not any("illegal" in w for w in result["warnings"])
    assert result["illegal"] == []


def test_metadata_preserved_when_flags_omitted(tmp_path, capsys):
    existing = (EXISTING
                .replace("project_key: NOKEY", "project_key: JIRA-PROJ")
                .replace("tracking_system: file-system", "tracking_system: jira")
                .replace("story_location: impl", "story_location: /custom/stories"))
    status_file = run_generate(tmp_path, existing=existing)
    out_json(capsys)
    data = load(status_file)
    assert data["project_key"] == "JIRA-PROJ"
    assert data["tracking_system"] == "jira"
    assert data["story_location"] == "/custom/stories"


def test_unknown_keys_and_comments_survive_regenerate(tmp_path, capsys):
    existing = EXISTING + "\n# my own note\nsprint_goal: Ship the beta\n"
    status_file = run_generate(tmp_path, existing=existing)
    out_json(capsys)
    data = load(status_file)
    assert data["sprint_goal"] == "Ship the beta"
    assert "# my own note" in status_file.read_text(encoding="utf-8")


def test_fresh_rebuild_ignores_existing_statuses(tmp_path, capsys):
    status_file = run_generate(tmp_path, existing=EXISTING, extra=("--fresh",))
    result = out_json(capsys)
    data = load(status_file)
    assert data["development_status"]["1-1-user-authentication"] == "backlog"
    assert data["development_status"]["epic-1"] == "backlog"
    assert result["fresh"] is True
    # action_items are retro history, not tracking state — carried even on --fresh
    assert data["action_items"][0]["status"] == "open"


def test_set_applies_explicit_statuses_even_downgrades(tmp_path, capsys):
    status_file = run_generate(
        tmp_path, existing=EXISTING,
        extra=("--fresh", "--set", "1-1-user-authentication=in-progress", "--set", "epic-1=in-progress"),
    )
    result = out_json(capsys)
    data = load(status_file)
    assert data["development_status"]["1-1-user-authentication"] == "in-progress"
    assert data["development_status"]["epic-1"] == "in-progress"
    assert "1-1-user-authentication=in-progress" in result["explicit_set"]


def test_set_rejects_unknown_key_and_illegal_status(tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        run_generate(tmp_path, extra=("--set", "9-9-nope=done"))
    assert excinfo.value.code == 1
    assert out_json(capsys)["ok"] is False
    with pytest.raises(SystemExit):
        run_generate(tmp_path, extra=("--set", "epic-1=review"))
    assert out_json(capsys)["ok"] is False


def test_action_items_carried_verbatim(tmp_path):
    status_file = run_generate(tmp_path, existing=EXISTING)
    data = load(status_file)
    assert data["action_items"][0]["action"] == "Add error-handling review; watch: quotes, commas"
    assert data["action_items"][0]["status"] == "open"


def test_story_file_on_disk_floors_ready_for_dev(tmp_path, capsys):
    status_file = run_generate(tmp_path, stories=["1-2-account-management"])
    result = out_json(capsys)
    data = load(status_file)
    assert data["development_status"]["1-2-account-management"] == "ready-for-dev"
    assert data["development_status"]["1-1-user-authentication"] == "backlog"
    assert result["upgraded_from_disk"] == ["1-2-account-management"]


def test_story_file_never_downgrades_done(tmp_path):
    status_file = run_generate(tmp_path, existing=EXISTING, stories=["1-1-user-authentication"])
    data = load(status_file)
    assert data["development_status"]["1-1-user-authentication"] == "done"


def test_illegal_existing_status_warns_and_resets(tmp_path, capsys):
    existing = EXISTING.replace("1-2-account-management: backlog", "1-2-account-management: shipped")
    status_file = run_generate(tmp_path, existing=existing)
    result = out_json(capsys)
    data = load(status_file)
    assert data["development_status"]["1-2-account-management"] == "backlog"
    assert any("illegal status 'shipped'" in w for w in result["warnings"])
    assert {"key": "1-2-account-management", "status": "shipped"} in result["illegal"]


def test_fenced_code_blocks_are_not_parsed(tmp_path, capsys):
    text = EPICS_FIXTURE + "\n```\n## Epic 9: Example Format\n### Story 9.1: Sample\n```\n"
    status_file = run_generate(tmp_path, epics_text=text)
    result = out_json(capsys)
    data = load(status_file)
    assert "epic-9" not in data["development_status"]
    assert result["epics"] == 2
    assert not any("Epic 9" in w for w in result["warnings"])


def test_non_ascii_titles_keep_distinct_keys(tmp_path, capsys):
    text = "## Epic 1: 基础\n### Story 1.1: 用户认证\n### Story 1.2: 账户管理\n"
    status_file = run_generate(tmp_path, epics_text=text)
    out_json(capsys)
    keys = list(load(status_file)["development_status"].keys())
    assert "1-1-用户认证" in keys and "1-2-账户管理" in keys


def test_suspect_heading_is_reported(tmp_path, capsys):
    text = EPICS_FIXTURE + "\n### Story Two point one: Bad Format\n"
    run_generate(tmp_path, epics_text=text)
    result = out_json(capsys)
    assert any("unparsed Epic/Story-like heading" in w for w in result["warnings"])


def test_dry_run_writes_nothing_and_reports_drift(tmp_path, capsys):
    epic_file = tmp_path / "epics.md"
    epic_file.write_text(EPICS_FIXTURE, encoding="utf-8")
    status_file = tmp_path / "sprint-status.yaml"
    status_file.write_text(EXISTING, encoding="utf-8")
    original = status_file.read_bytes()
    mod.main([
        "generate", "--epic-file", str(epic_file), "--status-file", str(status_file),
        "--stories-dir", str(tmp_path), "--project", "P", "--date", DATE, "--dry-run",
    ])
    result = out_json(capsys)
    assert result["dry_run"] is True and result["ok"] is True
    assert result["in_sync"] is False
    assert "epic-2" in result["new_entries"]
    assert result["dropped_orphans"] == [{"key": "9-9-ghost-story", "status": "done"}]
    assert status_file.read_bytes() == original


def test_dry_run_in_sync_after_generate(tmp_path, capsys):
    status_file = run_generate(tmp_path)
    capsys.readouterr()
    mod.main([
        "generate", "--epic-file", str(tmp_path / "epics.md"), "--status-file", str(status_file),
        "--stories-dir", str(tmp_path / "impl"), "--project", "My Project", "--date", DATE,
        "--dry-run",
    ])
    result = out_json(capsys)
    assert result["in_sync"] is True
    assert result["new_entries"] == [] and result["dropped_orphans"] == [] and result["illegal"] == []


def test_no_epics_fails_with_json(tmp_path, capsys):
    epic_file = tmp_path / "notes.md"
    epic_file.write_text("just prose, no epics", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        mod.main([
            "generate", "--epic-file", str(epic_file), "--status-file", str(tmp_path / "s.yaml"),
            "--stories-dir", str(tmp_path), "--project", "P", "--date", DATE,
        ])
    assert excinfo.value.code == 1
    assert out_json(capsys)["ok"] is False


def test_non_mapping_yaml_fails_with_json(tmp_path, capsys):
    epic_file = tmp_path / "epics.md"
    epic_file.write_text(EPICS_FIXTURE, encoding="utf-8")
    status_file = tmp_path / "sprint-status.yaml"
    status_file.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        mod.main([
            "generate", "--epic-file", str(epic_file), "--status-file", str(status_file),
            "--stories-dir", str(tmp_path), "--project", "P", "--date", DATE,
        ])
    assert excinfo.value.code == 1
    assert "not a mapping" in out_json(capsys)["error"]
    with pytest.raises(SystemExit):
        mod.main(["status", "--status-file", str(status_file)])
    assert "not a mapping" in out_json(capsys)["error"]


def test_argument_errors_emit_json(capsys):
    with pytest.raises(SystemExit) as excinfo:
        mod.main(["generate"])
    assert excinfo.value.code == 2
    assert out_json(capsys)["ok"] is False
    with pytest.raises(SystemExit) as excinfo:
        mod.main(["-h"])
    assert excinfo.value.code == 2
    assert out_json(capsys)["ok"] is False


STATUS_FIXTURE = """\
generated: 01-01-2026 09:00
last_updated: 07-30-2026 09:00
project: My Project
project_key: NOKEY
tracking_system: file-system
story_location: impl

development_status:
  epic-1: in-progress
  1-1-user-authentication: done
  1-2-account-management: drafted
  epic-1-retrospective: optional
  epic-2: backlog
  2-1-personality-system: backlog
  epic-2-retrospective: optional

action_items:
  - epic: 1
    action: "Tighten error handling"
    owner: "Charlie"
    status: open
  - epic: 1
    action: "Old item"
    owner: "Charlie"
    status: done
"""


def run_status(tmp_path, capsys, fixture=STATUS_FIXTURE, extra=()):
    status_file = tmp_path / "sprint-status.yaml"
    status_file.write_text(fixture, encoding="utf-8")
    mod.main(["status", "--status-file", str(status_file), "--date", DATE, *extra])
    return json.loads(capsys.readouterr().out)


def test_status_counts_and_recommendation(tmp_path, capsys):
    result = run_status(tmp_path, capsys)
    assert result["stories"] == {"done": 1, "ready-for-dev": 1, "backlog": 1}
    assert result["epics"] == {"in-progress": 1, "backlog": 1}
    assert result["retrospectives"] == {"optional": 2}
    assert result["recommendation"]["skill"] == "bmad-build"
    assert result["recommendation"]["story_key"] == "1-2-account-management"
    assert result["all_done"] is False


def test_status_maps_legacy_values(tmp_path, capsys):
    result = run_status(tmp_path, capsys)
    assert {"key": "1-2-account-management", "from": "drafted", "to": "ready-for-dev"} in result["legacy_mapped"]


def test_status_open_action_items(tmp_path, capsys):
    result = run_status(tmp_path, capsys)
    assert len(result["open_action_items"]) == 1
    assert result["open_action_items"][0]["action"] == "Tighten error handling"


def test_status_malformed_action_items_are_flagged_not_dropped(tmp_path, capsys):
    fixture = STATUS_FIXTURE + "  - \"just a string\"\n  - epic: 2\n    action: \"No status\"\n"
    result = run_status(tmp_path, capsys, fixture=fixture)
    assert any("not a mapping" in w for w in result["warnings"])
    assert any("missing or unknown status" in w for w in result["warnings"])
    assert len(result["open_action_items"]) == 1


def test_status_review_beats_ready(tmp_path, capsys):
    fixture = STATUS_FIXTURE.replace("2-1-personality-system: backlog", "2-1-personality-system: review")
    result = run_status(tmp_path, capsys, fixture=fixture)
    assert result["recommendation"]["skill"] == "bmad-code-review"
    assert result["recommendation"]["story_key"] == "2-1-personality-system"
    assert any("review" in r for r in result["risks"])


def test_status_in_progress_beats_all(tmp_path, capsys):
    fixture = STATUS_FIXTURE.replace("2-1-personality-system: backlog", "2-1-personality-system: in-progress")
    result = run_status(tmp_path, capsys, fixture=fixture)
    assert result["recommendation"]["skill"] == "bmad-build"
    assert result["recommendation"]["story_key"] == "2-1-personality-system"
    assert result["recommendation"]["reason"] == "resume the in-progress story"


def test_status_staleness_and_orphan_risks(tmp_path, capsys):
    fixture = (STATUS_FIXTURE
               .replace("last_updated: 07-30-2026 09:00", "last_updated: 01-02-2026 09:00")
               .replace("  epic-2-retrospective: optional",
                        "  epic-2-retrospective: optional\n  5-1-ghost: backlog"))
    result = run_status(tmp_path, capsys, fixture=fixture)
    assert any("stale" in r for r in result["risks"])
    assert any("orphaned story '5-1-ghost'" in r for r in result["risks"])


def test_status_unparseable_timestamp_warns_instead_of_silence(tmp_path, capsys):
    fixture = STATUS_FIXTURE.replace("last_updated: 07-30-2026 09:00", "last_updated: whenever")
    result = run_status(tmp_path, capsys, fixture=fixture)
    assert any("staleness check skipped" in w for w in result["warnings"])


def test_status_iso_and_date_typed_stamps_do_not_crash(tmp_path, capsys):
    fixture = (STATUS_FIXTURE
               .replace("generated: 01-01-2026 09:00", "generated: 2026-01-01")
               .replace("last_updated: 07-30-2026 09:00", "last_updated: 2026-01-02"))
    result = run_status(tmp_path, capsys, fixture=fixture)
    assert result["ok"] is True
    assert result["generated"] == "2026-01-01"
    assert any("stale" in r for r in result["risks"])  # ISO stamp still parses


def test_status_all_done_recommends_retro_then_nothing(tmp_path, capsys):
    fixture = (STATUS_FIXTURE
               .replace("1-2-account-management: drafted", "1-2-account-management: done")
               .replace("2-1-personality-system: backlog", "2-1-personality-system: done"))
    result = run_status(tmp_path, capsys, fixture=fixture)
    assert result["recommendation"]["skill"] == "bmad-retrospective"
    assert "epic-1-retrospective" in result["recommendation"]["reason"]
    fixture_done = fixture.replace("epic-1-retrospective: optional", "epic-1-retrospective: done") \
                          .replace("epic-2-retrospective: optional", "epic-2-retrospective: done")
    result = run_status(tmp_path, capsys, fixture=fixture_done)
    assert result["all_done"] is True and result["recommendation"] is None


def test_status_odd_retro_key_reports_instead_of_crashing(tmp_path, capsys):
    fixture = (STATUS_FIXTURE
               .replace("1-2-account-management: drafted", "1-2-account-management: done")
               .replace("2-1-personality-system: backlog", "2-1-personality-system: done")
               .replace("epic-1-retrospective: optional", "epic-1-retrospective: done")
               .replace("epic-2-retrospective: optional",
                        "epic-2-retrospective: done\n  epic-abc-retrospective: optional"))
    result = run_status(tmp_path, capsys, fixture=fixture)
    assert result["ok"] is True
    assert {"key": "epic-abc-retrospective", "status": "optional"} in result["unrecognized"]
    assert any("unrecognized key" in r for r in result["risks"])


def test_status_illegal_status_reported(tmp_path, capsys):
    fixture = STATUS_FIXTURE.replace("2-1-personality-system: backlog", "2-1-personality-system: shipped")
    result = run_status(tmp_path, capsys, fixture=fixture)
    assert {"key": "2-1-personality-system", "status": "shipped"} in result["illegal"]


def test_status_missing_file_fails_json(tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        mod.main(["status", "--status-file", str(tmp_path / "nope.yaml")])
    assert excinfo.value.code == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def run_validate(tmp_path, capsys, content):
    status_file = tmp_path / "sprint-status.yaml"
    if content is not None:
        status_file.write_text(content, encoding="utf-8")
    mod.main(["validate", "--status-file", str(status_file)])
    return json.loads(capsys.readouterr().out)


def test_validate_clean_file(tmp_path, capsys):
    clean = STATUS_FIXTURE.replace("1-2-account-management: drafted", "1-2-account-management: backlog")
    result = run_validate(tmp_path, capsys, clean)
    assert result["valid"] is True and result["problems"] == []


def test_validate_reports_problems_without_crashing(tmp_path, capsys):
    broken = (STATUS_FIXTURE
              .replace("2-1-personality-system: backlog", "2-1-personality-system: shipped")
              .replace("epic-2-retrospective: optional",
                       "epic-2-retrospective: optional\n  weird-key: done")
              .replace("last_updated: 07-30-2026 09:00", "last_updated: whenever"))
    result = run_validate(tmp_path, capsys, broken)
    assert result["valid"] is False
    assert any("illegal story status 'shipped'" in p for p in result["problems"])
    assert any("unrecognized key 'weird-key'" in p for p in result["problems"])
    assert any("'last_updated' timestamp" in p for p in result["problems"])
    assert {"key": "1-2-account-management", "from": "drafted", "to": "ready-for-dev"} in result["legacy_mapped"]


def test_validate_missing_file_and_bad_yaml(tmp_path, capsys):
    result = run_validate(tmp_path, capsys, None)
    assert result["valid"] is False and "does not exist" in result["problems"][0]
    result = run_validate(tmp_path, capsys, "development_status: [unclosed\n")
    assert result["valid"] is False and "not valid YAML" in result["problems"][0]
    result = run_validate(tmp_path, capsys, "- a\n- b\n")
    assert result["valid"] is False and any("not a mapping" in p for p in result["problems"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

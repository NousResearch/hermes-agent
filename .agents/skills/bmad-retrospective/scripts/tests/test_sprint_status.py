# /// script
# requires-python = ">=3.10"
# dependencies = ["pytest>=8.0", "ruamel.yaml>=0.18"]
# ///
"""Corruption-critical tests for sprint-status.py.

Each test runs the script as a subprocess via ``uv run`` against a temp copy of
an inline fixture, then re-reads the file to assert comments and formatting
survive and punctuation-heavy action values round-trip intact.

Run: uv run scripts/tests/test_sprint_status.py
 or: uv run --with pytest --with ruamel.yaml -m pytest scripts/tests/test_sprint_status.py
"""

import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from ruamel.yaml import YAML

SCRIPT = Path(__file__).resolve().parents[1] / "sprint_status.py"
# Vendored copy of bmad-sprint-planning's sprint-status-template.yaml — skills
# must not path into each other's directories (PATH-05). The repo-level
# test/test-template-sync.js keeps this fixture identical to the source.
TEMPLATE = Path(__file__).resolve().parent / "fixtures" / "sprint-status-template.yaml"

FIXTURE = """\
# Sprint Status Tracking
# STATUS DEFINITIONS:
#   backlog        - not yet started
#   ready-for-dev  - ready to be implemented
#   done           - completed
generated: "01-01-2026 09:00"
last_updated: "01-01-2026 09:00"
project: "Demo Project"
project_key: "DEMO"
tracking_system: "file"
story_location: "docs/stories"
development_status:
  epic-1: backlog
  1-1-user-authentication: done
  1-2-account-management: done
  epic-1-retrospective: optional
  epic-2: backlog
  2-1-dashboard: backlog
"""


# Two epics' worth of items, because the flag's headline use is epic N's retro
# closing epic N-1's items: nothing may scope a selector to --epic. The two
# "Scripted item" entries share their action text and differ only by epic, so the
# legacy epic+action selector has to discriminate on the epic to resolve at all.
ACTION_FIXTURE = """\
# Sprint Status Tracking
# STATUS DEFINITIONS:
#   open           - committed during a retrospective, not yet addressed
generated: "01-01-2026 09:00"
last_updated: "01-01-2026 09:00"
development_status:
  1-1-a: done
  epic-1-retrospective: optional
  2-1-b: done
  epic-2-retrospective: optional

# Action items committed during retrospectives
action_items:
  - id: "epic-1-retro-item-1-x"
    epic: 1
    action: "Scripted item"
    owner: "Amelia"
    status: "open"
    ref: "docs/epic-1-retro.md"
  - epic: 1
    action: "Pre-existing item"
    owner: "Charlie"
    status: open
  - id: "epic-2-retro-item-1-y"
    epic: 2
    action: "Scripted item"
    owner: "Dana"
    status: "in-progress"
    ref: "docs/epic-2-retro.md"
"""

# Three spellings of the same scalar, to pin that a status write never re-styles
# the line it lands on.
STYLE_FIXTURE = """\
last_updated: "01-01-2026 09:00"
development_status:
  1-1-a: done
  epic-1-retrospective: optional
action_items:
  - id: "double"
    epic: 1
    action: "Double"
    status: "open"
  - id: "single"
    epic: 1
    action: "Single"
    status: 'open'
  - id: "plain"
    epic: 1
    action: "Plain"
    status: open
"""


def _run(args):
    cmd = ["uv", "run", str(SCRIPT), *args]
    # LC_ALL=C keeps os.strerror text stable so error-string assertions do not
    # depend on the developer's locale.
    return subprocess.run(
        cmd, capture_output=True, text=True, env={**os.environ, "LC_ALL": "C"}
    )


def _module():
    """Import the script as a module, for the few properties that cannot be
    triggered through the CLI (a failure after the temp file already exists)."""
    spec = importlib.util.spec_from_file_location("sprint_status", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_fixture(tmp_path):
    target = tmp_path / "sprint-status.yaml"
    target.write_text(FIXTURE, encoding="utf-8")
    return target


def _write_action_fixture(tmp_path):
    target = tmp_path / "sprint-status.yaml"
    target.write_text(ACTION_FIXTURE, encoding="utf-8")
    return target


def _load(path):
    yaml = YAML(typ="rt")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.load(fh)


def _json(proc):
    """Parse the JSON-only stdout contract, surfacing a crash instead of hiding
    it behind a JSONDecodeError."""
    assert proc.stdout, f"empty stdout; stderr was: {proc.stderr}"
    assert "Traceback" not in proc.stderr, proc.stderr
    return json.loads(proc.stdout)


def test_detect_epic(tmp_path):
    target = _write_fixture(tmp_path)
    proc = _run(["detect-epic", "--file", str(target)])
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["epic"] == 1
    assert out["story_count"] == 2
    assert out["retro_key"] == "epic-1-retrospective"
    assert out["retro_status"] == "optional"
    assert set(out["done_stories"]) == {
        "1-1-user-authentication",
        "1-2-account-management",
    }


def test_detect_epic_rejects_typed_retrospective_status_as_json(tmp_path):
    fixture = (
        "development_status:\n"
        "  1-1-a: done\n"
        "  epic-1-retrospective: 2026-01-01\n"
    )
    target = tmp_path / "sprint-status.yaml"
    target.write_text(fixture, encoding="utf-8")

    proc = _run(["detect-epic", "--file", str(target)])

    assert proc.returncode == 1
    out = _json(proc)
    assert out["ok"] is False
    assert out["error"] == "epic-1-retrospective status must be a string or null"
    assert "restored" not in out
    assert target.read_text(encoding="utf-8") == fixture


# --- pending_stories: the unfinished-epic gate --------------------------------


def test_pending_stories_lists_the_selected_epics_unfinished_keys(tmp_path):
    # The gate the skill branches on before Phase 1: an epic whose highest done
    # story selected it, but which is not actually finished. Document order, so
    # the listing the user confirms matches the file they can open.
    fixture = (
        "development_status:\n"
        "  epic-2: backlog\n"
        "  2-1-a: done\n"
        "  2-2-b: backlog\n"
        "  2-3-c: ready-for-dev\n"
    )
    target = tmp_path / "sprint-status.yaml"
    target.write_text(fixture, encoding="utf-8")
    proc = _run(["detect-epic", "--file", str(target)])
    assert proc.returncode == 0, proc.stderr
    out = _json(proc)
    assert out["epic"] == 2
    assert out["pending_stories"] == ["2-2-b", "2-3-c"]
    # A non-story key sitting beside them never leaks in: STORY_RE gates entry.
    assert "epic-2" not in out["pending_stories"]


def test_pending_stories_is_empty_for_a_complete_epic(tmp_path):
    fixture = (
        "development_status:\n"
        "  2-1-a: done\n"
        "  2-2-b: done\n"
        "  epic-2-retrospective: optional\n"
    )
    target = tmp_path / "sprint-status.yaml"
    target.write_text(fixture, encoding="utf-8")
    proc = _run(["detect-epic", "--file", str(target)])
    assert proc.returncode == 0, proc.stderr
    out = _json(proc)
    assert out["epic"] == 2
    assert out["pending_stories"] == []


def test_pending_stories_ignores_other_epics(tmp_path):
    # FIXTURE detects epic 1 while 2-1-dashboard sits at backlog. The key is
    # scoped to the *selected* epic -- unlike done_stories, which spans the whole
    # file -- so another epic's unfinished work must never block this retro.
    target = _write_fixture(tmp_path)
    proc = _run(["detect-epic", "--file", str(target)])
    assert proc.returncode == 0, proc.stderr
    out = _json(proc)
    assert out["epic"] == 1
    assert out["pending_stories"] == []
    # done_stories keeps its whole-file scope, unchanged.
    assert set(out["done_stories"]) == {
        "1-1-user-authentication",
        "1-2-account-management",
    }


def test_pending_stories_present_when_no_epic_is_detected(tmp_path):
    # No done story anywhere: the shape stays uniform so a caller can read
    # pending_stories without first branching on epic.
    fixture = (
        "development_status:\n"
        "  1-1-a: backlog\n"
        "  2-1-b: ready-for-dev\n"
    )
    target = tmp_path / "sprint-status.yaml"
    target.write_text(fixture, encoding="utf-8")
    proc = _run(["detect-epic", "--file", str(target)])
    assert proc.returncode == 0, proc.stderr
    out = _json(proc)
    assert out["epic"] is None
    assert out["story_count"] == 0
    assert out["pending_stories"] == []


def test_detect_epic_flag_aims_pending_stories_at_a_supplied_epic(tmp_path):
    # Auto-detect would pick epic 2 (highest with a done story). --epic 1 aims
    # the unfinished-epic gate at the orchestrator's explicit choice instead —
    # the -H <epic> path that previously had no pending_stories at all.
    fixture = (
        "development_status:\n"
        "  1-1-a: done\n"
        "  1-2-b: backlog\n"
        "  1-3-c: review\n"
        "  2-1-a: done\n"
        "  2-2-b: done\n"
        "  epic-1-retrospective: optional\n"
    )
    target = tmp_path / "sprint-status.yaml"
    target.write_text(fixture, encoding="utf-8")
    proc = _run(["detect-epic", "--file", str(target), "--epic", "1"])
    assert proc.returncode == 0, proc.stderr
    out = _json(proc)
    assert out["epic"] == 1
    assert out["retro_key"] == "epic-1-retrospective"
    assert out["retro_status"] == "optional"
    assert out["pending_stories"] == ["1-2-b", "1-3-c"]
    # done_stories keeps its whole-file scope.
    assert set(out["done_stories"]) == {"1-1-a", "2-1-a", "2-2-b"}


def test_detect_epic_flag_lists_pending_when_no_story_is_done(tmp_path):
    # Without --epic, no done story means epic is null. With --epic, an
    # unfinished epic that never landed a done story is still addressable —
    # every story key of that epic is pending.
    fixture = (
        "development_status:\n"
        "  3-1-a: backlog\n"
        "  3-2-b: ready-for-dev\n"
    )
    target = tmp_path / "sprint-status.yaml"
    target.write_text(fixture, encoding="utf-8")
    proc = _run(["detect-epic", "--file", str(target), "--epic", "3"])
    assert proc.returncode == 0, proc.stderr
    out = _json(proc)
    assert out["epic"] == 3
    assert out["pending_stories"] == ["3-1-a", "3-2-b"]
    assert out["retro_key"] == "epic-3-retrospective"
    assert out["retro_status"] is None


def test_detect_epic_flag_empty_pending_for_a_complete_supplied_epic(tmp_path):
    fixture = (
        "development_status:\n"
        "  1-1-a: done\n"
        "  1-2-b: done\n"
        "  2-1-a: backlog\n"
    )
    target = tmp_path / "sprint-status.yaml"
    target.write_text(fixture, encoding="utf-8")
    proc = _run(["detect-epic", "--file", str(target), "--epic", "1"])
    assert proc.returncode == 0, proc.stderr
    out = _json(proc)
    assert out["epic"] == 1
    assert out["pending_stories"] == []


def test_detect_epic_flag_zero_story_count_marks_a_nonexistent_epic(tmp_path):
    # --epic 9 against a file that has no epic-9 stories: pending_stories is
    # empty exactly as it is for a finished epic, so story_count is the only
    # signal separating "complete" from "typo'd". The gate reads 0 as suspect,
    # never as done.
    fixture = (
        "development_status:\n"
        "  1-1-a: done\n"
        "  1-2-b: done\n"
    )
    target = tmp_path / "sprint-status.yaml"
    target.write_text(fixture, encoding="utf-8")
    proc = _run(["detect-epic", "--file", str(target), "--epic", "9"])
    assert proc.returncode == 0, proc.stderr
    out = _json(proc)
    assert out["epic"] == 9
    assert out["story_count"] == 0
    assert out["pending_stories"] == []
    assert out["retro_status"] is None
    # The finished epic it could be confused with reports its real count.
    proc = _run(["detect-epic", "--file", str(target), "--epic", "1"])
    out = _json(proc)
    assert out["story_count"] == 2
    assert out["pending_stories"] == []


def test_detect_epic_flag_rejects_non_positive_epic(tmp_path):
    target = _write_fixture(tmp_path)
    for bad in ("0", "-3"):
        proc = _run(["detect-epic", "--file", str(target), "--epic", bad])
        assert proc.returncode == 1, proc.stderr
        out = _json(proc)
        assert out["ok"] is False
        assert "epic" in out["error"]
        assert "restored" not in out


def test_update_rejects_non_positive_epic(tmp_path):
    target = _write_fixture(tmp_path)
    for bad in ("0", "-3"):
        proc = _run(["update", "--file", str(target), "--epic", bad])
        assert proc.returncode == 1, proc.stderr
        assert "positive integer" in _json(proc)["error"]
        assert target.read_text(encoding="utf-8") == FIXTURE


# --- The JSON-only contract covers the help paths ----------------------------


@pytest.mark.parametrize(
    "args",
    [["-h"], ["--help"], ["detect-epic", "-h"], ["detect-epic", "--help"]],
    ids=["top-short", "top-long", "sub-short", "sub-long"],
)
def test_help_flags_emit_json_not_usage(args):
    # argparse's built-in help action bypasses error() -- it prints usage text
    # to stdout and exits 0, which is exactly the contract this script sells.
    # add_help=False turns -h into an ordinary unrecognized argument instead.
    proc = _run(args)
    # Exit 2 specifically: the module docstring reserves 2 for argument errors
    # and 1 for I/O failures, and retro-document.md teaches callers to tell the
    # two apart, so collapsing them must fail here.
    assert proc.returncode == 2
    assert "usage:" not in proc.stdout
    out = _json(proc)
    assert out["ok"] is False
    assert out["error"]
    # An argparse rejection speaks for no file, so it carries no "restored" --
    # the same rule test_only_the_write_path_reports_restored pins for exit 1.
    assert "restored" not in out


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_update_help_flag_emits_json_not_usage(tmp_path, flag):
    # The update subparser too, driven with its required arguments present so
    # nothing but the help flag itself can be what argparse objects to.
    target = _write_fixture(tmp_path)
    proc = _run(["update", "--file", str(target), "--epic", "1", flag])
    assert proc.returncode == 2
    assert "usage:" not in proc.stdout
    out = _json(proc)
    assert out["ok"] is False
    assert flag in out["error"]
    assert "restored" not in out
    # A rejected invocation must not have written anything.
    assert target.read_text(encoding="utf-8") == FIXTURE


def test_update_sets_retro_and_appends_action(tmp_path):
    target = _write_fixture(tmp_path)
    payload = '[{"action":"Fix #42: colons: and # hashes","owner":"Amelia"}]'
    proc = _run(
        [
            "update",
            "--file",
            str(target),
            "--epic",
            "1",
            "--set-retro-done",
            "--add-action",
            payload,
        ]
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["ok"] is True
    assert out["retro_key_found"] is True
    assert out["retro_status_after"] == "done"
    assert out["action_items_added"] == 1

    # File must still parse cleanly (punctuation did not corrupt it).
    data = _load(target)
    assert data is not None

    # STATUS DEFINITIONS comment survived.
    raw = target.read_text(encoding="utf-8")
    assert "STATUS DEFINITIONS" in raw

    # Retro status flipped to done.
    assert data["development_status"]["epic-1-retrospective"] == "done"

    # The action value round-trips with literal '#' and ':' intact.
    action = data["action_items"][0]
    assert action["action"] == "Fix #42: colons: and # hashes"
    assert action["owner"] == "Amelia"
    assert action["epic"] == 1
    assert action["status"] == "open"


def test_update_rejects_typed_retrospective_status_before_writing(tmp_path):
    fixture = (
        "last_updated: 01-01-2026 09:00\n"
        "development_status:\n"
        "  1-1-a: done\n"
        "  epic-1-retrospective: 2026-01-01\n"
    )
    target = tmp_path / "sprint-status.yaml"
    target.write_text(fixture, encoding="utf-8")

    proc = _run(
        [
            "update",
            "--file",
            str(target),
            "--epic",
            "1",
            "--set-retro-done",
            "--add-action",
            '[{"action":"Must not be appended","owner":"Amelia"}]',
        ]
    )

    assert proc.returncode == 1
    out = _json(proc)
    assert out["ok"] is False
    assert out["restored"] is True
    assert out["error"] == "epic-1-retrospective status must be a string or null"
    assert target.read_text(encoding="utf-8") == fixture


def test_detect_epic_matches_split_story_keys(tmp_path):
    # A split-story key like 2-6a-... is first-class in BMAD (an oversized story
    # split into 2-6a / 2-6b) and must not be invisible to detection — otherwise
    # an epic whose only done stories are splits is silently skipped.
    fixture = (
        "development_status:\n"
        "  1-1-first: done\n"
        "  2-6a-split-auth: done\n"
        "  epic-2-retrospective: optional\n"
    )
    target = tmp_path / "sprint-status.yaml"
    target.write_text(fixture, encoding="utf-8")
    proc = _run(["detect-epic", "--file", str(target)])
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["epic"] == 2
    assert "2-6a-split-auth" in out["done_stories"]
    assert out["retro_key"] == "epic-2-retrospective"


def test_update_rejects_non_list_action_items(tmp_path):
    # A hand-corrupted action_items must fail on the JSON contract, not crash.
    fixture = (
        "development_status:\n"
        "  1-1-a: done\n"
        "  epic-1-retrospective: optional\n"
        'action_items: "oops-not-a-list"\n'
    )
    target = tmp_path / "sprint-status.yaml"
    target.write_text(fixture, encoding="utf-8")
    proc = _run(
        ["update", "--file", str(target), "--epic", "1",
         "--add-action", '[{"action":"x","owner":"y"}]']
    )
    assert proc.returncode == 1
    out = json.loads(proc.stdout)  # must be JSON, not a traceback
    assert out["ok"] is False
    assert "action_items" in out["error"]


def test_appended_items_carry_id_and_ref(tmp_path):
    target = _write_fixture(tmp_path)
    ref = "docs/stories/epic-1-retro-2026-07-21.md"
    proc = _run(
        ["update", "--file", str(target), "--epic", "1", "--set-retro-done",
         "--add-action", '[{"action":"Fix the seam","owner":"Amelia"}]',
         "--ref", ref, "--verdict", "accepted-with-open-items"]
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["verdict"] == "accepted-with-open-items"  # echoed, not written to a key

    item = _load(target)["action_items"][0]
    assert item["id"].startswith("epic-1-retro-item-1-")
    assert item["ref"] == ref
    # The retro key value stays "done" — verdict is not encoded into it.
    assert _load(target)["development_status"]["epic-1-retrospective"] == "done"


def test_free_spelled_verdict_is_rejected_before_the_file_is_touched(tmp_path):
    # The SKILL's prose verdict ("accepted with open items") and the frontmatter
    # token (accepted-with-open-items) used to be two spellings of one value; an
    # orchestrator branching on the echo would fall through both. Only the
    # frontmatter vocabulary passes; anything else fails with the file intact.
    target = _write_fixture(tmp_path)
    for bad in ("accepted with open items", "ship it", "ACCEPTED"):
        proc = _run(
            ["update", "--file", str(target), "--epic", "1", "--set-retro-done",
             "--verdict", bad]
        )
        assert proc.returncode == 1, f"accepted {bad!r}"
        out = _json(proc)
        assert out["ok"] is False and "--verdict" in out["error"]
        assert out["restored"] is True
    assert target.read_text(encoding="utf-8") == FIXTURE


def test_explicit_item_id_is_preserved(tmp_path):
    target = _write_fixture(tmp_path)
    proc = _run(
        ["update", "--file", str(target), "--epic", "1",
         "--add-action", '[{"action":"a","owner":"o","id":"custom-id-7"}]']
    )
    assert proc.returncode == 0, proc.stderr
    assert _load(target)["action_items"][0]["id"] == "custom-id-7"


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses file permission bits",
)
def test_write_failure_reports_restore_status(tmp_path):
    # If the write cannot happen, the caller must be told whether the original
    # was restored — a silent failure defeats the script's core guarantee.
    #
    # The write is atomic (temp file + os.replace), and os.replace needs write
    # permission on the *directory*, not on the target — a read-only target is
    # now replaceable. Making the containing directory read-only is what blocks
    # the write: mkstemp fails, while the restore write to the still-writable
    # target succeeds.
    holder = tmp_path / "holder"
    holder.mkdir()
    target = _write_fixture(holder)
    os.chmod(holder, 0o555)
    try:
        proc = _run(
            ["update", "--file", str(target), "--epic", "1", "--set-retro-done"]
        )
    finally:
        os.chmod(holder, 0o755)
    assert proc.returncode == 1
    out = _json(proc)
    assert out["ok"] is False
    assert out["restored"] is True
    # The file was never touched: the temp file could not even be created.
    assert target.read_text(encoding="utf-8") == FIXTURE
    assert [p.name for p in holder.iterdir()] == ["sprint-status.yaml"]


def test_punctuation_does_not_corrupt_file(tmp_path):
    # Explicit re-parse guarantee for YAML-breaking punctuation.
    target = _write_fixture(tmp_path)
    payload = '[{"action":"weird: value # with: hashes","owner":"Bob # Smith"}]'
    proc = _run(
        [
            "update",
            "--file",
            str(target),
            "--epic",
            "1",
            "--add-action",
            payload,
        ]
    )
    assert proc.returncode == 0, proc.stderr
    # Re-parse must succeed and preserve the literal punctuation.
    data = _load(target)
    assert data["action_items"][0]["action"] == "weird: value # with: hashes"
    assert data["action_items"][0]["owner"] == "Bob # Smith"


# --- Formatting fidelity -----------------------------------------------------


def test_template_round_trip_changes_only_last_updated(tmp_path):
    # The repo's own sprint-status template is the shape every generated file
    # inherits: 2-space sequence indent and a mid-file comment above
    # action_items. An update must touch nothing but last_updated — a re-indent
    # of a pre-existing, untouched entry defeats the preservation guarantee that
    # motivates "do not hand-edit this file".
    source = TEMPLATE.read_text(encoding="utf-8")
    target = tmp_path / "sprint-status.yaml"
    target.write_text(source, encoding="utf-8")

    proc = _run(
        ["update", "--file", str(target), "--epic", "1", "--date", "01-01-2026 09:00"]
    )
    assert proc.returncode == 0, proc.stderr
    assert _json(proc)["ok"] is True

    before = source.splitlines()
    after = target.read_text(encoding="utf-8").splitlines()
    assert len(before) == len(after)
    changed = [(b, a) for b, a in zip(before, after) if b != a]
    assert len(changed) == 1, changed
    assert changed[0][1] == "last_updated: 01-01-2026 09:00"
    # The pre-existing action item keeps its 2-space sequence indent.
    assert "  - epic: 1" in after


def test_mid_file_comment_survives_update(tmp_path):
    fixture = (
        "# header\n"
        "development_status:\n"
        "  1-1-a: done\n"
        "  epic-1-retrospective: optional\n"
        "\n"
        "# Action items committed during retrospectives\n"
        "action_items:\n"
        "  - epic: 1\n"
        '    action: "Pre-existing item"\n'
        '    owner: "Charlie"\n'
        "    status: open\n"
    )
    target = tmp_path / "sprint-status.yaml"
    target.write_text(fixture, encoding="utf-8")
    proc = _run(
        ["update", "--file", str(target), "--epic", "1", "--set-retro-done",
         "--add-action", '[{"action":"New item","owner":"Amelia"}]']
    )
    assert proc.returncode == 0, proc.stderr
    raw = target.read_text(encoding="utf-8")
    assert "# Action items committed during retrospectives" in raw
    assert "# header" in raw
    assert '    action: "Pre-existing item"' in raw


def test_legacy_offset_zero_file_is_canonicalized(tmp_path):
    # Files the previous version of this script wrote carry action_items at
    # column 0. The indent pin re-indents them to the template's shape on the
    # next write. That is a deliberate one-time canonicalization, not a silent
    # failure: the update still succeeds and no comment is lost.
    fixture = (
        "# header\n"
        "development_status:\n"
        "  1-1-a: done\n"
        "  epic-1-retrospective: optional\n"
        "action_items:\n"
        '- id: "legacy"\n'
        "  epic: 1\n"
        '  action: "written by the old code"\n'
        "  status: open\n"
    )
    target = tmp_path / "sprint-status.yaml"
    target.write_text(fixture, encoding="utf-8")
    proc = _run(["update", "--file", str(target), "--epic", "1", "--set-retro-done"])
    assert proc.returncode == 0, proc.stderr
    raw = target.read_text(encoding="utf-8")
    assert '  - id: "legacy"' in raw
    assert '    action: "written by the old code"' in raw
    assert "# header" in raw


def test_lost_comment_fails_with_restore(tmp_path):
    # A standalone comment inside a flow collection is genuinely dropped by the
    # round-trip. The leading-block check never saw it; the full multiset does,
    # and the original bytes must come back.
    fixture = (
        "# header\n"
        "development_status:\n"
        "  1-1-a: done\n"
        "  epic-1-retrospective: optional\n"
        "tags: [\n"
        "  # a standalone comment the round-trip drops\n"
        '  "alpha",\n'
        '  "beta",\n'
        "]\n"
    )
    target = tmp_path / "sprint-status.yaml"
    target.write_text(fixture, encoding="utf-8")
    proc = _run(
        ["update", "--file", str(target), "--epic", "1", "--set-retro-done"]
    )
    assert proc.returncode == 1
    out = _json(proc)
    assert out["ok"] is False
    assert out["restored"] is True
    assert "comment line lost" in out["error"]
    assert "a standalone comment the round-trip drops" in out["error"]
    assert target.read_text(encoding="utf-8") == fixture


# --- Malformed input stays on the JSON contract ------------------------------


@pytest.mark.parametrize("command", ["detect-epic", "update"])
def test_non_mapping_root_is_json_error(tmp_path, command):
    target = tmp_path / "sprint-status.yaml"
    target.write_text("- a\n- b\n", encoding="utf-8")
    args = ["--file", str(target)] + (["--epic", "1"] if command == "update" else [])
    proc = _run([command, *args])
    assert proc.returncode == 1
    out = _json(proc)
    assert out["ok"] is False
    assert "root document is not a mapping" in out["error"]
    assert target.read_text(encoding="utf-8") == "- a\n- b\n"


@pytest.mark.parametrize("command", ["detect-epic", "update"])
@pytest.mark.parametrize(
    "body",
    ['development_status: "not-a-mapping"\n', "development_status:\n  - a\n  - b\n"],
    ids=["scalar", "list"],
)
def test_non_mapping_development_status_is_json_error(tmp_path, command, body):
    target = tmp_path / "sprint-status.yaml"
    target.write_text(body, encoding="utf-8")
    args = ["--file", str(target)] + (["--epic", "1"] if command == "update" else [])
    proc = _run([command, *args])
    # update used to report ok:true here while silently doing nothing.
    assert proc.returncode == 1
    out = _json(proc)
    assert out["ok"] is False
    assert "development_status is not a mapping" in out["error"]


@pytest.mark.parametrize("command", ["detect-epic", "update"])
def test_directory_target_is_json_error(tmp_path, command):
    target = tmp_path / "a-directory"
    target.mkdir()
    args = ["--file", str(target)] + (["--epic", "1"] if command == "update" else [])
    proc = _run([command, *args])
    assert proc.returncode == 1
    out = _json(proc)
    assert out["ok"] is False
    assert out["error"]


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses file permission bits",
)
@pytest.mark.parametrize("command", ["detect-epic", "update"])
def test_unreadable_target_is_json_error(tmp_path, command):
    # The other half of the OSError widening: PermissionError, not just
    # IsADirectoryError, has to stay on the JSON contract.
    target = _write_fixture(tmp_path)
    os.chmod(target, 0o000)
    args = ["--file", str(target)] + (["--epic", "1"] if command == "update" else [])
    try:
        proc = _run([command, *args])
    finally:
        os.chmod(target, 0o644)
    assert proc.returncode == 1
    out = _json(proc)
    assert out["ok"] is False
    assert "denied" in out["error"].lower()


@pytest.mark.parametrize("command", ["detect-epic", "update"])
def test_invalid_utf8_is_json_error(tmp_path, command):
    target = tmp_path / "sprint-status.yaml"
    target.write_bytes(b"development_status:\n  1-1-a: d\xffone\n")
    args = ["--file", str(target)] + (["--epic", "1"] if command == "update" else [])
    proc = _run([command, *args])
    assert proc.returncode == 1
    out = _json(proc)
    assert out["ok"] is False
    assert "utf-8" in out["error"].lower()


# --- Atomic write ------------------------------------------------------------


def test_atomic_write_failure_leaves_target_byte_identical(tmp_path, monkeypatch):
    # The failure the atomic write exists for: something goes wrong after the
    # temp file has been written. Nothing may reach the target and no temp file
    # may survive. No CLI path reaches here -- a read-only directory fails at
    # mkstemp instead -- so this drives the helper directly.
    mod = _module()
    target = _write_fixture(tmp_path)

    def boom(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(mod.os, "replace", boom)
    with pytest.raises(OSError):
        mod._atomic_write(str(target), b"replacement bytes\n", 0o644)
    assert target.read_text(encoding="utf-8") == FIXTURE
    assert [p.name for p in tmp_path.iterdir()] == ["sprint-status.yaml"]


def test_dir_fsync_failure_after_rename_is_not_a_write_failure(tmp_path, monkeypatch):
    # Once os.replace has returned, the new bytes ARE the file. The directory
    # fsync that follows is durability polish; if it raised, cmd_update would
    # emit "restored": true about a write that in fact landed — the one lie the
    # restored contract exists to prevent. Deny opening the directory (the only
    # thing _atomic_write opens by path after the rename) and require success.
    mod = _module()
    target = _write_fixture(tmp_path)
    directory = os.path.dirname(os.path.realpath(str(target)))
    real_open = os.open

    def deny_directory_open(p, *args, **kwargs):
        if p == directory:
            raise OSError(5, "Input/output error")
        return real_open(p, *args, **kwargs)

    monkeypatch.setattr(mod.os, "open", deny_directory_open)
    mod._atomic_write(str(target), b"replacement bytes\n", 0o644)
    assert target.read_bytes() == b"replacement bytes\n"
    assert [p.name for p in tmp_path.iterdir()] == ["sprint-status.yaml"]


def test_restore_is_atomic(tmp_path, monkeypatch):
    # _restore is the rollback the reference sells as the safety net. A
    # truncating rewrite that dies halfway would destroy the very bytes it is
    # putting back, which is how a full disk used to corrupt the file.
    mod = _module()
    target = tmp_path / "sprint-status.yaml"
    target.write_text("damaged\n", encoding="utf-8")

    def boom(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(mod.os, "replace", boom)
    assert mod._restore(str(target), FIXTURE.encode("utf-8"), 0o644) is False
    # It reported failure honestly and left the file no worse than it found it.
    assert target.read_text(encoding="utf-8") == "damaged\n"
    assert [p.name for p in tmp_path.iterdir()] == ["sprint-status.yaml"]


def test_symlinked_target_is_written_through(tmp_path):
    # os.replace onto a symlink would detach the link and leave the real file
    # stale while reporting ok:true.
    real = tmp_path / "real-sprint-status.yaml"
    real.write_text(FIXTURE, encoding="utf-8")
    link = tmp_path / "sprint-status.yaml"
    link.symlink_to(real)
    proc = _run(["update", "--file", str(link), "--epic", "1", "--set-retro-done"])
    assert proc.returncode == 0, proc.stderr
    assert link.is_symlink(), "the symlink was replaced by a regular file"
    assert _load(real)["development_status"]["epic-1-retrospective"] == "done"


def test_atomic_write_preserves_mode_and_leaves_no_temp_file(tmp_path):
    # mkstemp creates 0600; without carrying the target's mode over, every
    # update would silently narrow the file.
    holder = tmp_path / "holder"
    holder.mkdir()
    target = _write_fixture(holder)
    os.chmod(target, 0o640)
    proc = _run(["update", "--file", str(target), "--epic", "1", "--set-retro-done"])
    assert proc.returncode == 0, proc.stderr
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert [p.name for p in holder.iterdir()] == ["sprint-status.yaml"]


# --- Result-JSON precision ---------------------------------------------------


def test_retro_key_found_is_null_without_the_flag(tmp_path):
    # No development_status key at all: the update must not conjure one, and
    # retro_key_found must say "not asked" rather than "absent".
    fixture = 'project: "Demo"\nlast_updated: "01-01-2026 09:00"\n'
    target = tmp_path / "sprint-status.yaml"
    target.write_text(fixture, encoding="utf-8")
    proc = _run(["update", "--file", str(target), "--epic", "1"])
    assert proc.returncode == 0, proc.stderr
    out = _json(proc)
    assert out["ok"] is True
    assert out["retro_key_found"] is None
    assert "development_status" not in target.read_text(encoding="utf-8")


def test_retro_key_found_is_false_when_the_key_is_absent(tmp_path):
    target = _write_fixture(tmp_path)
    proc = _run(["update", "--file", str(target), "--epic", "2", "--set-retro-done"])
    assert proc.returncode == 0, proc.stderr
    out = _json(proc)
    assert out["ok"] is True
    assert out["retro_key_found"] is False
    # Nothing was written into the mapping.
    assert "epic-2-retrospective" not in _load(target)["development_status"]


@pytest.mark.parametrize("command", ["detect-epic", "update"])
def test_only_the_write_path_reports_restored(tmp_path, command):
    # "restored" speaks to the state of a file the command may have written.
    # detect-epic never writes, so inventing the key there would mislead callers
    # that branch on it.
    target = tmp_path / "sprint-status.yaml"
    target.write_text("- a\n- b\n", encoding="utf-8")
    args = ["--file", str(target)] + (["--epic", "1"] if command == "update" else [])
    out = _json(_run([command, *args]))
    assert out["ok"] is False
    assert ("restored" in out) is (command == "update")


def test_pre_write_failure_reports_restored(tmp_path):
    # retro-document.md teaches callers that ok:false carries restored:true;
    # a failure before the write must not read as "the file may be incomplete".
    target = _write_fixture(tmp_path)
    proc = _run(
        ["update", "--file", str(target), "--epic", "1", "--add-action", "{not json"]
    )
    assert proc.returncode == 1
    out = _json(proc)
    assert out["ok"] is False
    assert out["restored"] is True
    assert target.read_text(encoding="utf-8") == FIXTURE


# --- Action-item validation and identity -------------------------------------


def test_non_latin_action_keeps_its_text_in_the_id(tmp_path):
    target = _write_fixture(tmp_path)
    payload = json.dumps(
        [{"action": "Улучшить обработку ошибок", "owner": "Amelia"}],
        ensure_ascii=False,
    )
    proc = _run(["update", "--file", str(target), "--epic", "1", "--add-action", payload])
    assert proc.returncode == 0, proc.stderr
    item_id = _load(target)["action_items"][0]["id"]
    assert item_id == "epic-1-retro-item-1-улучшить-обработку-ошибок"


def test_unsluggable_action_falls_back_to_a_hash(tmp_path):
    target = _write_fixture(tmp_path)
    payload = json.dumps([{"action": "!!! 🎉", "owner": "Amelia"}], ensure_ascii=False)
    proc = _run(["update", "--file", str(target), "--epic", "1", "--add-action", payload])
    assert proc.returncode == 0, proc.stderr
    item_id = _load(target)["action_items"][0]["id"]
    assert not item_id.endswith("-item")
    assert re.fullmatch(r"epic-1-retro-item-1-[0-9a-f]{8}", item_id), item_id


def test_empty_action_is_rejected(tmp_path):
    target = _write_fixture(tmp_path)
    proc = _run(
        ["update", "--file", str(target), "--epic", "1",
         "--add-action", '[{"action":"   ","owner":"x"}]']
    )
    assert proc.returncode == 1
    out = _json(proc)
    assert out["ok"] is False
    assert out["restored"] is True
    assert "action" in out["error"]
    assert target.read_text(encoding="utf-8") == FIXTURE


def test_non_string_action_is_rejected(tmp_path):
    # A JSON null would otherwise be str()'d into a literal "None" and written
    # as a real action item, which the new emptiness check alone lets through.
    target = _write_fixture(tmp_path)
    proc = _run(
        ["update", "--file", str(target), "--epic", "1",
         "--add-action", '[{"action":null,"owner":"x"}]']
    )
    assert proc.returncode == 1
    out = _json(proc)
    assert out["ok"] is False
    assert out["restored"] is True
    assert target.read_text(encoding="utf-8") == FIXTURE


def test_date_is_normalized_to_the_canonical_format(tmp_path):
    # strptime accepts unpadded spellings; writing those through would defeat
    # the point of validating the format.
    target = _write_fixture(tmp_path)
    proc = _run(
        ["update", "--file", str(target), "--epic", "1", "--date", "1-2-2026 9:05"]
    )
    assert proc.returncode == 0, proc.stderr
    assert _json(proc)["last_updated"] == "01-02-2026 09:05"
    assert _load(target)["last_updated"] == "01-02-2026 09:05"


def test_malformed_date_is_rejected(tmp_path):
    target = _write_fixture(tmp_path)
    proc = _run(
        ["update", "--file", str(target), "--epic", "1", "--date", "not-a-date"]
    )
    assert proc.returncode == 1
    out = _json(proc)
    assert out["ok"] is False
    assert out["restored"] is True
    assert "--date" in out["error"]
    assert target.read_text(encoding="utf-8") == FIXTURE


# --- Action-item status transitions ------------------------------------------


def test_set_action_status_applies_both_selector_forms(tmp_path):
    # The whole point of the flag: an item written by this script (selected by
    # id) and a legacy item that predates ids (selected by epic + exact action
    # text) both move off "open" in a single call.
    target = _write_action_fixture(tmp_path)
    payload = json.dumps(
        [
            {"id": "epic-1-retro-item-1-x", "status": "done"},
            {"epic": 1, "action": "Pre-existing item", "status": "in-progress"},
        ]
    )
    proc = _run(
        ["update", "--file", str(target), "--epic", "1", "--set-action-status", payload]
    )
    assert proc.returncode == 0, proc.stderr
    out = _json(proc)
    assert out["ok"] is True
    assert out["action_items_updated"] == 2
    assert out["action_items_added"] == 0

    items = _load(target)["action_items"]
    assert items[0]["status"] == "done"
    assert items[1]["status"] == "in-progress"
    # Every other key of both items survived untouched, in place.
    assert items[0]["id"] == "epic-1-retro-item-1-x"
    assert items[0]["epic"] == 1
    assert items[0]["action"] == "Scripted item"
    assert items[0]["owner"] == "Amelia"
    assert items[0]["ref"] == "docs/epic-1-retro.md"
    assert "id" not in items[1]
    assert items[1]["epic"] == 1
    assert items[1]["action"] == "Pre-existing item"
    assert items[1]["owner"] == "Charlie"
    # The epic-2 item shares its action text with items[0]; the epic-1 selector
    # must not have touched it.
    assert items[2]["status"] == "in-progress"
    assert items[2]["id"] == "epic-2-retro-item-1-y"
    assert len(items) == 3


def test_set_action_status_changes_exactly_one_line(tmp_path):
    # The status write is surgical: it must not re-style neighbouring lines, and
    # an item whose status was quoted keeps its quoting. last_updated is rewritten
    # with the same text it already held, so the whole file differs by one line.
    target = _write_action_fixture(tmp_path)
    proc = _run(
        ["update", "--file", str(target), "--epic", "1", "--date", "01-01-2026 09:00",
         "--set-action-status", '[{"id":"epic-1-retro-item-1-x","status":"done"}]']
    )
    assert proc.returncode == 0, proc.stderr

    before = ACTION_FIXTURE.splitlines()
    after = target.read_text(encoding="utf-8").splitlines()
    assert len(before) == len(after)
    changed = [(b, a) for b, a in zip(before, after) if b != a]
    assert changed == [('    status: "open"', '    status: "done"')], changed


def test_set_action_status_composes_with_retro_done_and_add_action(tmp_path):
    target = _write_action_fixture(tmp_path)
    proc = _run(
        ["update", "--file", str(target), "--epic", "1", "--set-retro-done",
         "--add-action", '[{"action":"Brand new item","owner":"Amelia"}]',
         "--set-action-status",
         '[{"epic":1,"action":"Pre-existing item","status":"done"}]']
    )
    assert proc.returncode == 0, proc.stderr
    out = _json(proc)
    assert out["ok"] is True
    assert out["retro_status_after"] == "done"
    assert out["action_items_added"] == 1
    assert out["action_items_updated"] == 1

    data = _load(target)
    assert data["development_status"]["epic-1-retrospective"] == "done"
    items = data["action_items"]
    assert len(items) == 4
    assert items[1]["status"] == "done"  # the targeted pre-existing item
    assert items[3]["action"] == "Brand new item"
    assert items[3]["status"] == "open"  # the appended item is always open


def test_set_action_status_cannot_target_an_item_added_in_the_same_run(tmp_path):
    # Selectors resolve against action_items as loaded, so the append cannot be
    # observed by the same invocation. Silently succeeding here would make the
    # flag a back door for writing a non-open status onto a brand-new item.
    target = _write_action_fixture(tmp_path)
    proc = _run(
        ["update", "--file", str(target), "--epic", "1",
         "--add-action", '[{"action":"Brand new","owner":"A","id":"brand-new"}]',
         "--set-action-status", '[{"id":"brand-new","status":"done"}]']
    )
    assert proc.returncode == 1
    out = _json(proc)
    assert out["ok"] is False
    assert out["restored"] is True
    assert "no action item matches" in out["error"]
    assert target.read_text(encoding="utf-8") == ACTION_FIXTURE


def test_set_action_status_rejects_unknown_id(tmp_path):
    target = _write_action_fixture(tmp_path)
    proc = _run(
        ["update", "--file", str(target), "--epic", "1",
         "--set-action-status", '[{"id":"not-in-the-file","status":"done"}]']
    )
    assert proc.returncode == 1
    out = _json(proc)
    assert out["ok"] is False
    assert out["restored"] is True
    assert "not-in-the-file" in out["error"]
    assert target.read_text(encoding="utf-8") == ACTION_FIXTURE


def test_set_action_status_rejects_selector_when_action_items_is_absent(tmp_path):
    # No action_items key at all must read as "no match", not as a crash.
    target = _write_fixture(tmp_path)
    proc = _run(
        ["update", "--file", str(target), "--epic", "1",
         "--set-action-status", '[{"id":"anything","status":"done"}]']
    )
    assert proc.returncode == 1
    out = _json(proc)
    assert out["ok"] is False
    assert out["restored"] is True
    assert "no action item matches" in out["error"]
    assert target.read_text(encoding="utf-8") == FIXTURE


def test_set_action_status_rejects_ambiguous_selector(tmp_path):
    # Two legacy items with identical epic + action text: guessing between them
    # would write the wrong row half the time.
    fixture = (
        "development_status:\n"
        "  1-1-a: done\n"
        "  epic-1-retrospective: optional\n"
        "action_items:\n"
        "  - epic: 1\n"
        '    action: "Same text"\n'
        '    owner: "Charlie"\n'
        "    status: open\n"
        "  - epic: 1\n"
        '    action: "Same text"\n'
        '    owner: "Dana"\n'
        "    status: open\n"
    )
    target = tmp_path / "sprint-status.yaml"
    target.write_text(fixture, encoding="utf-8")
    proc = _run(
        ["update", "--file", str(target), "--epic", "1",
         "--set-action-status", '[{"epic":1,"action":"Same text","status":"done"}]']
    )
    assert proc.returncode == 1
    out = _json(proc)
    assert out["ok"] is False
    assert out["restored"] is True
    assert "ambiguous" in out["error"]
    assert "Same text" in out["error"]
    assert "2 matches" in out["error"]
    assert target.read_text(encoding="utf-8") == fixture


def test_set_action_status_rejects_two_entries_hitting_the_same_item(tmp_path):
    # The id form and the epic/action form can name the same row; applying both
    # would overcount action_items_updated and hide a conflicting pair.
    target = _write_action_fixture(tmp_path)
    payload = json.dumps(
        [
            {"id": "epic-1-retro-item-1-x", "status": "done"},
            {"epic": 1, "action": "Scripted item", "status": "in-progress"},
        ]
    )
    proc = _run(
        ["update", "--file", str(target), "--epic", "1", "--set-action-status", payload]
    )
    assert proc.returncode == 1
    out = _json(proc)
    assert out["ok"] is False
    assert out["restored"] is True
    assert "same action item" in out["error"]
    assert target.read_text(encoding="utf-8") == ACTION_FIXTURE


def test_entry_with_both_selector_forms_uses_the_id(tmp_path):
    # A caller that copied a whole item through supplies both. The id is the
    # precise form and wins; the extra keys are ignored, not rejected. The two
    # forms are pointed at *different* rows so the precedence is observable:
    # the id names items[0], the epic/action pair names items[1].
    target = _write_action_fixture(tmp_path)
    payload = json.dumps(
        [
            {
                "id": "epic-1-retro-item-1-x",
                "epic": 1,
                "action": "Pre-existing item",
                "owner": "Charlie",
                "status": "done",
            }
        ]
    )
    proc = _run(
        ["update", "--file", str(target), "--epic", "1", "--set-action-status", payload]
    )
    assert proc.returncode == 0, proc.stderr
    assert _json(proc)["action_items_updated"] == 1
    items = _load(target)["action_items"]
    assert items[0]["status"] == "done"  # the id's item
    assert items[1]["status"] == "open"  # the epic/action item, untouched


def test_set_action_status_rejects_invalid_status(tmp_path):
    target = _write_action_fixture(tmp_path)
    proc = _run(
        ["update", "--file", str(target), "--epic", "1",
         "--set-action-status", '[{"id":"epic-1-retro-item-1-x","status":"closed"}]']
    )
    assert proc.returncode == 1
    out = _json(proc)
    assert out["ok"] is False
    assert out["restored"] is True
    assert "closed" in out["error"]
    # The allowed vocabulary is named so the caller can correct the call.
    assert "open, in-progress, done" in out["error"]
    assert target.read_text(encoding="utf-8") == ACTION_FIXTURE


def test_set_action_status_rejects_malformed_json(tmp_path):
    target = _write_action_fixture(tmp_path)
    proc = _run(
        ["update", "--file", str(target), "--epic", "1",
         "--set-action-status", "{not json"]
    )
    assert proc.returncode == 1
    out = _json(proc)
    assert out["ok"] is False
    assert out["restored"] is True
    assert "invalid --set-action-status JSON" in out["error"]
    assert target.read_text(encoding="utf-8") == ACTION_FIXTURE


_SELECTOR_SHAPE_ERROR = (
    "each --set-action-status entry must have a non-empty string id, "
    "or an integer epic and a non-empty string action"
)


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        (
            '{"id":"epic-1-retro-item-1-x","status":"done"}',
            "--set-action-status must be a JSON array",
        ),
        (
            '["epic-1-retro-item-1-x"]',
            "each --set-action-status entry must be an object",
        ),
        ('[{"status":"done"}]', _SELECTOR_SHAPE_ERROR),
        (
            '[{"id":"","status":"done"}]',
            "each --set-action-status id must be a non-empty string",
        ),
        (
            '[{"id":42,"status":"done"}]',
            "each --set-action-status id must be a non-empty string",
        ),
        (
            '[{"epic":"1","action":"Pre-existing item","status":"done"}]',
            _SELECTOR_SHAPE_ERROR,
        ),
        (
            '[{"epic":true,"action":"Pre-existing item","status":"done"}]',
            _SELECTOR_SHAPE_ERROR,
        ),
        ('[{"epic":1,"action":"   ","status":"done"}]', _SELECTOR_SHAPE_ERROR),
        (
            # No status key at all: the status validator runs before the selector
            # validator, so this is the status branch, not the selector branch.
            '[{"epic":1,"action":"Pre-existing item"}]',
            "invalid --set-action-status status None",
        ),
        (
            '[{"id":"epic-1-retro-item-1-x","status":3}]',
            "invalid --set-action-status status 3",
        ),
    ],
    ids=[
        "not-a-list",
        "entry-not-an-object",
        "no-selector",
        "empty-id",
        "non-string-id",
        "string-epic",
        "bool-epic",
        "blank-action",
        "no-status-key",
        "non-string-status",
    ],
)
def test_set_action_status_rejects_bad_shapes(tmp_path, payload, expected_error):
    # Each case pins its own message: collapsing the branches into one generic
    # error would leave a caller unable to tell which part of the array is wrong.
    target = _write_action_fixture(tmp_path)
    proc = _run(
        ["update", "--file", str(target), "--epic", "1",
         "--set-action-status", payload]
    )
    assert proc.returncode == 1
    out = _json(proc)
    assert out["ok"] is False
    assert out["restored"] is True
    assert expected_error in out["error"], out["error"]
    assert target.read_text(encoding="utf-8") == ACTION_FIXTURE


def test_selectors_are_not_scoped_to_the_epic_flag(tmp_path):
    # The flag's headline use: epic 2's retro closing epic 1's items. --epic only
    # names the retro key and stamps appended items; scoping selectors to it would
    # silently break the documented cross-epic workflow while every same-epic test
    # kept passing.
    target = _write_action_fixture(tmp_path)
    payload = json.dumps(
        [
            {"id": "epic-1-retro-item-1-x", "status": "done"},
            {"epic": 1, "action": "Pre-existing item", "status": "done"},
        ]
    )
    proc = _run(
        ["update", "--file", str(target), "--epic", "2", "--set-retro-done",
         "--set-action-status", payload]
    )
    assert proc.returncode == 0, proc.stderr
    assert _json(proc)["action_items_updated"] == 2

    data = _load(target)
    assert data["development_status"]["epic-2-retrospective"] == "done"
    items = data["action_items"]
    assert items[0]["status"] == "done"
    assert items[1]["status"] == "done"
    # Epic 2's own item is not swept along.
    assert items[2]["status"] == "in-progress"


def test_legacy_selector_discriminates_on_the_epic(tmp_path):
    # Two items share the action text "Scripted item" and differ only by epic, so
    # an epic-blind text match would be ambiguous -- or worse, silently pick one.
    target = _write_action_fixture(tmp_path)
    proc = _run(
        ["update", "--file", str(target), "--epic", "1", "--set-action-status",
         '[{"epic":2,"action":"Scripted item","status":"done"}]']
    )
    assert proc.returncode == 0, proc.stderr
    assert _json(proc)["action_items_updated"] == 1
    items = _load(target)["action_items"]
    assert items[2]["status"] == "done"
    assert items[0]["status"] == "open"  # the epic-1 namesake, untouched


def test_non_mapping_action_item_does_not_crash_the_selector(tmp_path):
    # A hand-edited scalar in the action_items list must be skipped, not
    # AttributeError'd into an empty stdout with a traceback.
    fixture = (
        "development_status:\n"
        "  1-1-a: done\n"
        "  epic-1-retrospective: optional\n"
        "action_items:\n"
        '  - "a bare string someone hand-edited in"\n'
        '  - id: "real"\n'
        "    epic: 1\n"
        '    action: "Real item"\n'
        "    status: open\n"
    )
    target = tmp_path / "sprint-status.yaml"
    target.write_text(fixture, encoding="utf-8")
    proc = _run(
        ["update", "--file", str(target), "--epic", "1",
         "--set-action-status", '[{"id":"real","status":"done"}]']
    )
    assert proc.returncode == 0, proc.stderr
    assert _json(proc)["action_items_updated"] == 1
    items = _load(target)["action_items"]
    assert items[0] == "a bare string someone hand-edited in"
    assert items[1]["status"] == "done"


def test_non_mapping_action_item_stays_on_the_json_contract_when_unmatched(tmp_path):
    # Same guard, reject path: the scalar must not be dereferenced while looking
    # for a selector that is not there.
    fixture = (
        "development_status:\n"
        "  1-1-a: done\n"
        "action_items:\n"
        "  - 42\n"
    )
    target = tmp_path / "sprint-status.yaml"
    target.write_text(fixture, encoding="utf-8")
    proc = _run(
        ["update", "--file", str(target), "--epic", "1",
         "--set-action-status", '[{"epic":1,"action":"Nothing","status":"done"}]']
    )
    assert proc.returncode == 1
    out = _json(proc)  # asserts stdout is JSON and stderr carries no traceback
    assert out["ok"] is False
    assert out["restored"] is True
    assert "no action item matches" in out["error"]
    assert target.read_text(encoding="utf-8") == fixture


def test_boolean_epic_in_the_file_does_not_match_epic_one(tmp_path):
    # True == 1 in Python, so without the file-side bool guard a hand-edited
    # "epic: true" row would be silently rewritten by a selector aimed at epic 1.
    fixture = (
        "development_status:\n"
        "  1-1-a: done\n"
        "action_items:\n"
        "  - epic: true\n"
        '    action: "Boolean epic"\n'
        "    status: open\n"
    )
    target = tmp_path / "sprint-status.yaml"
    target.write_text(fixture, encoding="utf-8")
    proc = _run(
        ["update", "--file", str(target), "--epic", "1", "--set-action-status",
         '[{"epic":1,"action":"Boolean epic","status":"done"}]']
    )
    assert proc.returncode == 1
    out = _json(proc)
    assert out["ok"] is False
    assert out["restored"] is True
    assert "no action item matches" in out["error"]
    assert target.read_text(encoding="utf-8") == fixture


def test_status_write_preserves_every_scalar_style(tmp_path):
    # The status write must land on the line without re-styling it, whichever way
    # the file spells the scalar.
    target = tmp_path / "sprint-status.yaml"
    target.write_text(STYLE_FIXTURE, encoding="utf-8")
    payload = json.dumps(
        [
            {"id": "double", "status": "done"},
            {"id": "single", "status": "done"},
            {"id": "plain", "status": "done"},
        ]
    )
    proc = _run(
        ["update", "--file", str(target), "--epic", "1", "--date",
         "01-01-2026 09:00", "--set-action-status", payload]
    )
    assert proc.returncode == 0, proc.stderr
    assert _json(proc)["action_items_updated"] == 3

    before = STYLE_FIXTURE.splitlines()
    after = target.read_text(encoding="utf-8").splitlines()
    assert len(before) == len(after)
    changed = [(b, a) for b, a in zip(before, after) if b != a]
    assert changed == [
        ('    status: "open"', '    status: "done"'),
        ("    status: 'open'", "    status: 'done'"),
        ("    status: open", "    status: done"),
    ], changed


def test_action_status_vocabulary_is_exactly_the_three():
    # bmad-sprint-planning is the authority. Widening this tuple would let the
    # script write a value sprint-planning's status view reports as illegal.
    assert _module().ACTION_STATUSES == ("open", "in-progress", "done")


def test_empty_status_array_is_a_no_op(tmp_path):
    target = _write_action_fixture(tmp_path)
    proc = _run(
        ["update", "--file", str(target), "--epic", "1", "--date",
         "01-01-2026 09:00", "--set-action-status", "[]"]
    )
    assert proc.returncode == 0, proc.stderr
    assert _json(proc)["action_items_updated"] == 0
    assert target.read_text(encoding="utf-8") == ACTION_FIXTURE


def test_in_progress_item_transitions_to_done(tmp_path):
    # Every other success case starts from "open"; the middle of the lifecycle
    # has to work too.
    target = _write_action_fixture(tmp_path)
    proc = _run(
        ["update", "--file", str(target), "--epic", "2", "--set-action-status",
         '[{"id":"epic-2-retro-item-1-y","status":"done"}]']
    )
    assert proc.returncode == 0, proc.stderr
    assert _json(proc)["action_items_updated"] == 1
    assert _load(target)["action_items"][2]["status"] == "done"


def test_action_items_updated_is_always_reported(tmp_path):
    # Consumers read the counter unconditionally, so it must be present even when
    # the flag was not passed.
    target = _write_fixture(tmp_path)
    proc = _run(["update", "--file", str(target), "--epic", "1", "--set-retro-done"])
    assert proc.returncode == 0, proc.stderr
    assert _json(proc)["action_items_updated"] == 0


def test_post_write_status_mismatch_restores(tmp_path, monkeypatch, capsys):
    # The last line of defence: the written file is re-parsed and every targeted
    # item is checked. No CLI path can fake a mismatch, so the re-parse is
    # doctored directly.
    mod = _module()
    target = _write_action_fixture(tmp_path)
    real_load_yaml = mod._load_yaml
    calls = {"n": 0}

    def flaky(path):
        yaml, data = real_load_yaml(path)
        calls["n"] += 1
        if calls["n"] == 2:  # the post-write re-parse
            data["action_items"][0]["status"] = "open"
        return yaml, data

    monkeypatch.setattr(mod, "_load_yaml", flaky)
    args = mod.build_parser().parse_args(
        ["update", "--file", str(target), "--epic", "1", "--date", "01-01-2026 09:00",
         "--set-action-status", '[{"id":"epic-1-retro-item-1-x","status":"done"}]']
    )
    with pytest.raises(SystemExit) as excinfo:
        mod.cmd_update(args)
    assert excinfo.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False
    assert out["restored"] is True
    assert "after write" in out["error"]
    assert target.read_text(encoding="utf-8") == ACTION_FIXTURE


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

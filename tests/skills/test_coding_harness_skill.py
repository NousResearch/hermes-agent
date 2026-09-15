"""Tests for the coding-harness skill.

Covers the hardline frontmatter rules for a shipped skill plus the harness
state helper, including the invariant that a failed increment can never be
recorded as kept (which is what marks an increment complete in `status`).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "skills" / "software-development" / "coding-harness"
SCRIPT_PATH = SKILL_DIR / "scripts" / "harness_state.py"


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("coding_harness_state", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ns(**kwargs) -> argparse.Namespace:
    kwargs.setdefault("state", None)
    return argparse.Namespace(**kwargs)


@pytest.fixture()
def state_file(tmp_path, harness) -> Path:
    path = tmp_path / "state.json"
    harness.cmd_init(_ns(state=str(path), goal="all tests green", force=False))
    return path


# --- frontmatter ----------------------------------------------------------


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"


def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline <=60): {desc!r}"
    assert desc.endswith("."), f"description should end with a period: {desc!r}"


def test_author_credits_contributor_with_handle(frontmatter) -> None:
    author = frontmatter["author"]
    assert "Teddy Tennant" in author, f"author should credit the contributor: {author!r}"
    assert "@teddytennant" in author, f"author should carry the GitHub handle: {author!r}"


def test_prose_names_native_hermes_tools() -> None:
    """Shell utilities the agent already has wrapped must not headline the prose."""
    body = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    prose = re.sub(r"```.*?```", "", body, flags=re.DOTALL)
    assert not re.search(r"\bgreps?\b", prose), "use `search_files`, not grep, in prose"
    assert "`search_files`" in prose
    assert "`read_file`" in prose


def test_shipped_reference_files_exist() -> None:
    for name in ("execution-loop.md", "verification-protocol.md", "change-manifest.md"):
        assert (SKILL_DIR / "references" / name).is_file(), f"missing reference: {name}"


# --- harness_state helper -------------------------------------------------


def test_init_then_add_increment_assigns_ids(harness, state_file) -> None:
    harness.cmd_add_increment(
        _ns(state=str(state_file), summary="add jwt helpers", predict="unit tests pass", risk="session test")
    )
    harness.cmd_add_increment(
        _ns(state=str(state_file), summary="swap middleware", predict=None, risk=None)
    )
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert [i["change_id"] for i in state["increments"]] == ["ch_001", "ch_002"]
    assert state["increments"][0]["verification"]["status"] == "pending"


def test_verdict_is_derived_from_status(harness, state_file) -> None:
    harness.cmd_add_increment(_ns(state=str(state_file), summary="one", predict=None, risk=None))
    harness.cmd_record_verification(
        _ns(state=str(state_file), change_id="ch_001", status="pass", note="pytest -q: 14 passed", verdict=None)
    )
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["increments"][0]["verification"]["verdict"] == "keep"


@pytest.mark.parametrize("status", ["fail", "partial"])
def test_non_passing_verification_cannot_be_kept(harness, state_file, status) -> None:
    """Regression: `record-verification ch_001 fail --verdict keep` used to be accepted,
    which let `status` count a known-broken increment as complete."""
    harness.cmd_add_increment(_ns(state=str(state_file), summary="one", predict=None, risk=None))
    with pytest.raises(SystemExit) as excinfo:
        harness.cmd_record_verification(
            _ns(state=str(state_file), change_id="ch_001", status=status, note="", verdict="keep")
        )
    assert "not valid for status" in str(excinfo.value)

    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["increments"][0]["verification"]["status"] == "pending", "state must not be mutated"


def test_failed_increment_stays_pending_in_status(harness, state_file) -> None:
    harness.cmd_add_increment(_ns(state=str(state_file), summary="one", predict=None, risk=None))
    harness.cmd_add_increment(_ns(state=str(state_file), summary="two", predict=None, risk=None))
    harness.cmd_record_verification(
        _ns(state=str(state_file), change_id="ch_001", status="pass", note="green", verdict=None)
    )
    harness.cmd_record_verification(
        _ns(state=str(state_file), change_id="ch_002", status="fail", note="pytest -q: 1 failed", verdict=None)
    )
    out = harness._render(json.loads(state_file.read_text(encoding="utf-8")))
    assert "increments: 2 (1 kept)" in out
    assert "NEXT: resume at ch_002" in out


def test_hand_edited_state_cannot_smuggle_a_failure_past_pending(harness, state_file) -> None:
    """Defense in depth: `_render()` re-checks status, not just the verdict."""
    harness.cmd_add_increment(_ns(state=str(state_file), summary="one", predict=None, risk=None))
    state = json.loads(state_file.read_text(encoding="utf-8"))
    state["increments"][0]["verification"] = {"status": "fail", "note": "", "verdict": "keep"}
    out = harness._render(state)
    assert "increments: 1 (0 kept)" in out
    assert "NEXT: resume at ch_001" in out


def test_init_refuses_to_clobber_without_force(harness, state_file) -> None:
    with pytest.raises(SystemExit):
        harness.cmd_init(_ns(state=str(state_file), goal="different goal", force=False))
    harness.cmd_init(_ns(state=str(state_file), goal="different goal", force=True))
    assert json.loads(state_file.read_text(encoding="utf-8"))["goal"] == "different goal"


def test_record_verification_on_unknown_increment_errors(harness, state_file) -> None:
    with pytest.raises(SystemExit):
        harness.cmd_record_verification(
            _ns(state=str(state_file), change_id="ch_999", status="pass", note="", verdict=None)
        )


def test_self_test_entrypoint_passes(harness, capsys) -> None:
    harness.cmd_self_test(_ns())
    assert "self-test OK" in capsys.readouterr().out

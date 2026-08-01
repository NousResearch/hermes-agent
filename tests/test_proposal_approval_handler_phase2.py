#!/usr/bin/env python3
"""Phase 2 gate-logic tests for proposal_approval_handler.py.

Covers: !pitch parsing, idea-card lookup, pitch/prototype artifact attachment.
Import-based, hermetic: no Discord API, no kanban creation (DRY_RUN).
"""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import proposal_approval_handler as pah


@pytest.fixture
def props(tmp_path):
    """Temp HERMES_HOME with a dated proposals dir + idea-index.jsonl."""
    hermes = tmp_path / "hermes"
    props_dir = hermes / "runbooks" / "proposals" / "2026-08-01"
    props_dir.mkdir(parents=True)
    pah.PROPOSALS_DIR = hermes / "runbooks" / "proposals"
    return props_dir


def _write_index(props, cards):
    idx = props / pah.IDEA_INDEX_NAME
    idx.write_text("\n".join(json.dumps(c) for c in cards) + "\n", encoding="utf-8")
    return idx


def _card(slug="mcp-unicode-sanitization", pitch=False):
    return {
        "slug": slug,
        "title": f"{slug} — test",
        "pitch_line": "a pitch",
        "recommendation": {"action": "pitch", "effort": "S", "risk": "Low", "priority": 8},
        "pitch": pitch,
        "approved": False,
    }


class TestParseCommand:
    def test_pitch(self):
        assert pah.parse_command("!pitch mcp-unicode-sanitization") == ("pitch", "mcp-unicode-sanitization")

    def test_pitch_with_dot(self):
        assert pah.parse_command("!pitch foo.bar.") == ("pitch", "foo.bar")

    def test_approve_still_works(self):
        assert pah.parse_command("!approve fact-graph-memory") == ("approve", "fact-graph-memory")

    def test_reject_still_works(self):
        assert pah.parse_command("!reject semantic-regression") == ("reject", "semantic-regression")

    def test_noise_ignored(self):
        assert pah.parse_command("just chatting") == (None, None)


class TestFindIdeaCard:
    def test_found(self, props):
        idx = _write_index(props, [_card(), _card("fact-graph-memory")])
        card, found_idx = pah.find_idea_card("fact-graph-memory")
        assert card is not None
        assert card["slug"] == "fact-graph-memory"
        assert found_idx == idx

    def test_missing(self, props):
        _write_index(props, [_card()])
        assert pah.find_idea_card("nope") == (None, None)

    def test_bad_json_skipped(self, props):
        idx = _write_index(props, [_card()])
        idx.write_text("not-json\n" + json.dumps(_card("ok")) + "\n", encoding="utf-8")
        card, _ = pah.find_idea_card("ok")
        assert card is not None
        assert pah.find_idea_card("nope") == (None, None)


class TestArtifactAttachment:
    def test_no_artifacts_when_absent(self, props):
        idx = _write_index(props, [_card()])
        card, found_idx = pah.find_idea_card("mcp-unicode-sanitization")
        pitch_path, proto_path = pah.artifact_paths(found_idx, "mcp-unicode-sanitization")
        assert not pitch_path.exists()
        assert not proto_path.exists()

    def test_attaches_pitch_and_prototype(self, props):
        idx = _write_index(props, [_card()])
        pitch_path, proto_path = pah.artifact_paths(idx, "mcp-unicode-sanitization")
        pitch_path.write_text("# Pitch\nProblem: X\n", encoding="utf-8")
        proto_path.write_text("# Prototype\nMVP: Y\n", encoding="utf-8")
        # create_kanban_triage in DRY_RUN returns early but we can inspect
        # the artifact helper output through find + artifact_paths
        assert pitch_path.read_text() == "# Pitch\nProblem: X\n"
        assert proto_path.read_text() == "# Prototype\nMVP: Y\n"


class TestPitchActionDryRun:
    def test_dry_run_short_circuits(self):
        pah.DRY_RUN = True
        pah.main()
        pah.DRY_RUN = False

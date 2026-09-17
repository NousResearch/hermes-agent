"""Tests for the spec-driven-dev skill's deterministic gate script.

Stdlib + pytest only, no live network, no real `specify` CLI invocation --
per skills/AGENTS.md authoring standards.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "software-development"
    / "spec-driven-dev"
)
GATE_SCRIPT = SKILL_DIR / "scripts" / "spec_decision_gate.py"
TRANSLATE_SCRIPT = SKILL_DIR / "scripts" / "translate_speckit_skills.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return _load_module(GATE_SCRIPT, "spec_decision_gate")


@pytest.fixture(scope="module")
def translator():
    return _load_module(TRANSLATE_SCRIPT, "translate_speckit_skills")


class TestSpecDecisionGate:
    def test_evidence_answerable_resolves_silently(self, gate):
        result = gate.evaluate({"is_evidence_answerable": True})
        assert result["status"] == "resolved"
        outcomes = [m[0] for m in result["matches"]]
        assert outcomes == ["resolve_silently"]

    def test_blocking_and_not_evidence_answerable_goes_to_clarify_now(self, gate):
        result = gate.evaluate(
            {"is_evidence_answerable": False, "blocks_every_next_step": True}
        )
        assert result["status"] == "resolved"
        outcomes = [m[0] for m in result["matches"]]
        assert outcomes == ["clarify_now"]

    def test_non_blocking_owner_call_goes_to_decision_hud(self, gate):
        result = gate.evaluate(
            {"is_evidence_answerable": False, "blocks_every_next_step": False}
        )
        assert result["status"] == "resolved"
        outcomes = [m[0] for m in result["matches"]]
        assert outcomes == ["decision_hud"]

    def test_missing_required_answer_is_incomplete(self, gate):
        result = gate.evaluate({})
        assert result["status"] == "incomplete"
        assert result["open_questions"]

    def test_resolve_silently_and_clarify_now_are_mutually_exclusive(self, gate):
        # An evidence-answerable question never also routes to clarify_now
        # or decision_hud -- these three outcomes partition the answer space.
        silent = gate.evaluate({"is_evidence_answerable": True})
        clarify = gate.evaluate(
            {"is_evidence_answerable": False, "blocks_every_next_step": True}
        )
        hud = gate.evaluate(
            {"is_evidence_answerable": False, "blocks_every_next_step": False}
        )
        outcomes = {
            tuple(m[0] for m in r["matches"]) for r in (silent, clarify, hud)
        }
        assert len(outcomes) == 3  # no overlap between the three verdicts


class TestTranslateSpeckitSkills:
    def test_translate_frontmatter_maps_fields(self, translator):
        source = (
            "---\n"
            "name: speckit-specify\n"
            "description: Creates the feature specification from a description that is much longer than sixty characters for sure\n"
            "---\n"
            "# Body content\n"
        )
        translated = translator.translate_frontmatter(source, "speckit-specify")

        assert "name: speckit-specify" in translated
        assert "metadata:" in translated
        assert "hermes:" in translated
        assert "# Body content" in translated

        desc_line = [
            line for line in translated.splitlines() if line.startswith("description:")
        ][0]
        desc_value = desc_line.split("description:", 1)[1].strip().strip('"')
        assert len(desc_value) <= 60

    def test_translate_frontmatter_no_frontmatter_passthrough(self, translator):
        source = "# No frontmatter here\n"
        assert translator.translate_frontmatter(source, "x") == source

    def test_translate_project_missing_dir_raises(self, translator, tmp_path):
        with pytest.raises(FileNotFoundError):
            translator.translate_project(tmp_path, tmp_path / "out")

    def test_translate_project_writes_expected_files(self, translator, tmp_path):
        speckit_dir = tmp_path / ".claude" / "skills" / "speckit-plan"
        speckit_dir.mkdir(parents=True)
        (speckit_dir / "SKILL.md").write_text(
            "---\nname: speckit-plan\ndescription: Plan step.\n---\nBody\n"
        )

        out_dir = tmp_path / ".hermes" / "skills"
        written = translator.translate_project(tmp_path, out_dir)

        assert len(written) == 1
        dest = Path(written[0])
        assert dest.exists()
        assert dest.parent.name == "speckit-plan"
        assert "Body" in dest.read_text()


class TestEarsSchema:
    def test_schema_is_valid_json(self):
        schema_path = SKILL_DIR / "references" / "ears-schema.json"
        data = json.loads(schema_path.read_text())
        assert data["title"] == "EARS Requirement"
        assert "id" in data["required"]
        assert "acceptance_criteria" in data["required"]

    def test_sample_requirement_matches_required_fields(self):
        schema_path = SKILL_DIR / "references" / "ears-schema.json"
        schema = json.loads(schema_path.read_text())
        sample = {
            "id": "FR-001",
            "pattern": "event_driven",
            "trigger": "a user submits an empty form",
            "actor": "the system",
            "response": "display a validation error",
            "acceptance_criteria": [
                {
                    "given": "an empty form",
                    "when": "the user submits it",
                    "then": "a validation error is shown",
                }
            ],
        }
        for field in schema["required"]:
            assert field in sample

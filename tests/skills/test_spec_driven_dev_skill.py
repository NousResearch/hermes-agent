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


GENERATE_TESTS_SCRIPT = SKILL_DIR / "scripts" / "generate_property_tests.py"
MOCK_SERVER_SCRIPT = SKILL_DIR / "scripts" / "generate_mock_server.py"


@pytest.fixture(scope="module")
def property_gen():
    return _load_module(GENERATE_TESTS_SCRIPT, "generate_property_tests")


@pytest.fixture(scope="module")
def mock_server(request):
    return _load_module(MOCK_SERVER_SCRIPT, "generate_mock_server")


class TestGeneratePropertyTests:
    SAMPLE_REQ = {
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

    def test_load_requirements_reads_single_object_files(self, property_gen, tmp_path):
        (tmp_path / "fr001.json").write_text(json.dumps(self.SAMPLE_REQ))
        reqs = property_gen.load_requirements(tmp_path)
        assert len(reqs) == 1
        assert reqs[0]["id"] == "FR-001"

    def test_load_requirements_reads_list_files(self, property_gen, tmp_path):
        (tmp_path / "batch.json").write_text(json.dumps([self.SAMPLE_REQ, self.SAMPLE_REQ]))
        reqs = property_gen.load_requirements(tmp_path)
        assert len(reqs) == 2

    def test_generate_produces_scaffold_with_placeholder(self, property_gen):
        output = property_gen.generate([self.SAMPLE_REQ])
        assert "def test_fr_001" in output
        assert "st.nothing()" in output
        assert "FR-001" in output
        assert "hypothesis" in output

    def test_slug_converts_id_to_valid_identifier(self, property_gen):
        assert property_gen._slug("FR-001") == "fr_001"


class TestGenerateMockServer:
    def test_check_prism_available_reflects_npx_presence(self, mock_server, monkeypatch):
        monkeypatch.setattr(mock_server.shutil, "which", lambda name: None)
        assert mock_server.check_prism_available() is False

        monkeypatch.setattr(mock_server.shutil, "which", lambda name: "/usr/bin/npx")
        assert mock_server.check_prism_available() is True

    def test_run_mock_server_missing_contract_returns_error(self, mock_server, tmp_path):
        missing = tmp_path / "does-not-exist.yaml"
        assert mock_server.run_mock_server(missing, 4010) == 1

    def test_run_mock_server_no_npx_returns_error(self, mock_server, tmp_path, monkeypatch):
        contract = tmp_path / "contract.yaml"
        contract.write_text("openapi: 3.0.0\n")
        monkeypatch.setattr(mock_server, "check_prism_available", lambda: False)
        assert mock_server.run_mock_server(contract, 4010) == 1


class TestPhilosophicalPreambleTemplate:
    def test_template_exists_and_has_required_sections(self):
        template_path = SKILL_DIR / "templates" / "philosophical-preamble.md"
        content = template_path.read_text()
        assert "# Philosophical Preamble" in content
        assert "## Why This Project Exists" in content
        assert "## What Would Make This a Failure" in content


class TestContractTestingReference:
    def test_reference_documents_client_first_ordering(self):
        ref_path = SKILL_DIR / "references" / "contract-testing.md"
        content = ref_path.read_text()
        assert "Layer 1" in content
        assert "Layer 2" in content
        assert "Client-First" in content or "client-first" in content.lower()

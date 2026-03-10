"""Tests for the ouroboros RL environment and council evaluator.

Tests environment setup, prompt formatting, reward computation,
DPO pair accumulation, and council evaluator functionality.

Note: atroposlib is an RL-only dependency not always installed locally.
Tests that require it are skipped if unavailable. Tests for council_evaluator
import it directly (no environments/__init__.py trigger).
"""

import json
import os
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure repo root on path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Check if atroposlib is available (RL dependency)
try:
    import atroposlib
    HAS_ATROPOS = True
except ImportError:
    HAS_ATROPOS = False


def _import_council_evaluator():
    """Import council_evaluator directly, bypassing environments/__init__.py."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "council_evaluator",
        str(_repo_root / "environments" / "council_evaluator.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =========================================================================
# Council Evaluator tests (no atroposlib needed)
# =========================================================================


class TestCouncilEvaluator:
    """Tests for environments/council_evaluator.py."""

    @pytest.fixture(autouse=True)
    def mock_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")

    def test_init_default(self):
        mod = _import_council_evaluator()
        evaluator = mod.CouncilEvaluator()
        assert evaluator.model == "nousresearch/hermes-3-llama-3.1-70b"
        assert evaluator._api_key == "test-key-123"
        assert "openrouter" in evaluator._base_url

    def test_init_custom(self):
        mod = _import_council_evaluator()
        evaluator = mod.CouncilEvaluator(
            model="custom/model",
            api_key="custom-key",
            base_url="https://custom.api/v1",
        )
        assert evaluator.model == "custom/model"
        assert evaluator._api_key == "custom-key"

    def test_init_with_persona_subset(self):
        mod = _import_council_evaluator()
        evaluator = mod.CouncilEvaluator(personas=["skeptic", "arbiter"])
        assert "skeptic" in evaluator._personas
        assert "arbiter" in evaluator._personas
        assert len(evaluator._personas) == 2

    def test_normalized_reward(self):
        mod = _import_council_evaluator()
        from tools.council_personas import CouncilVerdict

        evaluator = mod.CouncilEvaluator()

        verdict_high = CouncilVerdict(
            question="q", responses={}, arbiter_synthesis="s",
            confidence_score=85, conflict_detected=False,
        )
        assert abs(evaluator.normalized_reward(verdict_high) - 0.85) < 0.01

        verdict_low = CouncilVerdict(
            question="q", responses={}, arbiter_synthesis="s",
            confidence_score=20, conflict_detected=True,
        )
        assert abs(evaluator.normalized_reward(verdict_low) - 0.20) < 0.01

    def test_normalized_reward_clamped(self):
        mod = _import_council_evaluator()
        from tools.council_personas import CouncilVerdict

        evaluator = mod.CouncilEvaluator()

        verdict = CouncilVerdict(
            question="q", responses={}, arbiter_synthesis="s",
            confidence_score=150, conflict_detected=False,
        )
        assert evaluator.normalized_reward(verdict) == 1.0

        verdict_neg = CouncilVerdict(
            question="q", responses={}, arbiter_synthesis="s",
            confidence_score=-10, conflict_detected=False,
        )
        assert evaluator.normalized_reward(verdict_neg) == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_mocked(self):
        mod = _import_council_evaluator()
        evaluator = mod.CouncilEvaluator()

        mock_response = (
            "Evaluation of content.\n"
            "- Good depth of analysis provided\n"
            "- Needs more supporting evidence\n"
            "CONFIDENCE: 0.72\n"
            "DISSENT: false\n"
        )

        with patch.object(evaluator, "_llm_call", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            verdict = await evaluator.evaluate(
                content="Some agent output to evaluate thoroughly",
                question="Was the research thorough?",
            )

        assert verdict.confidence_score > 0
        assert verdict.question is not None
        assert len(verdict.responses) > 0

    @pytest.mark.asyncio
    async def test_gate_mocked(self):
        mod = _import_council_evaluator()
        evaluator = mod.CouncilEvaluator()

        mock_response = (
            "Safety check passed.\n"
            "CONFIDENCE: 0.8\n"
            "DISSENT: false\n"
        )

        with patch.object(evaluator, "_llm_call", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            gate_result = await evaluator.gate(
                action="Deploy to staging",
                context="All tests pass",
            )

        assert "allowed" in gate_result
        assert "confidence" in gate_result
        assert "reasoning" in gate_result

    def test_parse_response(self):
        mod = _import_council_evaluator()

        raw = (
            "Analysis here.\n"
            "- Key point one about the topic\n"
            "CONFIDENCE: 0.65\n"
            "DISSENT: true\n"
            "See https://example.com for more.\n"
        )
        resp = mod.CouncilEvaluator._parse_response("skeptic", raw)
        assert resp.persona_name == "skeptic"
        assert abs(resp.confidence - 0.65) < 0.01
        assert resp.dissents is True
        assert len(resp.sources) >= 1

    def test_extract_dpo_pairs_from_responses(self):
        mod = _import_council_evaluator()
        from tools.council_personas import PersonaResponse

        responses = {
            "advocate": PersonaResponse(
                persona_name="advocate", content="Strong for",
                confidence=0.9, dissents=False,
            ),
            "skeptic": PersonaResponse(
                persona_name="skeptic", content="Weak against",
                confidence=0.3, dissents=True,
            ),
            "arbiter": PersonaResponse(
                persona_name="arbiter", content="Synthesis",
                confidence=0.7, dissents=False,
            ),
        }
        pairs = mod.CouncilEvaluator.extract_dpo_pairs_from_responses("q", responses)
        assert len(pairs) >= 1
        assert pairs[0]["source"] == "council_evaluation"


# =========================================================================
# Ouroboros Environment tests (require atroposlib)
# =========================================================================


@pytest.mark.skipif(not HAS_ATROPOS, reason="atroposlib not installed")
class TestOuroborosEnvFull:
    """Tests requiring full atroposlib installation."""

    def test_env_imports(self):
        from environments.ouroboros_env import OuroborosEnv
        assert OuroborosEnv.name == "ouroboros"


class TestOuroborosEnvData:
    """Tests for ouroboros env data that don't require atroposlib.

    Import the data directly from the module file.
    """

    @pytest.fixture(autouse=True)
    def _load_module(self):
        """Load ouroboros_env module data without triggering atroposlib imports."""
        import importlib.util

        # Read the file and extract just the data constants
        env_file = _repo_root / "environments" / "ouroboros_env.py"
        source = env_file.read_text(encoding="utf-8")

        # Extract RESEARCH_QUESTIONS and EVAL_QUESTIONS via exec
        # (they're defined before any atroposlib imports)
        namespace = {}
        # Execute up to the first non-data import
        lines = source.split("\n")
        data_lines = []
        in_data = False
        for line in lines:
            if line.startswith("RESEARCH_QUESTIONS"):
                in_data = True
            if in_data:
                data_lines.append(line)
            if in_data and line.startswith("]") and not line.strip().startswith('"'):
                # Check if this closes EVAL_QUESTIONS
                if "EVAL_QUESTIONS" in "\n".join(data_lines):
                    data_lines.append(line)
                    break

        # Simpler approach: just regex extract
        import re
        rq_match = re.search(
            r"RESEARCH_QUESTIONS\s*=\s*\[(.*?)\]\s*\n\s*EVAL_QUESTIONS",
            source,
            re.DOTALL,
        )
        eq_match = re.search(
            r"EVAL_QUESTIONS\s*=\s*\[(.*?)\]",
            source,
            re.DOTALL,
        )

        self.research_questions = []
        self.eval_questions = []

        if rq_match:
            try:
                self.research_questions = eval(f"[{rq_match.group(1)}]")
            except Exception:
                pass
        if eq_match:
            try:
                self.eval_questions = eval(f"[{eq_match.group(1)}]")
            except Exception:
                pass

    def test_research_questions_loaded(self):
        assert len(self.research_questions) >= 20

    def test_question_structure(self):
        for q in self.research_questions:
            assert "question" in q
            assert "category" in q
            assert len(q["question"]) > 20

    def test_question_categories(self):
        categories = set(q["category"] for q in self.research_questions)
        assert "technology" in categories
        assert "economics" in categories
        assert "science" in categories

    def test_eval_questions_loaded(self):
        assert len(self.eval_questions) >= 3

    def test_eval_question_structure(self):
        for q in self.eval_questions:
            assert "question" in q
            assert "category" in q


class TestOuroborosHelpers:
    """Test static helper methods from OuroborosEnv without importing the class."""

    def test_extract_final_response(self):
        """Test _extract_final_response logic."""
        # Replicate the static method logic here
        def _extract_final_response(messages):
            for msg in reversed(messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    content = msg["content"]
                    if len(content) > 50:
                        return content
            parts = []
            for msg in messages:
                if msg.get("role") == "assistant" and msg.get("content"):
                    parts.append(msg["content"])
            return "\n".join(parts)

        messages = [
            {"role": "system", "content": "You are a researcher"},
            {"role": "user", "content": "Research this question"},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "web_search"}}]},
            {"role": "tool", "content": "Search results..."},
            {"role": "assistant", "content": "Based on my research, here is a comprehensive analysis of the topic that covers multiple perspectives and provides evidence-backed conclusions."},
        ]

        result = _extract_final_response(messages)
        assert "comprehensive analysis" in result

    def test_extract_final_response_empty(self):
        def _extract_final_response(messages):
            for msg in reversed(messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    content = msg["content"]
                    if len(content) > 50:
                        return content
            parts = []
            for msg in messages:
                if msg.get("role") == "assistant" and msg.get("content"):
                    parts.append(msg["content"])
            return "\n".join(parts)

        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Question"},
        ]
        result = _extract_final_response(messages)
        assert result == ""

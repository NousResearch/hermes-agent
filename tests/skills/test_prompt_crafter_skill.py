"""Tests for optional-skills/research/prompt-crafter/scripts/prompt_crafter.py.

All analysis/templating logic is pure and stdlib-only, so the suite exercises
real function paths and the CLI JSON output without any network or mocking.
"""

import io
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "research"
    / "prompt-crafter"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import prompt_crafter  # noqa: E402


def _run_main(argv):
    """Drive main() with argv and return parsed JSON stdout."""
    buf = io.StringIO()
    with mock.patch("sys.argv", ["prompt_crafter"] + argv):
        with mock.patch("sys.stdout") as out:
            out.write = buf.write
            prompt_crafter.main()
    return json.loads(buf.getvalue())


class TestAnalyzePrompt:
    def test_strong_prompt_scores_high(self):
        # Hits role, context (>30 words), constraints, output format, goal,
        # tone, examples, and chain-of-thought cues.
        prompt = (
            "You are a senior engineer. Review the following Python code using "
            "a strict, step by step approach. For example, check security first. "
            "Return your answer in JSON. Keep a professional tone and avoid fluff."
        )
        result = prompt_crafter.analyze_prompt(prompt)
        assert result["quality_score"] == 100
        assert result["checks_passed"] == result["checks_total"]
        assert result["verdict"] == "Strong prompt"

    def test_weak_prompt_low_score_and_suggestions(self):
        result = prompt_crafter.analyze_prompt("hi")
        assert result["quality_score"] < 40
        assert result["verdict"] == "Weak prompt, should be rewritten"
        assert len(result["suggestions"]) > 0
        assert "word_count" in result and "details" in result

    def test_returns_expected_keys(self):
        result = prompt_crafter.analyze_prompt("You are a helpful assistant.")
        for key in (
            "word_count",
            "quality_score",
            "checks_passed",
            "checks_total",
            "details",
            "suggestions",
            "verdict",
        ):
            assert key in result
        # detail entries carry the dimension name + pass state
        assert {d["check"] for d in result["details"]} == {
            c["name"] for c in prompt_crafter.QUALITY_CHECKS
        }

    def test_no_turkish_in_verdict(self):
        for score, _prompt in [(10, "x"), (50, "a b c d e f g h i j k l m n o p q r s t u v w x y z "
                                              "more words to cross threshold for context and goal here"),
                               (75, "You are a helper. Provide output as JSON. Be concise and avoid extra text "
                                    "with a professional tone and for example show one sample step by step.")]:
            _ = score
        # exercise each verdict branch via crafted prompts
        weak = prompt_crafter.analyze_prompt("no")
        avg = prompt_crafter.analyze_prompt(
            "You are an assistant. Summarize the long article about the history of computing and "
            "its impact on society with examples and a clear professional tone and return the result "
            "in JSON format and avoid unnecessary detail and think step by step through the reasoning."
        )
        good = prompt_crafter.analyze_prompt(
            "You are a professional assistant. Review the following code and explain your reasoning "
            "step by step. For example show one case. Return output in JSON format. Be concise and "
            "avoid extra text while keeping a friendly tone and a clear measurable goal in mind please."
        )
        for r in (weak, avg, good):
            assert any(
                turk in r["verdict"]
                for turk in ("Zayif", "Ortalama", "Iyi", "yazilmali", "gelistir")
            ) is False


class TestGetTemplate:
    def test_list_all(self):
        result = prompt_crafter.get_template()
        ids = {t["id"] for t in result["available_templates"]}
        assert ids == set(prompt_crafter.TEMPLATES.keys())

    def test_single_template(self):
        result = prompt_crafter.get_template("code-review")
        assert result["template"] == "Code Review"
        assert "code" in result["content"]

    def test_unknown_template_errors(self):
        result = prompt_crafter.get_template("nope")
        assert "error" in result
        assert "nope" in result["error"]


class TestGenerateVariations:
    def test_adds_missing_dimensions(self):
        # Bare prompt: no role, no output format, no constraints -> 3 variations.
        vars_list = prompt_crafter.generate_variations("Explain photosynthesis")
        styles = {v["style"] for v in vars_list}
        assert "Add Role" in styles
        assert "Specify Format" in styles
        assert "Add Constraint" in styles

    def test_strong_prompt_single_variation(self):
        prompt = (
            "You are an expert. Explain photosynthesis. Provide your answer in JSON. "
            "Be concise and avoid fluff. Use a professional tone with an example and "
            "think step by step."
        )
        vars_list = prompt_crafter.generate_variations(prompt)
        assert len(vars_list) == 1
        assert vars_list[0]["style"] == "Original (strong)"
        assert vars_list[0]["prompt"] == prompt

    def test_no_turkish_in_variation_styles(self):
        vars_list = prompt_crafter.generate_variations("say hi")
        for v in vars_list:
            assert any(
                turk in v["style"]
                for turk in ("Orijinal", "Format Belirt", "guclu")
            ) is False


class TestCliJsonOutput:
    def test_analyze_cli(self):
        result = _run_main(["analyze", "You are a helpful assistant that formats output as JSON."])
        assert isinstance(result, dict)
        assert "quality_score" in result

    def test_templates_cli(self):
        result = _run_main(["templates"])
        assert "available_templates" in result

    def test_templates_named_cli(self):
        result = _run_main(["templates", "--name", "brainstorm"])
        assert result["template"] == "Brainstorm Ideas"

    def test_variations_cli(self):
        result = _run_main(["variations", "Explain gravity"])
        assert "original" in result
        assert "variations" in result
        assert isinstance(result["variations"], list)

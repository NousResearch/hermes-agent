"""
Tests for eval_harness module (P2-2).

Covers: GoldenTask loading, EvalRun serialisation, EvalDelta comparison,
and the judge prompt builder.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from hermes_cli.eval_harness import (
    GoldenTask,
    EvalResult,
    EvalRun,
    EvalDelta,
    compare_runs,
    build_judge_prompt,
    run_eval,
)


# ---------------------------------------------------------------------------
# GoldenTask
# ---------------------------------------------------------------------------


class TestGoldenTask:
    def test_load_set_from_yaml(self):
        yaml_content = """
tasks:
  - id: test-001
    domain: code
    description: "Review a function"
    expected_behaviors:
      - "Find the bug"
      - "Suggest a fix"
    rubric:
      pass_if: "Both behaviors present"
      fail_if: "Misses either one"
    weight: 0.8
  - id: test-002
    domain: research
    description: "Research a topic"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            tasks = GoldenTask.load_set(path)
            assert len(tasks) == 2
            assert tasks[0].id == "test-001"
            assert tasks[0].domain == "code"
            assert len(tasks[0].expected_behaviors) == 2
            assert tasks[0].weight == 0.8
            assert tasks[1].id == "test-002"
            assert tasks[1].domain == "research"
        finally:
            os.unlink(path)

    def test_defaults(self):
        yaml_content = """
tasks:
  - id: minimal
    description: "Just a description"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            tasks = GoldenTask.load_set(path)
            assert len(tasks) == 1
            assert tasks[0].domain == "general"
            assert tasks[0].weight == 1.0
            assert tasks[0].expected_behaviors == []
            assert tasks[0].rubric == {}
        finally:
            os.unlink(path)

    def test_content_hash_is_stable(self):
        task = GoldenTask(
            id="test", domain="code", description="desc",
            expected_behaviors=["a", "b"],
        )
        h1 = task.content_hash
        h2 = task.content_hash
        assert h1 == h2
        assert len(h1) == 12

    def test_content_hash_changes_with_content(self):
        task1 = GoldenTask(id="test", domain="code", description="desc A")
        task2 = GoldenTask(id="test", domain="code", description="desc B")
        assert task1.content_hash != task2.content_hash

    def test_load_real_golden_set(self):
        """Verify the shipped golden task set loads correctly."""
        path = os.path.expanduser(
            "~/.hermes/kensei/eval/golden-tasks-code-v0.1.0.yaml"
        )
        if not os.path.exists(path):
            pytest.skip("Golden task set not found at " + path)
        tasks = GoldenTask.load_set(path)
        assert len(tasks) >= 5
        ids = {t.id for t in tasks}
        assert "code-review-001" in ids
        assert all(t.content_hash for t in tasks)


# ---------------------------------------------------------------------------
# EvalResult / EvalRun
# ---------------------------------------------------------------------------


class TestEvalRun:
    def test_pass_rate_zero_when_no_tasks(self):
        run = EvalRun(
            run_id="r1", timestamp="t", profile="p",
            judge_profile="j", task_count=0, passed_count=0,
        )
        assert run.pass_rate == 0.0

    def test_pass_rate_half(self):
        run = EvalRun(
            run_id="r1", timestamp="t", profile="p",
            judge_profile="j", task_count=4, passed_count=2,
        )
        assert run.pass_rate == 0.5

    def test_weighted_score(self):
        results = [
            EvalResult(task_id="a", domain="c", passed=True, score=0.8),
            EvalResult(task_id="b", domain="c", passed=True, score=1.0),
        ]
        run = EvalRun(
            run_id="r1", timestamp="t", profile="p",
            judge_profile="j", task_count=2, passed_count=2,
            results=results,
        )
        assert run.weighted_score == 0.9

    def test_save_and_load_roundtrip(self):
        results = [
            EvalResult(
                task_id="t1", domain="code", passed=True, score=0.9,
                judge_notes="Good", elapsed_seconds=1.5,
            ),
            EvalResult(
                task_id="t2", domain="code", passed=False, score=0.3,
                judge_notes="Missed injection", error="timeout",
            ),
        ]
        run = EvalRun(
            run_id="r-test", timestamp="2026-06-08T10:00:00Z",
            profile="test-profile", judge_profile="judge",
            task_count=2, passed_count=1, results=results,
            metadata={"version": "v0.1.0"},
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            run.save(path)
            loaded = EvalRun.load(path)
            assert loaded.run_id == "r-test"
            assert loaded.pass_rate == 0.5
            assert len(loaded.results) == 2
            assert loaded.results[0].task_id == "t1"
            assert loaded.results[0].passed is True
            assert loaded.results[1].passed is False
            assert loaded.metadata["version"] == "v0.1.0"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# EvalDelta / compare_runs
# ---------------------------------------------------------------------------


class TestCompareRuns:
    def make_run(self, run_id: str, results: list[EvalResult]) -> EvalRun:
        passed = sum(1 for r in results if r.passed)
        return EvalRun(
            run_id=run_id, timestamp="t", profile="p",
            judge_profile="j", task_count=len(results),
            passed_count=passed, results=results,
        )

    def test_no_regressions_recommends_keep(self):
        baseline = self.make_run("b", [
            EvalResult(task_id="a", domain="c", passed=True, score=0.8),
        ])
        candidate = self.make_run("c", [
            EvalResult(task_id="a", domain="c", passed=True, score=0.9),
        ])
        delta = compare_runs(baseline, candidate)
        assert delta.recommendation == "keep"

    def test_regression_recommends_rollback(self):
        baseline = self.make_run("b", [
            EvalResult(task_id="a", domain="c", passed=True, score=0.8),
        ])
        candidate = self.make_run("c", [
            EvalResult(task_id="a", domain="c", passed=False, score=0.3),
        ])
        delta = compare_runs(baseline, candidate)
        assert delta.recommendation == "rollback"
        assert "a" in delta.regressions

    def test_improvement_recommends_keep(self):
        baseline = self.make_run("b", [
            EvalResult(task_id="a", domain="c", passed=False, score=0.3),
        ])
        candidate = self.make_run("c", [
            EvalResult(task_id="a", domain="c", passed=True, score=0.9),
        ])
        delta = compare_runs(baseline, candidate)
        assert delta.recommendation == "keep"
        assert "a" in delta.improvements

    def test_no_change_recommends_review(self):
        baseline = self.make_run("b", [
            EvalResult(task_id="a", domain="c", passed=False, score=0.5),
        ])
        candidate = self.make_run("c", [
            EvalResult(task_id="a", domain="c", passed=False, score=0.5),
        ])
        delta = compare_runs(baseline, candidate)
        assert delta.recommendation == "review"

    def test_mixed_with_regression_recommends_rollback(self):
        baseline = self.make_run("b", [
            EvalResult(task_id="a", domain="c", passed=False, score=0.3),
            EvalResult(task_id="b", domain="c", passed=True, score=0.8),
        ])
        candidate = self.make_run("c", [
            EvalResult(task_id="a", domain="c", passed=True, score=0.9),
            EvalResult(task_id="b", domain="c", passed=False, score=0.4),
        ])
        delta = compare_runs(baseline, candidate)
        # Regression on task b should trigger rollback
        assert delta.recommendation == "rollback"
        assert "b" in delta.regressions
        assert "a" in delta.improvements


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------


class TestJudgePrompt:
    def test_includes_all_sections(self):
        task = GoldenTask(
            id="test", domain="code",
            description="Review this code",
            expected_behaviors=["Find bugs", "Suggest fixes"],
            rubric={"pass_if": "Found bugs", "fail_if": "Missed bugs"},
        )
        prompt = build_judge_prompt(task, "some output")
        assert "test" in prompt
        assert "Review this code" in prompt
        assert "Find bugs" in prompt
        assert "Found bugs" in prompt
        assert "Missed bugs" in prompt
        assert "some output" in prompt

    def test_truncates_long_output(self):
        task = GoldenTask(
            id="test", domain="code", description="d",
        )
        long_output = "x" * 10000
        prompt = build_judge_prompt(task, long_output)
        # Should not include the full 10k chars
        assert len(prompt) < 9000


# ---------------------------------------------------------------------------
# run_eval (dry_run)
# ---------------------------------------------------------------------------


class TestRunEval:
    def test_dry_run_returns_placeholder_results(self):
        tasks = [
            GoldenTask(id="t1", domain="code", description="task 1"),
            GoldenTask(id="t2", domain="code", description="task 2"),
        ]
        result = run_eval(tasks, profile="test", dry_run=True)
        assert result.task_count == 2
        assert result.passed_count == 2
        assert result.pass_rate == 1.0
        assert all("dry_run" in r.judge_notes for r in result.results)

    def test_dry_run_persists_saveable(self):
        tasks = [GoldenTask(id="t1", domain="code", description="task")]
        result = run_eval(tasks, profile="test", dry_run=True)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            result.save(path)
            loaded = EvalRun.load(path)
            assert loaded.run_id == result.run_id
            assert loaded.pass_rate == 1.0
        finally:
            os.unlink(path)

    def test_live_run_calls_real_llm_and_handles_failure(self):
        """Live eval attempts real LLM call, gracefully handles unavailability."""
        tasks = [GoldenTask(id="t1", domain="code", description="task")]
        # Without API keys in test env, _execute_golden_task raises → caught
        result = run_eval(tasks, profile="test", dry_run=False)
        assert result.task_count == 1
        # Task should have failed gracefully with an error message
        assert not result.results[0].passed
        assert result.results[0].error is not None
        assert result.results[0].score == 0.0

    def test_keyword_heuristic_fallback(self):
        """_keyword_heuristic_judge matches expected behaviors in output."""
        from hermes_cli.eval_harness import _keyword_heuristic_judge
        task = GoldenTask(
            id="t1", domain="code", description="Find SQL injection",
            expected_behaviors=[
                "Identifies the SQL injection vulnerability",
                "Suggests parameterised queries",
                "Does NOT suggest regex-based sanitisation",
            ],
        )
        good_output = (
            "I found a SQL injection vulnerability on line 42. "
            "I recommend using parameterised queries with prepared statements. "
            "Do not use regex sanitisation — it's brittle."
        )
        result = _keyword_heuristic_judge(task, good_output)
        assert result["passed"] is True
        assert result["score"] >= 0.5

        bad_output = "Use regex to sanitise the input: re.sub(r'[;\\'\\\"]', '', user_input)"
        result = _keyword_heuristic_judge(task, bad_output)
        # Should score low — regex sanitisation is the anti-pattern
        assert result["score"] <= 0.5

    def test_judge_output_short_circuits_placeholder(self):
        """_judge_output returns failed/0.0 for placeholder input."""
        from hermes_cli.eval_harness import _judge_output
        task = GoldenTask(id="t1", domain="code", description="task")
        result = _judge_output(task, "[EVAL_PLACEHOLDER] not executed", "test")
        assert result["passed"] is False
        assert result["score"] == 0.0
        assert "placeholder" in result["notes"].lower()


class TestProfileConfigResolution:
    """_get_profile_config must return a string model (not a dict) for both
    the flat and nested model: forms, plus base_url for custom providers."""

    def _write_profile(self, tmp_path, monkeypatch, name, body):
        monkeypatch.setenv("HOME", str(tmp_path))
        d = tmp_path / ".hermes" / "profiles" / name
        d.mkdir(parents=True)
        (d / "config.yaml").write_text(body)

    def test_flat_model_string(self, tmp_path, monkeypatch):
        from hermes_cli.eval_harness import _get_profile_config
        self._write_profile(
            tmp_path, monkeypatch, "flat",
            "provider: ollama-cloud\nmodel: deepseek-v4-flash\n",
        )
        cfg = _get_profile_config({}, "flat")
        assert cfg["model"] == "deepseek-v4-flash"
        assert isinstance(cfg["model"], str)
        assert cfg["provider"] == "ollama-cloud"

    def test_nested_model_dict(self, tmp_path, monkeypatch):
        from hermes_cli.eval_harness import _get_profile_config
        self._write_profile(
            tmp_path, monkeypatch, "nested",
            "model:\n  default: moonshotai/Kimi-K2.6\n"
            "  provider: custom:CommandCode\n"
            "  base_url: https://api.commandcode.ai/provider/v1\n",
        )
        cfg = _get_profile_config({}, "nested")
        assert cfg["model"] == "moonshotai/Kimi-K2.6"
        assert isinstance(cfg["model"], str)
        assert cfg["provider"] == "custom:CommandCode"
        assert cfg["base_url"] == "https://api.commandcode.ai/provider/v1"


class TestExecuteBaseUrlRouting:
    """_execute_golden_task must only forward base_url for custom: providers;
    a known provider with an explicit base_url 401s (call_llm falls back to
    OPENAI_API_KEY instead of the provider's credential pool)."""

    def _run(self, monkeypatch, provider, base_url):
        import hermes_cli.eval_harness as eh
        monkeypatch.setattr(
            eh, "_get_profile_config",
            lambda cfg, profile: {
                "provider": provider, "model": "m", "base_url": base_url, "timeout": 5,
            },
        )
        captured = {}

        class _Resp:
            class choices:
                pass
        def _fake_call_llm(**kw):
            captured.update(kw)
            r = type("R", (), {})()
            msg = type("M", (), {"content": "out"})()
            ch = type("C", (), {"message": msg})()
            r.choices = [ch]
            return r
        monkeypatch.setattr(eh, "_get_profile_config",
                            lambda cfg, profile: {"provider": provider, "model": "m",
                                                  "base_url": base_url, "timeout": 5})
        import agent.auxiliary_client as ac
        monkeypatch.setattr(ac, "call_llm", _fake_call_llm)
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})
        eh._execute_golden_task(GoldenTask(id="t", domain="code", description="d"), "p")
        return captured

    def test_known_provider_omits_base_url(self, monkeypatch):
        cap = self._run(monkeypatch, "opencode-go", "https://opencode.ai/zen/go/v1")
        assert cap.get("base_url") is None

    def test_custom_provider_forwards_base_url(self, monkeypatch):
        cap = self._run(monkeypatch, "custom:CommandCode", "https://api.commandcode.ai/provider/v1")
        assert cap.get("base_url") == "https://api.commandcode.ai/provider/v1"

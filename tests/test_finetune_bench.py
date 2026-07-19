"""
Unit tests for the finetune evaluation benchmark (FinetuneBenchEnv).

All tests are hermetic: scoring, aggregation, and verdict logic are exercised
as pure functions with synthetic case dicts and fake agent-loop results.
No network, no Docker, no real bench runs.
"""

import json
import sys
import types
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

# Add the bench dir (module under test) and the skill scripts dir (its
# `common` dependency) to the path, mirroring tests/test_finetune.py.
_skill_dir = Path(__file__).resolve().parent.parent / "optional-skills" / "mlops" / "finetune"
_bench_dir = str(_skill_dir / "bench")
_scripts_dir = str(_skill_dir / "scripts")
for _p in (_bench_dir, _scripts_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import finetune_bench_env as fbe  # noqa: E402


# ============================================================================
# Helpers
# ============================================================================

OK_TOOL = json.dumps({"exit_code": 0, "output": "file1\nfile2"})
ERR_TOOL = json.dumps({"exit_code": 127, "output": "bash: xyz: command not found"})


def make_env(tmp_path=None, **overrides):
    cfg = fbe.FinetuneBenchConfig(**overrides)
    env = fbe.FinetuneBenchEnv(cfg)
    if tmp_path is not None:
        # Keep scratch out of the real /tmp/finetune-bench root.
        env.run_base = Path(tmp_path) / "run-test"
    return env


def tc_msg(name="terminal", args='{"command": "ls"}'):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"function": {"name": name, "arguments": args}}],
    }


def tool_msg(content):
    return {"role": "tool", "content": content}


def asst(text):
    return {"role": "assistant", "content": text}


class SpyCtx:
    """VerifyContext stand-in that records commands instead of running them."""

    def __init__(self, default=None):
        self.calls = []
        self.default = default or {"exit_code": 0, "output": ""}

    def terminal(self, command, timeout=180):
        self.calls.append(command)
        return dict(self.default)

    def cleanup(self):
        pass


def tier2_output_case(case_id="te-x", expected_value=""):
    return {
        "id": case_id,
        "tier": 2,
        "category": "bash_correctness",
        "prompt": "do something",
        "expected": {"tool_name": "terminal", "should_call_tool": True},
        "verification": {"method": "output_match", "expected_value": expected_value},
    }


def tier3_output_case(case_id="e2e-x", expected_value=""):
    case = tier2_output_case(case_id, expected_value)
    case["tier"] = 3
    case["category"] = "research_and_summarize"
    case["expected"] = {"should_call_tool": True}
    return case


def case_result(**kw):
    defaults = dict(case_id="c", tier=1, category="correct_tool_simple", tags=[])
    defaults.update(kw)
    return fbe.CaseResult(**defaults)


# ============================================================================
# Finding 1 — empty expected_value is a MALFORMED case, never a vacuous pass
# ============================================================================

class TestEmptyExpectedValue:
    def test_successful_tool_output_does_not_pass_empty_assertion(self, tmp_path):
        """A case asserting nothing must not award execution credit for
        arbitrary successful output ('echo ok' must never score)."""
        env = make_env(tmp_path)
        msgs = [tc_msg(), tool_msg(OK_TOOL), asst("done")]
        res = env.compute_reward(tier2_output_case(), msgs, 2, 0, SpyCtx())
        assert res.tool_args_valid is False
        assert res.reward == pytest.approx(0.4)  # selection only

    def test_empty_assertion_recorded_as_malformed(self, tmp_path):
        env = make_env(tmp_path)
        msgs = [tc_msg(), tool_msg(OK_TOOL)]
        env.compute_reward(tier2_output_case("te-vac"), msgs, 1, 0, SpyCtx())
        assert "te-vac" in env.malformed_case_ids
        assert env._aggregate_metrics()["malformed_cases"] == 1

    def test_malformed_count_zero_for_real_assertions(self, tmp_path):
        env = make_env(tmp_path)
        case = tier2_output_case("te-ok", expected_value="file1")
        env.compute_reward(case, [tc_msg(), tool_msg(OK_TOOL)], 1, 0, SpyCtx())
        assert env.malformed_case_ids == set()
        assert env._aggregate_metrics()["malformed_cases"] == 0

    def test_garbage_tool_output_fails(self, tmp_path):
        env = make_env(tmp_path)
        msgs = [tc_msg(), tool_msg(ERR_TOOL)]
        res = env.compute_reward(tier2_output_case(), msgs, 2, 1, SpyCtx())
        assert res.tool_args_valid is False
        # Selection was still correct, but execution must not auto-pass.
        assert res.tool_selection_correct is True
        assert res.reward == pytest.approx(0.4)

    def test_no_tool_result_fails(self, tmp_path):
        env = make_env(tmp_path)
        res = env.compute_reward(
            tier2_output_case(), [asst("I refuse to run anything")], 1, 0, SpyCtx()
        )
        assert res.tool_args_valid is False

    def test_empty_tool_output_fails(self, tmp_path):
        env = make_env(tmp_path)
        msgs = [tc_msg(), tool_msg("   ")]
        res = env.compute_reward(tier2_output_case(), msgs, 2, 0, SpyCtx())
        assert res.tool_args_valid is False

    @pytest.mark.parametrize("bad_output", [
        "Error: something broke",
        "Traceback (most recent call last):\n  File ...",
        "sh: 1: frobnicate: command not found",
        json.dumps({"exit_code": 1, "output": "looks fine but exit 1"}),
        "process finished with exit code 2",
    ])
    def test_error_heuristics(self, tmp_path, bad_output):
        env = make_env(tmp_path)
        assert env._tool_output_ok(bad_output) is False

    def test_plain_text_success_ok(self, tmp_path):
        env = make_env(tmp_path)
        assert env._tool_output_ok("total 4\n-rw-r--r-- 1 u u 12 app.py") is True
        assert env._tool_output_ok(json.dumps({"exit_code": 0, "output": "hi"})) is True


class TestShortExpectedValueMatching:
    """te-001-style numeric expectations must not match digits embedded in
    timestamps/sizes, and must target the final answer / last tool result."""

    def test_embedded_digit_does_not_match(self, tmp_path):
        env = make_env(tmp_path)
        v = {"expected_value": "7"}
        msgs = [
            tc_msg(),
            tool_msg(json.dumps({
                "exit_code": 0,
                "output": "-rw-r--r-- 1 u u 4096 2024-07-12 10:37 app.py\ntotal 17",
            })),
        ]
        assert env._verify_output(msgs, v) is False

    def test_word_boundary_match_in_final_answer(self, tmp_path):
        env = make_env(tmp_path)
        v = {"expected_value": "7"}
        msgs = [tc_msg(), tool_msg(OK_TOOL), asst("The total is 7 lines.")]
        assert env._verify_output(msgs, v) is True

    def test_match_in_last_tool_result(self, tmp_path):
        env = make_env(tmp_path)
        v = {"expected_value": "7"}
        msgs = [tc_msg(), tool_msg(json.dumps({"exit_code": 0, "output": "7"}))]
        assert env._verify_output(msgs, v) is True

    def test_decimal_component_does_not_match(self, tmp_path):
        env = make_env(tmp_path)
        v = {"expected_value": "7"}
        msgs = [asst("Python 3.7 is installed")]
        assert env._verify_output(msgs, v) is False

    def test_earlier_tool_message_is_not_scanned(self, tmp_path):
        env = make_env(tmp_path)
        v = {"expected_value": "7"}
        msgs = [
            tc_msg(),
            tool_msg(json.dumps({"exit_code": 0, "output": "7"})),  # earlier
            tc_msg(),
            tool_msg(json.dumps({"exit_code": 0, "output": "nothing here"})),
        ]
        # Only the FINAL answer / LAST successful tool result count.
        assert env._verify_output(msgs, v) is False

    def test_long_expected_value_substring(self, tmp_path):
        env = make_env(tmp_path)
        v = {"expected_value": "Hello, World!"}
        msgs = [tc_msg(), tool_msg('{"exit_code": 0, "output": "Hello, World!"}')]
        assert env._verify_output(msgs, v) is True
        assert env._verify_output([tool_msg(ERR_TOOL)], v) is False


# ============================================================================
# Finding 2 — skill_invocation cases (should_call_tool: true, no tool_name)
# ============================================================================

class TestSkillInvocationScoring:
    CASE = {
        "id": "ts-100",
        "tier": 1,
        "category": "skill_invocation",
        "prompt": "Help me fine-tune my model",
        "expected": {"should_call_tool": True},
    }

    def test_any_tool_call_scores_correct(self, tmp_path):
        env = make_env(tmp_path)
        res = env.compute_reward(self.CASE, [tc_msg(name="skill")], 1, 0, SpyCtx())
        assert res.tool_selection_correct is True
        assert res.reward == 1.0

    def test_no_tool_call_scores_incorrect(self, tmp_path):
        env = make_env(tmp_path)
        res = env.compute_reward(self.CASE, [asst("Sure, here's how...")], 1, 0, SpyCtx())
        assert res.tool_selection_correct is False
        assert res.reward == 0.0


# ============================================================================
# Finding 3 — tier-3 output_match (empty checks) scores via transcript
# ============================================================================

class TestTier3OutputMatch:
    def test_matching_value_completes(self, tmp_path):
        env = make_env(tmp_path)
        case = tier3_output_case(expected_value="Django")
        msgs = [tc_msg(args='{"command": "cat frameworks.csv"}'),
                tool_msg('{"exit_code": 0, "output": "Django 82k stars"}'),
                asst("Here is the comparison table: Django | Flask | FastAPI")]
        res = env.compute_reward(case, msgs, 2, 0, SpyCtx())
        assert res.task_completed is True
        assert res.tool_args_valid is True
        assert res.reward == pytest.approx(1.0)

    def test_no_tool_use_and_no_match_fails(self, tmp_path):
        env = make_env(tmp_path)
        case = tier3_output_case(expected_value="Django")
        res = env.compute_reward(case, [asst("no idea")], 1, 0, SpyCtx())
        assert res.task_completed is False

    def test_expected_value_checked_against_final_answer(self, tmp_path):
        env = make_env(tmp_path)
        case = tier3_output_case(expected_value="Liskov")
        msgs = [tc_msg(), tool_msg("SOLID overview..."),
                asst("S: single resp... L: Liskov substitution principle ...")]
        assert env.compute_reward(case, msgs, 2, 0, SpyCtx()).task_completed is True

        msgs_bad = [tc_msg(), tool_msg("SOLID overview..."),
                    asst("SOLID means five principles.")]
        env.results.clear()
        assert env.compute_reward(case, msgs_bad, 2, 0, SpyCtx()).task_completed is False

    def test_functional_test_with_empty_checks_is_malformed(self, tmp_path):
        """A functional_test without checks AND without an expected value
        asserts nothing — it must be malformed, not a vacuous pass."""
        env = make_env(tmp_path)
        case = tier3_output_case()
        case["verification"] = {"method": "functional_test", "test_commands": [], "checks": []}
        ctx = SpyCtx()
        res = env.compute_reward(case, [tc_msg(), tool_msg(OK_TOOL)], 1, 0, ctx)
        assert res.task_completed is False
        assert ctx.calls == []  # nothing to execute
        assert case["id"] in env.malformed_case_ids

    def test_match_scope_final_answer_ignores_tool_output(self, tmp_path):
        env = make_env(tmp_path)
        case = tier3_output_case(expected_value="Tornado")
        case["verification"]["match_scope"] = "final_answer"
        # Value visible in a tool result (e.g. `cat` of the data file) but
        # never stated in the model's conclusion → no credit.
        msgs = [tc_msg(), tool_msg('{"exit_code": 0, "output": "Tornado,91500"}'),
                asst("I looked at the file.")]
        assert env.compute_reward(case, msgs, 2, 0, SpyCtx()).task_completed is False
        env.results.clear()
        msgs_ok = msgs[:-1] + [asst("The top framework is Tornado.")]
        assert env.compute_reward(case, msgs_ok, 2, 0, SpyCtx()).task_completed is True


# ============================================================================
# Finding 4 — functional verification runs exactly once
# ============================================================================

class TestSingleFunctionalExecution:
    def test_test_commands_run_once_for_both_metrics(self, tmp_path):
        env = make_env(tmp_path)
        case = {
            "id": "e2e-x",
            "tier": 3,
            "category": "workflow_automation",
            "prompt": "make a backup script",
            "expected": {"tool_name": "terminal", "should_call_tool": True},
            "verification": {
                "method": "functional_test",
                "test_commands": ["mkdir -p /tmp/finetune-bench/x && echo Hello"],
                "checks": [
                    {"type": "output_contains", "command_index": 0, "expected": ["Hello"]},
                ],
            },
        }
        ctx = SpyCtx(default={"exit_code": 0, "output": "Hello"})
        res = env.compute_reward(case, [tc_msg(), tool_msg(OK_TOOL)], 2, 0, ctx)
        # One test command → exactly one execution, shared by both metrics.
        assert len(ctx.calls) == 1
        assert res.tool_args_valid is True
        assert res.task_completed is True
        assert res.tool_args_valid == res.task_completed


# ============================================================================
# expect_failure — error-recovery cases must reward FAILING trajectories
# ============================================================================

def expect_failure_case(case_id="te-ef", regex="(permission|denied|read-?only)"):
    return {
        "id": case_id,
        "tier": 2,
        "category": "error_recovery",
        "prompt": "try something impossible and explain the failure",
        "expected": {"tool_name": "terminal", "should_call_tool": True},
        "verification": {
            "method": "output_match",
            "expect_failure": True,
            "expected_regex": regex,
        },
    }


class TestExpectFailure:
    def test_perfect_failing_trajectory_scores_full(self, tmp_path):
        """The correct behavior IS a failing command plus an explanation —
        it must earn full execution credit, not 0.4."""
        env = make_env(tmp_path)
        msgs = [
            tc_msg(args='{"command": "cat /etc/shadow"}'),
            tool_msg(json.dumps({"exit_code": 1, "output": "cat: /etc/shadow: Permission denied"})),
            asst("The read failed: permission denied — the file is root-only."),
        ]
        res = env.compute_reward(expect_failure_case(), msgs, 2, 1, SpyCtx())
        assert res.tool_args_valid is True
        assert res.reward == pytest.approx(1.0)

    def test_padding_with_successful_command_scores_zero_execution(self, tmp_path):
        """The old inverted incentive: an irrelevant successful command must
        NOT outscore the correct failing trajectory."""
        env = make_env(tmp_path)
        msgs = [
            tc_msg(args='{"command": "echo ok"}'),
            tool_msg(json.dumps({"exit_code": 0, "output": "ok"})),
            asst("Done! Everything went fine, no permission problems."),
        ]
        res = env.compute_reward(expect_failure_case(), msgs, 2, 0, SpyCtx())
        assert res.tool_args_valid is False
        assert res.reward == pytest.approx(0.4)  # selection only

    def test_failure_without_acknowledgment_fails(self, tmp_path):
        env = make_env(tmp_path)
        msgs = [
            tc_msg(),
            tool_msg(json.dumps({"exit_code": 1, "output": "Permission denied"})),
            asst("All done."),  # never explains the failure
        ]
        res = env.compute_reward(expect_failure_case(), msgs, 2, 1, SpyCtx())
        assert res.tool_args_valid is False

    def test_acknowledgment_match_is_case_insensitive(self, tmp_path):
        env = make_env(tmp_path)
        msgs = [
            tc_msg(),
            tool_msg(json.dumps({"exit_code": 1, "output": "boom"})),
            asst("It failed because the filesystem is READ-ONLY."),
        ]
        assert env._verify_output(msgs, {"expect_failure": True,
                                         "expected_regex": "read-?only"}) is True

    def test_fail_then_recover_trajectory_passes(self, tmp_path):
        """Cases like 'run it, fix it, rerun' have a failed call followed by
        a successful one — still a valid expect_failure trajectory."""
        env = make_env(tmp_path)
        msgs = [
            tc_msg(args='{"command": "python3 broken.py"}'),
            tool_msg(json.dumps({"exit_code": 1, "output": "SyntaxError: invalid syntax"})),
            tc_msg(args='{"command": "python3 broken.py"}'),
            tool_msg(json.dumps({"exit_code": 0, "output": "5"})),
            asst("broken.py was missing a colon — a syntax error. Fixed and reran: prints 5."),
        ]
        res = env.compute_reward(expect_failure_case(regex="(syntax|colon|fix)"),
                                 msgs, 3, 1, SpyCtx())
        assert res.tool_args_valid is True


# ============================================================================
# Functional check hardening — file_exists quoting and command_index bounds
# ============================================================================

class TestFunctionalCheckHardening:
    def test_file_exists_works_with_zero_test_commands(self, tmp_path):
        """A checks-only case (no test_commands) must be able to pass."""
        env = make_env(tmp_path)
        verification = {
            "method": "functional_test",
            "test_commands": [],
            "checks": [{"type": "file_exists", "path": "/tmp/finetune-bench/x/app.py"}],
        }
        ctx = SpyCtx(default={"exit_code": 0, "output": "EXISTS"})
        assert env._verify_functional(ctx, {}, verification) is True
        assert len(ctx.calls) == 1  # the probe itself

    def test_file_exists_path_is_shell_quoted(self, tmp_path):
        env = make_env(tmp_path)
        verification = {
            "method": "functional_test",
            "test_commands": [],
            "checks": [{"type": "file_exists",
                        "path": "/tmp/finetune-bench/my dir/$(rm -rf ~)/app.py"}],
        }
        ctx = SpyCtx(default={"exit_code": 1, "output": ""})
        env._verify_functional(ctx, {}, verification)
        probe = ctx.calls[0]
        # The path must be a single quoted token — no live substitution.
        assert "'" in probe
        assert "$(rm" in probe  # still present, but inside quotes
        import shlex as _shlex
        tokens = _shlex.split(probe.split("&&")[0])
        assert tokens[-1].endswith("app.py")

    def test_out_of_range_command_index_counts_as_failed(self, tmp_path):
        env = make_env(tmp_path)
        verification = {
            "method": "functional_test",
            "test_commands": [],
            "checks": [{"type": "output_contains", "command_index": 0,
                        "expected": ["anything"]}],
        }
        assert env._verify_functional(SpyCtx(), {}, verification) is False


# ============================================================================
# Docker enforcement — preflight, unsandboxed refusal, mid-run infra errors
# ============================================================================

class TestSandboxEnforcement:
    class _Proc:
        def __init__(self, rc, stderr=b""):
            self.returncode = rc
            self.stderr = stderr

    def test_docker_preflight_passes_when_daemon_up(self, monkeypatch, capsys):
        monkeypatch.setattr(fbe.subprocess, "run", lambda *a, **k: self._Proc(0))
        env = make_env()
        env._preflight_docker_check()  # must not raise
        assert "Docker daemon is responsive" in capsys.readouterr().out

    def test_docker_preflight_hard_exits_when_daemon_down(self, monkeypatch, capsys):
        monkeypatch.setattr(
            fbe.subprocess, "run",
            lambda *a, **k: self._Proc(1, b"Cannot connect to the Docker daemon"),
        )
        env = make_env()
        with pytest.raises(SystemExit) as ei:
            env._preflight_docker_check()
        assert ei.value.code == fbe.FinetuneBenchEnv.EXIT_SANDBOX_UNAVAILABLE
        assert "daemon" in capsys.readouterr().out.lower()

    def test_docker_preflight_hard_exits_when_cli_missing(self, monkeypatch):
        def no_cli(*a, **k):
            raise FileNotFoundError("docker")

        monkeypatch.setattr(fbe.subprocess, "run", no_cli)
        with pytest.raises(SystemExit) as ei:
            make_env()._preflight_docker_check()
        assert ei.value.code == fbe.FinetuneBenchEnv.EXIT_SANDBOX_UNAVAILABLE

    def test_enforce_uses_docker_preflight(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        called = []
        env = make_env()
        monkeypatch.setattr(env, "_preflight_docker_check", lambda: called.append(1))
        env._enforce_sandbox_backend()
        assert called == [1]

    def test_non_docker_backend_refused_without_override(self, monkeypatch, capsys):
        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.delenv("FINETUNE_BENCH_ALLOW_UNSANDBOXED", raising=False)
        env = make_env(terminal_backend="local")
        with pytest.raises(SystemExit) as ei:
            env._enforce_sandbox_backend()
        assert ei.value.code == fbe.FinetuneBenchEnv.EXIT_SANDBOX_UNAVAILABLE
        assert "FINETUNE_BENCH_ALLOW_UNSANDBOXED" in capsys.readouterr().out

    def test_non_docker_backend_allowed_with_env_override(self, monkeypatch, capsys):
        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.setenv("FINETUNE_BENCH_ALLOW_UNSANDBOXED", "1")
        make_env(terminal_backend="local")._enforce_sandbox_backend()  # no raise
        assert "UNSANDBOXED" in capsys.readouterr().out

    def test_non_docker_backend_allowed_with_config_override(self, monkeypatch, capsys):
        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.delenv("FINETUNE_BENCH_ALLOW_UNSANDBOXED", raising=False)
        env = make_env(terminal_backend="local", allow_unsandboxed=True)
        env._enforce_sandbox_backend()  # no raise
        assert "UNSANDBOXED" in capsys.readouterr().out

    def test_mid_run_docker_death_is_infra_error(self, tmp_path):
        """A tool result showing the daemon died must be classified as an
        infra error, not a model-quality zero."""
        env = make_env(tmp_path)
        msgs = [
            tc_msg(),
            tool_msg("Error: Cannot connect to the Docker daemon at "
                     "unix:///var/run/docker.sock. Is the docker daemon running?"),
        ]
        res = env.compute_reward(tier2_output_case("te-dd", "x"), msgs, 1, 1, SpyCtx())
        assert res.infra_error is True
        assert res.reward == 0.0
        m = env._aggregate_metrics()
        assert m["infra_errors"] == 1
        assert m["scored_cases"] == 0

    def test_in_container_docker_not_found_is_not_infra(self, tmp_path):
        """te-034 runs `docker ps` INSIDE the sandbox; 'command not found'
        must stay a normal (expected) tool failure."""
        env = make_env(tmp_path)
        msgs = [
            tc_msg(args='{"command": "docker ps"}'),
            tool_msg(json.dumps({"exit_code": 127,
                                 "output": "bash: docker: command not found"})),
            asst("Docker is not installed in this environment."),
        ]
        case = expect_failure_case("te-034x", regex="not (installed|found)")
        res = env.compute_reward(case, msgs, 2, 1, SpyCtx())
        assert res.infra_error is False
        assert res.tool_args_valid is True


# ============================================================================
# Finding 5 — infra errors: classification, exclusion, invalidation, verdict
# ============================================================================

class TestInfraErrors:
    @pytest.mark.parametrize("exc,expected", [
        (ConnectionResetError("peer reset"), True),
        (TimeoutError(), True),
        (Exception("Connection reset by peer"), True),
        (Exception("Request timed out after 120s"), True),
        (Exception("502 Bad Gateway"), True),
        (ValueError("bad json in tool call"), False),
        (KeyError("type"), False),
    ])
    def test_classification(self, exc, expected):
        assert fbe.FinetuneBenchEnv._is_infra_error(exc) is expected

    def test_infra_cases_excluded_from_denominators(self, tmp_path):
        env = make_env(tmp_path)
        env.results = [
            case_result(case_id="a", tool_selection_correct=True),
            case_result(case_id="b", tool_selection_correct=True),
            case_result(case_id="c", tool_selection_correct=True),
            case_result(case_id="d", infra_error=True),
        ]
        m = env._aggregate_metrics()
        assert m["tool_selection_accuracy"] == 1.0  # infra case not a miss
        assert m["infra_errors"] == 1
        assert m["scored_cases"] == 3
        assert m["total_cases"] == 4
        assert m["infra_error_rate"] == pytest.approx(0.25)
        # Neutral fields on infra cases must not poison other rates either.
        assert m["format_compliance"] == 1.0
        assert m["hallucination_rate"] == 0.0

    def test_run_invalidated_above_threshold(self, tmp_path):
        env = make_env(tmp_path)
        env.results = [case_result(case_id=str(i)) for i in range(8)]
        env.results += [case_result(case_id="x", infra_error=True),
                        case_result(case_id="y", infra_error=True)]
        metrics = env._aggregate_metrics()  # 20% infra
        with pytest.raises(SystemExit) as ei:
            env._validate_run(metrics)
        assert ei.value.code == 3

    def test_run_valid_at_or_below_threshold(self, tmp_path):
        env = make_env(tmp_path)
        env.results = [case_result(case_id=str(i)) for i in range(19)]
        env.results.append(case_result(case_id="x", infra_error=True))
        metrics = env._aggregate_metrics()  # exactly 5%
        env._validate_run(metrics)  # must not raise

    def test_verdict_hallucination_tolerance(self, tmp_path):
        env = make_env(tmp_path)

        def verdict_for(candidate, baseline=0.0):
            comparison = {"hallucination_rate": {
                "baseline": baseline, "candidate": candidate,
                "delta": candidate - baseline,
            }}
            return env._verdict(comparison)["no_hallucinations"]

        assert verdict_for(0.0) is True
        assert verdict_for(0.008) is True      # one flaky case out of 243
        assert verdict_for(0.02, 0.0) is False
        assert verdict_for(0.03, 0.025) is True   # <= baseline + 0.01


class TestVerdictThresholdBoundaries:
    @staticmethod
    def _verdict(env, key, delta):
        comparison = {key: {"baseline": 0.9, "candidate": 0.9 + delta, "delta": delta}}
        return env._verdict(comparison)

    def test_tool_selection_boundary(self, tmp_path):
        env = make_env(tmp_path)
        assert self._verdict(env, "tool_selection_accuracy", -0.03)["tool_selection"] is True
        assert self._verdict(env, "tool_selection_accuracy", -0.0301)["tool_selection"] is False

    def test_execution_and_completion_boundary(self, tmp_path):
        env = make_env(tmp_path)
        assert self._verdict(env, "tool_execution_success", -0.05)["tool_execution"] is True
        assert self._verdict(env, "tool_execution_success", -0.0501)["tool_execution"] is False
        assert self._verdict(env, "task_completion_rate", -0.05)["task_completion"] is True
        assert self._verdict(env, "task_completion_rate", -0.0501)["task_completion"] is False


# ============================================================================
# Finding 9 — hallucination vs parse-failure are distinct metrics
# ============================================================================

class TestHallucinationMetrics:
    def test_unknown_tool_name_flagged(self, tmp_path):
        env = make_env(tmp_path)
        case = tier2_output_case()
        msgs = [tc_msg(name="make_coffee"), tool_msg(OK_TOOL)]
        res = env.compute_reward(case, msgs, 1, 0, SpyCtx(),
                                 available_tools={"terminal", "web_search"})
        assert res.hallucinated_tool is True

    def test_known_tool_name_not_flagged(self, tmp_path):
        env = make_env(tmp_path)
        res = env.compute_reward(tier2_output_case(), [tc_msg(), tool_msg(OK_TOOL)],
                                 1, 0, SpyCtx(), available_tools={"terminal"})
        assert res.hallucinated_tool is False

    def test_aggregate_separates_parse_failures_from_hallucinations(self, tmp_path):
        env = make_env(tmp_path)
        env.results = [
            case_result(case_id="a"),
            case_result(case_id="b", tool_call_parseable=False),
            case_result(case_id="c", hallucinated_tool=True),
            case_result(case_id="d"),
        ]
        m = env._aggregate_metrics()
        assert m["tool_call_parse_failure_rate"] == pytest.approx(0.25)
        assert m["hallucination_rate"] == pytest.approx(0.25)


# ============================================================================
# Findings 5/6/8 — rollout behavior with a faked agent loop
# ============================================================================

@pytest.fixture
def fake_agent_modules(monkeypatch):
    """Inject fake run_agent / tools.terminal_tool modules so _rollout_case
    can run without the real agent stack, network, or Docker."""
    state = {"messages": [], "raise": None, "prompts": []}

    class FakeAgent:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.valid_tool_names = {"terminal", "web_search"}
            FakeAgent.instances.append(self)

        def run_conversation(self, prompt, system_message=None, task_id=None):
            state["prompts"].append(prompt)
            if state["raise"] is not None:
                raise state["raise"]
            return {"messages": state["messages"]}

    ra = types.ModuleType("run_agent")
    ra.AIAgent = FakeAgent
    tt = types.ModuleType("tools.terminal_tool")
    tt.register_task_env_overrides = lambda *a, **k: None
    tt.clear_task_env_overrides = lambda *a, **k: None
    tt.cleanup_vm = lambda *a, **k: None
    tools_pkg = types.ModuleType("tools")
    tools_pkg.terminal_tool = tt
    monkeypatch.setitem(sys.modules, "run_agent", ra)
    monkeypatch.setitem(sys.modules, "tools", tools_pkg)
    monkeypatch.setitem(sys.modules, "tools.terminal_tool", tt)
    FakeAgent.instances = []
    return state, FakeAgent


ROLLOUT_CASE = {
    "id": "ts-001",
    "tier": 1,
    "category": "correct_tool_simple",
    "prompt": "List files",
    "expected": {"tool_name": "terminal", "should_call_tool": True},
}


class TestRolloutCase:
    def test_scoring_exception_records_this_case(self, tmp_path, fake_agent_modules, monkeypatch):
        state, _ = fake_agent_modules
        state["messages"] = [tc_msg(), tool_msg(OK_TOOL)]
        env = make_env(tmp_path)
        # Pre-existing result from a previous case must not be returned.
        previous = case_result(case_id="previous-case")
        env.results.append(previous)

        def boom(*a, **k):
            raise KeyError("type")

        monkeypatch.setattr(env, "compute_reward", boom)
        res = env._rollout_case(dict(ROLLOUT_CASE))
        assert res is not previous
        assert res.case_id == "ts-001"
        assert res.reward == 0.0
        assert res.infra_error is False
        assert len(env.results) == 2
        assert env.results[-1] is res

    def test_infra_exception_marks_infra_error(self, tmp_path, fake_agent_modules):
        state, _ = fake_agent_modules
        state["raise"] = ConnectionResetError("connection reset by peer")
        env = make_env(tmp_path)
        res = env._rollout_case(dict(ROLLOUT_CASE))
        assert res.infra_error is True
        # Infra failures stay neutral on quality flags.
        assert res.format_valid is True
        assert res.tool_call_parseable is True

    def test_quality_exception_not_infra(self, tmp_path, fake_agent_modules):
        state, _ = fake_agent_modules
        state["raise"] = ValueError("model produced invalid output")
        env = make_env(tmp_path)
        res = env._rollout_case(dict(ROLLOUT_CASE))
        assert res.infra_error is False
        assert res.format_valid is False
        assert res.tool_call_parseable is False

    def test_seed_plumbed_into_request_overrides(self, tmp_path, fake_agent_modules):
        state, FakeAgent = fake_agent_modules
        state["messages"] = [tc_msg(), tool_msg(OK_TOOL)]
        env = make_env(tmp_path, seed=4242)
        env._rollout_case(dict(ROLLOUT_CASE))
        overrides = FakeAgent.instances[-1].kwargs["request_overrides"]
        assert overrides["seed"] == 4242

    def test_working_dir_remapped_into_run_base(self, tmp_path, fake_agent_modules):
        state, _ = fake_agent_modules
        state["messages"] = []
        env = make_env(tmp_path)
        case = dict(ROLLOUT_CASE)
        case["setup"] = {"working_dir": "/tmp/finetune-bench/greet-test"}
        env._rollout_case(case)
        remapped = str(env.run_base / "greet-test")
        assert remapped in state["prompts"][-1]
        assert Path(remapped).is_dir()

    def test_default_working_dir_is_per_run(self, tmp_path, fake_agent_modules):
        state, _ = fake_agent_modules
        state["messages"] = []
        env = make_env(tmp_path)
        env._rollout_case(dict(ROLLOUT_CASE))
        assert str(env.run_base / "ts-001") in state["prompts"][-1]

    def test_run_bases_are_unique_per_env(self):
        a = fbe.FinetuneBenchEnv(fbe.FinetuneBenchConfig())
        b = fbe.FinetuneBenchEnv(fbe.FinetuneBenchConfig())
        assert a.run_base != b.run_base
        assert str(a.run_base).startswith(str(fbe.RUN_ROOT))


class TestVerifyFunctionalRemap:
    def test_commands_and_paths_remapped(self, tmp_path):
        env = make_env(tmp_path)
        verification = {
            "method": "functional_test",
            "test_commands": ["cd /tmp/finetune-bench/foo && python app.py"],
            "checks": [
                {"type": "exit_code", "command_index": 0, "expected": 0},
                {"type": "file_exists", "path": "/tmp/finetune-bench/foo/app.py"},
            ],
        }
        ctx = SpyCtx(default={"exit_code": 0, "output": "EXISTS"})
        assert env._verify_functional(ctx, {}, verification) is True
        base = str(env.run_base)
        assert ctx.calls[0] == f"cd {base}/foo && python app.py"
        assert f"{base}/foo/app.py" in ctx.calls[1]
        assert "/tmp/finetune-bench/foo" not in " ".join(ctx.calls)


class TestWorkingDirGuard:
    """The setup phase rmtree-s the case working dir — it must refuse any
    path that does not resolve under this run's scratch base."""

    def _case_with_wd(self, wd):
        case = dict(ROLLOUT_CASE)
        case["setup"] = {"working_dir": wd}
        return case

    def test_outside_working_dir_skips_case(self, tmp_path, fake_agent_modules):
        state, _ = fake_agent_modules
        env = make_env(tmp_path)
        target = tmp_path / "precious"
        target.mkdir()
        (target / "keep.txt").write_text("do not delete")

        res = env._rollout_case(self._case_with_wd(str(target)))
        assert res is None                      # case skipped, not scored
        assert env.results == []                # no bogus result recorded
        assert (target / "keep.txt").exists()   # nothing was wiped
        assert state["prompts"] == []           # the agent never ran
        assert "ts-001" in env.malformed_case_ids
        assert env._aggregate_metrics()["malformed_cases"] == 1

    def test_home_working_dir_skips_case(self, tmp_path, fake_agent_modules,
                                          monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
        env = make_env(tmp_path)
        res = env._rollout_case(self._case_with_wd("~/projects"))
        assert res is None
        assert not (tmp_path / "fakehome" / "projects").exists()

    def test_dotdot_escape_skips_case(self, tmp_path, fake_agent_modules):
        env = make_env(tmp_path)
        wd = str(env.run_base / ".." / ".." / "escape")
        assert env._rollout_case(self._case_with_wd(wd)) is None

    def test_working_dir_under_run_base_is_allowed(self, tmp_path,
                                                   fake_agent_modules):
        state, _ = fake_agent_modules
        env = make_env(tmp_path)
        res = env._rollout_case(self._case_with_wd("/tmp/finetune-bench/ok-dir"))
        assert res is not None
        assert (env.run_base / "ok-dir").is_dir()


class TestVerdictFailClosed:
    def test_empty_intersection_fails(self, tmp_path):
        env = make_env(tmp_path)
        checks = env._verdict({})
        assert checks["overall"] is False

    def test_missing_metric_is_skipped_not_passed(self, tmp_path):
        """A metric absent from the baseline must not silently pass its
        gate (the old `.get("delta", 0) >= -threshold` fail-open bug)."""
        env = make_env(tmp_path)
        comparison = {
            "tool_selection_accuracy": {"baseline": 0.9, "candidate": 0.5,
                                        "delta": -0.4},
        }
        checks = env._verdict(comparison)
        assert checks["tool_selection"] is False
        # Gates whose metrics are missing don't appear at all.
        for name in ("tool_execution", "task_completion", "format_compliance",
                     "no_hallucinations", "canary"):
            assert name not in checks
        assert checks["overall"] is False

    def test_all_present_and_healthy_passes(self, tmp_path):
        env = make_env(tmp_path)
        comparison = {
            "tool_selection_accuracy": {"baseline": 0.9, "candidate": 0.91, "delta": 0.01},
            "tool_execution_success": {"baseline": 0.8, "candidate": 0.82, "delta": 0.02},
            "task_completion_rate": {"baseline": 0.7, "candidate": 0.7, "delta": 0.0},
            "format_compliance": {"baseline": 0.99, "candidate": 0.99, "delta": 0.0},
            "hallucination_rate": {"baseline": 0.0, "candidate": 0.0, "delta": 0.0},
            "canary_pass_rate": {"baseline": 1.0, "candidate": 1.0, "delta": 0.0},
        }
        checks = env._verdict(comparison)
        assert checks["overall"] is True


class TestEvaluateExitCode:
    def _run_evaluate(self, tmp_path, monkeypatch, baseline):
        env = make_env(tmp_path)
        monkeypatch.setattr(fbe, "BENCH_STATE_DIR", tmp_path / "state")
        env.prompt_bank = [dict(ROLLOUT_CASE)]
        env.baseline = baseline

        def fake_rollout(item):
            env.results.append(case_result(case_id=item["id"], tier=1,
                                           tool_selection_correct=False))

        monkeypatch.setattr(env, "_rollout_case", fake_rollout)
        rc = env.evaluate()
        results = sorted((tmp_path / "state" / "results").glob("bench_*.json"))
        data = json.loads(results[-1].read_text())
        return rc, data

    def test_fail_verdict_returns_nonzero_and_saves_verdict(self, tmp_path, monkeypatch):
        baseline = {"metrics": {
            "tool_selection_accuracy": 1.0, "tool_execution_success": 1.0,
            "task_completion_rate": 1.0, "format_compliance": 1.0,
            "hallucination_rate": 0.0, "canary_pass_rate": 1.0,
        }}
        rc, data = self._run_evaluate(tmp_path, monkeypatch, baseline)
        assert rc == fbe.FinetuneBenchEnv.EXIT_VERDICT_FAIL
        assert data["verdict"]["overall"] is False
        # Existing consumers' keys stay intact.
        assert "metrics" in data and "cases" in data and "timestamp" in data
        assert data["metrics"]["total_cases"] == 1

    def test_no_baseline_returns_zero_without_verdict_key(self, tmp_path, monkeypatch):
        rc, data = self._run_evaluate(tmp_path, monkeypatch, baseline=None)
        assert rc == 0
        assert "verdict" not in data


# ============================================================================
# Finding 10 — preflight auth header and HTTPError handling
# ============================================================================

class TestPreflight:
    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def test_sends_authorization_header(self, monkeypatch):
        captured = []

        def fake_urlopen(req, timeout=None):
            captured.append(req)
            return self._FakeResp()

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        env = make_env(api_key="sk-secret")
        env._preflight_health_check()
        assert captured[0].get_header("Authorization") == "Bearer sk-secret"

    def test_no_header_for_placeholder_key(self, monkeypatch):
        captured = []

        def fake_urlopen(req, timeout=None):
            captured.append(req)
            return self._FakeResp()

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        env = make_env(api_key="none")
        env._preflight_health_check()
        assert captured[0].get_header("Authorization") is None

    def test_http_error_reported_and_exits(self, monkeypatch, capsys):
        def raise_401(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", None, None)

        monkeypatch.setattr(urllib.request, "urlopen", raise_401)
        env = make_env(api_key="bad-key")
        with pytest.raises(SystemExit) as ei:
            env._preflight_health_check()
        assert ei.value.code == 2
        out = capsys.readouterr().out
        assert "401" in out
        assert "api_key" in out


# ============================================================================
# Finding 11 — prompt bank schema and scorability sweep
# ============================================================================

VALID_CHECK_TYPES = {"exit_code", "output_contains", "output_regex", "file_exists"}
VALID_MATCH_SCOPES = {None, "transcript", "final_answer"}


@pytest.fixture(scope="module")
def prompt_bank():
    with open(Path(_bench_dir) / "prompt_bank.yaml") as f:
        return yaml.safe_load(f)["cases"]


@pytest.fixture(scope="module")
def served_tools():
    """Tool names actually served by the toolsets the bench enables."""
    import toolsets

    with open(Path(_bench_dir) / "default.yaml") as f:
        enabled = yaml.safe_load(f)["env"]["enabled_toolsets"]
    return set(toolsets.resolve_multiple_toolsets(enabled))


class TestPromptBank:
    def test_ids_unique_and_fields_required(self, prompt_bank):
        ids = [c.get("id") for c in prompt_bank]
        assert len(ids) == len(set(ids)), "duplicate case ids"
        for c in prompt_bank:
            cid = c.get("id")
            assert cid, "case missing id"
            assert c.get("tier") in (1, 2, 3), cid
            assert c.get("category"), cid
            assert isinstance(c.get("prompt"), str) and c["prompt"].strip(), cid

    def test_tier1_cases_have_selection_expectation(self, prompt_bank):
        for c in prompt_bank:
            if c["tier"] != 1:
                continue
            exp = c.get("expected") or {}
            assert "should_call_tool" in exp or exp.get("tool_name"), (
                f"{c['id']}: tier-1 scoring needs should_call_tool or tool_name"
            )

    def test_no_case_expects_an_unserved_tool(self, prompt_bank, served_tools):
        """Pin for finding #3: a case must never demand a tool that the
        bench's enabled_toolsets don't serve (e.g. web_search) — such cases
        are unwinnable and count correct answers as hallucinations."""
        for c in prompt_bank:
            expected_tool = (c.get("expected") or {}).get("tool_name")
            if expected_tool:
                assert expected_tool in served_tools, (
                    f"{c['id']}: expects tool {expected_tool!r} which is not "
                    f"served by the bench toolsets {sorted(served_tools)}"
                )

    def test_tier2_plus_verification_well_formed(self, prompt_bank):
        for c in prompt_bank:
            if c["tier"] < 2:
                continue
            v = c.get("verification")
            assert v, f"{c['id']}: tier>=2 requires verification"
            assert v.get("method") in ("output_match", "functional_test"), c["id"]
            assert v.get("match_scope") in VALID_MATCH_SCOPES, c["id"]
            regex = v.get("expected_regex")
            if regex:
                import re as _re
                _re.compile(regex)  # must be a valid pattern
            for chk in v.get("checks") or []:
                assert chk.get("type") in VALID_CHECK_TYPES, c["id"]
                if chk["type"] == "exit_code":
                    assert "expected" in chk, c["id"]
                elif chk["type"] == "output_contains":
                    assert isinstance(chk.get("expected"), list), c["id"]
                elif chk["type"] == "output_regex":
                    assert chk.get("pattern"), c["id"]
                elif chk["type"] == "file_exists":
                    assert chk.get("path"), c["id"]
                if chk["type"] != "file_exists":
                    idx = chk.get("command_index", 0)
                    assert 0 <= idx < len(v.get("test_commands") or []), c["id"]

    def test_no_output_match_case_asserts_nothing(self, prompt_bank):
        """Pin for finding #1: every output_match verification must carry a
        real assertion. Empty expected_value used to reduce 'execution
        success' to 'any tool exited 0' across 84 cases."""
        for c in prompt_bank:
            if c["tier"] < 2:
                continue
            v = c["verification"]
            if v["method"] == "functional_test" and v.get("checks"):
                continue  # functional checks are the assertion
            expected = str(v.get("expected_value") or "").strip()
            regex = str(v.get("expected_regex") or "").strip()
            assert expected or regex, (
                f"{c['id']}: output_match with no expected_value/expected_regex "
                "is vacuous — it would score any successful command as a pass"
            )

    def test_expect_failure_cases_have_acknowledgment_assertion(self, prompt_bank):
        for c in prompt_bank:
            v = c.get("verification") or {}
            if not v.get("expect_failure"):
                continue
            assert c["category"] == "error_recovery", c["id"]
            assert str(v.get("expected_regex") or v.get("expected_value") or "").strip(), (
                f"{c['id']}: expect_failure needs a final-answer assertion"
            )

    def test_functional_cases_stay_inside_bench_scratch_root(self, prompt_bank):
        """Hermeticity pin: setup working_dirs and file_exists paths must
        live under /tmp/finetune-bench so the per-run remap applies."""
        for c in prompt_bank:
            wd = (c.get("setup") or {}).get("working_dir")
            if wd:
                assert wd.startswith("/tmp/finetune-bench/"), c["id"]
            v = c.get("verification") or {}
            for chk in v.get("checks") or []:
                if chk.get("type") == "file_exists":
                    assert str(chk["path"]).startswith("/tmp/finetune-bench/"), c["id"]

    def test_every_case_scorable_without_raising(self, prompt_bank, served_tools, tmp_path):
        env = make_env(tmp_path)
        transcripts = [
            [tc_msg(), tool_msg(OK_TOOL), asst("All done.")],   # normal run
            [asst("I cannot help with that.")],                  # no tools
            [tc_msg(args='{"command": "xyz"}'), tool_msg(ERR_TOOL)],  # garbage
            [],                                                  # empty rollout
        ]
        for case in prompt_bank:
            for msgs in transcripts:
                env.results.clear()
                res = env.compute_reward(
                    case, list(msgs), 1, 0,
                    SpyCtx(default={"exit_code": 1, "output": ""}),
                    available_tools=served_tools,
                )
                assert isinstance(res, fbe.CaseResult), case["id"]
                assert 0.0 <= res.reward <= 1.0, case["id"]
        # No bank case may be flagged malformed by the scorer.
        assert env.malformed_case_ids == set()

    def test_no_output_match_case_autopasses_on_garbage(self, prompt_bank, tmp_path):
        """The regression guard for finding #1: a padded transcript (one
        failing command, no real answer) must never satisfy an output_match
        verification — including the expect_failure cases, which demand an
        explanatory final answer."""
        env = make_env(tmp_path)
        garbage = [tc_msg(), tool_msg(ERR_TOOL)]
        swept = 0
        for case in prompt_bank:
            v = case.get("verification") or {}
            if case["tier"] < 2 or v.get("method") != "output_match":
                continue
            env.results.clear()
            res = env.compute_reward(case, list(garbage), 1, 1, SpyCtx())
            assert res.tool_args_valid is False, case["id"]
            assert res.task_completed is False, case["id"]
            swept += 1
        assert swept >= 60  # the bank's output_match population

    def test_no_output_match_case_autopasses_on_echo_ok(self, prompt_bank, tmp_path):
        """A model that runs `echo ok` and says 'done' must score zero
        execution credit on every output_match case."""
        env = make_env(tmp_path)
        padded = [
            tc_msg(args='{"command": "echo ok"}'),
            tool_msg(json.dumps({"exit_code": 0, "output": "ok"})),
            asst("Done."),
        ]
        for case in prompt_bank:
            v = case.get("verification") or {}
            if case["tier"] < 2 or v.get("method") != "output_match":
                continue
            env.results.clear()
            res = env.compute_reward(case, list(padded), 2, 0, SpyCtx())
            assert res.tool_args_valid is False, case["id"]


# ============================================================================
# Config plumbing
# ============================================================================

class TestConfig:
    def test_default_yaml_has_seed(self):
        with open(Path(_bench_dir) / "default.yaml") as f:
            data = yaml.safe_load(f)
        assert isinstance(data["env"].get("seed"), int)

    def test_seed_loads_from_config(self, tmp_path):
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("env:\n  seed: 99\n")
        cfg = fbe.FinetuneBenchConfig.load(cfg_path)
        assert cfg.seed == 99

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.validation import ValidationItem, ValidationResult
import pipeline.validate as validate


# ── helpers ─────────────────────────────────────────────────────────────────


def _write_tasks(tmp_path, tasks=None):
    data = {
        "generated_at": "2026-08-20T00:00:00+00:00",
        "total_cards": len(tasks) if tasks else 0,
        "tasks": tasks or [],
    }
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(p)


def _sample_task(skill="Hermes", req="deploy please", evidence=None):
    return {
        "skill_name": skill,
        "skill": skill,
        "user_request": req,
        "request": req,
        "friction_evidence": evidence or ["tool_error: write_file remote path failed"],
        "evidence": evidence or ["tool_error: write_file remote path failed"],
        "session_id": "sess_test_123",
        "tool_calls": [],
        "timestamp": 123.0,
    }


def _sample_diff(extra="pitfall"):
    return (
        "--- a/SKILL.md\n"
        "+++ b/SKILL.md\n"
        f"@@ -1,2 +1,3 @@\n"
        f" line\n"
        f"+{extra} remote deploy guard\n"
        f" line\n"
    )


# ── ValidationItem / ValidationResult dataclass ───────────────────────────────


def test_validation_item_to_dict():
    it = ValidationItem(task_index=0, score=85, passed=True, reason="good fix")
    d = it.to_dict()
    assert d["task_index"] == 0
    assert d["score"] == 85
    assert d["passed"] is True
    assert "good fix" in d["reason"]


def test_validation_item_reason_truncates():
    it = ValidationItem(0, 80, True, "x" * 5000)
    assert len(it.to_dict()["reason"]) <= 2000


def test_validation_item_from_dict_roundtrip():
    it = ValidationItem(1, 60, False, "not enough")
    it2 = ValidationItem.from_dict(it.to_dict())
    assert it2.score == 60
    assert it2.passed is False


def test_validation_result_to_dict_roundtrip():
    items = [ValidationItem(0, 85, True, "ok"), ValidationItem(1, 40, False, "no")]
    r = ValidationResult(
        generated_at=ValidationResult.now_iso(),
        skill_path="/a/SKILL.md",
        diff_path="/tmp/candidate.diff",
        gate_type="llm_judge",
        overall_passed=False,
        total_tasks=2,
        passed_tasks=1,
        pass_rate=0.5,
        threshold=70,
        min_pass_rate=0.6,
        limitation="test limitation",
        items=items,
        rejected_reason="pass_rate 0.50 < 0.60 — task 1: no",
    )
    d = r.to_dict()
    assert d["total_tasks"] == 2
    assert d["gate_type"] == "llm_judge"
    assert len(d["items"]) == 2
    r2 = ValidationResult.from_dict(d)
    assert r2.passed_tasks == 1
    assert r2.overall_passed is False


def test_validation_result_to_json_valid():
    r = ValidationResult(
        generated_at=ValidationResult.now_iso(),
        skill_path="/a/SKILL.md",
        diff_path="/d.diff",
        gate_type="llm_judge",
        overall_passed=True,
        total_tasks=1,
        passed_tasks=1,
        pass_rate=1.0,
        threshold=70,
        min_pass_rate=0.6,
        limitation="lim",
        items=[ValidationItem(0, 90, True, "good")],
        rejected_reason=None,
    )
    j = r.to_json()
    assert json.loads(j)["overall_passed"] is True


def test_validation_result_now_iso():
    assert "T" in ValidationResult.now_iso()


def test_validation_result_repr():
    r = ValidationResult(
        generated_at="2026-08-20T00:00:00+00:00",
        skill_path="/s/SKILL.md",
        diff_path="/d.diff",
        gate_type="llm_judge",
        overall_passed=True,
        total_tasks=2,
        passed_tasks=2,
        pass_rate=1.0,
        threshold=70,
        min_pass_rate=0.6,
        limitation="lim",
        items=[],
        rejected_reason=None,
    )
    assert "1.00" in repr(r) or "passed" in repr(r).lower()


# ── Loading ──────────────────────────────────────────────────────────────────


def test_load_tasks_missing_exits(tmp_path):
    with pytest.raises(SystemExit):
        validate.load_tasks(str(tmp_path / "nope.json"))


def test_load_tasks_invalid_json(tmp_path):
    p = tmp_path / "tasks.json"
    p.write_text("{bad", encoding="utf-8")
    with pytest.raises(SystemExit):
        validate.load_tasks(str(p))


def test_load_diff_missing_exits(tmp_path):
    with pytest.raises(SystemExit):
        validate.load_diff(str(tmp_path / "nope.diff"))


def test_load_diff_empty_exits(tmp_path):
    p = tmp_path / "candidate.diff"
    p.write_text("   \n", encoding="utf-8")
    with pytest.raises(SystemExit):
        validate.load_diff(str(p))


def test_load_diff_ok(tmp_path):
    p = tmp_path / "candidate.diff"
    p.write_text(_sample_diff(), encoding="utf-8")
    assert "SKILL.md" in validate.load_diff(str(p))


def test_load_proposal_none():
    assert validate.load_proposal(None) is None


def test_load_proposal_missing_warns(tmp_path, capsys):
    out = validate.load_proposal(str(tmp_path / "nope.json"))
    assert out is None


def test_load_proposal_invalid_json_warns(tmp_path):
    p = tmp_path / "proposal.json"
    p.write_text("{bad", encoding="utf-8")
    out = validate.load_proposal(str(p))
    assert out is None


# ── Prompt rendering ─────────────────────────────────────────────────────────


def test_render_judge_prompt_replaces_placeholders(tmp_path):
    tmpl = tmp_path / "tmpl.md"
    tmpl.write_text("REQ:{user_request} EV:{friction_evidence} DIFF:{candidate_diff} TH:{threshold}", encoding="utf-8")
    out = validate.render_judge_prompt(str(tmpl), "my req", "my ev", "my diff", 70)
    assert "my req" in out
    assert "my ev" in out
    assert "my diff" in out
    assert "70" in out


def test_render_judge_prompt_missing_exits(tmp_path):
    with pytest.raises(SystemExit):
        validate.render_judge_prompt(str(tmp_path / "nope.md"), "a", "b", "c", 70)


def test_extract_request_and_evidence():
    req, ev = validate.extract_request_and_evidence(_sample_task(req="hello", evidence=["e1", "e2"]))
    assert "hello" in req
    assert "e1" in ev

    # missing fields fallback
    req2, ev2 = validate.extract_request_and_evidence({})
    assert req2  # placeholder non-empty
    assert ev2


# ── Judge parsing ────────────────────────────────────────────────────────────


def test_parse_judge_output_json_fenced():
    raw = '```json\n{"score": 85, "passed": true, "reason": "fixes the pitfall"}\n```'
    score, passed, reason = validate.parse_judge_output(raw, 70)
    assert score == 85
    assert passed is True
    assert "pitfall" in reason


def test_parse_judge_output_bare_json():
    raw = '{"score": 42, "passed": false, "reason": "does not address friction"}'
    score, passed, reason = validate.parse_judge_output(raw, 70)
    assert score == 42
    assert passed is False


def test_parse_judge_output_string_passed():
    raw = '{"score": 80, "passed": "true", "reason": "looks good"}'
    score, passed, _ = validate.parse_judge_output(raw, 70)
    assert passed is True


def test_parse_judge_output_missing_passed_uses_threshold():
    raw = '{"score": 75, "reason": "ok"}'
    score, passed, _ = validate.parse_judge_output(raw, 70)
    assert passed is True
    raw2 = '{"score": 60, "reason": "weak"}'
    _, passed2, _ = validate.parse_judge_output(raw2, 70)
    assert passed2 is False


def test_parse_judge_output_clamps_score():
    raw = '{"score": 150, "passed": true, "reason": "too high"}'
    score, _, _ = validate.parse_judge_output(raw, 70)
    assert score == 100


def test_parse_judge_output_regex_fallback():
    raw = "score: 88\npassed: true\nThis diff adds the needed guard"
    score, passed, reason = validate.parse_judge_output(raw, 70)
    assert score == 88
    assert passed is True


def test_parse_judge_output_unparseable_fallback():
    raw = "no scores here at all, just prose"
    score, passed, reason = validate.parse_judge_output(raw, 70)
    assert score == 0
    assert passed is False
    assert reason


# ── Aggregation ──────────────────────────────────────────────────────────────


def test_aggregate_all_pass():
    items = [ValidationItem(0, 80, True, "ok"), ValidationItem(1, 90, True, "ok")]
    ok, reason = validate.aggregate(items, 70, 0.6)
    assert ok is True
    assert reason is None


def test_aggregate_empty_fails():
    ok, reason = validate.aggregate([], 70, 0.6)
    assert ok is False


def test_aggregate_partial_pass_rate_pass():
    # 2/3 = 0.66 >= 0.6 → PASS
    items = [
        ValidationItem(0, 80, True, "ok"),
        ValidationItem(1, 85, True, "ok"),
        ValidationItem(2, 30, False, "no"),
    ]
    ok, _ = validate.aggregate(items, 70, 0.6)
    assert ok is True


def test_aggregate_partial_fails():
    # 1/3 = 0.33 < 0.6 → FAIL
    items = [
        ValidationItem(0, 80, True, "ok"),
        ValidationItem(1, 30, False, "bad"),
        ValidationItem(2, 20, False, "bad"),
    ]
    ok, reason = validate.aggregate(items, 70, 0.6)
    assert ok is False
    assert reason is not None


def test_aggregate_writes_rejected_reason():
    items = [ValidationItem(0, 30, False, "nope")]
    ok, reason = validate.aggregate(items, 70, 0.6)
    assert "task 0" in reason


# ── call_omp (mocked) ───────────────────────────────────────────────────────


def test_call_omp_uses_list_args(tmp_path):
    prompt = tmp_path / "prompt.md"
    prompt.write_text("hello", encoding="utf-8")
    with patch("pipeline.validate.subprocess.run") as sp:
        sp.return_value = MagicMock(returncode=0, stdout='{"score":80,"passed":true,"reason":"ok"}', stderr="")
        validate.call_omp(prompt, str(tmp_path), validate.DEFAULT_MODEL, 10)
        args = sp.call_args[0][0]
        assert isinstance(args, list)
        assert args[0] == "omp"
        assert "--model" in args


def test_call_omp_not_found(tmp_path):
    prompt = tmp_path / "prompt.md"
    prompt.write_text("hello", encoding="utf-8")
    with patch("pipeline.validate.subprocess.run", side_effect=FileNotFoundError()):
        with pytest.raises(SystemExit):
            validate.call_omp(prompt, str(tmp_path), validate.DEFAULT_MODEL, 10)


def test_call_omp_timeout(tmp_path):
    prompt = tmp_path / "prompt.md"
    prompt.write_text("hello", encoding="utf-8")
    with patch("pipeline.validate.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="omp", timeout=10)):
        with pytest.raises(SystemExit):
            validate.call_omp(prompt, str(tmp_path), validate.DEFAULT_MODEL, 10)


def test_call_omp_empty_output_exits(tmp_path):
    prompt = tmp_path / "prompt.md"
    prompt.write_text("hello", encoding="utf-8")
    with patch("pipeline.validate.subprocess.run") as sp:
        sp.return_value = MagicMock(returncode=0, stdout="   ", stderr="")
        with pytest.raises(SystemExit):
            validate.call_omp(prompt, str(tmp_path), validate.DEFAULT_MODEL, 10)


# ── write_validation ────────────────────────────────────────────────────────


def test_write_validation(tmp_path):
    r = ValidationResult(
        generated_at=ValidationResult.now_iso(),
        skill_path="/a/SKILL.md",
        diff_path="/d.diff",
        gate_type="llm_judge",
        overall_passed=True,
        total_tasks=1,
        passed_tasks=1,
        pass_rate=1.0,
        threshold=70,
        min_pass_rate=0.6,
        limitation=validate.LIMITATION_TEXT,
        items=[ValidationItem(0, 90, True, "good")],
        rejected_reason=None,
    )
    out = validate.write_validation(r, str(tmp_path))
    data = json.loads(Path(out).read_text(encoding="utf-8"))
    assert data["overall_passed"] is True
    assert data["gate_type"] == "llm_judge"


# ── CLI integration (dry-run) ────────────────────────────────────────────────


def test_cli_dry_run_generates_outputs(tmp_path):
    tasks_path = _write_tasks(tmp_path, [_sample_task()])
    diff_path = tmp_path / "candidate.diff"
    diff_path.write_text(_sample_diff("pitfall"), encoding="utf-8")
    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({"skill_path": "/tmp/SKILL.md"}), encoding="utf-8")
    outdir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(ROOT / "pipeline" / "validate.py"),
        "--tasks",
        tasks_path,
        "--diff",
        str(diff_path),
        "--proposal",
        str(proposal),
        "--output-dir",
        str(outdir),
        "--dry-run",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, proc.stderr
    assert "[validate]" in proc.stdout
    assert (outdir / "validation.json").exists()
    data = json.loads((outdir / "validation.json").read_text(encoding="utf-8"))
    assert "overall_passed" in data
    assert "items" in data
    assert data["total_tasks"] == 1
    assert data["gate_type"] == "llm_judge"
    assert "limitation" in data
    assert data["threshold"] == 70
    # pass_rate consistency
    assert abs(data["pass_rate"] - data["passed_tasks"] / data["total_tasks"]) < 1e-6 if data["total_tasks"] else True
    # limitation text present
    assert "No real execution" in data["limitation"]


def test_cli_dry_run_fail_gate(tmp_path):
    # diff with no relevant keywords → low score → FAIL
    tasks_path = _write_tasks(tmp_path, [_sample_task(), _sample_task(), _sample_task()])
    diff_path = tmp_path / "candidate.diff"
    diff_path.write_text("--- a/SKILL.md\n+++ b/SKILL.md\n@@ -1 +1 @@\n+xyz unrelated\n", encoding="utf-8")
    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({"skill_path": "/tmp/SKILL.md"}), encoding="utf-8")
    outdir = tmp_path / "out2"
    cmd = [
        sys.executable,
        str(ROOT / "pipeline" / "validate.py"),
        "--tasks",
        tasks_path,
        "--diff",
        str(diff_path),
        "--proposal",
        str(proposal),
        "--output-dir",
        str(outdir),
        "--threshold",
        "99",
        "--pass-rate",
        "0.6",
        "--dry-run",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, proc.stderr
    data = json.loads((outdir / "validation.json").read_text(encoding="utf-8"))
    assert data["overall_passed"] is False
    assert data["rejected_reason"] is not None


def test_cli_dry_run_threshold_and_pass_rate(tmp_path):
    tasks_path = _write_tasks(tmp_path, [_sample_task(), _sample_task()])
    diff_path = tmp_path / "candidate.diff"
    diff_path.write_text(_sample_diff("pitfall"), encoding="utf-8")
    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({"skill_path": "/tmp/SKILL.md"}), encoding="utf-8")
    outdir = tmp_path / "out3"
    cmd = [
        sys.executable,
        str(ROOT / "pipeline" / "validate.py"),
        "--tasks",
        tasks_path,
        "--diff",
        str(diff_path),
        "--proposal",
        str(proposal),
        "--output-dir",
        str(outdir),
        "--threshold",
        "80",
        "--pass-rate",
        "0.5",
        "--dry-run",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, proc.stderr
    data = json.loads((outdir / "validation.json").read_text(encoding="utf-8"))
    assert data["threshold"] == 80
    assert data["min_pass_rate"] == 0.5


def test_cli_ninerouter_check(tmp_path):
    tasks_path = _write_tasks(tmp_path, [_sample_task()])
    diff_path = tmp_path / "candidate.diff"
    diff_path.write_text(_sample_diff(), encoding="utf-8")
    cmd = [
        sys.executable,
        str(ROOT / "pipeline" / "validate.py"),
        "--tasks",
        tasks_path,
        "--diff",
        str(diff_path),
        "--output-dir",
        str(tmp_path / "out4"),
    ]
    env = {k: v for k, v in os.environ.items() if k != "NINEROUTER_KEY"}
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, env=env)
    assert proc.returncode != 0
    assert "NINEROUTER_KEY" in proc.stderr


def test_cli_missing_tasks_exits(tmp_path):
    diff_path = tmp_path / "candidate.diff"
    diff_path.write_text(_sample_diff(), encoding="utf-8")
    cmd = [
        sys.executable,
        str(ROOT / "pipeline" / "validate.py"),
        "--tasks",
        str(tmp_path / "nope.json"),
        "--diff",
        str(diff_path),
        "--output-dir",
        str(tmp_path / "out5"),
        "--dry-run",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    assert proc.returncode != 0
    assert "ERROR" in proc.stderr


def test_cli_missing_diff_exits(tmp_path):
    tasks_path = _write_tasks(tmp_path, [_sample_task()])
    cmd = [
        sys.executable,
        str(ROOT / "pipeline" / "validate.py"),
        "--tasks",
        tasks_path,
        "--diff",
        str(tmp_path / "nope.diff"),
        "--output-dir",
        str(tmp_path / "out6"),
        "--dry-run",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    assert proc.returncode != 0
    assert "ERROR" in proc.stderr


def test_help_exits_zero():
    cmd = [sys.executable, str(ROOT / "pipeline" / "validate.py"), "--help"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0
    assert "--tasks" in proc.stdout
    assert "--threshold" in proc.stdout


def test_cli_no_proposal_still_works(tmp_path):
    tasks_path = _write_tasks(tmp_path, [_sample_task()])
    diff_path = tmp_path / "candidate.diff"
    diff_path.write_text(_sample_diff(), encoding="utf-8")
    outdir = tmp_path / "out_no_proposal"
    cmd = [
        sys.executable,
        str(ROOT / "pipeline" / "validate.py"),
        "--tasks",
        tasks_path,
        "--diff",
        str(diff_path),
        "--output-dir",
        str(outdir),
        "--dry-run",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, proc.stderr
    data = json.loads((outdir / "validation.json").read_text(encoding="utf-8"))
    assert data["total_tasks"] == 1


def test_cli_tasks_with_messages_field(tmp_path):
    # tasks that carry friction in different schema (messages-based)
    data = {
        "generated_at": "2026-08-20T00:00:00+00:00",
        "total_cards": 1,
        "tasks": [
            {
                "skill": "Hermes",
                "user_request": "help me deploy",
                "friction_evidence": ["remote path failed"],
                "messages": [{"role": "user", "content": "help me deploy"}],
            }
        ],
    }
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps(data), encoding="utf-8")
    diff_path = tmp_path / "candidate.diff"
    diff_path.write_text(_sample_diff(), encoding="utf-8")
    outdir = tmp_path / "out_msg"
    cmd = [
        sys.executable,
        str(ROOT / "pipeline" / "validate.py"),
        "--tasks",
        str(tasks_path),
        "--diff",
        str(diff_path),
        "--output-dir",
        str(outdir),
        "--dry-run",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, proc.stderr
    assert (outdir / "validation.json").exists()

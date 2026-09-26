import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib import staging as st
import pipeline.review as review


# ── helpers ─────────────────────────────────────────────────────────────────


def _sample_diff(extra: str = "pitfall") -> str:
    return (
        "diff --git a/SKILL.md b/SKILL.md\n"
        "index 42edf68..af94eca 100644\n"
        "--- a/SKILL.md\n"
        "+++ b/SKILL.md\n"
        "@@ -202,3 +202,9 @@ terminal(command=\"tmux new-session -d -s resumed 'hermes --resume 20260225_14305\n"
        f"+## Remote Host Operations — {extra}\n"
        "+- **Local tools** remote paths must go via ssh.\n"
        "+- Verify remotely.\n"
    )


def _write_validation(tmp_path, overall_passed=True, skill_path="/tmp/fake/SKILL.md", items=None):
    if items is None:
        items = [{"task_index": 0, "score": 90 if overall_passed else 30, "passed": overall_passed, "reason": "ok" if overall_passed else "nope"}]
    data = {
        "generated_at": "2026-08-20T11:43:04.125948+00:00",
        "skill_path": skill_path,
        "diff_path": str(tmp_path / "candidate.diff"),
        "gate_type": "llm_judge",
        "overall_passed": overall_passed,
        "total_tasks": len(items),
        "passed_tasks": sum(1 for it in items if it["passed"]),
        "pass_rate": (sum(1 for it in items if it["passed"]) / len(items)) if items else 0,
        "threshold": 70,
        "min_pass_rate": 0.6,
        "limitation": "No real execution replay",
        "items": items,
        "rejected_reason": None if overall_passed else "pass_rate 0.00 < 0.60",
    }
    p = tmp_path / "validation.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


def _write_proposal(tmp_path, skill_path="/tmp/fake/SKILL.md"):
    data = {
        "generated_at": "2026-08-20T11:39:20.070091+00:00",
        "skill_path": skill_path,
        "source_task_cards": 1,
        "diff_lines": 6,
        "summary": "Add remote host pitfalls",
        "focused_on": ["local vs remote", "rsync then verify"],
    }
    p = tmp_path / "proposal.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


def _write_diff(tmp_path, text=None):
    p = tmp_path / "candidate.diff"
    p.write_text(text or _sample_diff(), encoding="utf-8")
    return str(p)


# ── lib/staging unit tests ──────────────────────────────────────────────────


def test_skill_slug_skill_md():
    assert st.skill_slug("/Users/mac/.hermes/skills/autonomous-ai-agents/hermes-agent/SKILL.md") == "hermes-agent"


def test_skill_slug_generic():
    assert st.skill_slug("/tmp/foo/SKILL.md") == "foo"
    assert st.skill_slug("/tmp/my-skill.md") == "my-skill"


def test_staging_dir_name():
    name = st.staging_dir_name("/tmp/hermes-agent/SKILL.md", "20260820-114304")
    assert name == "hermes-agent-20260820-114304"


def test_now_ts_format():
    ts = st.now_ts()
    assert len(ts) == 15
    assert ts[8] == "-"


def test_ensure_dir_creates(tmp_path):
    p = tmp_path / "a" / "b"
    st.ensure_dir(p)
    assert p.is_dir()


def test_copy_file(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("hello", encoding="utf-8")
    dst = tmp_path / "dst" / "out.txt"
    st.copy_file(src, dst)
    assert dst.read_text(encoding="utf-8") == "hello"


def test_append_rejected_jsonl(tmp_path):
    base = tmp_path / "rejected"
    st.append_rejected_jsonl(base, {"ts": "t1", "reason": "r1"})
    st.append_rejected_jsonl(base, {"ts": "t2", "reason": "r2"})
    lines = (base / "rejected.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["ts"] == "t1"


def test_build_review_md_contains_fields(tmp_path):
    validation = {
        "overall_passed": True,
        "total_tasks": 1,
        "passed_tasks": 1,
        "pass_rate": 1.0,
        "threshold": 70,
        "items": [{"task_index": 0, "score": 90, "passed": True, "reason": "good"}],
        "rejected_reason": None,
    }
    proposal = {"summary": "Add pitfalls", "focused_on": ["remote host"]}
    md = st.build_review_md(
        ts="20260820-114304",
        skill_path="/tmp/SKILL.md",
        staging_dir="staging/foo-20260820-114304",
        validation=validation,
        proposal=proposal,
        candidate_diff=_sample_diff(),
    )
    assert "Skill 审查请求" in md
    assert "PASS" in md
    assert "candidate.diff" in md
    assert "apply --staging-dir" in md
    assert "reject --staging-dir" in md
    assert "Add pitfalls" in md


def test_build_review_md_fail_shows_rejected_reason():
    validation = {
        "overall_passed": False,
        "total_tasks": 1,
        "passed_tasks": 0,
        "pass_rate": 0.0,
        "threshold": 70,
        "items": [{"task_index": 0, "score": 30, "passed": False, "reason": "nope"}],
        "rejected_reason": "pass_rate 0.00 < 0.60",
    }
    md = st.build_review_md(
        ts="20260820-114304",
        skill_path="/tmp/SKILL.md",
        staging_dir="staging/foo",
        validation=validation,
        proposal=None,
        candidate_diff=_sample_diff(),
    )
    assert "FAIL" in md


# ── review stage: PASS → staging/ ──────────────────────────────────────────


def test_stage_pass_creates_staging(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    skill = tmp_path / "SKILL.md"
    skill.write_text("hello\n", encoding="utf-8")
    v = _write_validation(src, overall_passed=True, skill_path=str(skill))
    d = _write_diff(src)
    pr = _write_proposal(src, skill_path=str(skill))

    cmd = [
        sys.executable, str(ROOT / "pipeline" / "review.py"), "stage",
        "--validation", v, "--diff", d, "--proposal", pr,
        "--skill", str(skill), "--output-dir", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, proc.stderr
    assert "[review] Gate PASS" in proc.stdout
    stagings = list((out / "staging").glob("*-*"))
    assert len(stagings) == 1
    sd = stagings[0]
    assert (sd / "candidate.diff").exists()
    assert (sd / "validation.json").exists()
    assert (sd / "proposal.json").exists()
    assert (sd / "review.md").exists()
    assert "Skill 审查请求" in (sd / "review.md").read_text(encoding="utf-8")


def test_stage_fail_creates_rejected(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    skill = tmp_path / "SKILL.md"
    skill.write_text("hello\n", encoding="utf-8")
    v = _write_validation(src, overall_passed=False, skill_path=str(skill))
    d = _write_diff(src)
    pr = _write_proposal(src, skill_path=str(skill))

    cmd = [
        sys.executable, str(ROOT / "pipeline" / "review.py"), "stage",
        "--validation", v, "--diff", d, "--proposal", pr,
        "--skill", str(skill), "--output-dir", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, proc.stderr
    assert "Rejected" in proc.stdout or "FAIL" in proc.stdout
    assert not (out / "staging").exists() or len(list((out / "staging").glob("*"))) == 0
    # rejected dir + jsonl
    assert (out / "rejected").is_dir()
    assert (out / "rejected" / "rejected.jsonl").exists()
    lines = (out / "rejected" / "rejected.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert "reason" in entry
    assert "diff" in entry


def test_stage_without_proposal_still_passes(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    skill = tmp_path / "SKILL.md"
    skill.write_text("x\n", encoding="utf-8")
    v = _write_validation(src, overall_passed=True, skill_path=str(skill))
    d = _write_diff(src)

    cmd = [
        sys.executable, str(ROOT / "pipeline" / "review.py"), "stage",
        "--validation", v, "--diff", d,
        "--skill", str(skill), "--output-dir", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, proc.stderr
    assert (list((out / "staging").glob("*-*"))[0] / "review.md").exists()


def test_stage_missing_validation_exits(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    d = tmp_path / "candidate.diff"
    d.write_text(_sample_diff(), encoding="utf-8")
    cmd = [
        sys.executable, str(ROOT / "pipeline" / "review.py"), "stage",
        "--validation", str(tmp_path / "nope.json"), "--diff", str(d),
        "--output-dir", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert proc.returncode != 0
    assert "ERROR" in proc.stderr


def test_stage_missing_diff_exits(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    v = _write_validation(src, overall_passed=True)
    cmd = [
        sys.executable, str(ROOT / "pipeline" / "review.py"), "stage",
        "--validation", v, "--diff", str(tmp_path / "nope.diff"),
        "--output-dir", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert proc.returncode != 0
    assert "ERROR" in proc.stderr


def test_stage_rejected_dir_override(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    custom_rejected = tmp_path / "my-rejected"
    v = _write_validation(src, overall_passed=False)
    d = _write_diff(src)
    cmd = [
        sys.executable, str(ROOT / "pipeline" / "review.py"), "stage",
        "--validation", v, "--diff", d,
        "--output-dir", str(out), "--rejected-dir", str(custom_rejected),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stderr
    assert custom_rejected.is_dir()


def test_stage_skill_inferred_from_validation(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    skill = tmp_path / "SKILL.md"
    skill.write_text("x\n", encoding="utf-8")
    v = _write_validation(src, overall_passed=True, skill_path=str(skill))
    d = _write_diff(src)
    # no --skill, should infer from validation
    cmd = [
        sys.executable, str(ROOT / "pipeline" / "review.py"), "stage",
        "--validation", v, "--diff", d,
        "--output-dir", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stderr


# ── review apply ────────────────────────────────────────────────────────────


def test_apply_with_git(tmp_path):
    # Create a git repo so git apply works, and a SKILL.md
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, timeout=10)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo), capture_output=True, timeout=10)
    subprocess.run(["git", "config", "user.name", "t"], cwd=str(repo), capture_output=True, timeout=10)
    skill = repo / "SKILL.md"
    skill.write_text("line1\nline2\nline3\n", encoding="utf-8")
    subprocess.run(["git", "add", "SKILL.md"], cwd=str(repo), capture_output=True, timeout=10)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True, timeout=10)

    # Create staging with a diff that appends after line3
    staging = tmp_path / "out" / "staging" / "hermes-agent-20260820-114304"
    staging.mkdir(parents=True)
    diff_text = (
        "diff --git a/SKILL.md b/SKILL.md\n"
        "index 0000000..1111111 100644\n"
        "--- a/SKILL.md\n"
        "+++ b/SKILL.md\n"
        "@@ -1,3 +1,6 @@\n"
        " line1\n"
        " line2\n"
        " line3\n"
        "+## Remote Host Operations — pitfalls\n"
        "+- Local tools remote paths must go via ssh.\n"
        "+- Verify remotely.\n"
    )
    (staging / "candidate.diff").write_text(diff_text, encoding="utf-8")
    (staging / "validation.json").write_text(
        json.dumps({"skill_path": str(skill), "overall_passed": True}, ensure_ascii=False), encoding="utf-8"
    )
    (staging / "proposal.json").write_text(json.dumps({"skill_path": str(skill)}, ensure_ascii=False), encoding="utf-8")

    out_base = tmp_path / "out"
    cmd = [
        sys.executable, str(ROOT / "pipeline" / "review.py"), "apply",
        "--staging-dir", str(staging), "--skill", str(skill),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (skill.with_suffix(".md.bak")).exists() or Path(str(skill) + ".bak").exists()
    content = skill.read_text(encoding="utf-8")
    assert "Remote Host Operations" in content
    # staging should have been moved to adopted/
    assert not staging.exists()
    assert (out_base / "adopted").is_dir()


def test_apply_creates_backup(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, timeout=10)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo), capture_output=True, timeout=10)
    subprocess.run(["git", "config", "user.name", "t"], cwd=str(repo), capture_output=True, timeout=10)
    skill = repo / "SKILL.md"
    skill.write_text("a\nb\n", encoding="utf-8")
    subprocess.run(["git", "add", "SKILL.md"], cwd=str(repo), capture_output=True, timeout=10)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True, timeout=10)

    staging = tmp_path / "out" / "staging" / "s-20260820-000000"
    staging.mkdir(parents=True)
    diff_text = (
        "diff --git a/SKILL.md b/SKILL.md\n"
        "--- a/SKILL.md\n"
        "+++ b/SKILL.md\n"
        "@@ -1,2 +1,3 @@\n"
        " a\n"
        " b\n"
        "+c\n"
    )
    (staging / "candidate.diff").write_text(diff_text, encoding="utf-8")

    cmd = [sys.executable, str(ROOT / "pipeline" / "review.py"), "apply", "--staging-dir", str(staging), "--skill", str(skill)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, proc.stderr
    bak = Path(str(skill) + ".bak")
    assert bak.exists()
    assert bak.read_text(encoding="utf-8") == "a\nb\n"


def test_apply_missing_staging_fails(tmp_path):
    cmd = [
        sys.executable, str(ROOT / "pipeline" / "review.py"), "apply",
        "--staging-dir", str(tmp_path / "nope"), "--skill", str(tmp_path / "SKILL.md"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert proc.returncode != 0
    assert "ERROR" in proc.stderr


def test_apply_bad_diff_fails(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, timeout=10)
    skill = repo / "SKILL.md"
    skill.write_text("hello\n", encoding="utf-8")
    staging = tmp_path / "out" / "staging" / "s-20260820-000000"
    staging.mkdir(parents=True)
    (staging / "candidate.diff").write_text("not a valid diff at all\n", encoding="utf-8")
    cmd = [sys.executable, str(ROOT / "pipeline" / "review.py"), "apply", "--staging-dir", str(staging), "--skill", str(skill)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    assert proc.returncode != 0
    assert "ERROR" in proc.stderr


# ── review reject ───────────────────────────────────────────────────────────


def test_reject_moves_to_rejected(tmp_path):
    out = tmp_path / "out"
    staging = out / "staging" / "hermes-agent-20260820-114304"
    staging.mkdir(parents=True)
    (staging / "candidate.diff").write_text(_sample_diff(), encoding="utf-8")
    (staging / "validation.json").write_text(
        json.dumps({"skill_path": "/tmp/SKILL.md", "overall_passed": True}, ensure_ascii=False), encoding="utf-8"
    )

    cmd = [
        sys.executable, str(ROOT / "pipeline" / "review.py"), "reject",
        "--staging-dir", str(staging), "--reason", "not needed",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stderr
    assert not staging.exists()
    assert (out / "rejected").is_dir()
    rejected_entries = list((out / "rejected").glob("hermes-agent-*"))
    assert len(rejected_entries) == 1
    assert (out / "rejected" / "rejected.jsonl").exists()
    assert "not needed" in (out / "rejected" / "rejected.jsonl").read_text(encoding="utf-8")


def test_reject_default_reason(tmp_path):
    out = tmp_path / "out"
    staging = out / "staging" / "s-20260820-000000"
    staging.mkdir(parents=True)
    (staging / "candidate.diff").write_text(_sample_diff(), encoding="utf-8")

    cmd = [sys.executable, str(ROOT / "pipeline" / "review.py"), "reject", "--staging-dir", str(staging)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stderr
    assert (out / "rejected" / "rejected.jsonl").exists()


def test_reject_missing_staging_fails(tmp_path):
    cmd = [
        sys.executable, str(ROOT / "pipeline" / "review.py"), "reject",
        "--staging-dir", str(tmp_path / "nope"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert proc.returncode != 0


# ── CLI --help ──────────────────────────────────────────────────────────────


def test_help_top():
    cmd = [sys.executable, str(ROOT / "pipeline" / "review.py"), "--help"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0
    assert "stage" in proc.stdout
    assert "apply" in proc.stdout
    assert "reject" in proc.stdout


def test_help_stage():
    cmd = [sys.executable, str(ROOT / "pipeline" / "review.py"), "stage", "--help"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0
    assert "--validation" in proc.stdout
    assert "--diff" in proc.stdout


def test_help_apply():
    cmd = [sys.executable, str(ROOT / "pipeline" / "review.py"), "apply", "--help"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0
    assert "--staging-dir" in proc.stdout


def test_help_reject():
    cmd = [sys.executable, str(ROOT / "pipeline" / "review.py"), "reject", "--help"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0
    assert "--staging-dir" in proc.stdout


def test_no_args_exits():
    cmd = [sys.executable, str(ROOT / "pipeline" / "review.py")]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert proc.returncode != 0

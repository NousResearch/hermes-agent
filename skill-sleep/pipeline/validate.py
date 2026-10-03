"""skill-sleep VALIDATE: LLM judge gate for candidate diffs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Allow `python3 pipeline/validate.py` direct execution and `python3 -m pipeline.validate`
try:
    from lib.validation import ValidationItem, ValidationResult  # type: ignore
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lib.validation import ValidationItem, ValidationResult  # type: ignore

# ── Constants ───────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_THRESHOLD = 70
DEFAULT_MIN_PASS_RATE = 0.6
DEFAULT_TIMEOUT = 300
TEMPLATE_REL = Path("templates/judge_prompt.md")
LIMITATION_TEXT = (
    "No real execution replay — LLM judge evaluates candidate diff's "
    "expected impact on friction. This is weaker than SkillOpt paper's "
    "benchmark-gated validation."
)


# ── Loading ─────────────────────────────────────────────────────────────────


def load_tasks(tasks_path: str) -> dict:
    p = Path(tasks_path)
    if not p.exists():
        print(f"ERROR: tasks file not found: {tasks_path}", file=sys.stderr)
        sys.exit(1)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"ERROR: invalid tasks JSON: {e}", file=sys.stderr)
        sys.exit(1)


def load_diff(diff_path: str) -> str:
    p = Path(diff_path)
    if not p.exists():
        print(f"ERROR: diff file not found: {diff_path}", file=sys.stderr)
        sys.exit(1)
    text = p.read_text(encoding="utf-8")
    if not text.strip():
        print("ERROR: diff is empty", file=sys.stderr)
        sys.exit(1)
    return text


def load_proposal(proposal_path: str | None) -> dict | None:
    if not proposal_path:
        return None
    p = Path(proposal_path)
    if not p.exists():
        print(f"WARN: proposal file not found: {proposal_path} — continuing without it", file=sys.stderr)
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"WARN: invalid proposal JSON: {e} — continuing without it", file=sys.stderr)
        return None


# ── Prompt rendering ────────────────────────────────────────────────────────


def render_judge_prompt(
    template_path: str,
    user_request: str,
    friction_evidence: str,
    candidate_diff: str,
    threshold: int,
) -> str:
    p = Path(template_path)
    if not p.exists():
        print(f"ERROR: judge template not found: {template_path}", file=sys.stderr)
        sys.exit(1)
    text = p.read_text(encoding="utf-8")
    result = text.replace("{user_request}", user_request)
    result = result.replace("{friction_evidence}", friction_evidence)
    result = result.replace("{candidate_diff}", candidate_diff)
    result = result.replace("{threshold}", str(threshold))
    return result


def extract_request_and_evidence(task: dict) -> tuple[str, str]:
    req = str(task.get("user_request") or task.get("request") or "").strip()
    if not req:
        req = "(no user_request recorded)"
    ev = task.get("friction_evidence") or task.get("evidence") or []
    if isinstance(ev, list):
        evidence = "\n".join(str(x) for x in ev)
    else:
        evidence = str(ev)
    if not evidence.strip():
        evidence = "(no friction evidence recorded)"
    return req, evidence


# ── omp call ────────────────────────────────────────────────────────────────


def call_omp(
    prompt_path: Path,
    workdir: str,
    model: str,
    timeout: int,
) -> str:
    """Call `omp -p --cwd <workdir> --model <model> @prompt` and return stdout."""
    cmd = [
        "omp",
        "-p",
        "--cwd",
        workdir,
        "--model",
        model,
        f"@{prompt_path}",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "HERMES_NO_COLOR": "1"},
        )
    except FileNotFoundError:
        print("ERROR: 'omp' not found in PATH", file=sys.stderr)
        print("       Install: npm i -g @mariozechner/pi-coding-agent  (provides `omp`)", file=sys.stderr)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"ERROR: omp timed out after {timeout}s", file=sys.stderr)
        sys.exit(1)

    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        print(f"WARN: omp exited {proc.returncode}: {(proc.stderr or '')[:500]}", file=sys.stderr)
    if not output.strip():
        print("ERROR: omp produced no output", file=sys.stderr)
        sys.exit(1)
    return output


# ── Judge parsing ───────────────────────────────────────────────────────────


def parse_judge_output(raw: str, threshold: int) -> tuple[int, bool, str]:
    """Extract {score, passed, reason} from LLM judge output.

    Tries JSON extraction first (fenced or inline), falls back to regex.
    Returns (score, passed, reason).
    """
    text = raw.strip()

    # 1) Try to find a JSON object with score field
    # Look for fenced ```json ... ``` first, then bare {...}
    candidates: list[str] = []
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
        candidates.append(m.group(1))
    # Also try bare JSON objects (only ones containing "score")
    for m in re.finditer(r"\{[^{}]*\"score\"[^{}]*\}", text, re.DOTALL):
        blob = m.group(0)
        if blob not in candidates:
            candidates.append(blob)
    # Last resort: the largest {...} block
    if not candidates:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            candidates.append(m.group(0))

    for blob in candidates:
        try:
            obj = json.loads(blob)
            score = int(obj.get("score", 0))
            score = max(0, min(100, score))
            passed_raw = obj.get("passed", None)
            if isinstance(passed_raw, bool):
                passed = passed_raw
            elif isinstance(passed_raw, str):
                passed = passed_raw.lower() in ("true", "1", "yes")
            else:
                passed = score >= threshold
            reason = str(obj.get("reason") or obj.get("explanation") or "").strip()
            if not reason:
                reason = "judge returned score without reason"
            return score, passed, reason[:500]
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    # 2) Regex fallback: score: 85 / passed: true
    m_score = re.search(r"score\s*[:=]\s*(\d+)", text, re.IGNORECASE)
    m_passed = re.search(r"passed\s*[:=]\s*(true|false|yes|no|1|0)", text, re.IGNORECASE)
    if m_score:
        try:
            score = max(0, min(100, int(m_score.group(1))))
            if m_passed:
                p = m_passed.group(1).lower()
                passed = p in ("true", "yes", "1")
            else:
                passed = score >= threshold
            # reason = remaining text after score line, trimmed
            reason = text[m_score.end():].strip()[:500] or "parsed via regex fallback"
            # strip leading punctuation
            reason = re.sub(r"^[\s,:\-–—]+", "", reason)
            if len(reason) < 4:
                reason = "parsed via regex fallback"
            return score, passed, reason[:500]
        except ValueError:
            pass

    # 3) Complete fallback: treat as low confidence
    snippet = text[:300].replace("\n", " ").strip()
    return 0, False, f"could not parse judge output: {snippet[:200]}"


# ── Aggregation ─────────────────────────────────────────────────────────────


def aggregate(
    items: list[ValidationItem],
    threshold: int,
    min_pass_rate: float,
) -> tuple[bool, str | None]:
    total = len(items)
    if total == 0:
        return False, "no tasks to validate"
    passed = sum(1 for it in items if it.passed)
    rate = passed / total if total else 0.0
    if passed == total:
        return True, None
    if rate >= min_pass_rate:
        # >= threshold counts as PASS per spec: pass_rate >= min_pass_rate → PASS
        return True, None
    # FAIL — build rejected_reason for rejected buffer
    failed = [it for it in items if not it.passed]
    reasons = "; ".join(f"task {it.task_index}: {it.reason}" for it in failed[:3])
    return False, f"pass_rate {rate:.2f} < {min_pass_rate:.2f} — {reasons}"[:2000]


# ── Output writing ──────────────────────────────────────────────────────────


def write_validation(result: ValidationResult, output_dir: str) -> str:
    out = Path(output_dir) / "validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.to_json(indent=2) + "\n", encoding="utf-8")
    return str(out)


# ── CLI ─────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="skill-sleep VALIDATE: LLM judge gate for candidate diffs")
    p.add_argument("--tasks", required=False, default=None, help="Path to tasks.json (from MINE stage)")
    p.add_argument("--diff", required=False, default=None, help="Path to candidate.diff (from PROPOSE stage)")
    p.add_argument("--proposal", required=False, default=None, help="Path to proposal.json (from PROPOSE stage)")
    p.add_argument("--output-dir", required=False, default=".", help="Output directory for validation.json")
    p.add_argument("--model", required=False, default=DEFAULT_MODEL, help=f"omp model (default: {DEFAULT_MODEL})")
    p.add_argument("--template", required=False, default=None, help="Prompt template path (default: templates/judge_prompt.md)")
    p.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help=f"Per-task score threshold (default: {DEFAULT_THRESHOLD})")
    p.add_argument("--pass-rate", dest="min_pass_rate", type=float, default=DEFAULT_MIN_PASS_RATE, help=f"Minimum pass rate to gate PASS (default: {DEFAULT_MIN_PASS_RATE})")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"Timeout per omp call in seconds (default: {DEFAULT_TIMEOUT})")
    p.add_argument("--dry-run", action="store_true", help="Skip omp calls; use heuristic scoring for testing")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if not args.dry_run:
        if not os.environ.get("NINEROUTER_KEY"):
            print("ERROR: NINEROUTER_KEY not set", file=sys.stderr)
            print("      omp (muse-spark) requires NINEROUTER_KEY in env.", file=sys.stderr)
            print("      Export it or source ~/.zshenv:  export NINEROUTER_KEY=...", file=sys.stderr)
            sys.exit(1)

    # Resolve paths
    tasks_path = args.tasks
    if not tasks_path:
        for cand in [Path(args.output_dir) / "tasks.json", Path("tasks.json")]:
            if cand.exists():
                tasks_path = str(cand)
                break
        if not tasks_path:
            print("ERROR: --tasks is required (no tasks.json found)", file=sys.stderr)
            sys.exit(1)

    diff_path = args.diff
    if not diff_path:
        for cand in [Path(args.output_dir) / "candidate.diff", Path("candidate.diff")]:
            if cand.exists():
                diff_path = str(cand)
                break
        if not diff_path:
            print("ERROR: --diff is required (no candidate.diff found)", file=sys.stderr)
            sys.exit(1)

    proposal_path = args.proposal
    if not proposal_path:
        for cand in [Path(args.output_dir) / "proposal.json", Path("proposal.json")]:
            if cand.exists():
                proposal_path = str(cand)
                break

    print(f"[validate] Loading tasks from {tasks_path} ...")
    tasks_data = load_tasks(tasks_path)
    tasks: list[dict] = tasks_data.get("tasks", [])
    print(f"[validate] Got {len(tasks)} task card(s)")

    print(f"[validate] Loading diff from {diff_path} ...")
    candidate_diff = load_diff(diff_path)
    print(f"[validate] Diff: {len(candidate_diff)} chars")

    proposal_data = load_proposal(proposal_path) if proposal_path else None
    skill_path = ""
    if proposal_data:
        skill_path = str(proposal_data.get("skill_path", ""))
    if not skill_path:
        skill_path = str(tasks_data.get("skill_path", "")) or str(tasks_data.get("skill", ""))

    template_path = args.template or str(Path(__file__).resolve().parents[1] / TEMPLATE_REL)
    print(f"[validate] Threshold={args.threshold}, min_pass_rate={args.min_pass_rate}, model={args.model}")

    items: list[ValidationItem] = []

    if args.dry_run:
        print("[validate] Dry-run: using heuristic scoring (no omp calls)")
        for idx, task in enumerate(tasks):
            req, ev = extract_request_and_evidence(task)
            # Heuristic: if diff mentions any friction keyword, score high
            ev_lower = ev.lower()
            diff_lower = candidate_diff.lower()
            # Simple signal: overlap between evidence terms and diff
            score = 50
            if any(w in diff_lower for w in ["pitfall", "remote", "ssh", "scp", "deploy", "docker", "path"]):
                score = 80
            if any(w in ev_lower for w in ["pitfall", "remote", "path", "ssh", "scp"]) and any(
                w in diff_lower for w in ev_lower.split() if len(w) > 3
            ):
                score = 85
            # Empty request/evidence lowers score
            if req == "(no user_request recorded)":
                score = max(0, score - 20)
            passed = score >= args.threshold
            reason = "dry-run heuristic: diff appears relevant to friction" if passed else "dry-run heuristic: diff does not clearly address friction"
            items.append(ValidationItem(task_index=idx, score=score, passed=passed, reason=reason))
            print(f"[validate] Task {idx}: score={score} passed={passed}")
    else:
        for idx, task in enumerate(tasks):
            req, ev = extract_request_and_evidence(task)
            prompt_text = render_judge_prompt(template_path, req, ev, candidate_diff, args.threshold)
            with tempfile.TemporaryDirectory(prefix="skill-sleep-validate-") as tmpdir:
                prompt_file = Path(tmpdir) / "prompt.md"
                prompt_file.write_text(prompt_text, encoding="utf-8")
                print(f"[validate] Judging task {idx} via omp (timeout {args.timeout}s) ...")
                raw = call_omp(prompt_file, tmpdir, args.model, args.timeout)
                score, passed, reason = parse_judge_output(raw, args.threshold)
                print(f"[validate] Task {idx}: score={score} passed={passed} reason={reason[:120]}")
                items.append(ValidationItem(task_index=idx, score=score, passed=passed, reason=reason))

    overall_passed, rejected_reason = aggregate(items, args.threshold, args.min_pass_rate)
    total = len(items)
    passed_tasks = sum(1 for it in items if it.passed)
    pass_rate = (passed_tasks / total) if total else 0.0

    result = ValidationResult(
        generated_at=ValidationResult.now_iso(),
        skill_path=skill_path,
        diff_path=diff_path,
        gate_type="llm_judge",
        overall_passed=overall_passed,
        total_tasks=total,
        passed_tasks=passed_tasks,
        pass_rate=pass_rate,
        threshold=args.threshold,
        min_pass_rate=args.min_pass_rate,
        limitation=LIMITATION_TEXT,
        items=items,
        rejected_reason=rejected_reason,
    )

    out_path = write_validation(result, args.output_dir)
    print(f"[validate] Wrote {out_path}")
    gate_str = "PASS" if overall_passed else "FAIL"
    print(f"[validate] Gate: {gate_str} — {passed_tasks}/{total} passed (rate {pass_rate:.2f}, need {args.min_pass_rate:.2f})")
    print(f"[validate] Done — {result}")

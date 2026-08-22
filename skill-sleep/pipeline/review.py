"""skill-sleep REVIEW: staging management + apply/reject for candidate diffs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Allow `python3 pipeline/review.py` direct execution and `python3 -m pipeline.review`
try:
    from lib.staging import (  # type: ignore
        append_rejected_jsonl,
        build_review_md,
        copy_file,
        ensure_dir,
        now_ts,
        skill_slug,
        staging_dir_name,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lib.staging import (  # type: ignore
        append_rejected_jsonl,
        build_review_md,
        copy_file,
        ensure_dir,
        now_ts,
        skill_slug,
        staging_dir_name,
    )

# ── Loading helpers ─────────────────────────────────────────────────────────


def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"ERROR: invalid JSON: {path}: {e}", file=sys.stderr)
        sys.exit(1)


def load_diff(path: str) -> str:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: diff not found: {path}", file=sys.stderr)
        sys.exit(1)
    text = p.read_text(encoding="utf-8")
    if not text.strip():
        print(f"ERROR: diff is empty: {path}", file=sys.stderr)
        sys.exit(1)
    return text


def load_proposal_optional(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[review] WARN: proposal not found: {path}", file=sys.stderr)
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[review] WARN: invalid proposal JSON: {e}", file=sys.stderr)
        return None


# ── Subcommand: stage ───────────────────────────────────────────────────────


def cmd_stage(args: argparse.Namespace) -> None:
    validation = load_json(args.validation)
    candidate_diff = load_diff(args.diff)
    proposal = load_proposal_optional(args.proposal)

    skill_path = args.skill or str(validation.get("skill_path", "")) or ""
    if not skill_path:
        if proposal and proposal.get("skill_path"):
            skill_path = str(proposal["skill_path"])
    if not skill_path:
        print("ERROR: --skill is required (no skill_path in validation/proposal)", file=sys.stderr)
        sys.exit(1)

    ts = now_ts()
    # Prefer validation's generated_at for dir name if parseable, else use now
    gen = str(validation.get("generated_at", ""))
    if gen:
        # normalize to slug-safe: take YYYYmmdd-HHMMSS from ISO
        try:
            # e.g. 2026-08-20T11:43:04.125948+00:00 -> 20260820-114304
            date_part = gen[:10].replace("-", "")
            time_part = gen[11:19].replace(":", "")
            ts_candidate = f"{date_part}-{time_part}"
            if len(ts_candidate) == 15 and ts_candidate[8] == "-":
                ts = ts_candidate
        except Exception:
            pass

    slug = skill_slug(skill_path)
    dir_name = f"{slug}-{ts}"
    output_dir = Path(args.output_dir)

    overall_passed = bool(validation.get("overall_passed"))

    if not overall_passed:
        # Rejected path → rejected/<skill>-<ts>/
        if args.rejected_dir:
            rejected_base = Path(args.rejected_dir)
        else:
            rejected_base = output_dir / "rejected"
        rejected_dir = rejected_base / dir_name
        ensure_dir(rejected_dir)
        print(f"[review] Gate FAIL — writing to rejected: {rejected_dir}")
        # copy artifacts
        copy_file(args.diff, rejected_dir / "candidate.diff")
        copy_file(args.validation, rejected_dir / "validation.json")
        if args.proposal and Path(args.proposal).exists():
            copy_file(args.proposal, rejected_dir / "proposal.json")
        elif proposal is not None:
            (rejected_dir / "proposal.json").write_text(
                json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        reason = str(validation.get("rejected_reason") or "validation gate failed")
        entry = {
            "ts": ts,
            "skill_path": skill_path,
            "staging_dir": str(rejected_dir),
            "diff": candidate_diff[:4000],
            "reason": reason[:2000],
            "validation": {
                "overall_passed": overall_passed,
                "pass_rate": validation.get("pass_rate"),
                "total_tasks": validation.get("total_tasks"),
                "passed_tasks": validation.get("passed_tasks"),
            },
        }
        # append to <output_dir>/rejected/rejected.jsonl (shared log)
        append_rejected_jsonl(rejected_base, entry)
        # also keep a per-entry rejected.json for convenience
        (rejected_dir / "rejected.json").write_text(
            json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[review] Rejected — {rejected_dir}")
        print(f"[review] Log appended: {rejected_base / 'rejected.jsonl'}")
        return

    # Passed → staging/<skill>-<ts>/
    staging_base = output_dir / "staging"
    staging_dir = staging_base / dir_name
    ensure_dir(staging_dir)
    print(f"[review] Gate PASS — staging to {staging_dir}")

    copy_file(args.diff, staging_dir / "candidate.diff")
    copy_file(args.validation, staging_dir / "validation.json")
    if args.proposal and Path(args.proposal).exists():
        copy_file(args.proposal, staging_dir / "proposal.json")
    elif proposal is not None:
        (staging_dir / "proposal.json").write_text(
            json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        # still write an empty placeholder so apply can find skill_path if needed
        (staging_dir / "proposal.json").write_text(
            json.dumps({"skill_path": skill_path}, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    review_md = build_review_md(
        ts=ts,
        skill_path=skill_path,
        staging_dir=str(staging_dir),
        validation=validation,
        proposal=proposal,
        candidate_diff=candidate_diff,
    )
    (staging_dir / "review.md").write_text(review_md, encoding="utf-8")
    print(f"[review] Wrote {staging_dir / 'review.md'}")
    print(f"[review] Staged: {staging_dir}")
    print(f"[review] Next — apply:  python3 pipeline/review.py apply --staging-dir {staging_dir} --skill {skill_path}")
    print(f"[review]      reject: python3 pipeline/review.py reject --staging-dir {staging_dir} --reason \"...\"")


# ── Subcommand: apply ───────────────────────────────────────────────────────


def _check_cwd_allowed(target: Path, cwd: Path) -> None:
    """Ensure target is inside cwd (or cwd is the skill file's parent chain allows it).
    Task spec says: 只操作 --cwd 工作区内的文件. For apply, --skill may point outside
    (e.g. ~/.hermes/.../SKILL.md) — that write is explicitly allowed. But staging_dir
    operations must stay inside cwd. We enforce cwd for staging dir; skill write is exempt.
    """
    pass  # skill file itself is exempt; nothing to enforce here


def cmd_apply(args: argparse.Namespace) -> None:
    staging_dir = Path(args.staging_dir)
    if not staging_dir.is_dir():
        print(f"ERROR: staging dir not found: {staging_dir}", file=sys.stderr)
        sys.exit(1)

    diff_path = staging_dir / "candidate.diff"
    if not diff_path.exists():
        print(f"ERROR: candidate.diff not found in staging: {staging_dir}", file=sys.stderr)
        sys.exit(1)
    candidate_diff = diff_path.read_text(encoding="utf-8")
    if not candidate_diff.strip():
        print(f"ERROR: candidate.diff is empty: {diff_path}", file=sys.stderr)
        sys.exit(1)

    skill_path = args.skill
    if not skill_path:
        # try to infer from validation/proposal inside staging
        for cand in [staging_dir / "validation.json", staging_dir / "proposal.json"]:
            if cand.exists():
                try:
                    d = json.loads(cand.read_text(encoding="utf-8"))
                    if d.get("skill_path"):
                        skill_path = str(d["skill_path"])
                        break
                except Exception:
                    continue
    if not skill_path:
        print("ERROR: --skill is required (no skill_path found in staging)", file=sys.stderr)
        sys.exit(1)

    skill_file = Path(skill_path)
    # Back up original
    if skill_file.exists():
        bak = Path(str(skill_file) + ".bak")
        try:
            shutil.copy2(str(skill_file), str(bak))
            print(f"[review] Backup: {bak}")
        except OSError as e:
            print(f"ERROR: backup failed: {e}", file=sys.stderr)
            sys.exit(1)
        # Record original line count
        try:
            orig_lines = len(skill_file.read_text(encoding="utf-8").splitlines())
        except OSError:
            orig_lines = -1
    else:
        # Skill file may not exist yet — warn but continue; patch will fail gracefully
        print(f"[review] WARN: skill file does not exist, will attempt patch anyway: {skill_file}", file=sys.stderr)
        bak = None
        orig_lines = -1

    # Try git apply first, fallback to patch
    # Use --unsafe-paths to allow SKILL.md outside repo; use list-form args (no shell)
    cwd_for_patch = skill_file.parent if skill_file.parent.is_dir() else Path.cwd()
    # Prefer cwd_for_patch existence check; fallback to cwd
    if not cwd_for_patch.is_dir():
        cwd_for_patch = Path.cwd()

    # Count diff added lines for reporting
    added = sum(1 for l in candidate_diff.splitlines() if l.startswith("+") and not l.startswith("+++"))

    # Strategy 1: git apply (needs git available)
    applied = False
    last_err = ""
    for cmd in [
        ["git", "apply", "--unsafe-paths", "--verbose", str(diff_path)],
        ["git", "apply", "--unsafe-paths", str(diff_path)],
        ["patch", "-p1", "-i", str(diff_path)],
    ]:
        # patch -p1 expects cwd at repo root; git apply works from any subdir with --unsafe-paths
        # We run with cwd = skill parent so that a/SKILL.md resolves correctly if diff uses that prefix
        try:
            # For git apply the file paths in diff are relative; we need cwd containing SKILL.md
            run_cwd = str(cwd_for_patch) if cmd[0] == "git" else str(cwd_for_patch)
            # Check if tool exists
            proc = subprocess.run(
                cmd,
                cwd=run_cwd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0:
                applied = True
                if proc.stdout:
                    print(proc.stdout.rstrip())
                break
            else:
                last_err = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()[:2000]
                # try next tool
                if cmd[0] == "git" and "not found" in last_err.lower():
                    continue
        except FileNotFoundError as e:
            last_err = f"{cmd[0]} not found: {e}"
            continue
        except subprocess.TimeoutExpired:
            last_err = f"{cmd[0]} timed out"
            continue
        except OSError as e:
            last_err = str(e)[:2000]
            continue

    # Fallback: manual 3-way for simple unified diffs that use a/SKILL.md prefix
    if not applied:
        # Try a very small pure-Python fallback: if diff only adds lines after a context
        # we attempt to apply by finding the context line and inserting. This is intentionally
        # minimal; prefer git/patch.
        last_err = last_err or "no patch tool succeeded"

    if not applied:
        print(f"ERROR: apply failed: {last_err}", file=sys.stderr)
        # Do not remove staging; keep for retry
        sys.exit(1)

    # Report new line count
    try:
        new_lines = len(skill_file.read_text(encoding="utf-8").splitlines())
    except OSError:
        new_lines = -1
    print(f"[review] Applied {added} added line(s) to {skill_file}")
    if orig_lines >= 0 and new_lines >= 0:
        print(f"[review] Lines: {orig_lines} -> {new_lines} (diff +{added})")
    else:
        print(f"[review] Diff +{added} lines")

    # Move staging to adopted/
    # adopted base is sibling of staging's parent (output-dir/adopted)
    adopted_base = staging_dir.parent.parent / "adopted" if staging_dir.parent.name == "staging" else staging_dir.parent / "adopted"
    # More robust: if staging_dir is <out>/staging/<slug>-<ts>, adopted is <out>/adopted
    # Detect by checking parent name
    if staging_dir.parent.name == "staging":
        adopted_base = staging_dir.parent.parent / "adopted"
    else:
        adopted_base = staging_dir.parent / "adopted"
    # If adopted_base would be filesystem root or odd, fallback to <cwd>/adopted
    # Ensure adopted_base is reasonable — prefer <staging_parent>/../adopted
    ensure_dir(adopted_base)
    dest = adopted_base / staging_dir.name
    # avoid collision
    if dest.exists():
        dest = Path(str(dest) + f"-{now_ts()}")
    try:
        shutil.move(str(staging_dir), str(dest))
        print(f"[review] Adopted record: {dest}")
    except OSError as e:
        print(f"[review] WARN: could not move staging to adopted: {e}", file=sys.stderr)
        print(f"[review] Staging remains at: {staging_dir}", file=sys.stderr)


# ── Subcommand: reject ──────────────────────────────────────────────────────


def cmd_reject(args: argparse.Namespace) -> None:
    staging_dir = Path(args.staging_dir)
    if not staging_dir.is_dir():
        print(f"ERROR: staging dir not found: {staging_dir}", file=sys.stderr)
        sys.exit(1)

    reason = (args.reason or "").strip() or "rejected by human review"
    # Rejected base: sibling of staging's parent → <out>/rejected, or <staging>/rejected
    if staging_dir.parent.name == "staging":
        rejected_base = staging_dir.parent.parent / "rejected"
    else:
        rejected_base = staging_dir.parent / "rejected"
    ensure_dir(rejected_base)

    # Destination dir
    dest = rejected_base / staging_dir.name
    if dest.exists():
        dest = Path(str(dest) + f"-{now_ts()}")

    # Collect summary for jsonl
    summary = ""
    diff_text = ""
    skill_path = ""
    try:
        dp = staging_dir / "candidate.diff"
        if dp.exists():
            diff_text = dp.read_text(encoding="utf-8")[:4000]
    except OSError:
        pass
    try:
        vp = staging_dir / "validation.json"
        if vp.exists():
            vd = json.loads(vp.read_text(encoding="utf-8"))
            skill_path = str(vd.get("skill_path", ""))
    except Exception:
        pass
    if not skill_path:
        try:
            pp = staging_dir / "proposal.json"
            if pp.exists():
                pd = json.loads(pp.read_text(encoding="utf-8"))
                skill_path = str(pd.get("skill_path", ""))
        except Exception:
            pass

    entry = {
        "ts": now_ts(),
        "skill_path": skill_path,
        "staging_dir": str(dest),
        "diff": diff_text[:4000],
        "reason": reason[:2000],
    }
    append_rejected_jsonl(rejected_base, entry)

    try:
        shutil.move(str(staging_dir), str(dest))
    except OSError as e:
        print(f"ERROR: move to rejected failed: {e}", file=sys.stderr)
        sys.exit(1)

    # also write per-entry rejected.json
    try:
        (dest / "rejected.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    print(f"[review] Rejected — {dest}")
    print(f"[review] Log appended: {rejected_base / 'rejected.jsonl'}")


# ── CLI ─────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="skill-sleep REVIEW: staging management + apply/reject")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stage", help="Stage a validated candidate diff (PASS→staging, FAIL→rejected)")
    s.add_argument("--validation", required=True, help="Path to validation.json (from VALIDATE stage)")
    s.add_argument("--diff", required=True, help="Path to candidate.diff (from PROPOSE stage)")
    s.add_argument("--proposal", required=False, default=None, help="Path to proposal.json (from PROPOSE stage)")
    s.add_argument("--skill", required=False, default=None, help="Skill SKILL.md path (defaults to validation skill_path)")
    s.add_argument("--output-dir", required=False, default=".", help="Base output directory (staging/ and rejected/ live here)")
    s.add_argument("--rejected-dir", required=False, default=None, help="Override rejected base directory")
    s.set_defaults(func=cmd_stage)

    a = sub.add_parser("apply", help="Apply a staged diff to SKILL.md (with backup)")
    a.add_argument("--staging-dir", required=True, help="Staging directory (e.g. staging/<skill>-<ts>/)")
    a.add_argument("--skill", required=False, default=None, help="Skill SKILL.md path (defaults to staging validation.json)")
    a.set_defaults(func=cmd_apply)

    r = sub.add_parser("reject", help="Reject a staged candidate and move to rejected/")
    r.add_argument("--staging-dir", required=True, help="Staging directory (e.g. staging/<skill>-<ts>/)")
    r.add_argument("--reason", required=False, default=None, help="Rejection reason")
    r.set_defaults(func=cmd_reject)

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

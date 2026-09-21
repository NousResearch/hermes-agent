#!/usr/bin/env python3
"""
KENSEI Governance - Cross-Reference Engine + Change Registry
Takes Denji's profile review JSON and produces:
  1. Cross-reference report (self-eval vs Denji vs audit)
  2. Change registry (per-profile SOUL.md/skills/tools diffs)

Usage: python3 governance-crossref.py --review review.json --output-dir logboard/
"""

import json
import os
import sys
import datetime as dt
from pathlib import Path
from collections import defaultdict

HERMES_HOME = Path(os.environ.get("HERMES_HOME", "/home/kensei/.hermes"))
LOGBOARD = HERMES_HOME / "governance" / "logboard"


def load_self_evals():
    """Load all self-eval files from the logboard."""
    evals = {}
    for f in LOGBOARD.glob("self-eval-*.md"):
        stem = f.stem.replace("self-eval-", "")
        # Strip trailing date if present: "default-2026-06-08" → "default"
        parts = stem.rsplit("-", 3)
        if len(parts) == 4 and all(p.isdigit() for p in parts[1:]):
            profile = parts[0]
        else:
            profile = stem
        content = f.read_text()
        
        # Parse markdown sections
        lines = content.split("\n")
        claims = []
        current_section = ""
        in_content = False
        
        for line in lines:
            stripped = line.strip()
            
            # Track which section we're in
            if stripped.startswith("## "):
                current_section = stripped.replace("## ", "").lower()
                in_content = True
                continue
            if stripped.startswith("# "):
                in_content = False
                continue
            
            if not in_content or not stripped:
                continue
            
            # Extract non-STUB claims from key sections
            if any(s in current_section for s in ["tasks completed", "observations", "recommended"]):
                if "[STUB]" not in stripped and stripped.startswith("- "):
                    claims.append(stripped)
            
            # Flag STUB presence
            if "[STUB]" in stripped:
                if "STUB" not in profile:
                    claims.append("⚠️ STUB content found - self-eval is pro forma, not evidence-grounded")
        
        # Classify the self-eval quality
        is_stub = "[STUB]" in content
        eval_type = "STUB (pro forma, no evidence)" if is_stub else "EVIDENCE (grounded in data)"
        
        # Extract headline from first non-header, non-empty line
        headline = "No self-eval content parsed"
        for line in lines:
            s = line.strip()
            if not s or s.startswith("---"):
                continue
            if s.startswith("#"):
                continue
            headline = s[:100]
            break
        
        evals[profile] = {
            "file": str(f),
            "claims": claims,
            "headline": headline,
            "eval_type": eval_type,
            "is_stub": is_stub,
            "date": dt.datetime.fromtimestamp(f.stat().st_mtime).strftime("%d/%m/%Y"),
        }
    return evals


def load_skill_audit():
    """Load the latest skill audit JSON export."""
    audit_files = sorted(LOGBOARD.glob("skill-audit-*.json"), reverse=True)
    if not audit_files:
        return {}
    with open(audit_files[0]) as f:
        return json.load(f)


def cross_reference(denji_review, self_evals, skill_audit):
    """Produce cross-reference table and change registry."""
    rows = []
    changes = []

    for profile, review in denji_review.get("profiles", {}).items():
        self_eval = self_evals.get(profile, {})
        audit = skill_audit.get(profile, {})

        # Self-eval summary extraction
        eval_headline = self_eval.get("headline", "No self-eval found")
        eval_is_stub = self_eval.get("is_stub", False)
        eval_is_positive = any(w in eval_headline.lower() for w in
                               ["all good", "healthy", "stable", "✅", "working"]) or not eval_is_stub

        # Denji review scores
        denji_score = review.get("total_score", 0)
        honesty = review.get("honesty_score", 0)
        audit_score = audit.get("score", 0) if isinstance(audit, dict) else 0
        decision = review.get("decision", "UNKNOWN")

        # Mismatch detection
        mismatch_type = "aligned"
        if eval_is_stub:
            mismatch_type = "eval_quality"  # STUB = eval quality gap, not content mismatch
        elif eval_is_positive and denji_score < 50:
            mismatch_type = "major"
        elif eval_is_positive and denji_score < 70:
            mismatch_type = "minor"
        elif honesty < 10 and denji_score > 60:
            mismatch_type = "eval_quality"  # good profile, bad eval

        rows.append({
            "profile": profile,
            "self_eval": eval_headline[:80],
            "eval_type": self_eval.get("eval_type", "UNKNOWN"),
            "is_stub": eval_is_stub,
            "denji_score": denji_score,
            "audit_score": audit_score,
            "honesty": honesty,
            "decision": decision,
            "mismatch": mismatch_type,
        })

        # Change registry
        if decision in ("REWORK", "INTERVENE"):
            profile_changes = review.get("changes", {})
            if profile_changes:
                changes.append({
                    "profile": profile,
                    "score": denji_score,
                    "decision": decision,
                    "changes": profile_changes,
                })

    return rows, changes


def render_report(now, rows, changes):
    """Render cross-reference report as markdown."""
    healthy = [r for r in rows if r["decision"] == "HEALTHY"]
    observe = [r for r in rows if r["decision"] == "OBSERVE"]
    rework = [r for r in rows if r["decision"] == "REWORK"]
    intervene = [r for r in rows if r["decision"] == "INTERVENE"]

    lines = [
        f"# Governance Cross-Reference - {now.strftime('%d/%m/%Y')}",
        "",
        f"Active profiles: {len(rows)} | Healthy: {len(healthy)} | Observe: {len(observe)} | Rework: {len(rework)} | Intervene: {len(intervene)}",
        "",
    ]

    # Score table
    lines.append("## Profile Score Table")
    lines.append("")
    lines.append("| Profile | Denji Score | Audit Score | Honesty | Decision | Self-Eval Type | Self-Eval Says |")
    lines.append("|---------|------------|-------------|---------|----------|----------------|---------------|")
    for r in sorted(rows, key=lambda x: -x["denji_score"]):
        eval_preview = r["self_eval"][:50] if not r.get("is_stub") else "⚠️ STUB"
        lines.append(f"| {r['profile']} | {r['denji_score']} | {r['audit_score']} | {r['honesty']}/20 | {r['decision']} | {r.get('eval_type', '?')} | {eval_preview} |")
    lines.append("")

    # Mismatches
    mismatches = [r for r in rows if r["mismatch"] != "aligned"]
    if mismatches:
        lines.append("## Cross-Reference Mismatches")
        lines.append("")
        for r in mismatches:
            emoji = "🔴" if r["mismatch"] == "major" else "🟡"
            lines.append(f"- {emoji} **{r['profile']}** - self-eval says \"{r['self_eval'][:60]}\", Denji scores {r['denji_score']}/100")
        lines.append("")

    # Change registry
    if changes:
        lines.append("## Change Registry")
        lines.append("")
        for c in changes:
            lines.append(f"### {c['profile']} - Score {c['score']}/100 ({c['decision']})")
            lines.append("")
            for domain, items in c["changes"].items():
                lines.append(f"**{domain}:**")
                for item in items:
                    lines.append(f"- [ ] {item}")
                lines.append("")
            lines.append("")

    # Theatre detection
    theatre = [r for r in rows if r["mismatch"] == "major" and r["honesty"] < 8]
    if theatre:
        lines.append("## ⚠️ Theatre Detection")
        lines.append("")
        lines.append("These profiles rate themselves healthy but Denji's evidence-based review disagrees significantly:")
        for t in theatre:
            lines.append(f"- **{t['profile']}** - self-eval positive, Denji {t['denji_score']}/100, honesty {t['honesty']}/20")

    return "\n".join(lines)


def main():
    review_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not review_path:
        print("Usage: governance-crossref.py <denji-review.json>")
        sys.exit(1)

    with open(review_path) as f:
        denji_review = json.load(f)

    self_evals = load_self_evals()
    skill_audit = load_skill_audit()
    rows, changes = cross_reference(denji_review, self_evals, skill_audit)

    now = dt.datetime.now()
    report = render_report(now, rows, changes)

    report_path = LOGBOARD / f"cross-ref-{now.strftime('%Y%m%d')}.md"
    report_path.write_text(report)

    # Print summary for cron delivery. Speak only when there is drift to act on;
    # an all-aligned run is silent (the full report is still written to disk).
    healthy = len([r for r in rows if r["decision"] == "HEALTHY"])
    mismatches = len([r for r in rows if r["mismatch"] != "aligned"])
    if mismatches == 0:
        print("[SILENT]")
        return
    print(f"🔴 Governance cross-ref - {mismatches} mismatch(es) - {now.strftime('%d/%m/%Y %H:%M')}")
    print(f"{len(rows)} profiles | {healthy} healthy | {mismatches} mismatches")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

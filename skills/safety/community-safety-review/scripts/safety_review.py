#!/usr/bin/env python3
"""safety_review.py — lightweight community safety review for Web3/GameFi projects.

Reads public project information from a JSON file (or stdin) and produces a
neutral, structured **safety review note** in Markdown. It detects public
*risk signals* and *trust signals*, lists what information is missing, and
assigns a review level: LOW / MEDIUM / HIGH / HUMAN REVIEW REQUIRED.

Positioning (important):
  - This tool NEVER claims a project is a scam.
  - It identifies public risk signals only and produces a human-review note.
  - It uses public, reviewer-supplied information only — no private data,
    no scraping, no network calls, no automated enforcement.
  - Every report ends with a manual-verification reminder.

The "review level" describes how much careful human review is recommended.
It is not a verdict about a project or any person.

Usage:
    python safety_review.py sample-project.json
    python safety_review.py sample-project.json --report
    cat project.json | python safety_review.py -

Input JSON (all fields optional; provide what you have):
    {
      "project_name": "Example Project",
      "sources": ["https://example.com", "https://x.com/example"],
      "team": "Anonymous team, no names listed",
      "team_public": false,
      "documentation": "https://docs.example.com",
      "demo": "https://example.com/play",
      "github": "https://github.com/example/example",
      "github_active": true,
      "audit": "",
      "roadmap": "Q3: launch",
      "tokenomics": "unclear",
      "community": "Active Discord with real discussion",
      "description": "Free text describing the project ...",
      "marketing_text": "Promotional copy to scan for risky language ...",
      "reviewer_flags": ["broken_links"]
    }

`reviewer_flags` lets a human record signals that cannot be detected from text
alone (e.g. "broken_links", "copied_content", "fake_social_proof"). See
REVIEWER_FLAG_LABELS below for the recognised values.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# --- Language-based risk signals -------------------------------------------
# Each entry: key -> (human label, weight, [trigger phrases]).
# Phrases are matched case-insensitively with word boundaries, so short words
# do not match inside unrelated words (e.g. "ico" will not match "unicorn").
# A "strong" signal has weight 2; a softer/structural one has weight 1.

LANGUAGE_SIGNALS: dict[str, tuple[str, int, tuple[str, ...]]] = {
    "guaranteed_profit": (
        "Guaranteed-profit language",
        2,
        (
            "guaranteed profit", "guaranteed return", "guaranteed returns",
            "guaranteed apy", "risk-free", "risk free", "no risk", "can't lose",
            "cannot lose", "assured returns", "guaranteed gains",
        ),
    ),
    "unrealistic_rewards": (
        "Unrealistic reward claims",
        2,
        (
            "100x", "1000x", "10000%", "1000% apy", "get rich", "double your",
            "triple your", "to the moon", "moonshot", "life-changing money",
            "passive income for life", "huge gains",
        ),
    ),
    "referral_focus": (
        "Aggressive referral / recruitment focus",
        1,
        (
            "refer a friend", "referral bonus", "referral reward", "invite and earn",
            "affiliate bonus", "downline", "recruit", "mlm", "multi-level",
            "earn for every friend",
        ),
    ),
    "fake_partnership": (
        "Unverified partnership / endorsement claims",
        1,
        (
            "official partner", "backed by", "partnered with", "in partnership with",
            "endorsed by", "as featured on", "as seen on",
        ),
    ),
    "suspicious_token_sale": (
        "Suspicious token-sale wording",
        2,
        (
            "presale", "pre-sale", "private sale", "buy now before", "fair launch guaranteed",
            "token sale ends", "ico", "ido", "guaranteed allocation", "buy before it's too late",
        ),
    ),
    "pressure_tactics": (
        "Pressure tactics (urgency / scarcity)",
        2,
        (
            "limited time only", "limited time", "act now", "hurry", "only today",
            "last chance", "spots filling fast", "don't miss out", "dont miss out",
            "ends soon", "24 hours only", "before it's gone",
        ),
    ),
    "fake_social_proof": (
        "Possible fabricated social proof",
        2,
        (
            "thousands of happy investors", "trusted by millions", "join millions",
            "everyone is buying", "viral sensation", "fastest growing in history",
        ),
    ),
}

LANGUAGE_PATTERNS: dict[str, list[tuple[str, re.Pattern]]] = {
    key: [(p, re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE)) for p in phrases]
    for key, (_, _, phrases) in LANGUAGE_SIGNALS.items()
}

# Manual flags a human reviewer can supply for things text scanning cannot see.
REVIEWER_FLAG_LABELS: dict[str, tuple[str, int]] = {
    "broken_links": ("Broken or dead links reported", 1),
    "copied_content": ("Copied website / documentation language reported", 2),
    "fake_social_proof": ("Fabricated social proof reported", 2),
    "fake_partnership": ("Fake partnership claim reported", 2),
    "impersonation": ("Impersonation of a known project reported", 2),
}

# Words that mark a "team" field as anonymous / unclear.
ANON_MARKERS = ("anon", "anonymous", "unknown", "no team", "not listed", "hidden", "n/a")

# Words that mark tokenomics as unclear.
UNCLEAR_MARKERS = ("unclear", "unknown", "tbd", "n/a", "not disclosed", "none", "hidden")

# Key information categories used to decide whether there is enough public
# information to form any view at all.
INFO_FIELDS = (
    "sources", "team", "documentation", "demo", "github",
    "audit", "tokenomics", "roadmap", "community",
)


def _text(value) -> str:
    """Coerce a JSON value to a single lowercase-able string for scanning."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(_text(v) for v in value)
    return str(value)


def _has(value) -> bool:
    """True if a field carries usable, non-empty content."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    return bool(value)


def _looks_anonymous(team_value, team_public) -> bool:
    if team_public is True:
        return False
    if team_public is False:
        return True
    text = _text(team_value).lower()
    if not text.strip():
        return False  # absence is handled as "missing", not as a signal
    return any(marker in text for marker in ANON_MARKERS)


def _looks_unclear(value) -> bool:
    text = _text(value).lower()
    if not text.strip():
        return False
    return any(marker in text for marker in UNCLEAR_MARKERS)


def analyze(project: dict) -> dict:
    """Detect risk signals, trust signals, and missing information.

    Returns a dict with: risk_signals, trust_signals, missing, level, and a
    short explanation. Each signal is {"label", "evidence", "weight"}.
    """
    risk_signals: list[dict] = []
    trust_signals: list[str] = []

    # Combined free text used for language scanning.
    scan_blob = " ".join(
        _text(project.get(f))
        for f in ("description", "marketing_text", "documentation", "roadmap")
    )

    # 1) Language-based signals.
    for key, (label, weight, _phrases) in LANGUAGE_SIGNALS.items():
        hits = [phrase for phrase, pat in LANGUAGE_PATTERNS[key] if pat.search(scan_blob)]
        if hits:
            risk_signals.append({
                "label": label,
                "evidence": "matched: " + ", ".join(f'"{h}"' for h in hits[:4]),
                "weight": weight,
            })

    # 2) Reviewer-supplied manual flags.
    for raw in project.get("reviewer_flags") or []:
        flag = str(raw).strip().lower()
        if flag in REVIEWER_FLAG_LABELS:
            label, weight = REVIEWER_FLAG_LABELS[flag]
            risk_signals.append({
                "label": label,
                "evidence": "flagged by reviewer",
                "weight": weight,
            })

    # 3) Structural signals (presence / clarity of public information).
    team_public = project.get("team_public")
    if _looks_anonymous(project.get("team"), team_public):
        risk_signals.append({
            "label": "Anonymous or unclear team",
            "evidence": _text(project.get("team")) or "team marked non-public",
            "weight": 1,
        })
    elif team_public is True or (_has(project.get("team")) and not _looks_anonymous(project.get("team"), team_public)):
        trust_signals.append("Public team or named contributors")

    if not _has(project.get("documentation")):
        risk_signals.append({
            "label": "No public documentation provided",
            "evidence": "documentation field empty",
            "weight": 1,
        })
    else:
        trust_signals.append("Documentation available")

    if not _has(project.get("demo")):
        risk_signals.append({
            "label": "No working product/demo provided",
            "evidence": "demo field empty",
            "weight": 1,
        })
    else:
        trust_signals.append("Working demo / product link provided")

    github = project.get("github")
    github_active = project.get("github_active")
    if not _has(github):
        risk_signals.append({
            "label": "No GitHub provided",
            "evidence": "github field empty",
            "weight": 1,
        })
    elif github_active is False:
        risk_signals.append({
            "label": "Inactive GitHub reported",
            "evidence": f"{_text(github)} (marked inactive)",
            "weight": 1,
        })
    else:
        trust_signals.append("GitHub repository provided")
        if github_active is True:
            trust_signals.append("GitHub reported as active")

    if not _has(project.get("audit")):
        risk_signals.append({
            "label": "No audit / security information provided",
            "evidence": "audit field empty",
            "weight": 1,
        })
    else:
        trust_signals.append("Audit / security notes provided")

    if _looks_unclear(project.get("tokenomics")):
        risk_signals.append({
            "label": "Unclear tokenomics",
            "evidence": _text(project.get("tokenomics")),
            "weight": 1,
        })
    elif _has(project.get("tokenomics")):
        trust_signals.append("Transparent tokenomics described")

    if _has(project.get("roadmap")):
        trust_signals.append("Roadmap provided")
    if _has(project.get("community")):
        trust_signals.append("Community activity described")
    # Absence of guaranteed-profit language is itself a mild trust signal.
    if not any(s["label"].startswith("Guaranteed-profit") for s in risk_signals):
        trust_signals.append("No guaranteed-profit language detected")

    # 4) Missing information.
    missing = [f for f in INFO_FIELDS if not _has(project.get(f))]

    # 5) Review level.
    weight_total = sum(s["weight"] for s in risk_signals)
    strong_count = sum(1 for s in risk_signals if s["weight"] >= 2)
    info_present = len(INFO_FIELDS) - len(missing)

    if info_present < 3 and strong_count == 0:
        level = "HUMAN REVIEW REQUIRED"
        explanation = (
            "Not enough public information was provided to form any view. A human "
            "should gather more sources before assessing."
        )
    elif strong_count >= 2 or weight_total >= 6:
        level = "HIGH"
        explanation = (
            "Multiple notable risk signals were detected in the public information. "
            "Careful human review is strongly recommended before engaging."
        )
    elif weight_total >= 2:
        level = "MEDIUM"
        explanation = (
            "Some risk signals or gaps were detected. Manual verification is needed "
            "before reaching any conclusion."
        )
    else:
        level = "LOW"
        explanation = (
            "Few or no risk signals were detected in the public information, but "
            "human review is still recommended."
        )

    return {
        "risk_signals": risk_signals,
        "trust_signals": trust_signals,
        "missing": missing,
        "level": level,
        "explanation": explanation,
        "weight_total": weight_total,
        "strong_count": strong_count,
    }


def build_markdown(project: dict, result: dict, generated: str) -> str:
    """Render a neutral Markdown safety review note."""
    name = _text(project.get("project_name")).strip() or "Unnamed project"
    sources = project.get("sources") or []
    if isinstance(sources, str):
        sources = [sources]

    lines: list[str] = []
    lines.append("# Community Safety Review Note")
    lines.append("")
    lines.append(
        "> Neutral safety note based on **public information only**. This is "
        "**not** an accusation and **not** an enforcement action. It identifies "
        "risk signals for a human to verify. **Human review is required.**"
    )
    lines.append("")
    lines.append(f"**Project name:** {name}")
    lines.append(f"**Generated:** {generated}")
    lines.append(f"**Review level:** {result['level']}")
    lines.append("")

    lines.append("## Sources reviewed")
    lines.append("")
    if sources:
        for src in sources:
            lines.append(f"- {_text(src)}")
    else:
        lines.append("- none provided")
    lines.append("")

    lines.append("## Detected risk signals")
    lines.append("")
    if result["risk_signals"]:
        for sig in result["risk_signals"]:
            lines.append(f"- **{sig['label']}** — {sig['evidence']}")
    else:
        lines.append("- none detected in the provided public information")
    lines.append("")

    lines.append("## Positive trust signals")
    lines.append("")
    if result["trust_signals"]:
        for sig in result["trust_signals"]:
            lines.append(f"- {sig}")
    else:
        lines.append("- none detected in the provided public information")
    lines.append("")

    lines.append("## Missing information")
    lines.append("")
    if result["missing"]:
        for field in result["missing"]:
            lines.append(f"- {field.replace('_', ' ')}")
    else:
        lines.append("- none — all reviewed categories had some information")
    lines.append("")

    lines.append("## Explanation")
    lines.append("")
    lines.append(result["explanation"])
    lines.append("")

    lines.append("## Review level reference")
    lines.append("")
    lines.append("- **LOW** — few/no risk signals; human review still recommended.")
    lines.append("- **MEDIUM** — some signals or gaps; manual verification needed.")
    lines.append("- **HIGH** — multiple notable signals; careful human review strongly recommended.")
    lines.append("- **HUMAN REVIEW REQUIRED** — not enough public information to assess.")
    lines.append("")

    lines.append("## Final human-review note")
    lines.append("")
    lines.append(
        "This note was prepared from public, reviewer-supplied information to "
        "assist human reviewers. It does **not** accuse the project or any person, "
        "does **not** claim the project is a scam, and does **not** enforce any "
        "action. The detected signals are unverified and may have innocent "
        "explanations. A human must verify every signal against original sources "
        "and apply the community's own rules before making any decision. "
        "**Manual verification required.**"
    )
    lines.append("")
    return "\n".join(lines)


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "project"


def load_project(path: str) -> dict:
    """Load the project JSON from a file path or '-' for stdin."""
    if path == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("input JSON must be an object (a single project)")
    return data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a neutral community safety review note for a Web3/GameFi "
            "project from public information. Identifies risk signals only; never "
            "claims a project is a scam. Human review always required."
        )
    )
    parser.add_argument(
        "input",
        help="Path to the project JSON file, or '-' to read from stdin.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Also write the Markdown note to the reports/ folder.",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Directory for the report (default: ../reports next to this script).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Ensure UTF-8 output even on consoles that default to a legacy codepage
    # (e.g. Windows cp1252), so redirected/piped Markdown stays valid UTF-8.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    try:
        project = load_project(args.input)
    except FileNotFoundError:
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"error: could not read project JSON: {exc}", file=sys.stderr)
        return 2

    result = analyze(project)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    markdown = build_markdown(project, result, generated)

    print(markdown)

    if args.report:
        script_dir = Path(__file__).resolve().parent
        reports_dir = (
            Path(args.report_dir) if args.report_dir else script_dir.parent / "reports"
        )
        reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        slug = slugify(_text(project.get("project_name")))
        path = reports_dir / f"safety-review-{slug}-{stamp}.md"
        path.write_text(markdown, encoding="utf-8")
        print(f"\nReport written to {path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

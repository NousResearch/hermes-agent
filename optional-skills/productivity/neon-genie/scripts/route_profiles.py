#!/usr/bin/env python3
"""Suggest the smallest sufficient Neon Genie profile set from a request.

Packaging-only keyword router. Does not invent opportunities or run research.

Usage:
  python scripts/route_profiles.py --text "zero capital first cash"
  python scripts/route_profiles.py --request examples/product-audit.brief.yaml
  python scripts/route_profiles.py --text "audit" --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import paths as ng_paths  # noqa: E402

# Trigger phrases aligned with SKILL.md profile_router (case-insensitive substring).
PROFILE_TRIGGERS: dict[str, tuple[str, ...]] = {
    "product_architecture": (
        "product audit",
        "app design",
        "game design",
        "system design",
        "feature coherence",
        "product boundary",
        "product architecture",
        "product packet",
    ),
    "opportunity_mining": (
        "new venture",
        "unmet need",
        "market opportunity",
        "blocked transition",
        "opportunity mining",
        "weak signal",
        "roadmap",
        "business idea",
        "go into business",
        "for myself",
        "self-employed",
        "side project",
        "launch my",
        "turn my idea",
        "app idea",
        "product idea",
        "what should i build",
        "where do i start",
        "next steps for my idea",
        "how do i approach",
    ),
    "fragmentation": (
        "many portals",
        "repeated handoffs",
        "incompatible systems",
        "coordination problem",
        "fragmentation",
        "defrag",
    ),
    "zero_option": (
        "zero capital",
        "first cash",
        "immediate executable",
        "constrained launch",
        "zero option",
        "zero-option",
        "no capital",
        "limited money",
        "limited resources",
        "limited capital",
        "between jobs",
        "bootstrapped",
        "no budget",
        "can't afford",
        "cannot afford",
        "first revenue",
        "make money from",
    ),
    "agentic_services": (
        "agent workflow",
        "delegated outcome",
        "automation",
        "x402",
        "machine services",
        "agentic",
    ),
    "commercial": (
        "pricing",
        "buyer",
        "revenue",
        "costs",
        "market pressure",
        "business model",
        "commercial",
    ),
    "evidence_intelligence": (
        "grants",
        "boards",
        "philanthropy",
        "competitive research",
        "current external facts",
        "evidence intelligence",
        "market facts",
    ),
    "memetic": (
        "name",
        "hook",
        "pitch language",
        "public framing",
        "shareability",
        "memetic",
    ),
    "audit_delivery": (
        "client audit",
        "cost of inaction",
        "diagnostic package",
        "implementation offer",
        "audit delivery",
    ),
    "wayfinder_handoff": (
        "build plan",
        "engineering readiness",
        "execution packet",
        "wayfinder handoff",
        "wayfinder",
    ),
    "capital_sprint": (
        "capital sprint",
        "annual fund",
        "donation drive",
        "membership drive",
        "fundraising deadline",
        "501c3 campaign",
        "501(c)(3)",
        "impact object",
        "donor sprint",
        "nonprofit capital",
        "raise money by deadline",
    ),
}

# Prefer longer/phrase matches; avoid matching bare "name" alone unless word-ish.
WEAK_TRIGGERS = {"name", "hook", "buyer", "costs", "boards", "grants"}


def load_request_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    # Prefer preferred_profiles lines if present (YAML-ish list)
    return text


def preferred_from_yamlish(text: str) -> list[str]:
    """Extract preferred_profiles list items without requiring PyYAML."""
    lines = text.splitlines()
    out: list[str] = []
    in_block = False
    for line in lines:
        if re.match(r"^preferred_profiles\s*:", line):
            in_block = True
            inline = line.split(":", 1)[1].strip()
            if inline.startswith("[") and inline.endswith("]"):
                inner = inline[1:-1].strip()
                if inner:
                    out.extend(x.strip().strip("\"'") for x in inner.split(","))
                return [x for x in out if x]
            continue
        if in_block:
            m = re.match(r"^\s*-\s*[\"']?([a-z0-9_]+)[\"']?\s*$", line)
            if m:
                out.append(m.group(1))
                continue
            if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                break
            if line.strip() and not line.lstrip().startswith("-"):
                break
    return out


def match_profiles(text: str) -> list[str]:
    lower = text.lower()
    matched: list[str] = []
    for profile, triggers in PROFILE_TRIGGERS.items():
        for trig in sorted(triggers, key=len, reverse=True):
            if trig in WEAK_TRIGGERS:
                # word boundary for weak single tokens
                if re.search(rf"\b{re.escape(trig)}\b", lower):
                    matched.append(profile)
                    break
            elif trig in lower:
                matched.append(profile)
                break
    return matched


def ensure_privacy(profiles: list[str]) -> list[str]:
    """Always co-load privacy after core on every route result."""
    out = list(profiles)
    if "core" not in out:
        out = ["core"] + out
    if "privacy" not in out:
        # insert after core
        idx = out.index("core") + 1
        out = out[:idx] + ["privacy"] + out[idx:]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Route Neon Genie profiles")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Free-text request")
    src.add_argument("--request", type=Path, help="Path to request brief (yaml/text)")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument(
        "--auto-evidence",
        action="store_true",
        default=True,
        help="Note evidence_intelligence auto-load when external facts likely (default on)",
    )
    args = parser.parse_args(argv)

    preferred: list[str] = []
    if args.request:
        if not args.request.is_file():
            print(f"FAIL: request not found: {args.request}", file=sys.stderr)
            return 1
        raw = load_request_text(args.request)
        preferred = preferred_from_yamlish(raw)
        text = raw
    else:
        text = args.text or ""

    triggered = match_profiles(text)
    # Prefer explicit preferred_profiles when present
    selected = list(dict.fromkeys(["core", "privacy"] + preferred + triggered))

    known: list[str] = []
    try:
        known = list(
            json.loads(ng_paths.manifest_path().read_text(encoding="utf-8")).get("profiles") or []
        )
    except (OSError, json.JSONDecodeError, FileNotFoundError):
        known = []
    if known:
        unknown = [p for p in selected if p not in known]
        selected = [p for p in selected if p in known]
    else:
        unknown = []

    # Always co-load privacy after known-filter so it is never stripped as unknown
    # (privacy may land in manifest/profiles later; selected still requires it).
    selected = ensure_privacy(selected)

    notes: list[str] = []
    if "evidence_intelligence" not in selected and args.auto_evidence:
        notes.append(
            "evidence_intelligence auto-loads at runtime when external facts improve the result"
        )
    if unknown:
        notes.append(f"ignored unknown profiles: {', '.join(unknown)}")

    result = {
        "default": ["core", "privacy"],
        "preferred_profiles": preferred,
        "triggered": [p for p in triggered if p != "core"],
        "selected": selected,
        "notes": notes,
        "authority": "advisory_only",
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("Neon Genie profile route (advisory)")
        print(f"  selected: {', '.join(selected)}")
        if preferred:
            print(f"  preferred: {', '.join(preferred)}")
        if triggered:
            print(f"  triggered: {', '.join(p for p in triggered if p != 'core')}")
        for n in notes:
            print(f"  note: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

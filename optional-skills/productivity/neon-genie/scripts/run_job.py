#!/usr/bin/env python3
"""Operator-facing packaging run — normalize, route, workspace, receipt, envelope.

Does **not** invent product judgment (that stays in Hermes + SKILL.md).
Combines deterministic packaging stages for humans and orchestrators.

Usage:
  python scripts/run_job.py --recipe product-audit --out out/neon-genie/demo
  python scripts/run_job.py --brief examples/product-audit.brief.yaml --out out/...
  python scripts/run_job.py --text "zero capital first cash" --out out/...
  python scripts/run_job.py --brief path.yaml --packet p.json --type opportunity --validate

Exit codes:
  0 ok
  1 packaging failure (NG-PKG / NG-SCHEMA / recipe)
  2 usage
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
PY = sys.executable

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import build_envelope as be  # noqa: E402
import lineage  # noqa: E402
import recipe_common as rc  # noqa: E402
from recipe_run import RECIPES  # noqa: E402

# Map preferred profile → default recipe when --brief without --recipe
PROFILE_RECIPE_HINTS: list[tuple[str, str]] = [
    ("zero_option", "zero-option"),
    ("product_architecture", "product-audit"),
    ("fragmentation", "fragmentation"),
    # capital_sprint before commercial: capital briefs often also match commercial
    ("capital_sprint", "capital-sprint"),
    ("commercial", "commercial"),
    ("audit_delivery", "audit"),
    ("agentic_services", "agentic"),
    ("memetic", "memetic"),
    ("evidence_intelligence", "evidence"),
    ("opportunity_mining", "opportunity"),
    ("wayfinder_handoff", "product-audit"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fail(code: str, msg: str, exit_code: int = 1) -> int:
    print(f"FAIL: {code}: {msg}", file=sys.stderr)
    return exit_code


def write_hermes_next(out: Path, *, profiles: list[str], recipe: str | None) -> None:
    lines = [
        "# Hermes next steps (judgment lives in SKILL.md)",
        "",
        "This directory was prepared by the **packaging CLI** (`do run`).",
        "It does **not** invent opportunities or rewrite product intent.",
        "",
        "## Load the skill",
        "",
        "1. Open Hermes with Neon Genie loaded.",
        "2. Follow OPEN → ALIGN → ASCEND → CLEAR → SEAL.",
        f"3. Profiles suggested: `{', '.join(profiles)}`.",
        "4. Open `run-envelope.json` first when resuming this workspace.",
        "",
        "## Authority",
        "",
        "- advisory_only — no spend, publish, contact, or repo mutation",
        "- Label claims: OBSERVED | INFERRED | SPECULATIVE | NOT_COMPUTABLE",
        "- Public gap → research; private gap → DataRequest",
        "",
    ]
    if recipe:
        lines.append(f"Packaging recipe used: `{recipe}`.")
    else:
        lines.append(
            "No full packet recipe ran — only route + receipt + envelope scaffold."
        )
    (out / "HERMES_NEXT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def infer_recipe_from_route(route: dict[str, Any]) -> str | None:
    selected = list(route.get("selected") or [])
    for prof, recipe in PROFILE_RECIPE_HINTS:
        if prof in selected and recipe in RECIPES:
            return recipe
    return None


def run_scaffold(
    out: Path,
    *,
    brief: Path | None,
    text: str | None,
    route: dict[str, Any],
    packet: Path | None,
    packet_type: str | None,
    validate: bool,
) -> int:
    """Deterministic workspace without inventing domain packets."""
    out.mkdir(parents=True, exist_ok=True)
    profiles = list(route.get("selected") or ["core"])
    rc.write_json(out / "profile-route.json", route)

    request_doc: dict[str, Any] = {
        "created_at": utc_now(),
        "skill": "neon-genie",
        "authority": "advisory_only",
        "grants_execution": False,
        "text": text,
        "brief": rc.rel(brief) if brief else None,
        "selected_profiles": profiles,
        "note": "Packaging scaffold only — product judgment is Hermes + SKILL.md",
    }
    if brief and brief.is_file():
        dest = out / "brief.yaml"
        shutil.copy2(brief, dest)
        request_doc["brief_copied"] = "brief.yaml"
    rc.write_json(out / "request.json", request_doc)

    packets: list[Path] = []
    if packet:
        if not packet.is_file():
            return fail("NG-PKG-020", f"packet not found: {packet}")
        dest = out / packet.name
        if packet.resolve() != dest.resolve():
            shutil.copy2(packet, dest)
        packets.append(dest)
        if validate:
            ptype = packet_type or "opportunity"
            try:
                rc.validate_packet(dest, ptype)
            except RuntimeError as exc:
                return fail("NG-SCHEMA-010", str(exc))

    receipt_path = out / "run-receipt.json"
    try:
        rc.build_receipt(
            receipt_path,
            profiles,
            status="PROPOSED",
            promotion_state="RAW_SIGNAL",
            packets=packets or None,
            brief=brief,
        )
    except RuntimeError as exc:
        return fail("NG-PKG-021", f"receipt failed: {exc}")

    try:
        env_path = be.write_envelope(
            out,
            recipe=None,
            brief=brief,
            route=route,
            request_summary=text or (f"brief:{brief.name}" if brief else "do run scaffold"),
        )
    except Exception as exc:  # noqa: BLE001
        return fail("NG-SCHEMA-011", f"envelope failed: {exc}")

    write_hermes_next(out, profiles=profiles, recipe=None)
    env = json.loads(env_path.read_text(encoding="utf-8"))
    print("PASS: packaging run scaffold")
    print(f"  out: {out}")
    print(f"  profiles: {', '.join(profiles)}")
    print(f"  envelope: {env_path.name} ({env.get('run_id')})")
    print("  next: open HERMES_NEXT.md — judgment remains in Hermes + SKILL.md")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Neon Genie operator run (packaging only; advisory_only)"
    )
    parser.add_argument("--brief", type=Path, help="Request brief YAML")
    parser.add_argument("--text", help="Free-text request (routing only)")
    parser.add_argument(
        "--recipe",
        help=f"Named packaging recipe: {', '.join(sorted(RECIPES))}",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default out/neon-genie/run-<timestamp>)",
    )
    parser.add_argument(
        "--packet",
        type=Path,
        help="Optional external packet JSON to copy + validate",
    )
    parser.add_argument("--type", dest="packet_type", help="Packet type for --packet")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate --packet against schema",
    )
    parser.add_argument(
        "--no-auto-recipe",
        action="store_true",
        help="Do not pick a recipe from brief profiles; scaffold only",
    )
    args = parser.parse_args(argv)

    if not args.brief and not args.text and not args.recipe:
        return fail(
            "NG-PKG-022",
            "provide --recipe and/or --brief and/or --text",
            exit_code=2,
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or (SKILL_ROOT / "out" / "neon-genie" / f"run-{stamp}")
    if not out.is_absolute():
        out = SKILL_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    # --- Explicit recipe path ---
    recipe_name = args.recipe
    route: dict[str, Any] = {}

    brief = args.brief
    if brief and not brief.is_absolute():
        brief = SKILL_ROOT / brief

    if brief or args.text:
        if brief:
            if not brief.is_file():
                return fail("NG-PKG-023", f"brief not found: {brief}")
            try:
                route = rc.route_request(brief)
            except RuntimeError as exc:
                return fail("NG-ROUTE-001", str(exc))
        else:
            r = subprocess.run(
                [PY, str(SCRIPT_DIR / "neon_genie.py"), "do", "route", "--text", args.text or "", "--json"],
                cwd=SKILL_ROOT,
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                return fail("NG-ROUTE-001", r.stderr or r.stdout or "route failed")
            route = json.loads(r.stdout)

    if recipe_name is None and not args.no_auto_recipe and route:
        recipe_name = infer_recipe_from_route(route)

    if recipe_name:
        if recipe_name not in RECIPES:
            return fail(
                "NG-PKG-024",
                f"unknown recipe {recipe_name!r}; known: {', '.join(sorted(RECIPES))}",
            )
        try:
            code = RECIPES[recipe_name](out)
        except RuntimeError as exc:
            return fail("NG-PKG-025", str(exc))
        if code != 0:
            return code
        # Ensure Hermes next steps even after recipe
        profiles = list(route.get("selected") or [])
        if not profiles and (out / "profile-route.json").is_file():
            profiles = list(json.loads((out / "profile-route.json").read_text()).get("selected") or [])
        if not profiles:
            profiles = ["core"]
        write_hermes_next(out, profiles=profiles, recipe=recipe_name)
        if args.packet and args.validate:
            p = args.packet if args.packet.is_absolute() else SKILL_ROOT / args.packet
            try:
                rc.validate_packet(p, args.packet_type or "opportunity")
            except RuntimeError as exc:
                return fail("NG-SCHEMA-010", str(exc))
        print(f"PASS: do run (recipe={recipe_name})")
        print(f"  out: {out}")
        print(f"  open: {out / 'run-envelope.json'}")
        return 0

    # --- Scaffold only ---
    if not route:
        route = {"selected": ["core"], "note": "no route input"}
    return run_scaffold(
        out,
        brief=brief,
        text=args.text,
        route=route,
        packet=args.packet if args.packet is None or args.packet.is_absolute() else SKILL_ROOT / args.packet,
        packet_type=args.packet_type,
        validate=args.validate,
    )


if __name__ == "__main__":
    raise SystemExit(main())

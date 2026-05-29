#!/usr/bin/env python3
"""Build a review-only Signal Room rig acting contact sheet."""
from __future__ import annotations

import argparse
import html
import importlib.util
import json
from pathlib import Path
from typing import Any, NamedTuple


class ContactFrame(NamedTuple):
    filename: str
    label: str


REQUIRED_CONTACT_FRAMES = (
    ContactFrame("neutral_read.png", "neutral read"),
    ContactFrame("bill_shock.png", "bill shock"),
    ContactFrame("skeptical_point.png", "skeptical point"),
    ContactFrame("look_to_machine.png", "look to machine"),
    ContactFrame("lean_weight_shift.png", "lean / weight shift"),
    ContactFrame("mouth_closed.png", "mouth closed"),
    ContactFrame("mouth_a.png", "mouth a"),
    ContactFrame("mouth_o.png", "mouth o"),
)


def _load_rig_gate():
    gate_path = Path(__file__).with_name("signal_room_rig_acting_gate.py")
    spec = importlib.util.spec_from_file_location("signal_room_rig_acting_gate", gate_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _rel_href(image_path: Path, out_path: Path) -> str:
    try:
        return image_path.resolve().relative_to(out_path.resolve().parent).as_posix()
    except ValueError:
        return image_path.resolve().as_posix()


def _frame_tile(frame: ContactFrame, candidate_dir: Path, out_path: Path, index: int) -> str:
    columns = 2
    tile_w = 430
    tile_h = 330
    gap_x = 44
    gap_y = 34
    x = 88 + (index % columns) * (tile_w + gap_x)
    y = 292 + (index // columns) * (tile_h + gap_y)
    href = html.escape(_rel_href(candidate_dir / frame.filename, out_path), quote=True)
    label = html.escape(frame.label)
    filename = html.escape(frame.filename)
    return f"""
  <g class="tile" transform="translate({x} {y})">
    <rect width="{tile_w}" height="{tile_h}" rx="18" fill="#17232d" stroke="#d89a3a" stroke-opacity=".7" stroke-width="3"/>
    <image href="{href}" x="72" y="28" width="286" height="236" preserveAspectRatio="xMidYMid meet"/>
    <text x="28" y="288" class="label">{label}</text>
    <text x="28" y="316" class="file">{filename}</text>
  </g>"""


def build_contact_sheet_svg(candidate_dir: Path, out_path: Path, manifest: dict[str, Any] | None) -> str:
    license_status = manifest.get("license_status", "unrecorded") if manifest else "unrecorded"
    render_tool = manifest.get("render_tool", "unrecorded") if manifest else "unrecorded"
    candidate_name = manifest.get("candidate", candidate_dir.name) if manifest else candidate_dir.name
    tiles = "".join(
        _frame_tile(frame, candidate_dir, out_path, index) for index, frame in enumerate(REQUIRED_CONTACT_FRAMES)
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="1920" viewBox="0 0 1080 1920">
  <style>
    .title {{ fill: #f3eadb; font: 800 52px Arial, sans-serif; }}
    .meta {{ fill: #8fa0aa; font: 700 24px Arial, sans-serif; }}
    .label {{ fill: #f3eadb; font: 800 28px Arial, sans-serif; }}
    .file {{ fill: #8fa0aa; font: 600 18px Arial, sans-serif; }}
    .footer {{ fill: #f3eadb; font: 700 24px Arial, sans-serif; }}
  </style>
  <rect width="1080" height="1920" fill="#101820"/>
  <rect x="56" y="56" width="968" height="1808" fill="none" stroke="#d89a3a" stroke-opacity=".22" stroke-width="3"/>
  <text x="88" y="126" class="title">Signal Room rig acting review</text>
  <text x="88" y="174" class="meta">review-only / not public</text>
  <text x="88" y="218" class="meta">candidate: {html.escape(str(candidate_name))} | render: {html.escape(str(render_tool))}</text>
  {tiles}
  <text x="88" y="1770" class="footer">license: {html.escape(str(license_status))}</text>
  <text x="88" y="1810" class="meta">Gate: face and hands readable at phone size; gesture aims at machine area.</text>
</svg>
"""


def write_contact_sheet(candidate_dir: Path, out_path: Path) -> dict[str, Any]:
    rig_gate = _load_rig_gate()
    gate_result = rig_gate.evaluate_candidate_dir(candidate_dir)
    if not gate_result["passed"]:
        return {
            "passed": False,
            "candidate": candidate_dir.name,
            "output": str(out_path),
            "errors": gate_result["errors"],
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    svg = build_contact_sheet_svg(candidate_dir, out_path, gate_result["manifest"])
    out_path.write_text(svg)
    return {
        "passed": True,
        "candidate": candidate_dir.name,
        "output": str(out_path),
        "errors": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_dir", type=Path, help="candidate frame directory with rig_pass_manifest.json")
    parser.add_argument("--out", required=True, type=Path, help="write contact sheet SVG to this path")
    args = parser.parse_args()

    result = write_contact_sheet(args.candidate_dir, args.out)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

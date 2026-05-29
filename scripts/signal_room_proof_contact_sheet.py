#!/usr/bin/env python3
"""Build a proof retention contact sheet from sampled Signal Room frames."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def read_frame_plan(plan_path: Path) -> dict[str, Any]:
    return json.loads(plan_path.read_text())


def required_frame_ids(plan_path: Path) -> list[str]:
    plan = read_frame_plan(plan_path)
    return [str(frame["id"]) for frame in plan.get("frames", []) if frame.get("id")]


def _rel_href(image_path: Path, out_path: Path) -> str:
    try:
        return image_path.resolve().relative_to(out_path.resolve().parent).as_posix()
    except ValueError:
        return image_path.resolve().as_posix()


def _tile(frame: dict[str, Any], frames_dir: Path, out_path: Path, index: int) -> str:
    x = 70 + index * 360
    y = 164
    frame_id = str(frame["id"])
    filename = f"{frame_id}.png"
    href = html.escape(_rel_href(frames_dir / filename, out_path), quote=True)
    beat = html.escape(str(frame.get("beat", "")))
    question = html.escape(str(frame.get("review_question", "")))
    sample_time = html.escape(f"{float(frame.get('sample_time', 0.0)):g}s")
    return f"""
  <g transform="translate({x} {y})">
    <rect width="320" height="760" rx="16" fill="#17232d" stroke="#d89a3a" stroke-opacity=".72" stroke-width="3"/>
    <image href="{href}" x="34" y="28" width="252" height="448" preserveAspectRatio="xMidYMid meet"/>
    <text x="24" y="528" class="frame-id">{html.escape(frame_id)}</text>
    <text x="24" y="566" class="meta">{sample_time} / {beat}</text>
    <foreignObject x="24" y="594" width="272" height="120">
      <div xmlns="http://www.w3.org/1999/xhtml" class="question">{question}</div>
    </foreignObject>
    <text x="24" y="724" class="file">{html.escape(filename)}</text>
  </g>"""


def build_svg(plan: dict[str, Any], frames_dir: Path, out_path: Path) -> str:
    tiles = "".join(_tile(frame, frames_dir, out_path, index) for index, frame in enumerate(plan.get("frames", [])))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080">
  <style>
    .title {{ fill: #f3eadb; font: 800 44px Arial, sans-serif; }}
    .meta {{ fill: #8fa0aa; font: 700 20px Arial, sans-serif; }}
    .frame-id {{ fill: #f3eadb; font: 800 25px Arial, sans-serif; }}
    .file {{ fill: #8fa0aa; font: 600 16px Arial, sans-serif; }}
    .question {{ color: #f3eadb; font: 700 20px Arial, sans-serif; line-height: 1.18; }}
  </style>
  <rect width="1920" height="1080" fill="#101820"/>
  <text x="70" y="78" class="title">Signal Room proof retention contact sheet</text>
  <text x="70" y="118" class="meta">review-only: reject if these frames read as five similar cards</text>
  {tiles}
</svg>
"""


def write_proof_contact_sheet(plan_path: Path, frames_dir: Path, out_path: Path) -> dict[str, Any]:
    plan = read_frame_plan(plan_path)
    errors: list[str] = []
    for frame_id in required_frame_ids(plan_path):
        filename = f"{frame_id}.png"
        if not (frames_dir / filename).exists():
            errors.append(f"missing required frame: {filename}")

    if errors:
        return {"passed": False, "errors": errors, "output": str(out_path)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_svg(plan, frames_dir, out_path))
    return {"passed": True, "errors": [], "output": str(out_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = write_proof_contact_sheet(args.plan, args.frames, args.out)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

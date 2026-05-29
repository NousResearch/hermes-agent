#!/usr/bin/env python3
"""Create a static Signal Room review hub HTML page."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


DEFAULT_PRIMARY_REVIEW_FILES = {
    "watchable_draft": "fee_machine_v2_draft_with_audio.mp4",
    "contact_sheet": "proof_retention_contact_sheet.svg",
    "status_report": "REVIEW_STATUS.md",
    "motion_quality_plan": "MOTION_QUALITY_PASS.md",
    "motion_quality_plan_json": "motion_quality_plan.json",
    "scene_choreography": "fee_machine_v2_scaffold/scene_choreography.json",
    "editorial_review": "EDITORIAL_REVIEW_PACKET.md",
    "editorial_screening_guide": "EDITORIAL_SCREENING_GUIDE.md",
    "editorial_scorecard": "EDITORIAL_SCORECARD.json",
    "first_frame_review": "first_frame_candidates/first_frame_review.md",
    "pose_export_brief": "pose_export_intake/POSE_EXPORT_BRIEF.md",
    "pose_export_intake": "pose_export_intake/README.md",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _link(label: str, href: str) -> str:
    escaped_href = html.escape(href, quote=True)
    return f'<a href="{escaped_href}">{html.escape(label)}</a>'


def _primary_links(primary: dict[str, str]) -> str:
    labels = {
        "watchable_draft": "Watchable draft",
        "contact_sheet": "Proof contact sheet",
        "status_report": "Status report",
        "motion_quality_plan": "Motion quality pass",
        "motion_quality_plan_json": "Motion quality plan JSON",
        "scene_choreography": "Scene choreography JSON",
        "editorial_review": "Editorial review packet",
        "editorial_screening_guide": "Editorial screening guide",
        "editorial_scorecard": "Editorial scorecard",
        "first_frame_review": "First-frame review",
        "pose_export_brief": "Pose-export brief",
        "pose_export_intake": "Pose-export intake",
    }
    items = []
    for key, label in labels.items():
        href = primary.get(key)
        if href:
            items.append(f"<li>{_link(label, href)}</li>")
    return "\n".join(items)


def _first_frame_tiles(package_dir: Path) -> str:
    manifest_path = package_dir / "first_frame_candidates" / "first_frame_manifest.json"
    if not manifest_path.exists():
        return "<p>No first-frame candidates generated.</p>"
    manifest = read_json(manifest_path)
    tiles = []
    for candidate in manifest.get("candidates", []):
        href = f"first_frame_candidates/{candidate.get('filename')}"
        tiles.append(
            f"""<figure>
  <img src="{html.escape(href, quote=True)}" alt="{html.escape(str(candidate.get('id')))}" />
  <figcaption>{html.escape(str(candidate.get('id')))} · {candidate.get('sample_time')}s</figcaption>
</figure>"""
        )
    return "\n".join(tiles)


def _scene_choreography(package_dir: Path) -> str:
    choreography_path = package_dir / "fee_machine_v2_scaffold" / "scene_choreography.json"
    if not choreography_path.exists():
        return "<p>No scene choreography generated.</p>"
    choreography = read_json(choreography_path)
    rows = []
    for beat in choreography.get("beats", []):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(beat.get('id', '')))}</td>"
            f"<td>{html.escape(str(beat.get('start', '')))}-{html.escape(str(beat.get('end', '')))}s</td>"
            f"<td>{html.escape(str(beat.get('acting_objective', '')))}</td>"
            f"<td>{html.escape(str(beat.get('primary_motion', '')))}</td>"
            f"<td>{html.escape(str(beat.get('review_frame', '')))}</td>"
            "</tr>"
        )
    body = "\n".join(rows) or '<tr><td colspan="5">No staged beats listed.</td></tr>'
    return f"""<table>
  <thead>
    <tr><th>Beat</th><th>Time</th><th>Acting Objective</th><th>Primary Motion</th><th>Review Frame</th></tr>
  </thead>
  <tbody>
    {body}
  </tbody>
</table>"""


def _objective_evidence(package_index: dict[str, Any]) -> str:
    evidence = package_index.get("objective_evidence", {})
    labels = (
        ("choreography_beats_with_motion", "Choreography beats with motion"),
        ("choreography_beats_without_motion", "Choreography beats without motion"),
        ("choreography_beats_with_valid_review_frames", "Choreography beats with valid review frames"),
        ("choreography_beats_without_valid_review_frames", "Choreography beats without valid review frames"),
    )
    items = []
    for key, label in labels:
        items.append(f"<dt>{html.escape(label)}</dt><dd>{html.escape(str(evidence.get(key, 0)))}</dd>")
    return "\n".join(items)


def render_review_hub(package_dir: Path) -> str:
    package_index = read_json(package_dir / "review_package_index.json")
    handoff_path = package_dir / "handoff_manifest.json"
    if handoff_path.exists():
        handoff = read_json(handoff_path)
        primary = handoff.get("primary_review_files", DEFAULT_PRIMARY_REVIEW_FILES)
    else:
        primary = DEFAULT_PRIMARY_REVIEW_FILES
    blockers = package_index.get("blockers", [])
    blocker_items = "\n".join(f"<li>{html.escape(str(blocker))}</li>" for blocker in blockers) or "<li>none</li>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Signal Room Review Hub</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #101820; color: #f3eadb; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px; }}
    a {{ color: #d89a3a; font-weight: 700; }}
    section {{ border-top: 1px solid rgba(216,154,58,.35); padding: 24px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; background: #17232d; border: 1px solid rgba(216,154,58,.45); padding: 10px; }}
    img {{ width: 100%; aspect-ratio: 9 / 16; object-fit: cover; display: block; background: #0b1117; }}
    figcaption {{ margin-top: 8px; color: #8fa0aa; font-size: 14px; }}
    dl.evidence {{ display: grid; grid-template-columns: minmax(220px, 1fr) auto; gap: 8px 18px; max-width: 620px; }}
    dl.evidence dt {{ color: #8fa0aa; }}
    dl.evidence dd {{ margin: 0; font-weight: 800; color: #f3eadb; }}
    table {{ width: 100%; border-collapse: collapse; background: #17232d; }}
    th, td {{ border: 1px solid rgba(216,154,58,.32); padding: 10px; text-align: left; vertical-align: top; }}
    th {{ color: #d89a3a; }}
    code {{ color: #f3eadb; background: #17232d; padding: 2px 5px; }}
  </style>
</head>
<body>
  <main>
    <h1>Signal Room Review Hub</h1>
    <p>Status: review-only</p>

    <section>
      <h2>Primary Files</h2>
      <ul>
        {_primary_links(primary)}
      </ul>
    </section>

    <section>
      <h2>Current Blockers</h2>
      <ul>{blocker_items}</ul>
    </section>

    <section>
      <h2>First-Frame Candidates</h2>
      <div class="grid">
        {_first_frame_tiles(package_dir)}
      </div>
    </section>

    <section>
      <h2>Scene Choreography</h2>
      <p>{_link("Open choreography JSON", primary.get("scene_choreography", "fee_machine_v2_scaffold/scene_choreography.json"))}</p>
      {_scene_choreography(package_dir)}
    </section>

    <section>
      <h2>Objective Choreography Evidence</h2>
      <dl class="evidence">
        {_objective_evidence(package_index)}
      </dl>
    </section>

    <section>
      <h2>Review Order</h2>
      <ol>
        <li>Watch <code>{html.escape(primary.get("watchable_draft", ""))}</code>.</li>
        <li>Compare the proof contact sheet and first-frame candidates.</li>
        <li>Use <code>{html.escape(primary.get("editorial_screening_guide", ""))}</code> to fill <code>{html.escape(primary.get("editorial_scorecard", ""))}</code>.</li>
        <li>Run the editorial scorecard gate after decisions are filled.</li>
        <li>Send pose exports through <code>{html.escape(primary.get("pose_export_brief", primary.get("pose_export_intake", "")))}</code>.</li>
      </ol>
    </section>
  </main>
</body>
</html>
"""


def write_review_hub(package_dir: Path, out: Path) -> dict[str, Any]:
    out.write_text(render_review_hub(package_dir))
    return {"passed": True, "output": str(out)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = write_review_hub(args.package_dir, args.out)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

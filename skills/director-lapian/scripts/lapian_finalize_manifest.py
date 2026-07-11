#!/usr/bin/env python3
"""Update manifest.json with final director-lapian delivery artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import lapian_delivery_status as status


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def relative_or_absolute(path_value: str | None, project_dir: Path) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    try:
        return path.resolve().relative_to(project_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def build_manifest_update(project_dir: Path) -> dict:
    scan = status.scan_project(project_dir)
    paths = scan["paths"]
    audits = scan["audits"]
    deliverables = audits.get("deliverables", {})
    professional_md_audit = (deliverables.get("professional_analysis") or {}).get("markdown") or {}
    style_bible_md_audit = (deliverables.get("style_bible") or {}).get("markdown") or {}
    professional_reason_audit = professional_md_audit.get("professional_frame_reason_audit") or {}
    style_bible_structure = style_bible_md_audit.get("style_bible_audit") or {}
    return {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "project_dir": str(project_dir.resolve()),
        "reports": {
            "markdown_main": relative_or_absolute(paths.get("markdown_report"), project_dir),
            "docx_embedded": relative_or_absolute(paths.get("docx_embedded"), project_dir),
            "pdf_preview": relative_or_absolute(paths.get("pdf_preview"), project_dir),
            "qa_json": relative_or_absolute(paths.get("qa_json"), project_dir),
            "professional_analysis_md": relative_or_absolute(paths.get("professional_analysis_md"), project_dir),
            "professional_analysis_docx": relative_or_absolute(paths.get("professional_analysis_docx"), project_dir),
            "style_bible_md": relative_or_absolute(paths.get("style_bible_md"), project_dir),
            "style_bible_docx": relative_or_absolute(paths.get("style_bible_docx"), project_dir),
        },
        "deliverables": {
            "professional_analysis": {
                "exists": bool(professional_md_audit.get("exists")),
                "representative_frame_rows": professional_reason_audit.get("representative_frame_reason_rows", 0),
                "scene_block_count": professional_md_audit.get("scene_block_count", 0),
                "image_refs": professional_md_audit.get("markdown_images", 0),
            },
            "style_bible": {
                "exists": bool(style_bible_md_audit.get("exists")),
                "image_refs": style_bible_md_audit.get("markdown_images", 0),
                "asset_card_count": style_bible_structure.get("asset_card_count", 0),
                "missing_sections": style_bible_structure.get("missing_sections", []),
            },
        },
        "evidence": {
            "frames_dir": relative_or_absolute(paths.get("frames_dir"), project_dir),
            "frame_count": audits["frames"]["frame_count"],
            "audio_dir": relative_or_absolute(paths.get("audio_dir"), project_dir),
            "wav_files": [relative_or_absolute(p, project_dir) for p in audits["audio"]["wav_files"]],
            "levels_csv": [relative_or_absolute(p, project_dir) for p in audits["audio"]["levels_csv"]],
            "asr_files": [relative_or_absolute(p, project_dir) for p in audits["audio"]["asr_files"]],
        },
        "completion": {
            "done": scan["done"],
            "next_steps": scan["next_steps"],
            "pdf_valid": audits["pdf"]["exists"] and audits["pdf"]["has_pdf_header"] and audits["pdf"]["has_eof_marker"],
            "docx_blip_refs": audits["docx"]["docx_blip_refs"],
            "markdown_images": audits["markdown"].get("markdown_images", 0),
            "missing_markdown_images": len(audits["markdown"].get("missing_images", [])),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize director-lapian manifest delivery fields.")
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="defaults to <project-dir>/manifest.json")
    parser.add_argument("--dry-run", action="store_true", help="print merged manifest without writing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    out_path = args.out.resolve() if args.out else project_dir / "manifest.json"
    manifest = load_json(out_path)
    manifest["delivery"] = build_manifest_update(project_dir)

    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

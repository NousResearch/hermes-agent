#!/usr/bin/env python3
"""Scan a director-lapian project and report completed/missing delivery steps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import qa_lapian_delivery as qa


def find_latest_qa(project_dir: Path) -> Path | None:
    qa_dir = qa.find_child_by_prefix(project_dir, "06_") or project_dir / "06_QA审计"
    stable = qa_dir / "lapian_delivery_qa.json"
    if stable.is_file():
        return stable
    preferred = qa.find_latest_file(qa_dir, ["*qa*.json", "*QA*.json", "*审计*.json", "*audit*.json"])
    if preferred:
        return preferred
    return qa.find_latest_file(qa_dir, ["*.json"])


def scan_project(project_dir: Path) -> dict:
    project_dir = project_dir.resolve()
    defaults = qa.default_from_project(project_dir)
    frames_dir = defaults.get("frames_dir")
    audio_dir = defaults.get("audio_dir")
    report = defaults.get("report")
    docx = defaults.get("docx")
    pdf = defaults.get("pdf")
    manifest = defaults.get("manifest")
    latest_qa = find_latest_qa(project_dir)

    md_files = qa.list_markdown_reports(project_dir)
    docx_files = qa.list_docx_files(project_dir)
    professional_md = qa.latest_by_role(md_files, qa.ROLE_PROFESSIONAL)
    style_bible_md = qa.latest_by_role(md_files, qa.ROLE_STYLE_BIBLE)
    professional_docx = qa.latest_by_role(docx_files, qa.ROLE_PROFESSIONAL)
    style_bible_docx = qa.latest_by_role(docx_files, qa.ROLE_STYLE_BIBLE)

    frame_audit = qa.audit_frames(frames_dir.resolve() if frames_dir else None)
    markdown_audit = (
        qa.audit_markdown(
            report.resolve(),
            None,
            project_dir,
            frame_audit["frame_count"] or None,
        )
        if report
        else {"exists": False, "path": None}
    )
    docx_audit = qa.audit_docx(docx.resolve() if docx else None)
    pdf_audit = qa.audit_pdf(pdf.resolve() if pdf else None)
    audio_audit = qa.audit_audio(audio_dir.resolve() if audio_dir else None)
    professional_md_audit = (
        qa.audit_markdown(professional_md.resolve(), None, project_dir, None) if professional_md else None
    )
    style_bible_md_audit = (
        qa.audit_markdown(style_bible_md.resolve(), None, project_dir, None) if style_bible_md else None
    )
    professional_docx_audit = qa.audit_docx(professional_docx.resolve()) if professional_docx else None
    style_bible_docx_audit = qa.audit_docx(style_bible_docx.resolve()) if style_bible_docx else None

    done = {
        "frames": bool(frame_audit["exists"] and frame_audit["frame_count"]),
        "markdown_report": bool(markdown_audit.get("exists")),
        "docx_embedded": bool(docx_audit["exists"] and docx_audit["docx_blip_refs"]),
        "pdf_preview": bool(pdf_audit["exists"] and pdf_audit["has_pdf_header"] and pdf_audit["has_eof_marker"]),
        "audio_extract": bool(audio_audit["wav_files"] or audio_audit["levels_csv"]),
        "asr_files": bool(audio_audit["asr_files"]),
        "manifest": bool(manifest and manifest.exists()),
        "qa_json": bool(latest_qa and latest_qa.exists()),
        "professional_analysis_md": bool(professional_md_audit and professional_md_audit.get("exists")),
        "professional_analysis_docx": bool(
            professional_docx_audit and professional_docx_audit["exists"] and professional_docx_audit["docx_blip_refs"]
        ),
        "style_bible_md": bool(style_bible_md_audit and style_bible_md_audit.get("exists")),
        "style_bible_docx": bool(
            style_bible_docx_audit and style_bible_docx_audit["exists"] and style_bible_docx_audit["docx_blip_refs"]
        ),
    }

    next_steps: list[str] = []
    if not done["frames"]:
        next_steps.append("run prepare_lapian_evidence.py or video_frame_sampler.py to create frame evidence")
    if not done["markdown_report"]:
        next_steps.append("write the main Markdown report under 03_MD报告")
    if done["frames"] and done["markdown_report"] and markdown_audit.get("missing_images"):
        next_steps.append("fix missing Markdown image references before conversion")
    if done["markdown_report"] and not done["docx_embedded"]:
        next_steps.append("generate embedded-image DOCX with md_image_report_to_docx.py")
    if done["markdown_report"] and not done["professional_analysis_md"]:
        next_steps.append("write the independent 专业导演分析 Markdown (default deliverable; skip only if the user opted out)")
    if done["professional_analysis_md"] and not done["professional_analysis_docx"]:
        next_steps.append("generate embedded-image DOCX for 专业导演分析")
    if done["markdown_report"] and not done["style_bible_md"]:
        next_steps.append("write the independent 源片风格圣经 Markdown (default deliverable; skip only if the user opted out)")
    if done["style_bible_md"] and not done["style_bible_docx"]:
        next_steps.append("generate embedded-image DOCX for 源片风格圣经")
    if done["docx_embedded"] and not done["pdf_preview"]:
        next_steps.append("export or render a PDF preview from the DOCX")
    if not done["audio_extract"]:
        next_steps.append("extract audio evidence if sound/dialogue analysis is required")
    if not done["asr_files"]:
        next_steps.append("create or mark ASR/transcript status if dialogue is part of the request")
    if not done["qa_json"]:
        next_steps.append("run qa_lapian_delivery.py with --overwrite-out for final stable QA JSON")
    if not done["manifest"]:
        next_steps.append("run lapian_finalize_manifest.py to write final delivery paths into manifest.json")

    return {
        "project_dir": str(project_dir),
        "done": done,
        "next_steps": next_steps,
        "paths": {
            "frames_dir": str(frames_dir) if frames_dir else None,
            "audio_dir": str(audio_dir) if audio_dir else None,
            "markdown_report": str(report) if report else None,
            "docx_embedded": str(docx) if docx else None,
            "pdf_preview": str(pdf) if pdf else None,
            "manifest": str(manifest) if manifest else None,
            "qa_json": str(latest_qa) if latest_qa else None,
            "professional_analysis_md": str(professional_md) if professional_md else None,
            "professional_analysis_docx": str(professional_docx) if professional_docx else None,
            "style_bible_md": str(style_bible_md) if style_bible_md else None,
            "style_bible_docx": str(style_bible_docx) if style_bible_docx else None,
        },
        "audits": {
            "frames": frame_audit,
            "markdown": markdown_audit,
            "docx": docx_audit,
            "pdf": pdf_audit,
            "audio": audio_audit,
            "deliverables": {
                "professional_analysis": {
                    "markdown": professional_md_audit,
                    "docx": professional_docx_audit,
                },
                "style_bible": {
                    "markdown": style_bible_md_audit,
                    "docx": style_bible_docx_audit,
                },
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan director-lapian delivery completion status.")
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="optional JSON status output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = scan_project(args.project_dir)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["written_to"] = str(args.out.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

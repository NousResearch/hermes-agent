#!/usr/bin/env python3
"""Audit director-lapian deliverables before final handoff."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from urllib.parse import unquote


IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
MOJIBAKE_TOKENS = ["鎷", "鍥", "绱", "馃", "â€", "锛", "鈥", "閹", "閸", "棣"]
FRAME_REASON_REQUIRED_PARTS = ["情节", "构图", "导演意图", "含义"]
GENERIC_FRAME_REASON_PHRASES = [
    "作为本幕导演策略的视觉证据",
    "显示空间、道具、表情、色彩或类型转换中的关键一步",
    "这一帧很重要，体现了导演意图",
]

ROLE_MAIN = "main_report"
ROLE_PROFESSIONAL = "professional_analysis"
ROLE_STYLE_BIBLE = "style_bible"

STYLE_BIBLE_SECTION_NEEDLES = ["风格定位", "DNA", "资产索引", "场景级", "元素级", "图像生成", "视频生成", "跑偏"]
ASSET_CARD_RE = re.compile(r"^#{2,4}\s*(场景资产卡|元素资产卡)", re.MULTILINE)
SCENE_BLOCK_RE = re.compile(r"^##\s*【", re.MULTILINE)


def classify_report_role(name: str) -> str:
    if "风格圣经" in name:
        return ROLE_STYLE_BIBLE
    if "专业导演分析" in name:
        return ROLE_PROFESSIONAL
    return ROLE_MAIN


def clean_image_target(raw: str) -> str:
    value = raw.strip()
    if value.startswith("<") and ">" in value:
        value = value[1 : value.index(">")]
    else:
        quote_idx = value.find(" \"")
        if quote_idx != -1:
            value = value[:quote_idx]
    return unquote(value.strip())


def unique_file(path: Path, overwrite: bool = False) -> Path:
    if overwrite or not path.exists():
        return path
    for idx in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_v{idx:02d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise SystemExit(f"could not create unique output file near: {path}")


def find_child_by_prefix(project_dir: Path, prefix: str) -> Path | None:
    if not project_dir.exists():
        return None
    matches = [p for p in project_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not matches:
        return None
    return sorted(matches, key=lambda p: (p.stat().st_mtime, p.name))[-1]


def find_latest_file(folder: Path | None, patterns: list[str]) -> Path | None:
    if not folder or not folder.exists():
        return None
    files: list[Path] = []
    for pattern in patterns:
        files.extend(p for p in folder.glob(pattern) if p.is_file())
    if not files:
        return None
    return sorted(files, key=lambda p: (p.stat().st_mtime, p.name))[-1]


def list_markdown_reports(project_dir: Path | None) -> list[Path]:
    if not project_dir:
        return []
    markdown_dir = find_child_by_prefix(project_dir, "03_") or project_dir / "03_MD报告"
    if not markdown_dir.exists():
        return []
    files = [p for p in markdown_dir.glob("*.md") if p.is_file()]
    return sorted(files, key=lambda p: (p.stat().st_mtime, p.name))


def list_docx_files(project_dir: Path | None) -> list[Path]:
    if not project_dir:
        return []
    docx_dir = find_child_by_prefix(project_dir, "04_") or project_dir / "04_飞书Word交付"
    if not docx_dir.exists():
        return []
    files = [p for p in docx_dir.glob("*.docx") if p.is_file()]
    return sorted(files, key=lambda p: (p.stat().st_mtime, p.name))


def latest_by_role(paths: list[Path], role: str) -> Path | None:
    matches = [p for p in paths if classify_report_role(p.name) == role]
    return matches[-1] if matches else None


def resolve_local_image(raw: str, md_dir: Path, image_root: Path | None, project_dir: Path | None) -> Path | None:
    target = clean_image_target(raw)
    if not target or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", target):
        return None
    candidate = Path(target)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    candidates = [md_dir / target]
    if image_root:
        candidates.append(image_root / target)
    if project_dir:
        candidates.append(project_dir / target)
    for item in candidates:
        if item.exists():
            return item
    return None


def _has_reason_part(reason: str, part: str) -> bool:
    return re.search(re.escape(part) + r"\s*[：:]", reason) is not None


def extract_frame_reason_rows(text: str) -> list[str]:
    reasons: list[str] = []
    active = False
    for line in text.splitlines():
        stripped = line.strip()
        if "为什么选这一帧" in stripped and stripped.startswith("|"):
            active = True
            continue
        if not active:
            continue
        if not stripped:
            active = False
            continue
        if stripped.startswith("|---"):
            continue
        if not stripped.startswith("|"):
            active = False
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) >= 3 and "![" in cells[1]:
            reasons.append(cells[-1])
    return reasons


def audit_professional_frame_reasons(text: str, report: Path) -> dict:
    reasons = extract_frame_reason_rows(text)
    term_counts = {
        part: sum(1 for reason in reasons if _has_reason_part(reason, part))
        for part in FRAME_REASON_REQUIRED_PARTS
    }
    generic_hits = {
        phrase: sum(reason.count(phrase) for reason in reasons)
        for phrase in GENERIC_FRAME_REASON_PHRASES
        if any(phrase in reason for reason in reasons)
    }
    unique_reasons = len(set(reasons))
    total = len(reasons)
    duplicate_exact_count = total - unique_reasons
    repeated_samples: list[str] = []
    if duplicate_exact_count:
        seen: set[str] = set()
        duplicated: set[str] = set()
        for reason in reasons:
            if reason in seen and reason not in duplicated:
                repeated_samples.append(reason)
                duplicated.add(reason)
            seen.add(reason)
            if len(repeated_samples) >= 5:
                break
    four_part_rows = sum(
        1 for reason in reasons if all(_has_reason_part(reason, part) for part in FRAME_REASON_REQUIRED_PARTS)
    )
    return {
        "enabled": "专业导演分析" in report.name or "专业导演分析" in text,
        "representative_frame_reason_rows": total,
        "unique_reasons": unique_reasons,
        "duplicate_exact_count": duplicate_exact_count,
        "four_part_rows": four_part_rows,
        "required_term_counts": term_counts,
        "generic_phrase_hits": generic_hits,
        "repeated_reason_samples": repeated_samples,
    }


def audit_style_bible_structure(text: str) -> dict:
    hits = {needle: (needle in text) for needle in STYLE_BIBLE_SECTION_NEEDLES}
    return {
        "required_section_hits": hits,
        "missing_sections": [needle for needle, found in hits.items() if not found],
        "asset_card_count": len(ASSET_CARD_RE.findall(text)),
    }


def audit_markdown(report: Path, image_root: Path | None, project_dir: Path | None, expected_frame_count: int | None) -> dict:
    role = classify_report_role(report.name)
    result = {
        "path": str(report),
        "role": role,
        "exists": report.exists(),
        "utf8_readable": False,
        "markdown_images": 0,
        "local_images": 0,
        "remote_or_unresolved_images": 0,
        "missing_images": [],
        "contains_first_frame": False,
        "contains_last_frame": None,
        "suspected_mojibake": {},
        "required_section_hits": {},
        "professional_frame_reason_audit": {},
        "style_bible_audit": {},
        "scene_block_count": 0,
    }
    if not report.exists():
        return result
    text = report.read_text(encoding="utf-8")
    result["utf8_readable"] = True
    result["suspected_mojibake"] = {token: text.count(token) for token in MOJIBAKE_TOKENS if text.count(token)}
    result["professional_frame_reason_audit"] = audit_professional_frame_reasons(text, report)
    if role == ROLE_PROFESSIONAL:
        result["scene_block_count"] = len(SCENE_BLOCK_RE.findall(text))
    if role == ROLE_STYLE_BIBLE:
        result["style_bible_audit"] = audit_style_bible_structure(text)
    section_needles = [
        "素材诊断",
        "全片结构",
        "叙事",
        "视听",
        "主角",
        "情感",
        "主题",
        "可学习",
        "交付审计",
    ]
    result["required_section_hits"] = {needle: (needle in text) for needle in section_needles}
    result["contains_first_frame"] = "frame_000001" in text
    if expected_frame_count:
        result["contains_last_frame"] = f"frame_{expected_frame_count:06d}" in text

    md_dir = report.parent
    for match in IMAGE_RE.finditer(text):
        result["markdown_images"] += 1
        raw = match.group(2)
        target = clean_image_target(raw)
        resolved = resolve_local_image(raw, md_dir, image_root, project_dir)
        if resolved:
            result["local_images"] += 1
        elif re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", target):
            result["remote_or_unresolved_images"] += 1
        else:
            result["missing_images"].append(target)
    return result


def evaluate_markdown_audit(markdown_audit: dict, expected_report_images: int | None = None) -> tuple[list[str], list[str]]:
    """Turn one markdown audit into (failures, warnings) according to its deliverable role."""
    failures: list[str] = []
    warnings: list[str] = []
    label = Path(markdown_audit.get("path") or "report").name
    role = markdown_audit.get("role") or ROLE_MAIN
    if not markdown_audit.get("exists"):
        failures.append(f"{label}: markdown report not found")
        return failures, warnings
    if markdown_audit.get("missing_images"):
        failures.append(f"{label}: missing markdown images: {len(markdown_audit['missing_images'])}")
    image_count = markdown_audit.get("markdown_images", 0)
    if role == ROLE_MAIN and expected_report_images and image_count < expected_report_images:
        warnings.append(
            f"{label}: markdown image refs ({image_count}) are fewer than expected report images ({expected_report_images})"
        )
    if role == ROLE_STYLE_BIBLE:
        style_audit = markdown_audit.get("style_bible_audit") or {}
        if style_audit.get("missing_sections"):
            failures.append(
                f"{label}: style bible missing required sections: {', '.join(style_audit['missing_sections'])}"
            )
        if image_count == 0:
            failures.append(f"{label}: style bible has no image asset references")
    elif image_count == 0:
        warnings.append(f"{label}: report contains no image references")
    if markdown_audit.get("suspected_mojibake"):
        warnings.append(f"{label}: suspected mojibake tokens found in markdown")
    reason_audit = markdown_audit.get("professional_frame_reason_audit") or {}
    if reason_audit.get("enabled") and reason_audit.get("representative_frame_reason_rows"):
        reason_rows = reason_audit["representative_frame_reason_rows"]
        four_part_rows = reason_audit["four_part_rows"]
        if reason_audit.get("generic_phrase_hits"):
            failures.append(f"{label}: professional frame reasons contain generic template phrases")
        if four_part_rows < reason_rows:
            failures.append(
                f"{label}: professional frame reasons with 情节/构图/导演意图/含义 ({four_part_rows}) are fewer than representative rows ({reason_rows})"
            )
        if reason_rows >= 3 and reason_audit.get("unique_reasons", 0) / reason_rows < 0.9:
            failures.append(f"{label}: professional frame reasons appear too repetitive")
    return failures, warnings


def audit_frames(frames_dir: Path | None) -> dict:
    result = {
        "path": str(frames_dir) if frames_dir else None,
        "exists": bool(frames_dir and frames_dir.exists()),
        "frame_count": 0,
        "first_frame_exists": False,
        "last_frame": None,
    }
    if not frames_dir or not frames_dir.exists():
        return result
    frames = sorted((frames_dir / "frames").glob("frame_*.jpg"))
    if not frames:
        frames = sorted(frames_dir.glob("frame_*.jpg"))
    result["frame_count"] = len(frames)
    result["first_frame_exists"] = bool(frames and frames[0].name == "frame_000001.jpg")
    result["last_frame"] = str(frames[-1]) if frames else None
    return result


def audit_docx(docx: Path | None) -> dict:
    result = {
        "path": str(docx) if docx else None,
        "role": classify_report_role(docx.name) if docx else None,
        "exists": bool(docx and docx.exists()),
        "docx_media_files": 0,
        "docx_drawing_elements": 0,
        "docx_blip_refs": 0,
    }
    if not docx or not docx.exists():
        return result
    with zipfile.ZipFile(docx) as archive:
        names = archive.namelist()
        result["docx_media_files"] = len([n for n in names if n.startswith("word/media/")])
        try:
            document_xml = archive.read("word/document.xml").decode("utf-8", "ignore")
        except KeyError:
            document_xml = ""
    result["docx_drawing_elements"] = document_xml.count("<w:drawing")
    result["docx_blip_refs"] = document_xml.count("<a:blip")
    return result


def audit_pdf(pdf: Path | None) -> dict:
    result = {
        "path": str(pdf) if pdf else None,
        "exists": bool(pdf and pdf.exists()),
        "size_bytes": 0,
        "has_pdf_header": False,
        "has_eof_marker": False,
        "rough_page_markers": None,
    }
    if not pdf or not pdf.exists():
        return result
    data = pdf.read_bytes()
    result["size_bytes"] = len(data)
    result["has_pdf_header"] = data.startswith(b"%PDF-")
    result["has_eof_marker"] = b"%%EOF" in data[-2048:]
    result["rough_page_markers"] = data.count(b"/Type /Page")
    return result


def audit_audio(audio_dir: Path | None) -> dict:
    result = {
        "path": str(audio_dir) if audio_dir else None,
        "exists": bool(audio_dir and audio_dir.exists()),
        "wav_files": [],
        "levels_csv": [],
        "asr_files": [],
    }
    if not audio_dir or not audio_dir.exists():
        return result
    result["wav_files"] = [str(p) for p in sorted(audio_dir.glob("*_audio_16k_mono.wav"))]
    result["levels_csv"] = [str(p) for p in sorted(audio_dir.glob("*_audio_levels.csv"))]
    result["asr_files"] = [str(p) for p in sorted(audio_dir.glob("*_asr_*"))]
    return result


def load_manifest(path: Path | None) -> dict | None:
    if not path or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def expected_markdown_image_count(selection_plan: Path | None, fallback_frame_count: int | None) -> int | None:
    if selection_plan and selection_plan.exists():
        plan = json.loads(selection_plan.read_text(encoding="utf-8-sig"))
        selected_count = plan.get("selected_count")
        if isinstance(selected_count, int) and selected_count > 0:
            return selected_count
    return fallback_frame_count


def default_from_project(project_dir: Path | None) -> dict[str, Path | None]:
    if not project_dir:
        return {}
    frames_dir = find_child_by_prefix(project_dir, "01_") or project_dir / "01_逐秒抽帧"
    audio_dir = find_child_by_prefix(project_dir, "02_") or project_dir / "02_音频分析"
    pdf_dir = find_child_by_prefix(project_dir, "05_") or project_dir / "05_PDF预览"
    reports = list_markdown_reports(project_dir)
    docx_files = list_docx_files(project_dir)
    return {
        "frames_dir": frames_dir,
        "audio_dir": audio_dir,
        "manifest": project_dir / "manifest.json",
        "report": latest_by_role(reports, ROLE_MAIN) or (reports[-1] if reports else None),
        "docx": latest_by_role(docx_files, ROLE_MAIN) or (docx_files[-1] if docx_files else None),
        "pdf": find_latest_file(pdf_dir, ["*.pdf"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit director-lapian report delivery.")
    parser.add_argument("--project-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--docx", type=Path)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--selection-plan", type=Path, help="report-frame selection plan; applies to the main lapian report only")
    parser.add_argument("--expected-frame-count", type=int)
    parser.add_argument("--out", type=Path, help="optional JSON audit output path")
    parser.add_argument("--overwrite-out", action="store_true", help="write exactly to --out/default path instead of versioning")
    parser.add_argument("--strict", action="store_true", help="exit 1 when required checks fail")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_dir = args.project_dir.resolve() if args.project_dir else None
    defaults = default_from_project(project_dir)
    manifest_path = (args.manifest or defaults.get("manifest"))
    manifest = load_manifest(manifest_path)

    frames_dir = (args.frames_dir or defaults.get("frames_dir"))
    frame_audit = audit_frames(frames_dir.resolve() if frames_dir else None)
    expected_frame_count = args.expected_frame_count or frame_audit["frame_count"] or None
    selection_plan = args.selection_plan.resolve() if args.selection_plan else None
    expected_report_images = expected_markdown_image_count(selection_plan, expected_frame_count)

    image_root = args.image_root.resolve() if args.image_root else None
    if args.report:
        md_paths = [args.report.resolve()]
    else:
        md_paths = [p.resolve() for p in list_markdown_reports(project_dir)]
        if not md_paths and defaults.get("report"):
            md_paths = [defaults["report"].resolve()]
    markdown_audits = []
    for md_path in md_paths:
        role = classify_report_role(md_path.name)
        markdown_audits.append(
            audit_markdown(
                md_path,
                image_root,
                project_dir,
                expected_frame_count if role == ROLE_MAIN else None,
            )
        )
    main_markdown_audit = next(
        (audit for audit in reversed(markdown_audits) if audit.get("role") == ROLE_MAIN),
        markdown_audits[-1] if markdown_audits else {"exists": False, "path": None, "role": ROLE_MAIN},
    )

    if args.docx:
        docx_paths = [args.docx.resolve()]
    else:
        docx_paths = [p.resolve() for p in list_docx_files(project_dir)]
        if not docx_paths and defaults.get("docx"):
            docx_paths = [defaults["docx"].resolve()]
    docx_audits = [audit_docx(p) for p in docx_paths]
    main_docx_audit = next(
        (audit for audit in reversed(docx_audits) if audit.get("role") == ROLE_MAIN),
        docx_audits[-1] if docx_audits else audit_docx(None),
    )

    pdf = args.pdf or defaults.get("pdf")
    audio_dir = args.audio_dir or defaults.get("audio_dir")
    pdf_audit = audit_pdf(pdf.resolve() if pdf else None)
    audio_audit = audit_audio(audio_dir.resolve() if audio_dir else None)

    failures = []
    warnings = []
    if not md_paths:
        failures.append("markdown report not found")
    for markdown_audit in markdown_audits:
        md_failures, md_warnings = evaluate_markdown_audit(markdown_audit, expected_report_images)
        failures.extend(md_failures)
        warnings.extend(md_warnings)
    for docx_audit in docx_audits:
        if docx_audit["exists"] and docx_audit["docx_media_files"] == 0:
            failures.append(f"{Path(docx_audit['path']).name}: docx exists but has no embedded media")
    if args.pdf and not pdf_audit["exists"]:
        failures.append("explicit pdf path does not exist")
    if pdf_audit["exists"] and not pdf_audit["has_pdf_header"]:
        failures.append("pdf exists but does not start with %PDF-")
    if pdf_audit["exists"] and not pdf_audit["has_eof_marker"]:
        failures.append("pdf exists but EOF marker was not found near the end")
    if project_dir and not pdf_audit["exists"]:
        warnings.append("pdf preview not found under project directory")

    result = {
        "ok": not failures,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_loaded": manifest is not None,
        "selection_plan_path": str(selection_plan) if selection_plan else None,
        "expected_report_images": expected_report_images,
        "project_dir": str(project_dir) if project_dir else None,
        "frames": frame_audit,
        "markdown": main_markdown_audit,
        "markdown_reports": markdown_audits,
        "docx": main_docx_audit,
        "docx_files": docx_audits,
        "pdf": pdf_audit,
        "audio": audio_audit,
        "warnings": warnings,
        "failures": failures,
    }

    out_path = args.out
    if out_path is None and project_dir:
        out_path = project_dir / "06_QA审计" / "lapian_delivery_qa.json"
    if out_path:
        out_path = unique_file(out_path.resolve(), overwrite=args.overwrite_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["written_to"] = str(out_path)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.strict and failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

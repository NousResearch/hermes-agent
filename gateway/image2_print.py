"""Hermes-owned Image2 print-final lane utilities.

This module intentionally does not live under ``marketing-hub/scripts``. It is
used by the Feishu Image2 worker after a preview has been approved and a user
asks for a printable PSD/PDF final.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

_PRINT_ACTION_RE = re.compile(r"(定稿|印刷版|印刷稿|正式稿|正式印刷|打样|出\s*(?:PSD|PDF)|生成\s*(?:PSD|PDF)|proof\s*PDF)", re.I)
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[x×＊*]\s*(\d+(?:\.\d+)?)\s*(mm|毫米|cm|厘米|m|米)?", re.I)
_DPI_RE = re.compile(r"(\d{2,3})\s*dpi", re.I)
_PAGE_SIZES = {
    "A4": (210, 297),
    "Ａ４": (210, 297),
    "a4": (210, 297),
    "A3": (297, 420),
    "Ａ３": (297, 420),
    "a3": (297, 420),
}


def should_handle_print_request(text: str) -> bool:
    raw = str(text or "")
    return bool(_PRINT_ACTION_RE.search(raw))


def _unit_to_mm(value: float, unit: str | None) -> int:
    unit = (unit or "").lower()
    if unit in {"m", "米"}:
        return int(round(value * 1000))
    if unit in {"mm", "毫米"}:
        return int(round(value))
    return int(round(value * 10))


def default_dpi_for_size(width_mm: int, height_mm: int, *, quality_hint: str = "") -> int:
    largest = max(width_mm, height_mm)
    hint = str(quality_hint or "")
    if largest <= 420:
        return 300
    if largest < 800:
        return 200
    if any(word in hint for word in ("近看", "高质量", "精细", "高清", "200dpi")):
        return 200
    return 150


def pixels_for_mm(mm: int, dpi: int) -> int:
    return int(round(float(mm) / 25.4 * int(dpi)))


def parse_print_spec(text: str, *, image_size_px: tuple[int, int] | None = None) -> dict[str, Any]:
    raw = str(text or "")
    if not should_handle_print_request(raw):
        return {"status": "not_print_request"}

    width_mm: int | None = None
    height_mm: int | None = None
    page_name = ""
    for key, size in _PAGE_SIZES.items():
        if key in raw:
            width_mm, height_mm = size
            page_name = key.upper().replace("Ａ", "A")
            break

    if width_mm is None or height_mm is None:
        match = _SIZE_RE.search(raw)
        if match:
            unit = match.group(3) or "cm"
            width_mm = _unit_to_mm(float(match.group(1)), unit)
            height_mm = _unit_to_mm(float(match.group(2)), unit)

    if width_mm is None or height_mm is None:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(m|米|cm|厘米|mm|毫米)\s*宽", raw, re.I)
        if m and image_size_px and image_size_px[0] > 0 and image_size_px[1] > 0:
            width_mm = _unit_to_mm(float(m.group(1)), m.group(2))
            height_mm = int(round(width_mm * image_size_px[1] / image_size_px[0]))

    if width_mm is None or height_mm is None:
        return {
            "status": "need_clarification",
            "need_clarification": "size_required",
            "message": "要出最终印刷稿，需要尺寸。比如：60×90cm、80×120cm、100×150cm，或 A3。",
        }

    dpi_match = _DPI_RE.search(raw)
    dpi = int(dpi_match.group(1)) if dpi_match else default_dpi_for_size(width_mm, height_mm, quality_hint=raw)
    target_width_px = pixels_for_mm(width_mm, dpi)
    target_height_px = pixels_for_mm(height_mm, dpi)
    return {
        "status": "ok",
        "width_mm": int(width_mm),
        "height_mm": int(height_mm),
        "dpi": int(dpi),
        "target_width_px": int(target_width_px),
        "target_height_px": int(target_height_px),
        "bleed_mm": 5,
        "safe_margin_mm": 30 if max(width_mm, height_mm) >= 800 else 10,
        "output_psd_type": "flat_single_layer",
        "page_name": page_name,
        "upscale_mode": "fast_upscale",
        "terminology": "AI 超分 / 高分辨率重建",
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_safe_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _run(cmd: list[str], *, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=True)


def package_flat_print_outputs(
    *,
    job_dir: Path,
    approved_image_path: Path,
    spec: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Create V1 print files: highres PNG, flat PSD, proof PDF, small preview."""
    env = dict(environ or {})
    approved = Path(approved_image_path).expanduser()
    if not approved.is_file():
        raise FileNotFoundError(str(approved))
    if str(spec.get("output_psd_type") or "") != "flat_single_layer":
        raise ValueError("only flat_single_layer PSD is supported in V1")
    width_mm = int(spec["width_mm"])
    height_mm = int(spec["height_mm"])
    dpi = int(spec["dpi"])
    target_w = int(spec.get("target_width_px") or pixels_for_mm(width_mm, dpi))
    target_h = int(spec.get("target_height_px") or pixels_for_mm(height_mm, dpi))
    task_id = Path(job_dir).name

    root = Path(job_dir) / "print"
    highres_dir = root / "highres"
    proof_dir = root / "proof"
    psd_dir = root / "psd"
    reports_dir = root / "reports"
    for d in (highres_dir, proof_dir, psd_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)
    highres = highres_dir / "highres.png"
    preview = proof_dir / "preview_1200.png"
    proof_pdf = proof_dir / "proof.pdf"
    psd = psd_dir / f"{task_id}_print_{width_mm}x{height_mm}mm_{dpi}dpi_flat.psd"

    py = str(env.get("IMAGE2_PRINT_SYSTEM_PYTHON") or "/usr/bin/python3")
    script = """
import sys
from PIL import Image
src, highres, preview, proof, w, h, dpi = sys.argv[1:]
w=int(w); h=int(h); dpi=int(dpi)
im = Image.open(src).convert('RGB')
sw, sh = im.size
scale = max(w / sw, h / sh)
nw, nh = int(round(sw * scale)), int(round(sh * scale))
im = im.resize((nw, nh), Image.Resampling.LANCZOS)
left = max(0, (nw - w) // 2); top = max(0, (nh - h) // 2)
im = im.crop((left, top, left + w, top + h))
im.save(highres, 'PNG', dpi=(dpi, dpi))
scale2 = min(1200 / w, 1200 / h, 1.0)
pv = im.resize((max(1,int(round(w*scale2))), max(1,int(round(h*scale2)))), Image.Resampling.LANCZOS)
pv.save(preview, 'PNG')
im.save(proof, 'PDF', resolution=float(dpi))
"""
    run = runner or _run
    run([py, "-c", script, str(approved), str(highres), str(preview), str(proof_pdf), str(target_w), str(target_h), str(dpi)], timeout=int(env.get("IMAGE2_PRINT_PACKAGE_TIMEOUT") or 600))
    run(["/usr/bin/sips", "-s", "format", "psd", str(highres), "--out", str(psd)], timeout=300)
    run(["/usr/bin/sips", "-s", "dpiWidth", str(dpi), "-s", "dpiHeight", str(dpi), str(psd), "--out", str(psd)], timeout=120)

    highres_sha = sha256_file(highres)
    approved_sha = sha256_file(approved)
    result = {
        "status": "pass",
        "approved_image_path": str(approved),
        "approved_sha256": approved_sha,
        "highres_path": str(highres),
        "highres_sha256": highres_sha,
        "psd_path": str(psd),
        "pdf_path": str(proof_pdf),
        "preview_path": str(preview),
        "target_width_px": target_w,
        "target_height_px": target_h,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "dpi": dpi,
        "output_psd_type": "flat_single_layer",
        "sha_differs_from_approved": highres_sha != approved_sha,
    }
    _write_safe_json(root / "spec.json", dict(spec))
    if highres_sha == approved_sha:
        result["status"] = "fail"
        result["reason"] = "highres_sha_matches_approved_preview"
    _write_safe_json(reports_dir / "package_report.json", result)
    return result

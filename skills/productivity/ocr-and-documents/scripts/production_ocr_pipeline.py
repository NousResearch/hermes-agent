#!/usr/bin/env python3
"""Production-ish local OCR pipeline for Spearhead document batches.

Features:
- text-PDF short-circuit via pdftotext (do not OCR selectable text)
- image/PDF OCR via Poppler + Tesseract
- subprocess timeouts, resource-limit sandbox, no shell execution
- retry matrix over PSM modes with confidence quality gates
- per-page text + TSV + hOCR + ALTO + PAGE XML outputs
- JSON manifests for batch/document/page metrics

This is intentionally local/offline. It does not upload documents anywhere.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import xml.sax.saxutils as xml_escape
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

TEXT_EXTS = {".txt", ".md"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXTS = {".pdf"}
DEFAULT_PSMS = [3, 6, 11]
LOW_CONF_THRESHOLD = 80.0
LOW_CONF_RATIO_THRESHOLD = 0.10


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    seconds: float


def _sandbox_preexec(cpu_seconds: int = 120, address_space_mb: int = 2048) -> Callable[[], None]:
    def preexec() -> None:
        os.umask(0o077)
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))
        except Exception:
            pass
        try:
            limit = address_space_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        except Exception:
            pass
    return preexec


def run_command(
    cmd: Sequence[str],
    *,
    timeout: int,
    sandbox: bool = True,
    cwd: Path | None = None,
) -> CommandResult:
    """Run a command without shell, with timeout and optional resource limits."""
    start = time.monotonic()
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "TESSDATA_PREFIX": os.environ.get("TESSDATA_PREFIX", ""),
        # Keep Tesseract/OpenMP predictable under the process sandbox.
        "OMP_THREAD_LIMIT": os.environ.get("OMP_THREAD_LIMIT", "1"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
    }
    env = {k: v for k, v in env.items() if v}
    try:
        proc = subprocess.run(
            list(cmd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            shell=False,
            cwd=str(cwd) if cwd else None,
            env=env,
            preexec_fn=_sandbox_preexec(timeout) if sandbox and os.name == "posix" else None,
        )
        return CommandResult(proc.returncode, proc.stdout, proc.stderr, False, time.monotonic() - start)
    except subprocess.TimeoutExpired as exc:
        return CommandResult(124, exc.stdout or "", exc.stderr or "", True, time.monotonic() - start)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def document_output_dir(input_path: Path, out_root: Path) -> Path:
    """Return a collision-safe per-document output directory."""
    slug = input_path.stem.replace(" ", "_")[:80] or "document"
    doc_dir = out_root / slug
    if not doc_dir.exists():
        return doc_dir
    digest = hashlib.sha256(str(input_path.resolve()).encode("utf-8", errors="replace")).hexdigest()[:8]
    return out_root / f"{slug}-{digest}"


def metrics_from_tsv(tsv_text: str) -> tuple[dict, list[dict]]:
    reader = csv.DictReader(tsv_text.splitlines(), delimiter="\t")
    words: list[dict] = []
    confs: list[float] = []
    for row in reader:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            conf = float(row.get("conf") or -1)
        except ValueError:
            conf = -1.0
        if conf < 0:
            continue
        word = {
            "text": text,
            "conf": conf,
            "left": int(float(row.get("left") or 0)),
            "top": int(float(row.get("top") or 0)),
            "width": int(float(row.get("width") or 0)),
            "height": int(float(row.get("height") or 0)),
        }
        words.append(word)
        confs.append(conf)
    char_count = sum(len(w["text"]) for w in words)
    low = [c for c in confs if c < LOW_CONF_THRESHOLD]
    metrics = {
        "word_count": len(words),
        "char_count": char_count,
        "mean_confidence": round(statistics.mean(confs), 2) if confs else 0.0,
        "median_confidence": round(statistics.median(confs), 2) if confs else 0.0,
        "low_confidence_word_ratio": round(len(low) / len(confs), 4) if confs else 1.0,
    }
    return metrics, words


def needs_retry(metrics: dict) -> bool:
    if not metrics:
        return True
    word_count = metrics.get("word_count")
    no_words = word_count is not None and int(word_count or 0) == 0
    return (
        float(metrics.get("mean_confidence") or 0) < LOW_CONF_THRESHOLD
        or float(metrics.get("low_confidence_word_ratio") or 0) > LOW_CONF_RATIO_THRESHOLD
        or no_words
    )


def page_xml_from_words(words: list[dict], *, image_filename: str, width: int = 0, height: int = 0) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">',
        f'  <Page imageFilename="{xml_escape.escape(image_filename)}" imageWidth="{width}" imageHeight="{height}">',
        '    <TextRegion id="r1">',
    ]
    for i, word in enumerate(words, 1):
        left, top, w, h = word["left"], word["top"], word["width"], word["height"]
        points = f"{left},{top} {left+w},{top} {left+w},{top+h} {left},{top+h}"
        text = xml_escape.escape(word["text"])
        conf = word.get("conf", 0)
        lines.extend([
            f'      <Word id="w{i}" custom="conf:{conf}">',
            f'        <Coords points="{points}"/>',
            '        <TextEquiv>',
            f'          <Unicode>{text}</Unicode>',
            '        </TextEquiv>',
            '      </Word>',
        ])
    lines.extend(['    </TextRegion>', '  </Page>', '</PcGts>', ''])
    return "\n".join(lines)


def image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        return (0, 0)


def run_tesseract_attempt(
    image: Path,
    out_dir: Path,
    *,
    lang: str,
    psm: int,
    timeout: int,
    runner: Callable[..., CommandResult] = run_command,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"psm{psm}"
    cmd = [
        "tesseract",
        str(image),
        str(base),
        "-l",
        lang,
        "--psm",
        str(psm),
        "txt",
        "tsv",
        "hocr",
        "alto",
    ]
    result = runner(cmd, timeout=timeout, sandbox=True, cwd=out_dir)
    attempt = {
        "psm": psm,
        "lang": lang,
        "command": cmd,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "seconds": round(result.seconds, 3),
        "stderr": result.stderr[-2000:],
        "status": "ok" if result.returncode == 0 and not result.timed_out else "failed",
        "outputs": {},
        "metrics": {},
    }
    if attempt["status"] != "ok":
        return attempt

    suffix_map = {"txt": ".txt", "tsv": ".tsv", "hocr": ".hocr", "alto": ".xml"}
    for key, suffix in suffix_map.items():
        p = base.with_suffix(suffix)
        if p.exists():
            attempt["outputs"][key] = str(p)

    tsv_path = base.with_suffix(".tsv")
    if tsv_path.exists():
        metrics, words = metrics_from_tsv(tsv_path.read_text(encoding="utf-8", errors="replace"))
        width, height = image_size(image)
        page_xml = page_xml_from_words(words, image_filename=image.name, width=width, height=height)
        page_xml_path = base.with_suffix(".page.xml")
        page_xml_path.write_text(page_xml, encoding="utf-8")
        attempt["outputs"]["page_xml"] = str(page_xml_path)
        attempt["metrics"] = metrics
    return attempt


def ocr_page_with_retries(
    image: Path,
    out_dir: Path,
    *,
    lang: str,
    psms: Sequence[int],
    timeout: int,
    attempt_fn: Callable[..., dict] = run_tesseract_attempt,
) -> dict:
    attempts = []
    selected = None
    for psm in psms:
        attempt_dir = out_dir / f"attempt-psm{psm}"
        attempt = attempt_fn(image, attempt_dir, lang=lang, psm=psm, timeout=timeout)
        attempts.append(attempt)
        if attempt.get("status") == "ok" and not needs_retry(attempt.get("metrics", {})):
            selected = attempt
            break
    if selected is None:
        ok_attempts = [a for a in attempts if a.get("status") == "ok"]
        selected = max(ok_attempts, key=lambda a: a.get("metrics", {}).get("mean_confidence", 0), default=attempts[-1])
    return {
        "image": str(image),
        "selected_psm": selected.get("psm") if selected else None,
        "selected_status": selected.get("status") if selected else "failed",
        "metrics": selected.get("metrics", {}) if selected else {},
        "outputs": selected.get("outputs", {}) if selected else {},
        "attempts": attempts,
        "needs_review": needs_retry(selected.get("metrics", {})) if selected else True,
    }


def extract_text_pdf(input_path: Path, out_dir: Path, timeout: int) -> dict | None:
    out_txt = out_dir / "pdftotext-layout.txt"
    result = run_command(["pdftotext", "-layout", str(input_path), str(out_txt)], timeout=timeout, sandbox=True, cwd=out_dir)
    if result.returncode != 0 or not out_txt.exists():
        return None
    text = out_txt.read_text(encoding="utf-8", errors="replace")
    chars = len(text.strip())
    if chars < 40:
        return None
    return {
        "method": "pdftotext-layout",
        "status": "ok",
        "metrics": {"char_count": chars, "word_count": len(text.split())},
        "outputs": {"txt": str(out_txt)},
        "command": ["pdftotext", "-layout", str(input_path), str(out_txt)],
        "seconds": result.seconds,
    }


def render_pdf(input_path: Path, out_dir: Path, *, dpi: int, timeout: int, max_pages: int | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "page"
    cmd = ["pdftoppm", "-png", "-r", str(dpi)]
    if max_pages:
        cmd += ["-f", "1", "-l", str(max_pages)]
    cmd += [str(input_path), str(prefix)]
    result = run_command(cmd, timeout=timeout, sandbox=True, cwd=out_dir)
    if result.returncode != 0:
        raise RuntimeError(f"pdftoppm failed: {result.stderr[-1000:]}")

    def page_sort_key(path: Path) -> tuple[int, str]:
        try:
            return (int(path.stem.rsplit("-", 1)[-1]), path.name)
        except ValueError:
            return (10**9, path.name)

    pages = sorted(out_dir.glob("page-*.png"), key=page_sort_key)
    if not pages:
        raise RuntimeError("pdftoppm produced no page images")
    return pages


def process_document(
    input_path: Path,
    out_root: Path,
    *,
    lang: str,
    psms: Sequence[int],
    dpi: int,
    timeout: int,
    max_pages: int | None = None,
) -> dict:
    doc_dir = document_output_dir(input_path, out_root)
    doc_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "input": str(input_path),
        "sha256": sha256_file(input_path),
        "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "status": "ok",
        "pages": [],
        "outputs": {},
        "errors": [],
    }
    try:
        if input_path.suffix.lower() in PDF_EXTS:
            text_pdf = extract_text_pdf(input_path, doc_dir, timeout)
            if text_pdf:
                record["method"] = "text_pdf"
                record["outputs"].update(text_pdf["outputs"])
                record["metrics"] = text_pdf["metrics"]
                write_document_manifest(doc_dir, record)
                return record
            images = render_pdf(input_path, doc_dir / "rendered", dpi=dpi, timeout=timeout, max_pages=max_pages)
        elif input_path.suffix.lower() in IMAGE_EXTS:
            images = [input_path]
        elif input_path.suffix.lower() in TEXT_EXTS:
            record["method"] = "plain_text"
            record["outputs"]["txt"] = str(input_path)
            record["metrics"] = {"char_count": len(input_path.read_text(encoding="utf-8", errors="replace"))}
            write_document_manifest(doc_dir, record)
            return record
        else:
            raise ValueError(f"Unsupported extension: {input_path.suffix}")

        record["method"] = "tesseract_ocr"
        for idx, image in enumerate(images, 1):
            page_out = doc_dir / f"page-{idx:04d}"
            page = ocr_page_with_retries(image, page_out, lang=lang, psms=psms, timeout=timeout)
            page["page"] = idx
            record["pages"].append(page)
        finalize_ocr_pages(record)
        write_document_manifest(doc_dir, record)
        return record
    except Exception as exc:
        record["status"] = "failed"
        record["errors"].append(str(exc))
        write_document_manifest(doc_dir, record)
        return record


def finalize_ocr_pages(record: dict) -> None:
    pages = record.get("pages") or []
    record["needs_review"] = any(p.get("needs_review") for p in pages)
    if pages and all(p.get("selected_status") != "ok" for p in pages):
        record["status"] = "failed"
        record["needs_review"] = True
        record.setdefault("errors", []).append("all OCR pages failed")


def iter_inputs(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in (PDF_EXTS | IMAGE_EXTS | TEXT_EXTS):
            yield p


def write_document_manifest(doc_dir: Path, record: dict) -> Path:
    path = doc_dir / "manifest.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_batch_manifest(out_root: Path, documents: list[dict]) -> Path:
    manifest = {
        "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "pipeline": "production_ocr_pipeline.py",
        "documents": documents,
        "summary": {
            "count": len(documents),
            "ok": sum(1 for d in documents if d.get("status") == "ok"),
            "failed": sum(1 for d in documents if d.get("status") == "failed"),
            "needs_review": sum(1 for d in documents if d.get("needs_review")),
        },
    }
    path = out_root / "batch-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def check_tools() -> dict:
    tools = {name: shutil.which(name) for name in ["tesseract", "pdftotext", "pdftoppm", "pdfinfo"]}
    tools["missing"] = [k for k, v in tools.items() if k != "missing" and not v]
    return tools


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="PDF/image/text file or directory")
    parser.add_argument("--output-dir", type=Path, default=Path("ocr-output"))
    parser.add_argument("--lang", default="ces+eng")
    parser.add_argument("--psm", dest="psms", action="append", type=int, help="Retry PSM; repeatable. Default: 3,6,11")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=120, help="Seconds per subprocess/page attempt")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    if args.check:
        tool_report = check_tools()
        print(json.dumps(tool_report, indent=2))
        return 0 if not tool_report["missing"] else 2

    if args.input is None:
        parser.error("input is required unless --check is used")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = list(iter_inputs(args.input))
    if not input_paths:
        print(f"ERROR: no supported input files found in {args.input}", file=sys.stderr)
        return 1

    documents = []
    for input_path in input_paths:
        documents.append(
            process_document(
                input_path,
                args.output_dir,
                lang=args.lang,
                psms=args.psms or DEFAULT_PSMS,
                dpi=args.dpi,
                timeout=args.timeout,
                max_pages=args.max_pages,
            )
        )
    manifest = write_batch_manifest(args.output_dir, documents)
    print(str(manifest))
    return 1 if any(d.get("status") == "failed" for d in documents) else 0


if __name__ == "__main__":
    raise SystemExit(main())

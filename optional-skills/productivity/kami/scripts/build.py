#!/usr/bin/env python3
"""kami build & check

Usage:
    python3 scripts/build.py                      # build all examples (HTML + diagrams + PPTX)
    python3 scripts/build.py resume               # build one template, print pages + fonts
    python3 scripts/build.py --check              # scan templates for CSS rule violations
    python3 scripts/build.py --check -v           # verbose (show each scanned file)
    python3 scripts/build.py --sync               # check CSS token drift across templates
    python3 scripts/build.py --verify             # build all + page count + font checks
    python3 scripts/build.py --verify resume-en   # single target full verification
    python3 scripts/build.py --check-placeholders path/to/doc.html
    python3 scripts/build.py --check-orphans      # scan example PDFs for orphan text
    python3 scripts/build.py --check-orphans path/to/doc.pdf
    python3 scripts/build.py --check-density       # warn on pages with >25% trailing whitespace
    python3 scripts/build.py --check-density path/to/doc.pdf
    python3 scripts/build.py --check-rhythm       # warn on monotonous slide sequences
    python3 scripts/build.py --check-rhythm slides slides-en
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from shared import COOL_GRAY_BLOCKLIST, DIAGRAMS, EXAMPLES, ROOT, TEMPLATES, TOKENS_FILE

# name -> (source, max_pages). max_pages=0 means no hard check.
HTML_TARGETS: dict[str, tuple[str, int]] = {
    # Chinese
    "one-pager":    ("one-pager.html", 1),
    "letter":       ("letter.html", 1),
    "long-doc":     ("long-doc.html", 0),
    "portfolio":    ("portfolio.html", 0),
    "resume":       ("resume.html", 2),
    # English
    "one-pager-en": ("one-pager-en.html", 1),
    "letter-en":    ("letter-en.html", 1),
    "long-doc-en":  ("long-doc-en.html", 0),
    "portfolio-en": ("portfolio-en.html", 0),
    "resume-en":    ("resume-en.html", 2),
    # Equity Report
    "equity-report":    ("equity-report.html", 3),
    "equity-report-en": ("equity-report-en.html", 3),
    # Changelog
    "changelog":    ("changelog.html", 2),
    "changelog-en": ("changelog-en.html", 2),
    # Slides (WeasyPrint)
    "slides-weasy":    ("slides-weasy.html", 0),
    "slides-weasy-en": ("slides-weasy-en.html", 0),
}
PPTX_TARGETS: dict[str, str] = {
    "slides":    "slides.py",
    "slides-en": "slides-en.py",
}

# Diagram HTMLs live in a separate directory and have no page-count contract.
DIAGRAM_TARGETS: dict[str, str] = {
    "diagram-architecture": "architecture.html",
    "diagram-flowchart":    "flowchart.html",
    "diagram-quadrant":     "quadrant.html",
    "diagram-bar-chart":    "bar-chart.html",
    "diagram-line-chart":   "line-chart.html",
    "diagram-donut-chart":  "donut-chart.html",
    "diagram-state-machine": "state-machine.html",
    "diagram-timeline":      "timeline.html",
    "diagram-swimlane":      "swimlane.html",
    "diagram-tree":          "tree.html",
    "diagram-layer-stack":   "layer-stack.html",
    "diagram-venn":          "venn.html",
    "diagram-candlestick":   "candlestick.html",
    "diagram-waterfall":     "waterfall.html",
}


# ------------------------- build -------------------------

def infer_author() -> str:
    """Infer author name from git config or environment.

    Priority:
    1. git config user.name
    2. KAMI_AUTHOR env var
    3. fallback to "Kami"
    """
    try:
        result = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except FileNotFoundError:
        pass

    if env_author := os.environ.get("KAMI_AUTHOR"):
        return env_author

    return "Kami"


def set_pdf_metadata(pdf_path: Path, author: str | None = None) -> None:
    """Set PDF metadata using pypdf, only if placeholders are still present."""
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        return

    if not pdf_path.exists():
        return

    reader = PdfReader(str(pdf_path))

    # Read existing metadata from WeasyPrint
    existing = reader.metadata or {}

    # Check if we need to update anything
    needs_update = False
    metadata = dict(existing)  # Copy all existing metadata

    # Only override author if it's still a placeholder
    if author and existing.get("/Author"):
        author_value = str(existing["/Author"])
        if "{{" in author_value and "}}" in author_value:
            metadata["/Author"] = author
            needs_update = True

    # Always set Producer and Creator to Kami
    if metadata.get("/Producer") != "Kami":
        metadata["/Producer"] = "Kami"
        needs_update = True
    if metadata.get("/Creator") != "Kami":
        metadata["/Creator"] = "Kami"
        needs_update = True

    if not needs_update:
        return

    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)

    writer.add_metadata(metadata)

    with open(pdf_path, "wb") as f:
        writer.write(f)


def build_html(name: str, source: str, max_pages: int,
               src_dir: Path = TEMPLATES) -> bool:
    try:
        from weasyprint import HTML
        from pypdf import PdfReader
    except ImportError:
        print("ERROR: missing deps: pip install weasyprint pypdf --break-system-packages")
        return False

    src = src_dir / source
    if not src.exists():
        print(f"ERROR: {name}: source not found ({src})")
        return False

    EXAMPLES.mkdir(parents=True, exist_ok=True)
    out = EXAMPLES / f"{name}.pdf"

    # weasyprint resolves @font-face relative to CWD. Run from the source dir
    # so fonts placed next to the HTML are found.
    HTML(str(src), base_url=str(src.parent)).write_pdf(str(out))

    # Set PDF metadata (only replaces placeholders, preserves filled values)
    author = infer_author()
    set_pdf_metadata(out, author=author)

    n = len(PdfReader(str(out)).pages)
    msg = f"OK: {name}: {n} pages"
    if max_pages and n > max_pages:
        msg = f"ERROR: {name}: {n} pages (limit {max_pages})"
        print(msg)
        return False
    print(msg)
    return True


def build_slides(name: str = "slides") -> bool:
    source = PPTX_TARGETS.get(name)
    if source is None:
        print(f"ERROR: {name}: unknown slides target")
        return False
    src = TEMPLATES / source
    if not src.exists():
        print(f"ERROR: {name}: source not found ({src})")
        return False

    EXAMPLES.mkdir(parents=True, exist_ok=True)
    out = EXAMPLES / f"{name}.pptx"
    result = subprocess.run(
        [sys.executable, str(src)],
        cwd=str(src.parent),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: {name}: {result.stderr.strip() or 'script failed'}")
        return False
    # The script writes output.pptx in cwd; move to examples/ under our name.
    generated = src.parent / "output.pptx"
    if generated.exists():
        generated.replace(out)
        print(f"OK: {name}: generated {out.name}")
        return True
    print(f"ERROR: {name}: output.pptx not produced")
    return False


def build_all() -> int:
    failures = 0
    for name, (source, max_pages) in HTML_TARGETS.items():
        if not build_html(name, source, max_pages):
            failures += 1
    for name, source in DIAGRAM_TARGETS.items():
        if not build_html(name, source, 0, src_dir=DIAGRAMS):
            failures += 1
    for name in PPTX_TARGETS:
        if not build_slides(name):
            failures += 1
    return failures


def build_single(name: str) -> int:
    if name in HTML_TARGETS:
        source, max_pages = HTML_TARGETS[name]
        ok = build_html(name, source, max_pages)
        if ok:
            show_fonts(EXAMPLES / f"{name}.pdf")
        return 0 if ok else 1
    if name in DIAGRAM_TARGETS:
        source = DIAGRAM_TARGETS[name]
        ok = build_html(name, source, 0, src_dir=DIAGRAMS)
        return 0 if ok else 1
    if name in PPTX_TARGETS:
        return 0 if build_slides(name) else 1
    known = list(HTML_TARGETS) + list(DIAGRAM_TARGETS) + list(PPTX_TARGETS)
    print(f"ERROR: unknown target: {name}. Known: {', '.join(known)}")
    return 2


def show_fonts(pdf: Path) -> None:
    if not pdf.exists():
        return
    try:
        out = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True, check=False)
        if out.returncode == 0:
            print("--- pdffonts ---")
            print(out.stdout.rstrip())
    except FileNotFoundError:
        pass  # pdffonts not installed; silent


# ------------------------- sync -------------------------

ROOT_BLOCK = re.compile(r":root\s*\{([^}]*)\}", re.DOTALL)
CSS_VAR = re.compile(r"--([\w-]+)\s*:\s*([^;]+);")
PY_RGB = re.compile(
    r"^([A-Z][A-Z_]+)\s*=\s*RGBColor\(\s*0x([0-9a-fA-F]{2})\s*,"
    r"\s*0x([0-9a-fA-F]{2})\s*,\s*0x([0-9a-fA-F]{2})\s*\)",
    re.MULTILINE,
)
# Python const name -> tokens.json key. Only constants that mirror a CSS token.
PY_TOKEN_MAP = {
    "PARCHMENT": "--parchment",
    "IVORY": "--ivory",
    "BRAND": "--brand",
    "NEAR_BLACK": "--near-black",
    "DARK_WARM": "--dark-warm",
    "CHARCOAL": "--charcoal",
    "OLIVE": "--olive",
    "STONE": "--stone",
}


def sync_check(verbose: bool = False) -> int:
    if not TOKENS_FILE.exists():
        print(f"ERROR: tokens.json not found at {TOKENS_FILE.relative_to(ROOT)}")
        return 1

    try:
        canonical: dict[str, str] = json.loads(TOKENS_FILE.read_text())
    except json.JSONDecodeError as exc:
        print(f"ERROR: tokens.json is malformed: {exc}")
        return 1

    targets: list[Path] = list(TEMPLATES.glob("*.html"))
    if DIAGRAMS.exists():
        targets.extend(DIAGRAMS.glob("*.html"))
    py_targets: list[Path] = list(TEMPLATES.glob("*.py"))

    drift: list[tuple[str, str, str, str]] = []  # (file, token, expected, actual)

    for path in sorted(targets):
        text = path.read_text(encoding="utf-8", errors="replace")
        block_match = ROOT_BLOCK.search(text)
        if not block_match:
            if verbose:
                print(f"  (skip {path.name}: no :root block)")
            continue
        root_block = block_match.group(1)
        found: dict[str, str] = {
            m.group(1): m.group(2).strip()
            for m in CSS_VAR.finditer(root_block)
        }
        rel = path.relative_to(ROOT)
        for token, expected in canonical.items():
            name = token.lstrip("-")
            actual = found.get(name)
            # Only flag if the template defines the token but with a wrong value.
            # Templates that don't use a token don't need to define it.
            if actual is not None and actual.lower() != expected.lower():
                drift.append((str(rel), token, expected, actual))

    for path in sorted(py_targets):
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(ROOT)
        for m in PY_RGB.finditer(text):
            name = m.group(1)
            token = PY_TOKEN_MAP.get(name)
            if token is None:
                continue
            expected = canonical.get(token)
            if expected is None:
                continue
            actual = f"#{m.group(2)}{m.group(3)}{m.group(4)}"
            if actual.lower() != expected.lower():
                drift.append((str(rel), token, expected, actual))

    if not drift:
        scanned = len(targets) + len(py_targets)
        print(f"OK: tokens in sync across {scanned} template(s)")
        return 0

    print(f"\n[token-drift] {len(drift)}")
    for file, token, expected, actual in drift:
        print(f"  {file}: {token} expected {expected}, got {actual}")

    return 1


# ------------------------- verify -------------------------

PLACEHOLDER = re.compile(r"\{\{[^}]+\}\}")

# Primary fonts expected in embedded PDF font names
CN_PRIMARY_FONTS = {"TsangerJinKai02"}
EN_PRIMARY_FONTS = {"Charter"}


def _pdf_font_names(pdf_path: Path) -> set[str]:
    def _resolve_pdf_obj(obj):
        if obj is None:
            return None
        try:
            return obj.get_object() if hasattr(obj, "get_object") else obj
        except Exception:
            return obj

    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        fonts: set[str] = set()
        for page in reader.pages:
            resources = _resolve_pdf_obj(page.get("/Resources"))
            if resources is None or not hasattr(resources, "get"):
                continue
            font_dict = _resolve_pdf_obj(resources.get("/Font"))
            if font_dict is None or not hasattr(font_dict, "values"):
                continue
            for obj in font_dict.values():
                resolved = _resolve_pdf_obj(obj)
                if resolved is None or not hasattr(resolved, "get"):
                    continue
                base = resolved.get("/BaseFont")
                if base:
                    fonts.add(str(base).lstrip("/"))
        return fonts
    except Exception as exc:
        print(f"  WARN: could not read font names from PDF: {exc}")
        return set()


def _check_font_sources(html_path: Path) -> list[str]:
    """Return list of local @font-face src files that are missing on disk."""
    text = html_path.read_text(encoding="utf-8", errors="replace")
    missing: list[str] = []
    for url in re.findall(r"""url\(["']?([^"')]+)["']?\)""", text):
        if url.startswith(("http://", "https://", "data:", "#")):
            continue
        resolved = (html_path.parent / url).resolve()
        if not resolved.exists():
            missing.append(url)
    return missing


def verify_target(name: str, source: str, max_pages: int, src_dir: Path) -> list[str]:
    issues: list[str] = []
    src = src_dir / source
    if not src.exists():
        issues.append(f"source not found: {src}")
        return issues

    try:
        from weasyprint import HTML
        from pypdf import PdfReader
    except ImportError:
        issues.append("missing deps: pip install weasyprint pypdf --break-system-packages")
        return issues

    EXAMPLES.mkdir(parents=True, exist_ok=True)
    out = EXAMPLES / f"{name}.pdf"

    # Warn about missing local font files before rendering
    missing_fonts = _check_font_sources(src)
    if missing_fonts:
        for mf in missing_fonts:
            print(f"  [FONT MISS] {name}: {mf} not found")
        print(f"  [FONT MISS] To fix: bash scripts/ensure-fonts.sh")
        print(f"  [FONT MISS] Or install fallback: brew install --cask font-source-han-serif-sc")

    HTML(str(src), base_url=str(src.parent)).write_pdf(str(out))

    # Set PDF metadata (only replaces placeholders, preserves filled values)
    author = infer_author()
    set_pdf_metadata(out, author=author)

    # page count check
    n = len(PdfReader(str(out)).pages)
    if max_pages and n > max_pages:
        over = n - max_pages
        hint = ""
        if "resume" in name and over == 1:
            hint = '; add class="resume--dense" to <body> or tighten .proj-text line-height to 1.38'
        issues.append(f"page overflow: {n} pages (limit {max_pages}){hint}")

    # font check
    embedded = _pdf_font_names(out)
    fallback_present = any(
        kw in font for font in embedded
        for kw in ("Georgia", "Palatino", "TsangerJinKai", "YuMincho", "Hiragino", "SourceHan", "Noto", "Charter", "Songti")
    )

    # Diagram templates are language-neutral and often rely on fallback stacks,
    # so only enforce that at least one recognizable serif/sans fallback exists.
    is_diagram = src_dir == DIAGRAMS
    if is_diagram:
        if not fallback_present:
            issues.append(f"no recognizable font embedded in {out.name}")
        return issues

    is_en = name.endswith("-en")
    expected = EN_PRIMARY_FONTS if is_en else CN_PRIMARY_FONTS
    if not any(exp in font_name for exp in expected for font_name in embedded):
        primary = next(iter(expected))
        if not fallback_present:
            issues.append(f"no recognizable font embedded in {out.name}")
        else:
            issues.append(f"primary font ({primary}) not embedded; using fallback")

    return issues


def verify_slides_target(name: str) -> list[str]:
    return [] if build_slides(name) else ["slides build failed"]


def verify_all(target: str | None = None) -> int:
    targets_to_run: dict[str, tuple[str, int, Path] | None] = {}
    if target:
        if target in HTML_TARGETS:
            src, mp = HTML_TARGETS[target]
            targets_to_run[target] = (src, mp, TEMPLATES)
        elif target in DIAGRAM_TARGETS:
            targets_to_run[target] = (DIAGRAM_TARGETS[target], 0, DIAGRAMS)
        elif target in PPTX_TARGETS:
            targets_to_run[target] = None
        else:
            print(f"ERROR: unknown target: {target}")
            return 2
    else:
        for name, (src, mp) in HTML_TARGETS.items():
            targets_to_run[name] = (src, mp, TEMPLATES)
        for name, src in DIAGRAM_TARGETS.items():
            targets_to_run[name] = (src, 0, DIAGRAMS)
        for name in PPTX_TARGETS:
            targets_to_run[name] = None

    failures = 0
    rows: list[tuple[str, str]] = []
    for name, config in targets_to_run.items():
        if config is None:
            issues = verify_slides_target(name)
        else:
            source, max_pages, src_dir = config
            issues = verify_target(name, source, max_pages, src_dir)
        if issues:
            rows.append((f"ERROR: {name}", "; ".join(issues)))
            failures += 1
        else:
            rows.append((f"OK: {name}", "ok"))

    for status, detail in rows:
        print(f"{status}: {detail}")

    return 0 if failures == 0 else 1


def check_placeholders(paths: list[str]) -> int:
    if not paths:
        print("ERROR: provide at least one HTML file to scan")
        return 2

    failures = 0
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            print(f"ERROR: {raw}: file not found")
            failures += 1
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = list(dict.fromkeys(PLACEHOLDER.findall(text)))
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        if hits:
            print(f"ERROR: {rel}: unfilled placeholder(s): {', '.join(hits)}")
            failures += 1
        else:
            print(f"OK: {rel}: no placeholders")

    return 0 if failures == 0 else 1


# ------------------------- orphan check -------------------------

def check_orphans(paths: list[str]) -> int:
    """Scan PDF for text blocks whose last line has <= 2 words and < 15 chars."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("ERROR: PyMuPDF required: pip install pymupdf --break-system-packages")
        return 2

    if not paths:
        # Default: scan all example PDFs
        if EXAMPLES.exists():
            paths = [str(p) for p in sorted(EXAMPLES.glob("*.pdf"))]
        if not paths:
            print("ERROR: no PDF files to scan")
            return 2

    total = 0
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            print(f"ERROR: {raw}: not found")
            continue
        doc = fitz.open(str(path))
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("blocks")
            for bx0, by0, bx1, by1, text, block_no, block_type in blocks:
                if block_type != 0:  # text blocks only
                    continue
                lines = text.strip().splitlines()
                if len(lines) < 2:
                    continue
                last = lines[-1].strip()
                words = last.split()
                if len(words) <= 2 and len(last) < 15:
                    total += 1
                    print(f"  {rel} p{page_num + 1}: orphan: \"{last}\" ({len(words)} word(s), {len(last)} chars)")
        doc.close()

    if total == 0:
        print(f"OK: no orphans found across {len(paths)} PDF(s)")
        return 0

    print(f"\n{total} orphan(s) found across {len(paths)} PDF(s)")
    return 1


# ------------------------- density check -------------------------

# Parchment background RGB for pixel comparison.
_BG_R, _BG_G, _BG_B = 0xF5, 0xF4, 0xED
_BG_TOLERANCE = 10


def check_density(paths: list[str]) -> int:
    """Scan PDF pages for sparse content (large trailing whitespace from
    break-inside:avoid pushing content to the next page)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("ERROR: PyMuPDF required: pip install pymupdf --break-system-packages")
        return 2

    if not paths:
        if EXAMPLES.exists():
            paths = [str(p) for p in sorted(EXAMPLES.glob("*.pdf"))]
        if not paths:
            print("ERROR: no PDF files to scan")
            return 2

    warnings = 0
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            print(f"ERROR: {raw}: not found")
            continue
        doc = fitz.open(str(path))
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        for page_num in range(len(doc)):
            if page_num == 0:
                continue
            page = doc[page_num]
            pix = page.get_pixmap(dpi=36)
            w, h = pix.width, pix.height
            if h == 0:
                continue
            samples = pix.samples
            stride = pix.stride
            n = pix.n

            last_content_y = 0
            for y in range(h - 1, -1, -1):
                row_start = y * stride
                is_bg = True
                for x in range(0, w, 4):
                    offset = row_start + x * n
                    if (abs(samples[offset] - _BG_R) > _BG_TOLERANCE
                            or abs(samples[offset + 1] - _BG_G) > _BG_TOLERANCE
                            or abs(samples[offset + 2] - _BG_B) > _BG_TOLERANCE):
                        is_bg = False
                        break
                if not is_bg:
                    last_content_y = y
                    break

            empty = (h - last_content_y) / h
            if empty > 0.50:
                print(f"  SPARSE: {rel} p{page_num + 1}: {empty:.0%} trailing whitespace")
                warnings += 1
            elif empty > 0.25:
                print(f"  WARN: {rel} p{page_num + 1}: {empty:.0%} trailing whitespace")
                warnings += 1
        doc.close()

    if warnings == 0:
        print(f"OK: no density issues across {len(paths)} PDF(s)")
        return 0

    print(f"\n{warnings} density warning(s) across {len(paths)} PDF(s)")
    return 1


# ------------------------- check -------------------------

RGBA_BG_DIRECT = re.compile(r"background(?:-color)?\s*:\s*[^;]*rgba\s*\(", re.IGNORECASE)
RGBA_VAR_DEF = re.compile(r"--([\w-]+)\s*:\s*[^;]*rgba\s*\(", re.IGNORECASE)
BG_VAR_USE = re.compile(r"background(?:-color)?\s*:\s*[^;]*var\s*\(\s*--([\w-]+)", re.IGNORECASE)
RGBA_BORDER_DIRECT = re.compile(r"border(?:-\w+)?\s*:\s*[^;]*rgba\s*\(", re.IGNORECASE)
BORDER_VAR_USE = re.compile(r"border(?:-\w+)?\s*:\s*[^;]*var\s*\(\s*--([\w-]+)", re.IGNORECASE)
LINE_HEIGHT_LOOSE = re.compile(r"line-height\s*:\s*1\.[6-9]\d*", re.IGNORECASE)
UNICODE_ARROW = re.compile(r"→")  # U+2192; should not appear in EN template body
HEX_ANY = re.compile(r"#[0-9a-fA-F]{3,6}\b")
# Thin closed border: border shorthand (not single-side) with sub-1pt width — pitfall #2
THIN_CLOSED_BORDER = re.compile(
    r"border(?!-(?:left|right|top|bottom))\s*:\s*[^;]*0\.\d+pt",
    re.IGNORECASE,
)
BORDER_RADIUS_PROP = re.compile(r"border-radius\s*:", re.IGNORECASE)


@dataclass
class Finding:
    file: Path
    line: int
    rule: str
    excerpt: str


def scan_file(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # Pass 1: collect variable names that hold rgba(...) so the tag-background
    # bug can be detected through one level of indirection.
    rgba_vars: set[str] = set()
    for raw in lines:
        m = RGBA_VAR_DEF.search(raw)
        if m:
            rgba_vars.add(m.group(1))

    is_en = path.name.endswith("-en.html")

    # Pass 2: per-line rule checks
    for i, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue

        if RGBA_BG_DIRECT.search(raw):
            findings.append(Finding(path, i, "rgba-background",
                                    "rgba() used directly on background (tag double-rectangle bug)"))

        bg_var = BG_VAR_USE.search(raw)
        if bg_var and bg_var.group(1) in rgba_vars:
            findings.append(Finding(path, i, "rgba-background",
                                    f"background: var(--{bg_var.group(1)}) resolves to rgba() (tag double-rectangle bug)"))

        if RGBA_BORDER_DIRECT.search(raw):
            findings.append(Finding(path, i, "rgba-border",
                                    "rgba() used on border (violates solid-color invariant)"))

        border_var = BORDER_VAR_USE.search(raw)
        if border_var and border_var.group(1) in rgba_vars:
            findings.append(Finding(path, i, "rgba-border",
                                    f"border: var(--{border_var.group(1)}) resolves to rgba() (solid-color invariant)"))

        if is_en and UNICODE_ARROW.search(raw):
            # skip CSS comment lines (/* ... */) and the arrow-in-CSS-content patterns
            stripped = raw.lstrip()
            if not stripped.startswith("/*") and not stripped.startswith("*") and "content:" not in raw:
                findings.append(Finding(path, i, "arrow-unicode-in-en",
                                        "→ (U+2192) in English template; use 'to' or '->' per patterns §2"))

        m = LINE_HEIGHT_LOOSE.search(raw)
        if m:
            findings.append(Finding(path, i, "line-height-too-loose",
                                    f"{m.group(0)} exceeds 1.55 ceiling"))

        for hex_match in HEX_ANY.finditer(raw):
            h = hex_match.group(0).lower()
            if h in COOL_GRAY_BLOCKLIST:
                findings.append(Finding(path, i, "cool-gray",
                                        f"{h} is a cool / neutral gray, use warm undertone"))

    # Pass 3: thin-border-radius block scan (pitfall #2 double-ring).
    # For each thin closed border line, scan backward to the block open and
    # forward to the block close, checking for border-radius in the same block.
    for i, raw in enumerate(lines):
        if not THIN_CLOSED_BORDER.search(raw):
            continue
        if "skip-thin-border-radius" in raw:
            continue
        found = False
        # Scan backward; stop at { or } (entering/leaving a block).
        for j in range(i - 1, max(0, i - 6) - 1, -1):
            if "{" in lines[j] or "}" in lines[j]:
                break
            if BORDER_RADIUS_PROP.search(lines[j]):
                found = True
                break
        # Scan forward; stop at } (leaving the block).
        if not found:
            for j in range(i + 1, min(len(lines), i + 6)):
                if "}" in lines[j]:
                    break
                if BORDER_RADIUS_PROP.search(lines[j]):
                    found = True
                    break
        if found:
            findings.append(Finding(path, i + 1, "thin-border-radius",
                "thin border (<1pt) with border-radius -- pitfall #2 double-ring risk"))
    return findings


def check_all(verbose: bool) -> int:
    targets: list[Path] = []
    for p in TEMPLATES.glob("*.html"):
        targets.append(p)
    for p in TEMPLATES.glob("*.py"):
        targets.append(p)
    if DIAGRAMS.exists():
        for p in DIAGRAMS.glob("*.html"):
            targets.append(p)

    findings: list[Finding] = []
    for p in sorted(targets):
        file_findings = scan_file(p)
        findings.extend(file_findings)
        if verbose:
            print(f"scanned {p.relative_to(ROOT)}: {len(file_findings)} finding(s)")

    if not findings:
        print(f"OK: no violations across {len(targets)} templates")
        return 0

    by_rule: dict[str, list[Finding]] = {}
    for f in findings:
        by_rule.setdefault(f.rule, []).append(f)

    print(f"ERROR: {len(findings)} violation(s) across {len({f.file for f in findings})} file(s)")
    for rule, items in by_rule.items():
        print(f"\n[{rule}] {len(items)}")
        for f in items:
            rel = f.file.relative_to(ROOT)
            print(f"  {rel}:{f.line}  {f.excerpt}")
    return 1


# ------------------------- rhythm check -------------------------

# Layout functions that count as "divider" slides (break monotony).
_DIVIDER_FUNCS = {"chapter_slide"}
# Layout functions that count as "density variation" slides.
_DENSITY_VARIATION_FUNCS = {"quote_slide", "metrics_slide"}
# Layout function call pattern in slides.py source.
_SLIDE_CALL = re.compile(r"^\s*(\w+_slide)\s*\(")


def _parse_slide_sequence(src: Path) -> list[str]:
    """Return the ordered list of slide-function names called in main()."""
    text = src.read_text(encoding="utf-8", errors="replace")
    in_main = False
    sequence: list[str] = []
    for line in text.splitlines():
        if re.match(r"^def main\s*\(", line):
            in_main = True
            continue
        if in_main and re.match(r"^def \w", line):
            break
        if in_main:
            m = _SLIDE_CALL.match(line)
            if m:
                sequence.append(m.group(1))
    return sequence


def check_rhythm(targets: list[str]) -> int:
    """Scan slide templates for monotony: too many consecutive content_slides,
    missing dividers, and missing density variation.

    Usage: python3 scripts/build.py --check-rhythm [slides] [slides-en]
    When no targets are given, checks all PPTX_TARGETS.
    """
    names = targets if targets else list(PPTX_TARGETS.keys())
    failures = 0

    for name in names:
        source = PPTX_TARGETS.get(name)
        if source is None:
            print(f"ERROR: {name}: not a known slides target")
            failures += 1
            continue
        src = TEMPLATES / source
        if not src.exists():
            print(f"ERROR: {name}: source not found ({src})")
            failures += 1
            continue

        seq = _parse_slide_sequence(src)
        if not seq:
            print(f"WARN: {name}: no slide calls found in main()")
            continue

        issues: list[str] = []

        # Rule 1: no run of more than 5 consecutive content_slides.
        run = 0
        max_run = 0
        for fn in seq:
            if fn == "content_slide":
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        if max_run > 5:
            issues.append(f"longest content_slide run is {max_run} (limit 5)")

        # Rule 2: decks >= 12 slides need at least one chapter_slide divider.
        if len(seq) >= 12 and not any(fn in _DIVIDER_FUNCS for fn in seq):
            issues.append(f"{len(seq)} slides with no chapter_slide divider")

        # Rule 3: deck must contain at least one density-variation slide.
        if not any(fn in _DENSITY_VARIATION_FUNCS for fn in seq):
            issues.append("no quote_slide or metrics_slide for density variation")

        if issues:
            for issue in issues:
                print(f"WARN: {name}: {issue}")
            failures += 1
        else:
            print(f"OK: {name}: rhythm ok ({len(seq)} slides, max run {max_run})")

    return 0 if failures == 0 else 1


# ------------------------- entry -------------------------

def main(argv: list[str]) -> int:
    args = argv[1:]
    if not args:
        return build_all()
    if args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if args[0] == "--check":
        verbose = "-v" in args[1:] or "--verbose" in args[1:]
        css_result = check_all(verbose)
        sync_result = sync_check(verbose)
        return max(css_result, sync_result)
    if args[0] == "--sync":
        verbose = "-v" in args[1:] or "--verbose" in args[1:]
        return sync_check(verbose)
    if args[0] == "--verify":
        target = args[1] if len(args) > 1 and not args[1].startswith("-") else None
        return verify_all(target)
    if args[0] == "--check-orphans":
        return check_orphans(args[1:])
    if args[0] == "--check-density":
        return check_density(args[1:])
    if args[0] in ("--check-placeholders", "--verify-filled"):
        return check_placeholders(args[1:])
    if args[0] == "--check-rhythm":
        slide_targets = [a for a in args[1:] if not a.startswith("-")]
        return check_rhythm(slide_targets)
    return build_single(args[0])


if __name__ == "__main__":
    sys.exit(main(sys.argv))

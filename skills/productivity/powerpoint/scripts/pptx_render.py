#!/usr/bin/env python3
"""Render every slide of a .pptx to per-slide PNG images.

Pipeline: LibreOffice (soffice --headless --convert-to pdf) turns the deck
into a PDF, then poppler (pdftoppm, or pdftocairo as an alternate) splits
the PDF into one PNG per slide.

Output is JSON. When both tools are present:
  {"rendered": true, "files": ["render/slide-1.png", ...]}
When either tool is missing the script still exits 0 and reports:
  {"rendered": false, "missing": ["soffice"], "guidance": "..."}
so callers can degrade gracefully (fall back to pptx_read.py --outline).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile


def find_tools():
    """Return (soffice, splitter, missing) using shutil.which."""
    soffice = shutil.which("soffice")
    splitter = shutil.which("pdftoppm") or shutil.which("pdftocairo")
    missing = []
    if not soffice:
        missing.append("soffice")
    if not splitter:
        missing.append("pdftoppm (or pdftocairo)")
    return soffice, splitter, missing


def render(pptx_path, out_dir, prefix, dpi):
    soffice, splitter, missing = find_tools()
    if missing:
        return {
            "rendered": False, "missing": missing,
            "guidance": "Install LibreOffice (soffice) and poppler-utils "
                        "(pdftoppm/pdftocairo) to render slides. Without "
                        "them, verify decks with pptx_read.py --outline "
                        "instead.",
        }

    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf",
             "--outdir", tmp, pptx_path],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=300)
        pdfs = glob.glob(os.path.join(tmp, "*.pdf"))
        if proc.returncode != 0 or not pdfs:
            raise SystemExit(f"soffice PDF conversion failed: {proc.stderr}")
        pdf = pdfs[0]
        out_prefix = os.path.join(out_dir, prefix)
        proc = subprocess.run(
            [splitter, "-png", "-r", str(dpi), pdf, out_prefix],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=300)
        if proc.returncode != 0:
            raise SystemExit(f"{os.path.basename(splitter)} failed: "
                             f"{proc.stderr}")

    files = sorted(glob.glob(out_prefix + "*.png"))
    return {"rendered": True, "files": files}


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Render each slide of a .pptx to a PNG via "
                    "soffice + pdftoppm/pdftocairo.",
        epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pptx", help="path to the .pptx file")
    parser.add_argument("--outdir", default="render",
                        help="directory for PNGs (default: ./render)")
    parser.add_argument("--prefix", default="slide",
                        help="PNG filename prefix (default: slide)")
    parser.add_argument("--dpi", type=int, default=100,
                        help="render resolution (default: 100)")
    args = parser.parse_args(argv)

    result = render(args.pptx, args.outdir, args.prefix, args.dpi)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

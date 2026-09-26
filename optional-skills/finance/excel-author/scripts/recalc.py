#!/usr/bin/env python3
"""Recalculate an .xlsx file's formulas using LibreOffice headless.

Usage: python recalc.py <path.xlsx> [timeout_seconds]

openpyxl writes formula strings but does not compute them. Downstream scripts
that open the file with data_only=True get None for every formula cell until
something has actually calculated the workbook. Excel does this on open;
headless pipelines need LibreOffice (or similar) to do it explicitly.

Exits 0 on success (workbook recomputed and resaved in place), non-zero on
failure. Writes status JSON to stdout either way.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def find_libreoffice() -> str | None:
    for cmd in ("libreoffice", "soffice"):
        path = shutil.which(cmd)
        if path:
            return path
    return None


def recalc(xlsx_path: str, timeout: int = 60) -> dict:
    src = Path(xlsx_path).resolve()
    if not src.exists():
        return {"status": "error", "error": f"File not found: {src}"}

    lo = find_libreoffice()
    if lo is None:
        return {
            "status": "error",
            "error": "libreoffice not found on PATH — install it or recalc in a real Excel session",
        }

    with tempfile.TemporaryDirectory() as td:
        try:
            subprocess.run(
                [
                    lo,
                    "--headless",
                    "--calc",
                    "--convert-to",
                    "xlsx",
                    str(src),
                    "--outdir",
                    td,
                ],
                check=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": f"libreoffice timed out after {timeout}s"}
        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "error": f"libreoffice exited {e.returncode}: {e.stderr.decode(errors='replace')[:500]}",
            }

        produced = Path(td) / src.name
        if not produced.exists():
            return {"status": "error", "error": "libreoffice did not produce output file"}

        shutil.copy(produced, src)

    return {"status": "success", "file": str(src)}


def main():
    if len(sys.argv) < 2:
        print("Usage: python recalc.py <path.xlsx> [timeout_seconds]", file=sys.stderr)
        sys.exit(2)
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    result = recalc(sys.argv[1], timeout=timeout)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()

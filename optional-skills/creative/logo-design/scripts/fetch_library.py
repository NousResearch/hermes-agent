from __future__ import annotations

"""Fetch the upstream reference-logo library into this skill's assets/library/.

The 1,400+ SVG logos used by search_library.py / preview_sheet.py / svg_audit.py
are third-party trademarks (see references/TRADEMARKS.md) and are therefore
not vendored with the skill. This script downloads the pinned upstream
tarball once and extracts only skills/logo-design/assets/library/** into
<skill-dir>/assets/library/, where the upstream scripts expect it.

Usage:
    python3 <skill-dir>/scripts/fetch_library.py [--force] [--dest DIR]

Prints exactly one JSON object on stdout:
    {"status": "fetched" | "present", "library_dir": ..., "svg_count": N, "sha": ...}
or, on failure (non-zero exit):
    {"status": "error", "error": "..."}

Standard library only; works on Python 3.9+.
"""

import argparse
import json
import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

REPO = "kaankiziltug/logo-design-skill"
SHA = "0ecf52e9a4b3ac92b714f7cc6e3148ab8c774134"
TARBALL_URL = f"https://codeload.github.com/{REPO}/tar.gz/{SHA}"
LIBRARY_SUBPATH = "skills/logo-design/assets/library/"
MIN_SVG_COUNT = 1400
MARKER_NAME = ".upstream-sha"
USER_AGENT = "hermes-agent-logo-design-skill/1.0 (+https://github.com/NousResearch/hermes-agent)"

SKILL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DEST = SKILL_DIR / "assets" / "library"
TRADEMARKS_SRC = SKILL_DIR / "references" / "TRADEMARKS.md"


def emit(payload: dict, code: int = 0) -> int:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return code


def count_svgs(library_dir: Path) -> int:
    svg_dir = library_dir / "svg"
    if not svg_dir.is_dir():
        return 0
    return sum(1 for p in svg_dir.iterdir() if p.suffix.lower() == ".svg" and p.is_file())


def is_present(library_dir: Path) -> bool:
    marker = library_dir / MARKER_NAME
    if not (library_dir / "catalog.json").is_file() or not marker.is_file():
        return False
    try:
        with open(marker, encoding="utf-8") as fh:
            return fh.read().strip() == SHA
    except OSError:
        return False


def scratch_dir(dest: Path) -> Path:
    """Directory for the temporary tarball: the skill's assets/ dir, or $HERMES_HOME/cache."""
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        candidate = Path(hermes_home) / "cache"
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            pass
    assets = dest.parent
    assets.mkdir(parents=True, exist_ok=True)
    return assets


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, open(target, "wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 256)


def _library_relpath(member_name: str) -> str | None:
    """Return the path relative to the library root, or None if the member is outside it."""
    parts = member_name.split("/", 1)
    if len(parts) != 2:
        return None
    rest = parts[1]
    if not rest.startswith(LIBRARY_SUBPATH):
        return None
    rel = rest[len(LIBRARY_SUBPATH):]
    return rel or None


def _safe_relpath(rel: str) -> bool:
    if not rel or rel.startswith(("/", "\\")):
        return False
    pieces = rel.replace("\\", "/").split("/")
    return all(piece not in ("", ".", "..") for piece in pieces)


def extract_library(tarball: Path, dest: Path) -> int:
    """Extract only library members into dest. Returns number of files written."""
    written = 0
    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()
    use_filter = hasattr(tarfile, "data_filter")
    with tarfile.open(tarball, mode="r:gz") as archive:
        for member in archive:
            rel = _library_relpath(member.name)
            if rel is None or not _safe_relpath(rel):
                continue
            if member.issym() or member.islnk():
                continue
            if member.isdir():
                (dest / rel).mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            target = (dest / rel)
            if dest_resolved not in target.resolve().parents:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            member = archive.getmember(member.name)
            if use_filter:
                member = tarfile.data_filter(member, str(dest_resolved))  # type: ignore[attr-defined]
                if member is None:
                    continue
            source = archive.extractfile(member)
            if source is None:
                continue
            with source, open(target, "wb") as out:
                shutil.copyfileobj(source, out)
            written += 1
    return written


def copy_trademark_notice(dest: Path) -> None:
    if TRADEMARKS_SRC.is_file():
        shutil.copyfile(TRADEMARKS_SRC, dest / "TRADEMARKS.md")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download the pinned upstream logo library into this skill's assets/library/.",
    )
    parser.add_argument("--force", action="store_true", help="re-download even if the library is present")
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="destination directory (default: <skill-dir>/assets/library)",
    )
    args = parser.parse_args(argv)
    dest: Path = args.dest.expanduser().resolve()

    if not args.force and is_present(dest):
        return emit({
            "status": "present",
            "library_dir": str(dest),
            "svg_count": count_svgs(dest),
            "sha": SHA,
        })

    tmp_dir = scratch_dir(dest)
    tarball = tmp_dir / f"logo-design-library-{SHA[:12]}.tar.gz.part"
    try:
        try:
            download(TARBALL_URL, tarball)
        except (urllib.error.URLError, OSError) as exc:
            return emit({"status": "error", "error": f"download failed: {exc}", "url": TARBALL_URL}, 1)

        if args.force and dest.is_dir():
            shutil.rmtree(dest)
        try:
            written = extract_library(tarball, dest)
        except (tarfile.TarError, OSError) as exc:
            return emit({"status": "error", "error": f"extraction failed: {exc}"}, 1)
    finally:
        try:
            tarball.unlink()
        except OSError:
            pass

    svg_count = count_svgs(dest)
    missing = [name for name in ("catalog.json", "stats.json") if not (dest / name).is_file()]
    if missing or svg_count < MIN_SVG_COUNT:
        return emit({
            "status": "error",
            "error": f"library incomplete: missing={missing} svg_count={svg_count} (need >= {MIN_SVG_COUNT})",
            "library_dir": str(dest),
            "files_written": written,
        }, 2)

    copy_trademark_notice(dest)
    with open(dest / MARKER_NAME, "w", encoding="utf-8") as fh:
        fh.write(SHA + "\n")

    return emit({
        "status": "fetched",
        "library_dir": str(dest),
        "svg_count": svg_count,
        "sha": SHA,
    })


if __name__ == "__main__":
    sys.exit(main())

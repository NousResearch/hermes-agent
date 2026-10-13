#!/usr/bin/env -S bash -c 'exec "$BASH" "$(dirname "$0")/run-in-hermes-env" python3 "$0" "$@"'
"""Splice the canonical store-root resolver into the standalone bootstraps.

The resolver mirrors ``hermes_constants.get_default_hermes_root()``. Two
bootstrap scripts cannot import it — ``install.sh`` is piped to bash and
``install.ps1`` is fetched with ``irm | iex``, so neither has a sibling to
source — and they therefore carry a copy. Hand-maintaining two copies is what
this generator removes: ``scripts/bootstrap/store_root.sh`` is the single
source, and the copy inside each standalone installer is derived from it
between markers.

Files that run from a checkout do not carry a copy at all: ``setup-hermes.sh``
and ``scripts/_activation.sh`` source the canonical file directly.

    python3 scripts/gen-bootstrap-resolver.py          # rewrite the copies
    python3 scripts/gen-bootstrap-resolver.py --check  # exit 1 on drift
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap"
CANONICAL_SH = BOOTSTRAP / "store_root.sh"
CANONICAL_PS1 = BOOTSTRAP / "store_root.ps1"
CANONICAL_PINS = BOOTSTRAP / "uv_state_pins.ps1"

BEGIN_MARK = "# --- BEGIN GENERATED: store-root resolver (scripts/gen-bootstrap-resolver.py) ---"
END_MARK = "# --- END GENERATED: store-root resolver ---"
PINS_BEGIN = "# --- BEGIN GENERATED: uv state pins (scripts/gen-bootstrap-resolver.py) ---"
PINS_END = "# --- END GENERATED: uv state pins ---"

# install.ps1 is fetched with `irm | iex`, so it cannot source a sibling either:
# both its resolver and its uv-state pins are spliced from the canonical files,
# exactly as install.sh's are. setup-hermes.ps1 runs from a checkout and dot-
# sources them, so it carries no copy (same split as the sh side).
_SPLICE_SH = (REPO_ROOT / "scripts" / "install.sh",)
_SPLICE_PS1 = (REPO_ROOT / "scripts" / "install.ps1",)


# The canonical file marks its own body with the same markers the spliced copy
# carries inside it, so the test extracts the truth the same way from both —
# no bare text.index() that a renamed function would break silently.
CANON_BEGIN = "# --- BEGIN canonical store-root resolver (scripts/gen-bootstrap-resolver.py) ---"
CANON_END = "# --- END canonical store-root resolver ---"


CANON_PS1_BEGIN = "# --- BEGIN canonical store-root resolver (scripts/gen-bootstrap-resolver.py) ---"
CANON_PS1_END = "# --- END canonical store-root resolver ---"
CANON_PINS_BEGIN = "# --- BEGIN canonical uv state pins (scripts/gen-bootstrap-resolver.py) ---"
CANON_PINS_END = "# --- END canonical uv state pins ---"


def _canonical_body(path: Path, begin_mark: str, end_mark: str) -> str:
    """The marked region of a canonical file: exactly what the installer carries."""
    text = path.read_text(encoding="utf-8-sig")
    begin = text.index(begin_mark) + len(begin_mark)
    end = text.index(end_mark)
    return text[begin:end].strip("\r\n")


def _fragment(body: str, begin: str, end: str, source: str) -> str:
    return "\n".join([
        begin,
        f"# Derived from {source}. DO NOT EDIT BY HAND:",
        "# run scripts/gen-bootstrap-resolver.py after editing the canonical file.",
        body,
        end,
    ])


def _splice(path: Path, fragment: str, check: bool, begin_mark: str, end_mark: str) -> bool:
    """Replace the marked block in *path*. Returns True when up to date."""
    raw = path.read_bytes()
    had_bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    begin = text.index(begin_mark)  # ValueError on a missing marker is the point
    end = text.index(end_mark) + len(end_mark)
    newline = "\r\n" if "\r\n" in text else "\n"
    fragment = fragment.replace("\n", newline)
    updated = text[:begin] + fragment + text[end:]
    if updated == text:
        return True
    if not check:
        path.write_bytes((b"\xef\xbb\xbf" if had_bom else b"") + updated.encode("utf-8"))
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit 1 when a spliced copy drifts from the canonical resolver")
    args = parser.parse_args()

    results = {}
    resolver_sh = _fragment(
        _canonical_body(CANONICAL_SH, CANON_BEGIN, CANON_END),
        BEGIN_MARK, END_MARK, "scripts/bootstrap/store_root.sh")
    resolver_ps1 = _fragment(
        _canonical_body(CANONICAL_PS1, CANON_PS1_BEGIN, CANON_PS1_END),
        BEGIN_MARK, END_MARK,
        "scripts/bootstrap/store_root.ps1")
    pins_ps1 = _fragment(
        _canonical_body(CANONICAL_PINS, CANON_PINS_BEGIN, CANON_PINS_END),
        PINS_BEGIN, PINS_END, "scripts/bootstrap/uv_state_pins.ps1")
    # key -> (canonical source the copy derives from, fresh?)
    results: dict[str, tuple[str, bool]] = {}
    for path in _SPLICE_SH:
        results[path.relative_to(REPO_ROOT).as_posix()] = (
            CANONICAL_SH.relative_to(REPO_ROOT).as_posix(),
            _splice(path, resolver_sh, args.check, BEGIN_MARK, END_MARK))
    for path in _SPLICE_PS1:
        where = path.relative_to(REPO_ROOT).as_posix()
        results[where] = (CANONICAL_PS1.relative_to(REPO_ROOT).as_posix(),
                          _splice(path, resolver_ps1, args.check, BEGIN_MARK, END_MARK))
        results[f"{where} (uv pins)"] = (
            CANONICAL_PINS.relative_to(REPO_ROOT).as_posix(),
            _splice(path, pins_ps1, args.check, PINS_BEGIN, PINS_END))
    stale = {name: source for name, (source, fresh) in results.items() if not fresh}
    if args.check and stale:
        detail = "\n".join(f"  {name}: drifted from {source}" for name, source in stale.items())
        print(
            "generated bootstrap-resolver copies drifted from their canonical files:\n"
            f"{detail}\nrun: python3 scripts/gen-bootstrap-resolver.py",
            file=sys.stderr,
        )
        return 1
    for name in stale:
        print(f"updated {name}")
    if not stale:
        print("store-root resolver copies up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

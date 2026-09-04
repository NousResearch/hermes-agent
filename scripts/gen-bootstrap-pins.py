#!/usr/bin/env python3
"""Generate the bootstrap pin fragments inside the installers.

The installers bootstrap uv (and, on Windows, git) BEFORE any checkout
exists, so they cannot read pm/lock.json at run time: they are fetched
standalone (`curl | sh`, `irm | iex`) and there is no JSON parser they can
rely on that early (no jq guarantee; python is what uv installs on Windows).
Instead, this script derives a plain-data fragment from pm/lock.json — the
SAME authority the pm package manager uses for every managed tool — and
splices it between markers in each installer. The bytes are stored, the
truth is derived, and the drift test (tests/test_bootstrap_pins_fragment.py)
fails when they disagree.

Run after bumping a bootstrapped tool in pm/lock.json:

    python3 scripts/gen-bootstrap-pins.py          # rewrite fragments
    python3 scripts/gen-bootstrap-pins.py --check  # exit 1 on drift
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PINS_PATH = REPO_ROOT / "pm" / "lock.json"

BEGIN_MARK = "# --- BEGIN GENERATED: bootstrap pins (scripts/gen-bootstrap-pins.py) ---"
END_MARK = "# --- END GENERATED: bootstrap pins ---"

_POSIX_TARGETS = ("linux-x64", "linux-arm64", "darwin-x64", "darwin-arm64")
_WINDOWS_TARGETS = ("win32-x64", "win32-arm64")


def _load_lock() -> dict:
    return json.loads(PINS_PATH.read_text(encoding="utf-8-sig"))


def _load_uv_pin() -> dict:
    entry = _load_lock()["packages"]["uv"]
    for target in _POSIX_TARGETS + _WINDOWS_TARGETS:
        _validate_artifact("uv", entry, target)
    return entry


def _load_git_pin() -> dict:
    # Windows targets only. install.ps1 bootstraps the pinned git-for-windows
    # artifact because Windows needs it to clone. macOS and Linux use the
    # machine's git, so the pin table declares those targets as reasoned gaps
    # and there is nothing for the sh installers to stage.
    entry = _load_lock()["packages"]["git"]
    for target in _WINDOWS_TARGETS:
        _validate_artifact("git", entry, target)
    return entry


def _validate_artifact(tool: str, entry: dict, target: str) -> None:
    spec = entry["artifacts"][target]  # KeyError on a missing target is the point
    if spec.get("missing"):
        # The table says upstream ships nothing here, but this generator was
        # told to emit that target. One of the two is wrong, and a KeyError
        # further down would not say which.
        raise ValueError(
            f"{tool} pin for {target}: the table declares a gap "
            f"({spec['missing']}), so there is nothing to generate"
        )
    if not spec["url"].startswith("https://"):
        raise ValueError(f"{tool} pin for {target}: url must be https")
    if len(spec["sha256"]) != 64:
        raise ValueError(f"{tool} pin for {target}: sha256 must be 64 hex chars")


def _sh_fragment(uv: dict) -> str:
    lines = [
        BEGIN_MARK,
        "# Derived from pm/lock.json. DO NOT EDIT BY HAND:",
        "# run scripts/gen-bootstrap-pins.py after a pin bump.",
        f'UV_PIN_VERSION="{uv["version"]}"',
        "",
        "# Sets UV_PIN_URL + UV_PIN_SHA256 for a <os>-<arch> target key.",
        "uv_bootstrap_pin() {",
        '    case "$1" in',
    ]
    for target in _POSIX_TARGETS:
        entry = uv["artifacts"][target]
        lines += [
            f"        {target})",
            f'            UV_PIN_URL="{entry["url"]}"',
            f'            UV_PIN_SHA256="{entry["sha256"]}"',
            "            ;;",
        ]
    lines += [
        "        *)",
        '            UV_PIN_URL=""',
        '            UV_PIN_SHA256=""',
        "            return 1",
        "            ;;",
        "    esac",
        "}",
        END_MARK,
    ]
    return "\n".join(lines)


def _ps1_fragment(uv: dict, git: dict) -> str:
    lines = [
        BEGIN_MARK,
        "# Derived from pm/lock.json. DO NOT EDIT BY HAND:",
        "# run scripts/gen-bootstrap-pins.py after a pin bump.",
        f'$script:UvPinVersion = "{uv["version"]}"',
        "$script:UvPinFiles = @{",
    ]
    for target in _WINDOWS_TARGETS:
        entry = uv["artifacts"][target]
        lines += [
            f'    "{target}" = @{{',
            f'        Url    = "{entry["url"]}"',
            f'        Sha256 = "{entry["sha256"]}"',
            "    }",
        ]
    lines += [
        "}",
        "",
        f'$script:GitPinVersion = "{git["version"]}"',
        "$script:GitPinFiles = @{",
    ]
    for target in _WINDOWS_TARGETS:
        entry = git["artifacts"][target]
        lines += [
            f'    "{target}" = @{{',
            f'        Url    = "{entry["url"]}"',
            f'        Sha256 = "{entry["sha256"]}"',
            "    }",
        ]
    lines += [
        "}",
        END_MARK,
    ]
    return "\n".join(lines)


def _splice(path: Path, fragment: str, check: bool) -> bool:
    """Replace the marked block in *path*. Returns True when up to date."""
    raw = path.read_bytes()
    had_bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    begin = text.index(BEGIN_MARK)  # ValueError on missing marker is the point
    end = text.index(END_MARK) + len(END_MARK)
    # Match the file's line endings (install.ps1 is CRLF) so the generated
    # block does not leave the script with mixed newlines.
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
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 when the committed fragments drift from pm/lock.json",
    )
    args = parser.parse_args()

    uv = _load_uv_pin()
    git = _load_git_pin()
    results = {
        "scripts/install.sh": _splice(
            REPO_ROOT / "scripts" / "install.sh", _sh_fragment(uv), args.check
        ),
        "scripts/install.ps1": _splice(
            REPO_ROOT / "scripts" / "install.ps1", _ps1_fragment(uv, git), args.check
        ),
        # setup-hermes.sh deliberately holds NO fragment: it runs after a
        # checkout exists, so it reads the pin straight from pm/lock.json
        # at run time (the pm.sh-derived bootstrap). The fragment exists
        # only for the curl|sh / irm|iex installers that bootstrap BEFORE
        # any checkout exists.
    }
    stale = [name for name, fresh in results.items() if not fresh]
    if args.check and stale:
        print(
            "bootstrap pin fragments drifted from pm/lock.json in: "
            f"{', '.join(stale)}\nrun: python3 scripts/gen-bootstrap-pins.py",
            file=sys.stderr,
        )
        return 1
    for name in stale:
        print(f"updated {name}")
    if not stale:
        print("fragments up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

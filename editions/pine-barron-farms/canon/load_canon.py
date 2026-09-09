#!/usr/bin/env python3
"""Load the Pine Barron Farms studio canon packet into the agent's context.

This edition's SOUL.md says *"I follow the loaded canon packet exactly"* — but the
packet is deploy-time content (`canon/PINE_BARRON_FARMS_CANON.md`, dropped in by an
admin, never in the public repo). Nothing loaded it. This is the loader.

It is used two ways, both existing supported mechanisms — **no core engine change**:

* by the ``pbf-canon`` skill (``editions/pine-barron-farms/skills/pbf-canon/SKILL.md``),
  whose ``!`...``` inline-shell snippet runs this script so the receipt + packet
  excerpt land in the preloaded-skills portion of the system prompt; and
* as a plain session-start command an operator (or ``north-forge.cmd``) can run:
  ``python editions/pine-barron-farms/canon/load_canon.py``.

Output contract (always exit 0 — a missing packet is not an error):

* packet present  ->  a ``canon loaded: <path>, sha256=<hex>, <n> bytes`` receipt
  line, then the packet between BEGIN/END markers (truncated to ``--max-chars``
  with an explicit "read the full file" note when it does not fit).
* packet absent   ->  a clear ``⚠ canon packet NOT FOUND`` warning naming every
  path checked and stating plainly that only the SOUL.md persona/method are in
  force.

Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

PACKET_NAMES = (
    "PINE_BARRON_FARMS_CANON.md",
    "PINE_BARRON_FARMS_CANON.txt",
    "pine_barron_farms_canon.md",
    "CANON.md",
)

BEGIN = "--- BEGIN PINE BARRON FARMS CANON ---"
END = "--- END PINE BARRON FARMS CANON ---"

# Inline-shell output is capped by the engine (~4000 chars); the skill passes a
# budget below that so the receipt line always survives. A direct run defaults to
# the full packet.
DEFAULT_MAX_CHARS = 1_000_000


def _canon_dirs(explicit: str | None) -> list[Path]:
    """Every directory that might hold the packet, most-specific first."""
    out: list[Path] = []

    def add(p: Path | None) -> None:
        if p is not None:
            p = p.expanduser()
            if p not in out:
                out.append(p)

    if explicit:
        add(Path(explicit))

    # This script lives at <profile>/canon/load_canon.py — its own directory.
    add(Path(__file__).resolve().parent)

    # Running profile: north-forge.cmd pins HERMES_HOME to the profile dir.
    home = os.environ.get("HERMES_HOME", "").strip()
    if home:
        add(Path(home) / "canon")
        add(Path(home))

    # The pbf-canon skill runs this with HERMES_SKILL_DIR set to
    # <profile>/skills/pbf-canon — the packet is two levels up under canon/.
    skill_dir = os.environ.get("HERMES_SKILL_DIR", "").strip()
    if skill_dir:
        add(Path(skill_dir).parent.parent / "canon")

    return out


def find_packet(explicit_dir: str | None = None) -> Path | None:
    for d in _canon_dirs(explicit_dir):
        try:
            if not d.is_dir():
                continue
        except OSError:
            continue
        for name in PACKET_NAMES:
            cand = d / name
            if cand.is_file():
                return cand
    return None


def render(explicit_dir: str | None = None, max_chars: int = DEFAULT_MAX_CHARS) -> tuple[str, bool]:
    """Return (text, found). ``text`` is exactly what should enter context."""
    packet = find_packet(explicit_dir)
    if packet is None:
        checked = "\n".join(f"  - {d}" for d in _canon_dirs(explicit_dir))
        drop_dir = _canon_dirs(explicit_dir)[0]
        return (
            "WARNING: canon packet NOT FOUND. Checked:\n"
            f"{checked}\n"
            "This edition is running on its SOUL.md persona and method ONLY — there "
            "are no studio-specific facts (characters, plates, locations, episode "
            "history, open patches) to enforce. If the user asks anything "
            "canon-dependent, say the packet is not loaded and ask them to add it "
            f"(admin: drop {PACKET_NAMES[0]} into {drop_dir}).",
            False,
        )

    raw = packet.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8", errors="replace")
    n_lines = text.count("\n") + 1
    receipt = (
        f"canon loaded: {packet}, sha256={sha}, {len(raw)} bytes, {n_lines} lines"
    )

    body = text
    note = ""
    if len(text) > max_chars:
        body = text[:max_chars].rstrip()
        note = (
            f"\n\n[canon excerpt truncated to {max_chars} chars — read the full "
            f"packet at {packet} with the file tool before any canon-dependent work]"
        )

    return f"{receipt}\n{BEGIN}\n{body}\n{END}{note}", True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="load_canon.py",
        description="Load the Pine Barron Farms canon packet into context (or warn if absent).",
    )
    ap.add_argument("--canon-dir", default=None,
                    help="explicit directory to look in (default: profile canon/, HERMES_HOME, skill dir)")
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                    help="truncate the packet excerpt to this many characters (the skill passes a small budget)")
    args = ap.parse_args(argv)

    # Canon content (and this script's own output) is UTF-8; a Windows console
    # defaults to cp1252 and would crash on an em-dash. Force UTF-8 out.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    text, _found = render(args.canon_dir, max_chars=max(1, args.max_chars))
    print(text)
    return 0  # a missing packet is a documented state, never a failure


if __name__ == "__main__":
    raise SystemExit(main())

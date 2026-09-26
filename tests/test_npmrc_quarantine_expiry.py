"""The npm supply-chain quarantine must not carry expired holes.

``.npmrc`` sets ``min-release-age``: a quarantine that keeps a freshly
published version out of every install until it has aged. Each
``min-release-age-exclude[]`` entry punches a hole in that gate for one
package pattern, and the holes exist for a *temporary* reason — a security fix
lands in a release younger than the window, so the resolver cannot see it yet.

Nothing ever expired them. Every dated entry in ``.npmrc`` and
``website/.npmrc`` sat 30+ days past the date its own comment named for its
removal, so the quarantine was off for 34 package patterns (four of them whole
wildcard scopes) on any ``npm install``. The discipline that writes the expiry
was present; the process that acts on it was not.

These tests are that process. A hole must name the release date it was opened
for (``rel YYYY-MM-DD``), and it must be gone once that date is older than the
window the same file declares.
"""

import datetime
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

_SETTING = re.compile(r"^min-release-age\s*=\s*(\d+)\s*$")
_EXCLUDE = re.compile(r"^min-release-age-exclude\[\]\s*=\s*(\S+)\s*$")
_RELEASE_DATE = re.compile(r"\brel\s+(\d{4}-\d{2}-\d{2})\b")

# Trees that are not ours to police: a dependency may ship its own .npmrc, and
# on a machine that has installed the workspaces those are tens of thousands of
# files to walk.
_UNWALKED = {".git", "node_modules", ".venv", "venv"}


@dataclass(frozen=True)
class Hole:
    """One ``min-release-age-exclude[]`` entry and the date it was opened for."""

    line_no: int
    pattern: str
    released_on: datetime.date | None  # None when the comment carries no marker


def parse_quarantine(text: str) -> tuple[int | None, list[Hole]]:
    """Return ``(window_days, holes)`` for one ``.npmrc``.

    ``window_days`` is the file's own ``min-release-age`` (``None`` when it
    declares none). A hole's release date comes from the ``rel YYYY-MM-DD``
    marker in the contiguous comment block directly above it; consecutive
    entries share that block, and any other line ends it.
    """
    window: int | None = None
    block: list[str] = []
    holes: list[Hole] = []
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if line.startswith("#"):
            block.append(line)
            continue
        excluded = _EXCLUDE.match(line)
        if excluded:
            marker = _RELEASE_DATE.search("\n".join(block))
            released_on = (
                datetime.date.fromisoformat(marker.group(1)) if marker else None
            )
            holes.append(Hole(line_no, excluded.group(1), released_on))
            continue
        setting = _SETTING.match(line)
        if setting:
            window = int(setting.group(1))
        block = []
    return window, holes


def expired_holes(text: str, today: datetime.date) -> list[Hole]:
    """Holes whose named release is now older than the file's own window.

    Such a hole can no longer change what installs — the version it was opened
    for clears the age gate on its own — so all it still does is exempt
    whatever that package publishes next.
    """
    window, holes = parse_quarantine(text)
    if window is None:
        return []
    cutoff = today - datetime.timedelta(days=window)
    return [h for h in holes if h.released_on is not None and h.released_on < cutoff]


def undated_holes(text: str) -> list[Hole]:
    """Holes with no ``rel YYYY-MM-DD`` marker — nothing can ever expire them."""
    return [h for h in parse_quarantine(text)[1] if h.released_on is None]


def _repo_npmrc_files() -> list[Path]:
    found = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in _UNWALKED]
        if ".npmrc" in filenames:
            found.append(Path(dirpath) / ".npmrc")
    return sorted(found)


def _npmrc_params():
    files = _repo_npmrc_files()
    assert files, "no .npmrc found — the quarantine guard is checking nothing"
    return [pytest.param(p, id=str(p.relative_to(REPO_ROOT))) for p in files]


@pytest.mark.parametrize("npmrc", _npmrc_params())
def test_no_quarantine_hole_has_outlived_its_own_release_date(npmrc):
    stale = expired_holes(npmrc.read_text(encoding="utf-8"), datetime.date.today())
    assert not stale, (
        f"{npmrc.relative_to(REPO_ROOT)} keeps the supply-chain quarantine open for "
        f"{[h.pattern for h in stale]}. Each entry's own comment dates the release it "
        "was opened for, and that release is now older than this file's "
        "min-release-age, so the exclusion no longer changes what installs — it only "
        "exempts whatever those packages publish next. Delete the entries at lines "
        f"{[h.line_no for h in stale]} and re-run `npm ci` to confirm the lockfile "
        "does not move."
    )


@pytest.mark.parametrize("npmrc", _npmrc_params())
def test_every_quarantine_hole_names_the_release_it_exists_for(npmrc):
    """An undated hole is a permanent one: no later reader can tell when it is spent.

    Twenty of the root file's 24 entries named the version they needed
    ("remove this when 10.0.4 is > 2wks old") without ever writing its release
    date, which is why nothing — human or CI — could act on them.
    """
    undated = undated_holes(npmrc.read_text(encoding="utf-8"))
    assert not undated, (
        f"{npmrc.relative_to(REPO_ROOT)} opens the quarantine for "
        f"{[h.pattern for h in undated]} without a release date. Add the release date "
        "of the version the hole exists to let through to the comment above it, as "
        "`rel YYYY-MM-DD`, so it can be expired: "
        "`# <pkg> <ver> fixes <GHSA-...>. remove when > 2wks old (rel 2026-08-03)`."
    )


def test_the_expiry_window_comes_from_the_file_not_from_a_constant():
    """The same hole expires or not depending on the file's own declared window.

    Pins the relationship rather than the number: a repo that lengthens
    ``min-release-age`` must not start reporting holes that are still doing
    their job, and one that shortens it must start reporting more.
    """
    today = datetime.date(2026, 9, 5)
    npmrc = (
        "min-release-age={window}\n"
        "# fast-uri 3.1.5 fixes GHSA-7p8r-x3mc-p8w7. remove when > 2wks old"
        " (rel 2026-08-20)\n"
        "min-release-age-exclude[]=fast-uri\n"
    )

    assert [h.pattern for h in expired_holes(npmrc.format(window=14), today)] == [
        "fast-uri"
    ]
    assert expired_holes(npmrc.format(window=90), today) == []


def test_a_hole_with_no_release_date_can_never_expire():
    """The undated shape is invisible to the expiry check, so it is its own failure.

    ``# ink needs`` above two exclusions carried no date and no version, so
    ``expired_holes`` has nothing to compare and would pass it forever. Catching
    that shape is ``undated_holes``' job.
    """
    undated = "min-release-age=14\n# ink needs\nmin-release-age-exclude[]=postcss\n"

    assert expired_holes(undated, datetime.date(2027, 1, 1)) == []
    assert [h.pattern for h in undated_holes(undated)] == ["postcss"]

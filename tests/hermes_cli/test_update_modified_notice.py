"""Guard: every `hermes update` path that reports user-modified skills must
also tell the user how to find them.

`hermes update` keeps (does not overwrite) bundled skills the user edited and
prints a ``~ N user-modified (kept)`` count. There are two independent update
code paths in ``hermes_cli/main.py`` that print this notice (the git-pull path
in ``_cmd_update_impl`` and the unpack/install path). Both must point the user
at ``hermes skills list-modified`` so the count is actionable — otherwise,
depending on which path a user hits, they may never learn the discovery command
exists.

This is an *invariant* test (the two sibling notices must agree), not a literal
snapshot: it asserts the relationship "count line ⇒ discovery hint", so it
keeps holding if the wording is reworded, as long as both sites stay in sync.
"""

import re
from pathlib import Path

import hermes_cli.main as main_mod
import hermes_cli.update_cmd as update_mod
import hermes_cli.update_orchestrator as update_orchestrator_mod
import hermes_cli.update_zip as update_zip_mod


_COUNT_RE = re.compile(r"user-modified \(kept\)")
_HINT_RE = re.compile(r"hermes skills list-modified")


def _source_files() -> list[tuple[Path, list[str]]]:
    # The compatibility facade and its implementation modules jointly own the
    # update surface. Scan every current home of this invariant.
    return [
        (
            Path(mod.__file__),
            Path(mod.__file__).read_text(encoding="utf-8").splitlines(),
        )
        for mod in (
            main_mod,
            update_mod,
            update_orchestrator_mod,
            update_zip_mod,
        )
    ]


def _missing_hint_sites(
    source_files: list[tuple[Path, list[str]]],
) -> tuple[int, list[tuple[Path, int, str]]]:
    count = 0
    missing: list[tuple[Path, int, str]] = []
    for path, lines in source_files:
        for idx, line in enumerate(lines):
            if not _COUNT_RE.search(line):
                continue
            count += 1
            # Keep the window inside the owning file. Crossing into the next
            # source could let an unrelated hint hide a missing sibling hint.
            window = "\n".join(lines[idx : idx + 5])
            if not _HINT_RE.search(window):
                missing.append((path, idx + 1, window))
    return count, missing


def test_every_user_modified_notice_points_at_list_modified():
    count, missing = _missing_hint_sites(_source_files())

    # The notice must exist somewhere (guard against it being deleted outright),
    # but we deliberately do NOT assert a fixed *count* of sites: consolidating
    # the duplicated print paths into a shared helper is a welcome refactor and
    # must not fail this test. The invariant is per-site, not how many sites.
    assert count, (
        "no 'user-modified (kept)' notice found in main.py — the update "
        "summary that surfaces kept user edits appears to have been removed"
    )
    assert not missing, "\n\n".join(
        "a 'user-modified (kept)' notice at "
        f"{path}:{line_number} does not point users at "
        "`hermes skills list-modified` within the following lines:\n"
        f"{window}"
        for path, line_number, window in missing
    )


def test_notice_hint_cannot_be_borrowed_from_the_next_file(tmp_path):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    count, missing = _missing_hint_sites(
        [
            (first, ["user-modified (kept)"]),
            (second, ["hermes skills list-modified"]),
        ]
    )

    assert count == 1
    assert [(path, line_number) for path, line_number, _ in missing] == [
        (first, 1)
    ]

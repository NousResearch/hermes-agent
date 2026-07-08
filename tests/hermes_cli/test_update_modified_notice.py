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


_COUNT_RE = re.compile(r"user-modified \(kept\)")
_HINT_RE = re.compile(r"ht skills list-modified")


def _source_lines() -> list[str]:
    return Path(main_mod.__file__).read_text(encoding="utf-8").splitlines()


def test_every_user_modified_notice_points_at_list_modified():
    lines = _source_lines()
    count_sites = [i for i, ln in enumerate(lines) if _COUNT_RE.search(ln)]

    # The notice must exist somewhere (guard against it being deleted outright),
    # but we deliberately do NOT assert a fixed *count* of sites: consolidating
    # the duplicated print paths into a shared helper is a welcome refactor and
    # must not fail this test. The invariant is per-site, not how many sites.
    assert count_sites, (
        "no 'user-modified (kept)' notice found in main.py — the update "
        "summary that surfaces kept user edits appears to have been removed"
    )

    for idx in count_sites:
        # The count print and its discovery hint sit on adjacent lines; allow a
        # small window so wording/formatting tweaks don't break the check.
        window = "\n".join(lines[idx : idx + 5])
        assert _HINT_RE.search(window), (
            "a 'user-modified (kept)' notice near line "
            f"{idx + 1} of main.py does not point users at "
            "`ht skills list-modified` within the following lines — the "
            "update paths have drifted apart again:\n" + window
        )

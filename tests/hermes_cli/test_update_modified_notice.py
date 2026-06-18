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
_HINT_RE = re.compile(r"hermes skills list-modified")


def _source_lines() -> list[str]:
    return Path(main_mod.__file__).read_text(encoding="utf-8").splitlines()


def test_every_user_modified_notice_points_at_list_modified():
    lines = _source_lines()
    count_sites = [i for i, ln in enumerate(lines) if _COUNT_RE.search(ln)]

    # There are at least two such notices today; the bug was that only one of
    # them carried the discovery hint. Assert each is followed (within a small
    # window — the count print + the hint print) by the list-modified pointer.
    assert len(count_sites) >= 2, (
        "expected at least two 'user-modified (kept)' notices in main.py; "
        f"found {len(count_sites)}"
    )

    for idx in count_sites:
        window = "\n".join(lines[idx : idx + 5])
        assert _HINT_RE.search(window), (
            "a 'user-modified (kept)' notice near line "
            f"{idx + 1} of main.py does not point users at "
            "`hermes skills list-modified` within the following lines — the "
            "two update paths have drifted apart again:\n" + window
        )

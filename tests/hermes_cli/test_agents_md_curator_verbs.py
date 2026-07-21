"""Docs/code consistency guard for the curator verb list in AGENTS.md.

`hermes_cli/curator.py` is the source of truth for the `hermes curator <verb>`
subcommands. AGENTS.md (the "CLI:" bullet under the curator section) documents
those verbs in prose. The two drifted once already: `list-archived` was added
to the CLI (registered as a real subparser with a handler) but never made it
into the AGENTS.md list.

This test parses the registered subparser verbs out of curator.py and asserts
every one is documented in AGENTS.md, so the prose can never silently fall
behind the code again.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CURATOR_SRC = REPO_ROOT / "hermes_cli" / "curator.py"
AGENTS_MD = REPO_ROOT / "AGENTS.md"


def _registered_verbs() -> list[str]:
    """Verbs registered via ``subs.add_parser("<verb>", ...)`` in curator.py."""
    src = CURATOR_SRC.read_text(encoding="utf-8")
    return re.findall(r"""subs\.add_parser\(\s*["']([a-z][a-z-]*)["']""", src)


def _documented_verbs() -> set[str]:
    """Verbs listed in the AGENTS.md curator "CLI:" bullet."""
    text = AGENTS_MD.read_text(encoding="utf-8")
    # Grab the prose between "verbs are:" and the sentence-ending period.
    match = re.search(r"hermes curator <verb>.*?verbs are:(.*?)\.\n", text, re.DOTALL)
    assert match is not None, "could not locate the curator verbs list in AGENTS.md"
    return set(re.findall(r"`([a-z][a-z-]*)`", match.group(1)))


def test_registered_verbs_found():
    # Sanity: the parser actually picked up the known core verbs.
    verbs = _registered_verbs()
    assert "list-archived" in verbs
    assert "status" in verbs


def test_agents_md_documents_every_curator_verb():
    registered = _registered_verbs()
    documented = _documented_verbs()
    missing = [v for v in registered if v not in documented]
    assert not missing, (
        "AGENTS.md curator verb list is out of date with hermes_cli/curator.py; "
        f"missing verbs: {missing}"
    )

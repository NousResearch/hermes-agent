"""scripts/ci/catalog_dead_repos.py: which catalog entries get delisted, from canned GitHub answers."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ci"))

import catalog_dead_repos as dead_repos

# repo name -> what GitHub knows: GraphQL node (None = NOT_FOUND) and REST status of /repos/o/<name>
GITHUB = {
    "live": ({"object": {"oid": "x"}}, 200),
    "gone": (None, 404),
    "force-pushed": ({"object": None}, 200),
    "old-name": (None, 200),  # renamed: GraphQL misses it, REST follows the redirect, pin is there
}


def _catalog(tmp_path: Path) -> Path:
    for name in GITHUB:
        (tmp_path / f"{name}.yaml").write_text(
            f'name: {name}\nrepo: https://github.com/o/{name}\nsha: "{"a" * 40}"\n', encoding="utf-8")
    (tmp_path / "removed.yaml").write_text("removed: []\n", encoding="utf-8")
    return tmp_path


def _fetch(extra_errors: list[dict] | None = None):
    def fetch(method, url, body):
        if url.endswith("/graphql"):
            aliases = dict(re.findall(r'(r\d+): repository\(owner: "o", name: "([^"]+)"\)', body["query"]))
            data = {alias: GITHUB[name][0] for alias, name in aliases.items()}
            errors = [{"type": "NOT_FOUND"} for node in data.values() if node is None]
            return 200, {"data": data, "errors": errors + (extra_errors or [])}
        name = url.split("/repos/o/")[1].split("/")[0]
        return GITHUB[name][1], {}
    return fetch


def test_only_deleted_repos_and_vanished_pins_are_delisted(tmp_path):
    dead = dead_repos.find_dead(dead_repos.load_entries(_catalog(tmp_path)), _fetch())
    assert {(d.entry.path.stem, d.reason) for d in dead} == {
        ("gone", dead_repos.REPO_GONE), ("force-pushed", dead_repos.PIN_GONE)}


def test_an_unclear_lookup_delists_nothing(tmp_path):
    with pytest.raises(dead_repos.LookupFailed):
        dead_repos.find_dead(dead_repos.load_entries(_catalog(tmp_path)), _fetch([{"type": "RATE_LIMITED"}]))

"""A stale absorbed-submodule gitdir in a ``--filter=tree:0`` install, through the real ``hermes update``.

The repository once had a submodule (``tinker-atropos``). Checkouts that ran ``git submodule update``
back then still carry ``.git/modules/tinker-atropos`` and ``submodule.tinker-atropos.active=true``, long
after upstream dropped ``.gitmodules``. With git's default ``fetch.recurseSubmodules=on-demand``, a
non-empty ``.git/modules/`` makes every fetch diff each newly fetched commit against its parents to
look for changed gitlinks (``submodule.c: calculate_changed_submodule_paths``). A treeless clone holds
none of those trees, so git lazy-fetches them one commit at a time, after the refs have already
moved. Against GitHub each round trip can take seconds, and a routine update of a few hundred commits
ran into the 300 s network cap on every attempt.

Property: the updater's fetches never walk new history for submodule changes. Hermes ships no
submodules, so an update (and ``--check``) of N new commits costs a handful of fetch requests, not one
per commit. A request count against the number of new commits, never wall time.
"""

from __future__ import annotations

import pytest

from tests.e2e.core.upgrade.git import _git_world as G

pytestmark = G.PYTESTMARK

NEW_COMMITS = 30


@pytest.fixture(scope="module")
def w(tmp_path_factory):
    with G.world(tmp_path_factory.mktemp("git-stale-gitdir"), base=G.I.head_sha()) as world:
        assert world.install_shape["filter"] == "tree:0", f"expected a treeless install: {world.install_shape}"
        yield world


def _seed_stale_submodule_gitdir(w: G.World) -> None:
    """What a checkout that once ran ``git submodule update`` still has."""
    w.git("init", "-q", "--bare", str(w.checkout / ".git" / "modules" / "tinker-atropos"))
    w.git("config", "submodule.tinker-atropos.active", "true")
    w.git("config", "submodule.tinker-atropos.url", "https://example.invalid/tinker-atropos")


def _publish_history(w: G.World, label: str) -> str:
    target = ""
    for i in range(NEW_COMMITS):
        target = w.publish(f"release: e2e stale gitdir {label} {i}", {f"e2e-stale-gitdir/{label}-{i}.txt": f"{i}\n"})
    return target


def _fetch_requests(w: G.World, mark: int) -> list:
    return [r for r in w.srv.since(mark) if r.command == "fetch"]


def test_update_does_not_fetch_once_per_new_commit(w):
    w.reset_clean()
    _seed_stale_submodule_gitdir(w)
    target = _publish_history(w, "update")
    mark = w.srv.mark()

    cp = w.update()

    assert cp.returncode == 0 and G.SUCCESS in G.output(cp) and w.head() == target, \
        f"update over {NEW_COMMITS} new commits failed:\n{w.diag(cp, mark)}"
    fetches = _fetch_requests(w, mark)
    assert len(fetches) < NEW_COMMITS // 2, (
        f"updating {NEW_COMMITS} new commits made {len(fetches)} fetch requests: the fetch walked every new "
        f"commit's trees for submodule changes\n{w.diag(cp, mark)}")


def test_update_check_does_not_fetch_once_per_new_commit(w):
    w.reset_clean()
    _seed_stale_submodule_gitdir(w)
    _publish_history(w, "check")
    mark = w.srv.mark()

    cp = w.sb.cli("update", "--check", timeout=900)

    assert "Update available" in G.output(cp), f"--check did not report the new commits:\n{w.diag(cp, mark)}"
    fetches = _fetch_requests(w, mark)
    assert len(fetches) < NEW_COMMITS // 2, (
        f"--check over {NEW_COMMITS} new commits made {len(fetches)} fetch requests: the fetch walked every new "
        f"commit's trees for submodule changes\n{w.diag(cp, mark)}")

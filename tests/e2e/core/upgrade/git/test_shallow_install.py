"""Shallow and stale installs through the real ``hermes update``.

N-1's ``scripts/install.sh`` clones ``--depth 1 --single-branch``: that is the checkout every
N-1 user has. Three installs run concurrently over smart HTTP:

* ``depth1``: N-1's own depth-1 install, updated to HEAD and then to one more release. The first
  update must not download the whole project history to move one release (#123254: on a real
  network that transfer hits the 300 s fetch cap, so a stale shallow install can never update), and
  the second must succeed after the first one unshallowed the clone (#124272: the unshallow applied
  ``--filter=tree:0`` to a clone whose objects sit in a non-promisor pack, and the next fetch dies
  in ``pack-objects ... should_include_obj``).
* ``checked``: the same install after three passive ``hermes update --check`` runs while upstream
  moved (each is a depth-1 fetch that appends a graft), then updated to HEAD. The history is
  shared, so the update must not claim orphan divergence (#105951's symptom).
* ``prefetched``: a HEAD install (non-shallow ``tree:0``) with a local commit, where the user ran
  ``git fetch --depth 1 origin main`` (the documented workaround for slow fetches) before
  ``hermes update``. The local commit shares history with upstream: it must stay recoverable, and
  the update must not declare orphan divergence (#123346).
"""

from __future__ import annotations

import pytest

from tests.e2e.core._pending_fixes import known_failure
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.git import _git_world as G

pytestmark = G.PYTESTMARK

ORPHAN = "orphan divergence"


def _depth1(root) -> dict:
    cm, w = G.start_world(root / "depth1", base=G.n1_base(), installer_ref=G.n1_tag())
    run = {"cm": cm, "w": w, "shape0": w.install_shape}
    w.set_main(I.head_sha())
    run["mark1"] = w.srv.mark()
    run["cp1"], run["head1"] = w.update(), w.head()
    run["fetches1"] = [r for r in w.srv.since(run["mark1"]) if r.command == "fetch"]
    run["target2"] = w.publish("release: e2e shallow next", {"e2e-shallow-next.txt": "release\n"})
    run["mark2"] = w.srv.mark()
    run["cp2"], run["head2"] = w.update(), w.head()
    run["version"] = w.version()
    return run


def _checked(root) -> dict:
    cm, w = G.start_world(root / "checked", base=G.n1_base(), installer_ref=G.n1_tag())
    run = {"cm": cm, "w": w, "shape0": w.install_shape, "checks": []}
    between = I.git("rev-list", "--first-parent", f"{G.n1_base()}..{I.head_sha()}", cwd=I.H.WORKTREE).split()
    for k in (len(between) * 3 // 4, len(between) // 2, len(between) // 4):
        w.set_main(between[k])
        run["checks"].append(w.sb.cli("update", "--check", timeout=600))
    shallow = w.checkout / ".git" / "shallow"
    run["grafts"] = len(shallow.read_text(encoding="utf-8").split()) if shallow.is_file() else 0
    w.set_main(I.head_sha())
    run["cp1"], run["head1"] = w.update(), w.head()
    return run


def _prefetched(root) -> dict:
    cm, w = G.start_world(root / "prefetched", base=I.head_sha())
    run = {"cm": cm, "w": w}
    (w.checkout / "e2e-local-before-prefetch.txt").write_text("my local commit\n", encoding="utf-8")
    w.git("add", "-A")
    w.git("commit", "-q", "-m", "my local patch")
    run["local"] = w.head()
    run["target"] = w.publish("release: e2e shallow prefetch", {"e2e-shallow-prefetch.txt": "release\n"})
    w.git("fetch", "-q", "--depth", "1", "origin", "main")
    run["prefetch_shape"] = w.shape()
    run["cp1"], run["head1"] = w.update(), w.head()
    return run


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("git-shallow")
    results = G.in_parallel({"depth1": lambda: _depth1(root), "checked": lambda: _checked(root),
                             "prefetched": lambda: _prefetched(root)})
    yield results
    for res in results.values():
        if isinstance(res, dict):
            res["cm"].__exit__(None, None, None)


def _run(runs, name) -> dict:
    run = runs[name]
    if isinstance(run, BaseException):
        raise run
    return run


def test_stale_depth1_install_updates_without_fetching_the_whole_history(runs):
    run = _run(runs, "depth1")
    w: G.World = run["w"]
    cp1 = run["cp1"]
    assert run["shape0"]["shallow"] == "true", f"N-1's installer did not make a shallow clone: {run['shape0']}"
    assert cp1.returncode == 0 and run["head1"] == I.head_sha(), f"N-1 -> HEAD update failed:\n{w.diag(cp1, run['mark1'])}"
    main_fetch = run["fetches1"][0].body_bytes if run["fetches1"] else 0
    bound = 2 * w.install_bytes
    with known_failure(r"downloaded .* the depth-1 install itself",
                       "#123254: on a depth-1 install N-1's plain `git fetch origin main` downloads ~the whole "
                       "history. HEAD unshallows (commits only) before that fetch; the fetch runs pre-swap, so "
                       "this flips once a release carrying the fix is N-1"):
        assert main_fetch <= bound, (
            f"moving one release, the update's main fetch downloaded {main_fetch} bytes, more than twice the "
            f"{w.install_bytes} bytes of the depth-1 install itself:\n{w.diag(cp1, run['mark1'])}")


def test_second_update_after_the_unshallowing_update_succeeds(runs):
    run = _run(runs, "depth1")
    w: G.World = run["w"]
    cp1, cp2 = run["cp1"], run["cp2"]
    assert cp1.returncode == 0 and run["head1"] == I.head_sha(), f"N-1 -> HEAD update failed:\n{w.diag(cp1, run['mark1'])}"
    assert cp2.returncode == 0 and run["head2"] == run["target2"], (
        f"the update after the unshallowing update failed (rc={cp2.returncode}):\n"
        f"{G.output(cp2)[-1500:]}\n{w.diag(cp2, run['mark2'])}")
    version = run["version"]
    assert version.returncode == 0 and G.TRACEBACK not in G.output(version), w.diag(version)


def test_passive_checks_do_not_push_the_next_update_into_orphan_divergence(runs):
    run = _run(runs, "checked")
    w: G.World = run["w"]
    cp1 = run["cp1"]
    for cp in run["checks"]:
        assert cp.returncode == 0 and G.TRACEBACK not in G.output(cp), f"`hermes update --check` failed:\n{w.diag(cp)}"
    assert cp1.returncode == 0 and run["head1"] == I.head_sha(), f"N-1 -> HEAD update failed:\n{w.diag(cp1)}"
    with known_failure(r"claims orphan divergence",
                       "#124645: each depth-1 `--check` appends a graft the fetch reflog pins, and N-1's "
                       "update resets as orphan divergence. HEAD's prune expires those reflogs and HEAD "
                       "unshallows before the pull; it is N-1's own check and pull code, so this flips once "
                       "a release carrying the fix is N-1"):
        assert ORPHAN not in G.output(cp1) and not w.refs("refs/hermes-update-backups/orphan-*"), (
            f"after {len(run['checks'])} passive checks ({run['grafts']} grafts in .git/shallow) the update "
            f"claims orphan divergence on a history it shares with upstream:\n{w.diag(cp1)}")


def test_depth1_prefetch_never_turns_shared_history_into_orphan_divergence(runs):
    run = _run(runs, "prefetched")
    w: G.World = run["w"]
    cp1 = run["cp1"]
    assert run["prefetch_shape"]["shallow"] == "true", f"the --depth 1 pre-fetch did not graft: {run['prefetch_shape']}"
    assert cp1.returncode == 0 and run["head1"] == run["target"], f"update failed:\n{w.diag(cp1)}"
    keepers = w.refs_containing(run["local"])
    assert keepers, f"local commit {run['local'][:12]} is reachable only from the reflog:\n{w.diag(cp1)}"
    assert ORPHAN not in G.output(cp1) and not w.refs("refs/hermes-update-backups/orphan-*"), (
        f"after a --depth 1 pre-fetch the update claims orphan divergence and force-resets main "
        f"(local commit {run['local'][:12]} shares history with upstream):\n{w.diag(cp1)}")

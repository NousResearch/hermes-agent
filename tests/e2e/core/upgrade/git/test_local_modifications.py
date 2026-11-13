"""Local modifications through a real ``hermes update``: tracked edits, untracked files, local commits.

One HEAD install (HEAD's ``scripts/install.sh``: a ``--filter=tree:0`` clone over smart HTTP) is
shared by the cells; each cell starts from a clean ``main`` (``World.reset_clean``), makes the
user's change, publishes one upstream release on top and runs ``hermes update --yes``.

The property is the user's: whatever the update does with their work, it is either back in the
working tree or left behind a ref the output names, and the final banner / exit code never says
"Update complete" while their patch sits parked in a stash.
"""

from __future__ import annotations

import pytest

from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.git import _git_world as G

pytestmark = G.PYTESTMARK

USER_LINE = "user patch: keep my local tweak\n"


@pytest.fixture(scope="module")
def w(tmp_path_factory):
    with G.world(tmp_path_factory.mktemp("git-localmods"), base=I.head_sha()) as world:
        yield world


def _stash_texts(w: G.World, rel: str) -> list[str]:
    """``rel`` as recorded in every stash entry (tracked part, then the untracked-files parent),
    whitespace-stripped like every ``World.git`` output."""
    out = []
    for i in range(len(w.git("stash", "list").splitlines())):
        for spec in (f"stash@{{{i}}}:{rel}", f"stash@{{{i}}}^3:{rel}"):
            text = w.git("show", spec, check=False)
            if text:
                out.append(text)
    return out


def test_non_conflicting_edits_and_untracked_files_are_restored(w):
    w.reset_clean()
    readme = w.checkout / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\n" + USER_LINE, encoding="utf-8")
    notes = w.checkout / "my_notes.txt"
    notes.write_text("my own notes\n", encoding="utf-8")
    target = w.publish("release: e2e localmods clean", {"e2e-localmods-clean.txt": "release\n"})

    cp = w.update()

    assert cp.returncode == 0 and w.head() == target, f"update did not land on {target}:\n{w.diag(cp)}"
    assert USER_LINE in readme.read_text(encoding="utf-8"), f"tracked edit not restored:\n{w.diag(cp)}"
    assert notes.is_file() and notes.read_text(encoding="utf-8") == "my own notes\n", \
        f"untracked file not restored:\n{w.diag(cp)}"
    assert not w.git("stash", "list"), f"restored changes still parked in a stash:\n{w.diag(cp)}"


def test_conflicting_patch_is_recoverable_and_never_reported_complete(w):
    w.reset_clean()
    readme = w.checkout / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\n" + USER_LINE, encoding="utf-8")
    target = w.publish("release: e2e localmods conflict", {"README.md": "# upstream rewrote the README\n"})

    cp = w.update()

    assert w.head() == target, f"code update did not land:\n{w.diag(cp)}"
    in_tree = USER_LINE in readme.read_text(encoding="utf-8")
    parked = any(USER_LINE.strip() in t for t in _stash_texts(w, "README.md"))
    assert in_tree or parked, f"the user's patch is neither in the tree nor in any stash:\n{w.diag(cp)}"
    if parked and not in_tree:
        assert not G.reported_success(cp), (
            f"reported 'Update complete' (rc={cp.returncode}) while the user's patch is parked in a stash:\n"
            f"{w.diag(cp)}")


def test_untracked_file_colliding_with_a_new_upstream_file_is_not_lost(w):
    w.reset_clean()
    rel = "e2e-localmods-collide.txt"
    mine = "USER PRIVATE CONTENT that upstream never saw\n"
    (w.checkout / rel).write_text(mine, encoding="utf-8")
    target = w.publish("release: e2e localmods collide", {rel: "upstream adds this path\n"})

    cp = w.update()

    assert w.head() == target, f"code update did not land:\n{w.diag(cp)}"
    in_tree = (w.checkout / rel).read_text(encoding="utf-8") == mine
    parked = any(t == mine.strip() for t in _stash_texts(w, rel))
    assert in_tree or parked, (
        f"the user's untracked file {rel} is gone: the tree has upstream's copy and no stash keeps it:\n"
        f"{w.diag(cp)}")


def test_local_commit_on_main_is_kept_behind_a_named_ref(w):
    w.reset_clean()
    (w.checkout / "e2e-local-patch.txt").write_text("my local commit\n", encoding="utf-8")
    w.git("add", "-A")
    w.git("commit", "-q", "-m", "my local patch")
    local = w.head()
    target = w.publish("release: e2e localmods diverged", {"e2e-localmods-diverged.txt": "release\n"})

    cp = w.update()

    assert cp.returncode == 0 and w.head() == target, f"update did not land on {target}:\n{w.diag(cp)}"
    keepers = [r for r in w.refs_containing(local) if r != "refs/heads/main"]
    assert keepers, f"local commit {local[:12]} is reachable only from the reflog:\n{w.diag(cp)}"
    assert any(r.removeprefix("refs/") in G.output(cp) or r in G.output(cp) for r in keepers), (
        f"local commit {local[:12]} is kept by {keepers}, but the output never names that ref:\n{w.diag(cp)}")
    version = w.version()
    assert version.returncode == 0 and G.TRACEBACK not in G.output(version), w.diag(version)

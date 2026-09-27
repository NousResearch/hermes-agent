"""Truthful outcome of ``hermes update`` when the git transport fails.

The origin is a real smart-HTTP ``git http-backend``; faults are injected at the HTTP layer, where
GitHub's own failures happen: 429 rate limiting, 5xx on the ref advertisement or the fetch, a
connection dropped mid-pack, and a drop during the promisor lazy fetch that a ``--filter=tree:0``
install makes while checking out the new release. One HEAD install is shared by the cells.

Property: a failed transfer is a non-zero exit with no success banner and no success receipt; the
install is left runnable on the commit it had, with a clean tree, and no false diagnosis. Once the
fault clears a plain retry heals, and repeated interrupted fetches do not make that retry re-download
a pathological amount (a byte bound against a clean update of the same shape, never wall time).
"""

from __future__ import annotations

import os
import time

import pytest

from tests.e2e.core._pending_fixes import known_failure
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.git import _git_world as G
from tests.e2e.core.upgrade.git._smart_http import Fault

pytestmark = G.PYTESTMARK

FAULTS = {
    "429-on-fetch": lambda: Fault(status=429, command="fetch", times=99),
    "502-on-ref-advertisement": lambda: Fault(status=502, path_has="info/refs", times=99),
    "503-on-ls-refs": lambda: Fault(status=503, command="ls-refs", times=99),
    "cut-mid-pack": lambda: Fault(cut_after=150, command="fetch", times=99),
    # The commits arrive; the trees/blobs the tree:0 checkout then lazy-fetches are cut off.
    "cut-mid-checkout-lazy-fetch": lambda: Fault(cut_after=20000, command="fetch", skip=1, times=99),
}


@pytest.fixture(scope="module")
def w(tmp_path_factory):
    with G.world(tmp_path_factory.mktemp("git-truthful"), base=I.head_sha()) as world:
        yield world


def _assert_failed_truthfully(w: G.World, cp, before: str, mark: int) -> None:
    out = G.output(cp)
    diag = w.diag(cp, mark)
    assert cp.returncode != 0, f"update exited 0 although the transfer failed:\n{diag}"
    assert G.SUCCESS not in out, f"success banner printed although the transfer failed:\n{diag}"
    assert w.receipt().get("outcome") != "success", f"receipt records success for a failed transfer:\n{diag}"
    assert w.head() == before and w.branch() == "main", f"failed update moved HEAD:\n{diag}"
    assert not w.status(), f"failed update left the tree dirty:\n{diag}"
    version = w.version()
    assert version.returncode == 0 and G.TRACEBACK not in G.output(version), \
        f"install not runnable after the failed update:\n{w.diag(version, mark)}"


@pytest.mark.parametrize("fault", list(FAULTS))
def test_transport_failure_is_reported_as_failure(w, fault):
    w.reset_clean()
    w.srv.clear_faults()
    before = w.head()
    w.publish(f"release: e2e truthful {fault}", {f"e2e-truthful-{fault}.txt": "release\n"})
    armed = w.srv.arm(FAULTS[fault]())
    mark = w.srv.mark()

    cp = w.update()

    w.srv.clear_faults()
    assert armed.fired, f"the {fault} fault never fired; the cell exercised nothing:\n{w.diag(cp, mark)}"
    _assert_failed_truthfully(w, cp, before, mark)
    with known_failure(r"claims .*history diverged",
                       "gated on #124642: a ff-merge that failed for a non-divergence reason (lazy fetch cut) is "
                       "reported as diverged history, a rescue ref is written and `git reset --hard` is advised"):
        assert "diverged" not in G.output(cp) and not w.refs("refs/hermes-update-backups"), (
            f"a transport failure claims local history diverged (and writes a rescue ref):\n{w.diag(cp, mark)}")


def test_retry_after_interrupted_fetches_heals_without_refetching_history(w):
    w.reset_clean()
    w.srv.clear_faults()
    # Baseline: a clean update of a one-file release, bytes as the server sent them.
    clean_target = w.publish("release: e2e truthful baseline", {"e2e-truthful-baseline.txt": "release\n"})
    mark = w.srv.mark()
    cp = w.update()
    assert cp.returncode == 0 and w.head() == clean_target, f"baseline update failed:\n{w.diag(cp, mark)}"
    baseline = w.bytes_since(mark)

    before = w.head()
    target = w.publish("release: e2e truthful after cuts", {"e2e-truthful-after-cuts.txt": "release\n"})
    # Every checkout lazy fetch is cut mid-pack, after ~20 KB of pack data in whole pkt-lines (so
    # index-pack has started and each cut leaves a dead temp pack); the commits arrive on the first attempt.
    w.srv.arm(Fault(cut_after=20000, command="fetch", skip=1, times=99))
    for attempt in range(3):
        mark = w.srv.mark()
        cp = w.update()
        assert cp.returncode != 0 and w.head() == before, f"interrupted attempt {attempt + 1} did not fail cleanly:\n" \
            f"{w.diag(cp, mark)}"
    w.srv.clear_faults()
    pack_dir = w.checkout / ".git" / "objects" / "pack"
    dead = sorted(p.name for p in pack_dir.glob("tmp_*"))
    assert dead, (
        "the interrupted fetches left no temp packs; the cell exercised nothing (a cut the client saw before "
        "any whole pack pkt-line never starts index-pack: see 'pack bytes in whole pkt-lines' below)\n"
        f"objects/pack: {sorted(p.name for p in pack_dir.iterdir())}\n{w.diag(cp, mark)}")
    old = time.time() - 3600  # the user retries later: dead transfers are past any in-flight window
    for p in pack_dir.glob("tmp_*"):
        os.utime(p, (old, old))

    mark = w.srv.mark()
    cp = w.update()

    healed = w.bytes_since(mark)
    assert cp.returncode == 0 and G.SUCCESS in G.output(cp) and w.head() == target, \
        f"retry after the fault cleared did not heal:\n{w.diag(cp, mark)}"
    assert healed <= baseline * 1.5 + 64 * 1024, (
        f"after 3 interrupted fetches the healing update downloaded {healed} bytes, vs {baseline} for a clean update "
        f"of the same shape: it re-fetched far more than the release:\n{w.diag(cp, mark)}")
    debris = sorted(p.name for p in pack_dir.glob("tmp_*") if p.stat().st_mtime <= old + 1)
    assert not debris, f"dead transfers' temp packs survived a successful update: {debris}\n{w.diag(cp, mark)}"

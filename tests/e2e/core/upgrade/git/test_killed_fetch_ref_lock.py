"""A real ``hermes update`` killed inside its fetch's ref transaction, then the user's next update.

Failure class: the Windows crash cell ``mid_fetch`` (``tests/e2e/core/windows_update``) kills the
update's tree while its ``git fetch`` runs. When the kill lands while git holds
``refs/remotes/origin/main.lock``, that file outlives git, and git 2.51+ rejects the ref on every
later fetch: "fetching ref refs/remotes/origin/main failed: reference already exists". No update
could fetch again until a user found and deleted the file.

Here the kill point is deterministic: git's own ``reference-transaction`` hook (in the checkout's
``.git/hooks``, nothing in the product) holds the fetch in the ``prepared`` state, when every ref
lock of the transaction is taken. The whole sandboxed update tree is then SIGKILLed (a closed
console, ``taskkill /T /F``), and the next plain ``hermes update`` of the same home must complete
to the target.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.git.test_hostile_lock import Rig, _finish

pytestmark = [
    pytest.mark.platforms("linux"),
    pytest.mark.live_system_guard_bypass,
    pytest.mark.skipif(H.sandbox_required_reason() is not None, reason=str(H.sandbox_required_reason())),
]

HOOK = """#!/bin/sh
input=$(cat)
case "$1:$input" in
  prepared:*refs/remotes/origin/main*)
    if [ -e "{hold}" ]; then
      : > "{held}"
      while [ -e "{hold}" ]; do sleep 0.05; done
    fi ;;
esac
"""


def test_an_update_killed_holding_its_fetchs_ref_lock_does_not_block_the_next(tmp_path):
    rig = Rig(tmp_path)
    origin = Path(I.git("remote", "get-url", "origin", cwd=rig.checkout))
    tree = I.git("rev-parse", "main^{tree}", cwd=origin)
    target = I.git("-c", "user.name=e2e", "-c", "user.email=e2e@example.invalid", "commit-tree", tree, "-p", "main", "-m", "next release", cwd=origin)
    I.git("update-ref", "refs/heads/main", target, cwd=origin)
    hold, held = tmp_path / "hold-ref-transaction", tmp_path / "held-ref-transaction"
    hook = rig.checkout / ".git" / "hooks" / "reference-transaction"
    hook.write_text(HOOK.format(hold=hold, held=held), encoding="utf-8")
    hook.chmod(0o755)
    lock = rig.checkout / ".git" / "refs" / "remotes" / "origin" / "main.lock"

    hold.touch()
    # Rig.update's process is not a group leader, so H.kill_tree could not reach it.
    first = subprocess.Popen(
        H.sandbox_argv([sys.executable, "-m", "hermes_cli.main", "update", "--yes", "--branch", "main",
                        "--no-gateway-restart"], writable=[tmp_path]),
        env=rig.env("home-a"), cwd=rig.checkout, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        H.wait_for(lambda: held.exists() or first.poll() is not None, timeout=240,
                   what="the update's fetch to reach its ref transaction")
        assert first.poll() is None, f"the update exited before its fetch locked the ref:\n{first.communicate()[0][-4000:]}"
        assert lock.exists(), "premise: the held fetch holds refs/remotes/origin/main.lock"
    finally:
        H.kill_tree(first)
        first.communicate()
        hold.unlink()
    assert lock.exists(), "premise: the killed fetch left its ref lock behind"

    code, out = _finish(rig.update("home-a"), timeout=600)
    head = I.git("rev-parse", "HEAD", cwd=rig.checkout)
    assert code == 0 and head == target, f"the update after the killed one did not complete (rc={code}, HEAD {head}):\n{out[-6000:]}"
    assert not lock.exists()

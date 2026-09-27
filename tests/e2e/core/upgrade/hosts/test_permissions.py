"""Install trees the running user cannot write: turns keep working, updates fail and say why.

Failure class: users and permissions. A second uid sharing an install (a root gateway whose
workers run as other uids, #120151's topology), an admin-provisioned machine or an image leaves
the code and PM state (checkout, tool store, ``installs/``) readable but not writable by the user
who runs Hermes, while their own data under HERMES_HOME stays writable. On such a tree:

* a turn must work: deciding that an install is current is a read, and nothing on the launch
  path may need write access to the install (gated on #124635 while every launch opens PM
  lock files for writing);
* ``hermes update`` must exit non-zero with a message that names the permission problem, never
  report success, never crash with a traceback, and leave the install on its old commit;
* once the tree is writable again, the same ``hermes update`` succeeds: the refused attempt left
  nothing behind that blocks the next one.

One real install through HEAD's ``scripts/install.sh``; upstream then publishes a release. The
read-only state is real file modes (``chmod -R a-w``), which is exactly what a different uid
sees on a tree it does not own.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from tests.e2e.core._pending_fixes import known_failure
from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.hosts import _hosts as X
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [
    pytest.mark.platforms("linux"),
    pytest.mark.live_system_guard_bypass,
    pytest.mark.skipif(H.sandbox_required_reason() is not None, reason=str(H.sandbox_required_reason())),
    pytest.mark.skipif(shutil.which("git") is None, reason="git required"),
    pytest.mark.skipif(I.real_uv() is None, reason="uv required"),
]

# What the install owns; HERMES_HOME itself (config, .env, sessions, logs) stays the user's.
CODE_DIRS = ("hermes-agent", "tools", "installs")
ACTIONABLE = re.compile(r"(?i)permission denied|read-only|not writable|no write access|EACCES|Errno 13")


def _chmod_code(sb: I.Sandbox, *, writable: bool) -> None:
    for name in CODE_DIRS:
        subprocess.run(["chmod", "-R", "u+w" if writable else "a-w", str(sb.hermes_home / name)], check=True)


@pytest.fixture(scope="module")
def provider():
    with FakeLLMServer(default_text="fake reply for the permissions suite") as srv:
        yield srv


@pytest.fixture(scope="module")
def world(tmp_path_factory, provider):
    root = tmp_path_factory.mktemp("perms")
    origin = X.make_origin(root)
    sb = X.new_sandbox(root / "sb", origin)
    out = {"sb": sb, "install": I.run_installer(sb)}
    try:
        if out["install"].returncode == 0:
            X.configure(sb, provider)
            X.turn(sb, provider, "turn on a writable tree")
            out["before"] = I.git("rev-parse", "HEAD", cwd=sb.checkout)
            out["target"] = I.publish_commit(origin, root, "release: permissions bump",
                                             {"docs/e2e-host-perms-marker.txt": "release 1\n"})
            n = len(provider.main_requests())
            _chmod_code(sb, writable=False)
            try:
                out["ro_turn"] = sb.run([sb.hermes, "-z", "turn on a read-only install tree"], timeout=900)
                out["ro_turn_requests"] = len(provider.main_requests()) - n
            finally:
                _chmod_code(sb, writable=True)
            # Only the checkout read-only: the launch path works (#124635 needs installs/ too), so
            # this reaches `hermes update`'s own handling of a tree it cannot write.
            subprocess.run(["chmod", "-R", "a-w", str(sb.checkout)], check=True)
            try:
                out["ro_update"] = sb.cli("update", "--yes", "--branch", "main", timeout=X.UPDATE_TIMEOUT)
                out["ro_head"] = I.git("rev-parse", "HEAD", cwd=sb.checkout)
            finally:
                _chmod_code(sb, writable=True)
            out["rw_update"] = sb.cli("update", "--yes", "--branch", "main", timeout=X.UPDATE_TIMEOUT)
        yield out
    finally:
        if sb.hermes_home.exists():
            _chmod_code(sb, writable=True)


def test_read_only_install_tree_still_runs_a_turn(world, provider):
    assert world["install"].returncode == 0, "install.sh failed:\n" + I.describe(world["install"])
    cp = world["ro_turn"]
    err = cp.stdout + cp.stderr
    assert I.TRACEBACK not in err, "a turn crashed on a read-only install tree:\n" + I.describe(cp)
    with known_failure(r"read-only install tree: the turn failed .*Permission denied: .*/installs/.*\.lock",
                       "gated on #124635: every launch opens PM lock files under installs/ for writing, so a "
                       "uid without write access to the install cannot start Hermes"):
        # Either it works, or it says the install is not writable by this user without sending them
        # to a remedy that needs the same write access (`hermes pm repair`, "finish the update").
        if cp.returncode != 0:
            misleading = re.findall(r"run `hermes (?:pm repair|update)`[^\n]*", err)
            assert ACTIONABLE.search(err) and not misleading, (
                f"read-only install tree: the turn failed (rc={cp.returncode}) and pointed at remedies that need the "
                f"same write access {misleading}: {cp.stderr.strip()[-800:].replace(chr(10), ' | ')}\n" + I.describe(cp))
    if cp.returncode == 0:
        assert provider.default_text in cp.stdout, "the reply never reached stdout:\n" + I.describe(cp)
        assert world["ro_turn_requests"] == 1, f"the turn reached the provider {world['ro_turn_requests']} times (want 1)"


def test_update_on_a_read_only_tree_refuses_with_a_reason_and_recovers(world, provider):
    assert world["install"].returncode == 0, "install.sh failed:\n" + I.describe(world["install"])
    up = world["ro_update"]
    out = up.stdout + up.stderr
    assert up.returncode != 0, "hermes update reported success on an install tree it cannot write:\n" + I.describe(up)
    assert I.TRACEBACK not in out, "hermes update crashed on a read-only install tree:\n" + I.describe(up)
    assert ACTIONABLE.search(out), "hermes update failed on a read-only tree without saying why:\n" + I.describe(up)
    assert world["ro_head"] == world["before"], "a refused update still moved the checkout"
    again = world["rw_update"]
    assert again.returncode == 0 and I.TRACEBACK not in again.stdout + again.stderr, (
        "after the tree became writable again, hermes update still failed:\n" + I.describe(again))
    assert I.git("rev-parse", "HEAD", cwd=world["sb"].checkout) == world["target"], (
        "update exited 0 but HEAD is not the new release")
    X.turn(world["sb"], provider, "turn after the recovered update")

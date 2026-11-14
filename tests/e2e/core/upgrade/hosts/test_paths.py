"""Install, update, gateway and a turn in a home whose path has spaces and non-ASCII bytes.

Failure class: path handling. Users run Hermes from homes like ``/home/José Müller`` or a
CJK-named directory. Every path the installer and updater write (clone target, uv/python/node
store, PM generation, the ``~/.local/bin/hermes`` launcher, the rc-file PATH line, sys.path) must
survive a space and non-ASCII bytes.

One real install through HEAD's ``scripts/install.sh`` with the whole sandbox (HOME, HERMES_HOME,
TMPDIR) under ``José Müller 漢字 dir``; then a one-shot turn, a new login shell resolving
``hermes``, an upstream release and ``hermes update``, a turn on the new commit and
``hermes gateway run``. The symlinked-home shapes live in ``test_paths_symlinked.py`` so the two
installs run in parallel.
"""

from __future__ import annotations

import os
import shutil

import pytest

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

ODD_DIR = "José Müller 漢字 dir"


@pytest.fixture(scope="module")
def provider():
    with FakeLLMServer(default_text="fake reply for the host-paths suite") as srv:
        yield srv


@pytest.fixture(scope="module")
def odd_home(tmp_path_factory, provider):
    root = tmp_path_factory.mktemp("paths-odd")
    origin = X.make_origin(root)
    sb = X.new_sandbox(root / ODD_DIR, origin)
    (sb.home / ".bashrc").write_text("# ~/.bashrc\ncase $- in *i*) ;; *) return;; esac\n", encoding="utf-8")
    (sb.home / ".profile").write_text('if [ -f "$HOME/.bashrc" ]; then . "$HOME/.bashrc"; fi\n', encoding="utf-8")
    first = I.run_installer(sb)
    return {"sb": sb, "origin": origin, "root": root, "install": first}


def test_install_update_gateway_and_turn_under_a_non_ascii_spaced_home(odd_home, provider):
    sb, first = odd_home["sb"], odd_home["install"]
    assert ODD_DIR in str(sb.home), "harness: sandbox HOME is not under the odd directory"
    assert first.returncode == 0, "install.sh failed under a non-ASCII, spaced HOME:\n" + I.describe(first)
    assert I.git("rev-parse", "HEAD", cwd=sb.checkout) == I.head_sha(), "installed checkout is not the published commit"
    X.configure(sb, provider)
    t1 = X.turn(sb, provider, "first turn under a non-ASCII home")
    assert not X.reran_completion(t1), "the first launch after install re-ran the completion:\n" + I.describe(t1)

    # A new login shell finds the launcher through the rc line the installer wrote.
    probe = X.login_shell(sb, "command -v hermes; hermes --version >/dev/null && echo LAUNCH-OK")
    assert probe.returncode == 0, H.describe(probe)
    lines = probe.stdout.strip().splitlines()
    assert lines and lines[0] == sb.hermes and "LAUNCH-OK" in lines, (
        f"a new login shell does not run the installed launcher (want {sb.hermes}):\n" + H.describe(probe))

    # Every sys.path entry of the selected interpreter names a real location.
    seen = X.interpreter_paths(sb)
    assert not X.missing_entries(seen["sys_path"]), f"sys.path names locations that do not exist: {X.missing_entries(seen['sys_path'])}"
    assert os.path.exists(seen["executable"]), f"sys.executable does not exist: {seen['executable']!r}"

    target = I.publish_commit(odd_home["origin"], odd_home["root"], "release: host-paths bump",
                              {"docs/e2e-host-paths-marker.txt": "release 1\n"})
    up = sb.cli("update", "--yes", "--branch", "main", timeout=X.UPDATE_TIMEOUT)
    assert up.returncode == 0 and I.TRACEBACK not in up.stdout + up.stderr, (
        "hermes update failed under a non-ASCII, spaced HOME:\n" + I.describe(up))
    assert I.git("rev-parse", "HEAD", cwd=sb.checkout) == target, "update exited 0 but HEAD is not the new release"
    t2 = X.turn(sb, provider, "turn after updating under a non-ASCII home")
    assert not X.reran_completion(t2), "the first launch after a finished update re-ran the completion:\n" + I.describe(t2)
    seen = X.interpreter_paths(sb)
    assert not X.missing_entries(seen["sys_path"]), f"after update, sys.path names missing locations: {X.missing_entries(seen['sys_path'])}"

    gw = X.Gateway(sb)
    try:
        st = gw.start()
        assert st.get("pid") and gw.proc.poll() is None, f"gateway state says running but the process is gone: {st}\n{gw.tail()}"
    finally:
        gw.stop()

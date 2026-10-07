"""The PM runtime lock is where PM commands look for it (PM lifecycle, class 6; #124075).

The field report: after ``rabbit gateway install --force`` the gateway crash-looped with "no
dependency environment is committed" and ``rabbit pm repair`` / ``pm doctor`` / ``pm status`` all
died on ``FileNotFoundError: <install>/environments/<env>/workspace/pm/uv.lock``.

``--force`` was incidental: every generation's ``workspace/`` copy was materialised without
``pm/uv.lock`` (the copy dropped every file named ``uv.lock``), so ANY ``rabbit pm`` command run
by the selected generation's own ``rabbit`` (the one on PATH for every child a Rabbit process
spawns, and the one a unit pointing at the managed environment runs) died before it started.
After a dependency-changing ``rabbit update``:

* ``rabbit gateway install --force`` (service manager is a failing shim: the unit write is what
  is exercised) leaves the install bootable: ``rabbit pm status`` / ``pm doctor`` run and
  ``rabbit gateway run`` reaches running;
* ``rabbit pm status`` / ``pm doctor`` also run from the managed environment's ``rabbit``.
"""

from __future__ import annotations

import shutil

import pytest

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.pm import _pm as P
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [
    pytest.mark.platforms("linux"),
    pytest.mark.live_system_guard_bypass,
    pytest.mark.skipif(H.sandbox_required_reason() is not None, reason=str(H.sandbox_required_reason())),
    pytest.mark.skipif(shutil.which("git") is None, reason="git required"),
    pytest.mark.skipif(I.real_uv() is None, reason="uv required"),
]


def _pm_failures(sb: I.Sandbox, rabbit: str) -> list[str]:
    bad = []
    for args in (("pm", "status"), ("pm", "doctor")):
        cp = sb.run([rabbit, *args], timeout=300)
        if cp.returncode != 0 or I.TRACEBACK in cp.stdout + cp.stderr:
            tail = (cp.stdout + cp.stderr).strip().splitlines()[-1:] or [""]
            bad.append(f"`rabbit {' '.join(args)}` rc={cp.returncode}: {tail[0][:300]}\n{I.describe(cp)}")
    return bad


@pytest.fixture(scope="module")
def provider():
    with FakeLLMServer(default_text="fake reply for the pm gateway-install suite") as srv:
        yield srv


@pytest.fixture(scope="module")
def updated(tmp_path_factory, provider):
    root = tmp_path_factory.mktemp("pm-gwinstall")
    sb, origin = P.install_head(root)
    P.configure(sb, provider.base_url)
    installed = P.selected_generation(sb)
    P.publish_dependency_release(origin, root, 1)
    P.ok(P.update(sb, env=P.lazy_env(sb)), "rabbit update failed")
    assert P.selected_generation(sb) != installed, "harness: the update did not select a new generation"
    return sb


def test_gateway_install_force_leaves_the_install_bootable(updated):
    sb = updated
    cp = P.run_env(sb, [sb.rabbit, "gateway", "install", "--force"], P.lazy_env(sb), timeout=600)
    assert I.TRACEBACK not in cp.stdout + cp.stderr, I.describe(cp)
    bad = _pm_failures(sb, sb.rabbit)
    assert not bad, "rabbit pm commands fail after `gateway install --force`:\n" + "\n".join(bad)
    gw = P.Gateway(sb, P.lazy_env(sb), sb.root / "after-install-force.log")
    try:
        gw.wait_running()
    finally:
        gw.stop()


def test_pm_commands_run_from_the_managed_environment(updated):
    sb = updated
    exe = str(P.selected_generation(sb) / "venv" / "bin" / "rabbit")
    bad = _pm_failures(sb, exe)
    assert not bad, "rabbit pm commands die from the managed environment's rabbit:\n" + "\n".join(bad)

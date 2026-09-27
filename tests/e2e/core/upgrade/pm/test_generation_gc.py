"""Repeated updates and repairs don't grow the install without bound (PM lifecycle,
failure class 5; #123340).

Every dependency rebuild publishes a NEW generation under ``installs/<key>/environments/`` (about
half a gigabyte with ``[all]``). The previous generation must stay while a running process may
still read it (leases) and for a day after it was built, and must be collected after that, by the
command that superseded it, not only by a manual ``hermes pm gc``.

The seeded state is the one a daily-updating user has: before each rebuild the existing
generations' ``.lease-managed`` stamps are aged two days (as if they were built on earlier days),
then the user runs ``hermes update`` / ``hermes pm repair``. After each rebuild the aged,
unselected generations must be gone, the selected one kept and working (a real ``hermes -z`` turn
through the loopback provider). Two updates: the first one's leftover is the generation the updater
itself ran from (leased, legitimately kept); the second shows whether anything reclaims it.
Restarts are covered in ``test_restart_generations.py``. Neither ``hermes update`` nor ``hermes pm repair`` runs the collector today (gated on #124668);
the generation the rebuilding process itself ran from is leased by it and legitimately survives.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

import pytest

from tests.e2e.core._pending_fixes import known_failure
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

TWO_DAYS = 2 * 86400


def _age_all_generations(sb: I.Sandbox) -> list[str]:
    """Pretend every existing generation was built two days ago; returns their names."""
    envs = P.state_dir(sb) / "environments"
    old = time.time() - TWO_DAYS
    aged = []
    for gen in envs.iterdir():
        marker = gen / ".lease-managed"
        if marker.is_file():
            os.utime(marker, (old, old))
            aged.append(gen.name)
    return sorted(aged)


@pytest.fixture(scope="module")
def provider():
    with FakeLLMServer(default_text="fake reply for the pm gc suite") as srv:
        yield srv


@pytest.fixture(scope="module")
def home(tmp_path_factory, provider):
    root = tmp_path_factory.mktemp("pm-gc")
    sb, origin = P.install_head(root)
    P.configure(sb, provider.base_url)
    return {"sb": sb, "origin": origin, "root": root}


def _rebuild_collects(sb: I.Sandbox, provider: FakeLLMServer, what: str, rebuild) -> tuple:
    before_sel = P.selected_generation(sb).name
    aged = _age_all_generations(sb)
    assert before_sel in aged, f"harness: selected generation {before_sel} carries no lease stamp: {aged}"
    cp = rebuild()
    assert cp.returncode == 0, f"{what} failed:\n" + P.diagnostics(sb, cp)
    selected = P.selected_generation(sb).name
    assert selected != before_sel, f"harness: {what} did not build a new generation\n" + P.diagnostics(sb, cp)
    left = P.generations(sb)
    # The generation the rebuilding process itself ran from is leased by it until it exits; every
    # OTHER superseded generation older than a day must be gone.
    kept_old = sorted((set(aged) & set(left)) - {before_sel})
    assert selected in left, f"{what} collected the generation it selected: {selected} not in {left}"
    P.turn(sb, provider, f"turn after {what}")
    return kept_old, left, selected, before_sel, cp


def _assert_collected(sb: I.Sandbox, what: str, result) -> None:
    kept_old, left, selected, before_sel, cp = result
    with known_failure(r"kept superseded generations built two days ago",
                       "gated on #124668: hermes update / pm repair never run the generation collector"):
        assert not kept_old, (
            f"{what} kept superseded generations built two days ago: {kept_old} (all: {left}, selected "
            f"{selected}, previous {before_sel}); every rebuild adds one and nothing reclaims them\n"
            + P.diagnostics(sb, cp))


def test_repeated_updates_collect_superseded_generations(home, provider):
    sb = home["sb"]
    results = []
    for n in (1, 2):
        target = P.publish_dependency_release(home["origin"], home["root"], n)
        results.append(_rebuild_collects(sb, provider, f"`hermes update` #{n}",
                                         lambda: P.update(sb, env=P.lazy_env(sb))))
        assert I.git("rev-parse", "HEAD", cwd=sb.checkout) == target
    for n, result in enumerate(results, 1):
        _assert_collected(sb, f"`hermes update` #{n}", result)


def test_repeated_repairs_collect_superseded_generations(home, provider):
    sb = home["sb"]
    results = [_rebuild_collects(sb, provider, f"`hermes pm repair` #{n}",
                                 lambda: P.run_env(sb, [sb.hermes, "pm", "repair"], P.lazy_env(sb),
                                                   timeout=P.UPDATE_TIMEOUT))
               for n in (1, 2)]
    for n, result in enumerate(results, 1):
        _assert_collected(sb, f"`hermes pm repair` #{n}", result)

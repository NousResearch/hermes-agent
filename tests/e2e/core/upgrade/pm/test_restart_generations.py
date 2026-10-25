"""Restarts never build or accumulate dependency generations (PM lifecycle, failure class 5; #123340).

A current install relaunched over and over (the CLI, and the gateway a service manager restarts)
must reuse the selected generation: no dependency sync, no source-update tail, no new generation.
A gateway started while a source-update tail is still owed (``source-completion-pending``) may
finish that tail once, on its first start, and must neither build a generation nor re-run the tail
on later starts (#123340: it re-ran it, building a new generation, on every restart).
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


@pytest.fixture(scope="module")
def provider():
    with FakeLLMServer(default_text="fake reply for the pm restart suite") as srv:
        yield srv


@pytest.fixture(scope="module")
def home(tmp_path_factory, provider):
    root = tmp_path_factory.mktemp("pm-restarts")
    sb, _origin = P.install_head(root)
    P.configure(sb, provider.base_url)
    return {"sb": sb}


def test_restarts_never_build_or_accumulate_generations(home, provider):
    sb = home["sb"]
    before = P.generations(sb)
    rebuilt = []
    for n in range(3):
        cp = P.turn(sb, provider, f"plain relaunch {n}")
        rebuilt += [f"relaunch {n}: {line}" for line in cp.stderr.splitlines()
                    if "source-update dependencies" in line or "interrupted source update" in line]
    assert not rebuilt, "a plain relaunch of a current install re-ran the dependency sync / update tail:\n" + "\n".join(rebuilt)
    assert P.generations(sb) == before, f"plain relaunches changed the generation set: {before} -> {P.generations(sb)}"


def test_gateway_restarts_with_an_owed_tail_do_not_rebuild(home):
    """#123340: a gateway started with ``source-completion-pending`` present re-ran the tail (and
    built a generation) on every start. The first start may finish the tail; none may add a
    generation, and later starts must owe nothing."""
    sb = home["sb"]
    P.arm_pending_tail(sb)
    before = P.generations(sb)
    boots = []
    for n in range(3):
        gw = P.Gateway(sb, P.lazy_env(sb), sb.root / f"gc-gateway-{n}.log")
        try:
            gw.wait_running()
        finally:
            gw.stop()
        boots.append({"gens": P.generations(sb), "pending": P.pending_marker(sb).exists(),
                      "tail": "finishing an interrupted source update" in gw.output()})
    grown = [b["gens"] for b in boots if b["gens"] != before]
    assert not grown, f"gateway restarts built dependency generations: {before} -> {grown}\n" + P.diagnostics(sb)
    assert not boots[0]["pending"], f"the first gateway start left the owed tail pending: {boots}"
    assert not any(b["tail"] for b in boots[1:]), f"every gateway start re-runs the source-update tail: {boots}"

"""Interrupted dependency updates recover (PM lifecycle, failure class 2).

``hermes update`` on a release that changes the dependency set runs a PM transaction: it arms the
``source-completion-pending`` marker, stages a NEW generation under
``installs/<key>/environments/`` (download + ``uv sync``), publishes it by rewriting facts.json,
then runs the source-update tail (launchers, TUI/web builds) and clears the marker.

Each cell SIGKILLs the whole sandboxed updater (power cut, closed laptop lid) at one phase
boundary, observed from outside through the files PM itself writes:

* ``stage``: a new generation directory exists, facts.json still selects the old one;
* ``publish``: facts.json has just been flipped to the new generation, the tail is still owed.

(The gateway relaunch hand-off after an update is lane ``handoff``'s suite.)

The user-visible contract after the kill, as a user retries later (no lazy-install override, so
the real launch path runs): the very next ``hermes`` works and reaches the provider; ``hermes
update`` exits 0 at the target commit and leaves no pending marker; the launch after that does
not re-run the completion again (a marker that loops forever is #123933's failure), and a
half-built generation is never the one selected.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time

import pytest

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.pm import _pm as P
from tests.e2e.core.upgrade.test_upgrade_path import _KILLED_RUN_PREFIX, _RETRY_PREFIX
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [
    pytest.mark.platforms("linux"),
    pytest.mark.live_system_guard_bypass,
    pytest.mark.skipif(H.sandbox_required_reason() is not None, reason=str(H.sandbox_required_reason())),
    pytest.mark.skipif(shutil.which("git") is None, reason="git required"),
    pytest.mark.skipif(I.real_uv() is None, reason="uv required"),
]

COMPLETION_NOTICES = ("finishing an interrupted source update", "completing source-update dependencies")


@pytest.fixture(scope="module")
def provider():
    with FakeLLMServer(default_text="fake reply for the pm interrupt suite") as srv:
        yield srv


@pytest.fixture(scope="module")
def home(tmp_path_factory, provider):
    root = tmp_path_factory.mktemp("pm-interrupt")
    sb, origin = P.install_head(root)
    P.configure(sb, provider.base_url)
    return {"sb": sb, "origin": origin, "root": root, "releases": 0}


def _retry(sb: I.Sandbox, *args: str, timeout: float = P.UPDATE_TIMEOUT) -> subprocess.CompletedProcess:
    """A later retry by the user: a fresh process whose pid is never the dead updater's."""
    return P.run_env(sb, [*_RETRY_PREFIX, sb.hermes, *args], P.lazy_env(sb), timeout=timeout)


def _turn(sb: I.Sandbox, provider: FakeLLMServer, marker: str) -> subprocess.CompletedProcess:
    n = len(provider.main_requests())
    cp = _retry(sb, "-z", marker)
    new = provider.main_requests()[n:]
    assert cp.returncode == 0 and I.TRACEBACK not in cp.stdout + cp.stderr, P.diagnostics(sb, cp)
    assert len(new) == 1 and marker in json.dumps(new[0]["messages"]), (
        f"`hermes -z` did not reach the provider exactly once ({len(new)} requests)\n" + P.diagnostics(sb, cp))
    return cp


def _kill_update_at(sb: I.Sandbox, phase: str) -> dict:
    gen0, known = P.selected_generation(sb), set(P.generations(sb))
    log = sb.root / f"killed-update-{phase}.log"

    def staged():
        try:
            return bool(set(P.generations(sb)) - known) and P.selected_generation(sb) == gen0
        except (OSError, ValueError):
            return False  # facts.json mid-replace

    def published():
        try:
            return P.selected_generation(sb) != gen0 and P.pending_marker(sb).exists()
        except (OSError, ValueError):
            return False  # facts.json mid-replace

    with log.open("w") as out:
        proc = subprocess.Popen(
            H.sandbox_argv([*_KILLED_RUN_PREFIX, sb.hermes, "update", "--yes", "--branch", "main"], writable=[sb.root]),
            env=P.lazy_env(sb), cwd=str(sb.root), stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
            text=True, start_new_session=True)
        try:
            P.kill_when(proc, staged if phase == "stage" else published, timeout=P.UPDATE_TIMEOUT,
                        what=f"the {phase} phase of the dependency update")
        except AssertionError as exc:
            raise AssertionError(f"{exc}\n--- update log ---\n{log.read_text(errors='replace')[-5000:]}") from None
    return {"gen0": gen0, "known": known, "at_kill": P.selected_generation(sb), "gens_at_kill": P.generations(sb),
            "pending_at_kill": P.pending_marker(sb).exists(), "log": log.read_text(errors="replace")}


@pytest.mark.parametrize("phase", ["stage", "publish"])
def test_sigkill_mid_dependency_update_recovers(home, provider, phase):
    sb = home["sb"]
    assert not P.pending_marker(sb).exists(), "harness: install is not healthy before the cell\n" + P.diagnostics(sb)
    home["releases"] += 1
    target = P.publish_dependency_release(home["origin"], home["root"], home["releases"])
    killed = _kill_update_at(sb, phase)
    assert killed["pending_at_kill"], f"the update was killed at {phase} but owed no tail marker:\n{killed}"
    if phase == "stage":
        assert killed["at_kill"] == killed["gen0"], f"harness: killed after publish, not mid-stage: {killed}"
    else:
        assert killed["at_kill"] != killed["gen0"], f"harness: killed before publish: {killed}"

    # The very next launch works and reaches the provider.
    first = _turn(sb, provider, f"first turn after a kill at {phase}")
    # The user re-runs the update: it converges at the target and owes nothing.
    again = _retry(sb, "update", "--yes", "--branch", "main")
    assert again.returncode == 0 and I.TRACEBACK not in again.stdout + again.stderr, (
        f"`hermes update` after a kill at {phase} failed\n" + P.diagnostics(sb, first, again))
    assert I.git("rev-parse", "HEAD", cwd=sb.checkout) == target, P.diagnostics(sb, again)
    assert not P.pending_marker(sb).exists(), (
        f"`hermes update` exited 0 after a kill at {phase} but left source-completion-pending behind\n"
        + P.diagnostics(sb, first, again))
    selected = P.selected_generation(sb)
    if phase == "stage":
        half_built = set(killed["gens_at_kill"]) - killed["known"]
        assert selected.name not in half_built, (
            f"the generation half-built by the killed update was selected: {selected.name}\n" + P.diagnostics(sb, again))
    imports = P.managed_imports(sb, "ruamel.yaml", "pydantic", "openai", "httpx")
    assert set(imports.values()) == {"ok"}, f"selected generation cannot import core deps: {imports}"

    # The launch after that owes nothing: no completion loop, no new generation per launch.
    gens = P.generations(sb)
    t0 = time.monotonic()
    later = _turn(sb, provider, f"second turn after a kill at {phase}")
    noisy = [n for n in COMPLETION_NOTICES if n in later.stderr]
    assert not noisy, (
        f"a finished update still re-runs the source-update completion on every launch ({noisy}, "
        f"{time.monotonic() - t0:.0f}s)\n" + P.diagnostics(sb, later))
    assert P.generations(sb) == gens, f"a plain launch built a new generation: {gens} -> {P.generations(sb)}"

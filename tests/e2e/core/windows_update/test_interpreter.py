"""Which site-packages a Windows gateway and CLI load after an update.

Failure class: interpreter. Real Windows machines carry two traps the managed runtime
must ignore:

* a different, standalone Python on PATH ahead of anything Hermes installed (#123185);
* the pre-PM in-tree ``hermes-agent\\venv`` an older install left behind (#123965, #123972).

After ``hermes update`` neither the gateway the user starts through ``hermes.exe`` nor a
CLI turn may add the stale venv to ``sys.path`` or publish it to its children, with the
standalone Python first on PATH. The stale venv carries a ``.pth`` hook that
drops a ``<pid>`` marker whenever any process adds its site dir, so "loaded the stale
venv" is observed directly rather than inferred from a crash.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import psutil
import pytest

from tests.e2e.core.windows_update._machine import (
    REQUIRES_OPT_IN,
    Journey,
    Machine,
    fail_with,
    new_machine,
    one_shot_turn,
)
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [pytest.mark.platforms("windows"), pytest.mark.integration,
              pytest.mark.live_system_guard_bypass, REQUIRES_OPT_IN]

_WORKER_DEATH = re.compile(r"^.*Supervised task \S+ died.*$", re.MULTILINE)


def _system_python_dir() -> Path:
    """The newest standalone CPython in the runner's tool cache: the "Python on PATH"."""
    cache = Path(os.environ.get("RUNNER_TOOL_CACHE") or r"C:\hostedtoolcache\windows")
    found = sorted(cache.glob("Python/3.*/x64/python.exe"),
                   key=lambda p: tuple(int(x) for x in re.findall(r"\d+", p.parent.parent.name)[:3]))
    assert found, f"harness: no standalone CPython under {cache} to put on PATH"
    return found[-1].parent


def _seed_stale_venv(machine: Machine) -> Path:
    """A pre-PM in-tree venv as older installs left it, with a load detector."""
    markers = machine.root / "stale-venv-loads"
    markers.mkdir(parents=True, exist_ok=True)
    venv = machine.install_dir / "venv"
    site = venv / "Lib" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    (venv / "Scripts").mkdir(exist_ok=True)
    (venv / "pyvenv.cfg").write_text(
        "home = C:\\Python311\ninclude-system-site-packages = false\nversion = 3.11.15\n", encoding="utf-8")
    (machine.install_dir / _PROBE).write_text(
        f"import os; open(os.path.join({str(markers)!r}, str(os.getpid())), 'w').close()\n", encoding="utf-8")
    return markers


_PROBE = Path("venv", "Lib", "site-packages", "zz_e2e_stale_venv_probe.pth")


def _loaded_by(markers: Path) -> set[int]:
    return {int(p.name) for p in markers.iterdir() if p.name.isdigit()} if markers.is_dir() else set()


@pytest.fixture(scope="module")
def journey(tmp_path_factory):
    system_python = _system_python_dir()
    with FakeLLMServer() as srv:
        machine = new_machine(tmp_path_factory.mktemp("interp"), srv.base_url, label="interp", system_git=True)
        j = Journey(machine)
        try:
            install = j.step("install", machine.install)
            if j.ok("install"):
                j.step("install_ok", lambda: j.require("install", install.returncode == 0,
                                                        f"install.ps1 exited {install.returncode}", install))
            if j.ok("install_ok"):
                markers = _seed_stale_venv(machine)
                j.results["markers"] = markers
                # From here on the user has installed a standalone Python and put it first on PATH.
                machine.path_prepend = [str(system_python), str(system_python / "Scripts")]
                j.results["system_python"] = system_python
                machine.advance()
                update = j.step("update", machine.update)
                if j.ok("update"):
                    j.step("update_ok", lambda: j.require("update", update.returncode == 0,
                                                           f"hermes update exited {update.returncode}", update))
                if j.ok("update_ok"):
                    with machine.gateway_phase():
                        j.results["probe_present"] = (machine.install_dir / _PROBE).is_file()
                        j.step("spawn", machine.spawn_gateway)
                        state = j.step("state", machine.wait_gateway_running)
                        if j.ok("state"):
                            time.sleep(15)  # supervised workers start ~2 s after boot; give them room to die
                            j.step("gateway_proc", lambda: _inspect(int(state["pid"])))
                            j.results["loaded_by_gateway"] = _loaded_by(markers)
                        machine.kill_owned()  # nothing of this machine outlives its gateway phase
                    j.step("turn", lambda: one_shot_turn(machine, srv, "turn-stale-venv"))
            yield j
        finally:
            machine.teardown()


def _inspect(pid: int) -> dict:
    proc = psutil.Process(pid)
    env = {k.upper(): v for k, v in proc.environ().items()}
    return {"pid": pid, "exe": proc.exe(), "cmdline": proc.cmdline(),
            "VIRTUAL_ENV": env.get("VIRTUAL_ENV", ""), "PYTHONPATH": env.get("PYTHONPATH", "")}


def test_gateway_does_not_load_stale_in_tree_venv(journey: Journey) -> None:
    m, info = journey.machine, journey["gateway_proc"]
    stale = str(m.install_dir / "venv")
    published = [k for k in ("VIRTUAL_ENV", "PYTHONPATH")
                 if os.path.normcase(stale) in os.path.normcase(info[k])]
    assert journey.results["probe_present"], fail_with(
        m, f"harness: the stale-venv probe {_PROBE} was gone before the gateway started")
    loaded = sorted(journey.results["loaded_by_gateway"])
    log = m.hermes_home / "logs" / "gateway.log"
    deaths = _WORKER_DEATH.findall(log.read_text(encoding="utf-8", errors="replace")) if log.is_file() else []
    assert not loaded and not published, fail_with(
        m, f"the gateway loaded the stale pre-PM in-tree venv {stale} (site dir added by pids "
           f"{loaded}, gateway pid {info['pid']}; published via {published or 'nothing'}; worker deaths: {deaths[:3]})")
    assert not deaths, fail_with(m, f"gateway supervised workers died: {deaths[:5]}")


def test_cli_turn_ignores_stale_venv_and_path_python(journey: Journey) -> None:
    m, turn, markers = journey.machine, journey["turn"], journey.results["markers"]
    others = sorted(_loaded_by(markers) - set(journey.results.get("loaded_by_gateway", ())))
    assert turn.ok, fail_with(
        m, f"a turn with a stale in-tree venv and a system Python on PATH failed (reply printed="
           f"{turn.reply_id in turn.run.stdout}, prompt reached provider={turn.reached_wire})", turn.run)
    assert not others, fail_with(m, f"CLI processes loaded the stale pre-PM in-tree venv: pids {others}", turn.run)

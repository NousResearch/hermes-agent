"""Real s6/UID-10000 dependency-generation handoff probe for the Docker harness."""

import asyncio
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
from tests.waiting import wait_until

ROOT = Path("/tmp/generation-proof")
HOME = ROOT / "data"
REPO = ROOT / "repo"
os.environ["HERMES_HOME"] = str(HOME)
os.environ["HERMES_S6_SUPERVISED_CHILD"] = "1"
from pm.environments import (
    activate_dependencies,
    install_state_dir,
    runtime_facts_path,
    site_packages,
)
from pm.generation_lifecycle import GenerationObserver


def event(kind, **data):
    with (ROOT / "events").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(kind=kind, pid=os.getpid(), **data)) + "\n")


def wait_for(predicate):
    return wait_until(predicate, timeout=90, what="supervised generation handoff")


def actor(role):
    from gateway.dependency_generation_exit import dependency_generation_watcher
    from hermes_cli.backend_retirement import RetirementFence
    from hermes_cli import web_server_idle_proof
    from hermes_cli.web_server_generation_exit import start_generation_watchdog

    activate_dependencies(REPO)
    proof_dependency = importlib.import_module("proof_dependency")

    observer = GenerationObserver(REPO)
    assert observer.boot is not None
    event(
        "boot",
        role=role,
        generation=observer.boot.parent.name,
        value=proof_dependency.VALUE,
    )
    busy = lambda: (ROOT / "busy").exists()
    if role == "gateway":
        runner = SimpleNamespace(
            _running=True,
            _draining=False,
            _restart_task_started=False,
            _active_work_count=lambda: int(busy()),
            adapters={},
            _profile_adapters={},
            _background_tasks=set(),
            session_runtime_descriptor={},
        )
        runner._mark_api_runs_shutdown_requested = lambda: (
            ROOT / "gateway-draining"
        ).touch()
        runner.request_restart = lambda **kw: os._exit(75)

        async def run():
            await dependency_generation_watcher(runner, observer=observer, poll_s=0.05)

        asyncio.run(run())
    else:

        def idle_proof(*args, **kwargs) -> dict:
            if busy():
                (ROOT / "dashboard-waiting").touch()
            return {"idle": not busy()}

        server = SimpleNamespace(should_exit=False)
        with patch.object(web_server_idle_proof, "idle_proof", side_effect=idle_proof):
            thread = start_generation_watchdog(
                server,
                observer=observer,
                fence=RetirementFence(),
                owned=True,
                poll_s=0.05,
            )
            thread.join()
        os._exit(75)


def controller():
    ROOT.mkdir()
    HOME.mkdir()
    REPO.mkdir()
    (ROOT / "busy").touch()
    (ROOT / "events").touch()
    envs = {}
    for name in ("old", "new"):
        env = install_state_dir(REPO) / "environments" / name / "venv"
        env.mkdir(parents=True)
        (env / "pyvenv.cfg").write_text("version = 3.13", encoding="utf-8")
        packages = site_packages(env)
        packages.mkdir(parents=True)
        (packages / "proof_dependency.py").write_text(
            "VALUE = " + repr(name), encoding="utf-8"
        )
        marker = env.parent / ".lease-managed"
        marker.touch()
        stamp = time.time() - (172800 if name == "old" else 0)
        os.utime(marker, (stamp, stamp))
        envs[name] = env

    def publish(name):
        from hermes_cli.runtime_state import runtime_lock

        with runtime_lock(REPO):
            path = runtime_facts_path(REPO)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps({"packages": {"venv": {"environment": str(envs[name])}}}),
                encoding="utf-8",
            )
            temporary.replace(path)

    publish("old")
    for path in [ROOT, *ROOT.rglob("*")]:
        os.chown(path, 10000, 10000)
    scan = ROOT / "services"
    scan.mkdir()
    for role in ("gateway", "dashboard"):
        directory = scan / role
        directory.mkdir()
        run = directory / "run"
        run.write_text(
            "#!/bin/sh\nexec /command/s6-setuidgid hermes /opt/hermes/.venv/bin/python "
            "/work/tests/docker/generation_lifecycle_probe.py " + role + "\n",
            encoding="utf-8",
        )
        run.chmod(0o755)
    supervisor = subprocess.Popen(["/command/s6-svscan", str(scan)])

    def boots():
        return [
            json.loads(line)
            for line in (ROOT / "events").read_text(encoding="utf-8").splitlines()
            if line
        ]

    try:
        first = wait_for(lambda: len(boots()) == 2 and boots())
        assert all(row["generation"] == row["value"] == "old" for row in first), first
        publish("new")
        wait_for(
            lambda: (
                (ROOT / "gateway-draining").exists()
                and (ROOT / "dashboard-waiting").exists()
            )
        )
        assert len(boots()) == 2, boots()
        from hermes_cli.runtime_state import collect_generations

        collect_generations(REPO)
        assert envs["old"].exists(), "active code was removed"
        (ROOT / "busy").unlink()
        rows = wait_for(lambda: len(boots()) >= 4 and boots())
        assert all(row["generation"] == row["value"] == "new" for row in rows[2:]), rows
        wait_for(lambda: not envs["old"].exists())
        assert envs["new"].exists()
        print(
            json.dumps(
                {
                    "result": "PASS",
                    "uid": 10000,
                    "real_s6_restarts": 2,
                    "busy_work_preserved": True,
                    "old_collected_automatically": True,
                    "boots": rows,
                }
            ),
            flush=True,
        )
    finally:
        subprocess.run(["/command/s6-svscanctl", "-t", str(scan)], check=False)
        supervisor.wait(timeout=15)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        actor(sys.argv[1])
    else:
        controller()

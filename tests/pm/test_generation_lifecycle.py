"""Actual bootstrap/lease lifetime, committed selections and rebuild headroom."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_cli.runtime_state import collect_generations
from pm.environments import remember_activation
from pm.generation_lifecycle import GenerationObserver
from pm.plugin_inputs import Members
from tests.pm.generation_fixture import make_install


def test_real_boot_identity_survives_publication_until_reader_exits(
    tmp_path, monkeypatch
):
    install = make_install(tmp_path, monkeypatch)
    script = """
import json, sys
from pathlib import Path
from pm.environments import activate_dependencies
activate_dependencies(Path(sys.argv[1]))
from pm.generation_lifecycle import GenerationObserver
observer = GenerationObserver(Path(sys.argv[1]))
print(json.dumps({"known": observer.known, "boot": str(observer.boot)}), flush=True)
sys.stdin.readline()
print(observer.probe(), flush=True)
"""

    def start():
        return subprocess.Popen(
            [sys.executable, "-c", script, str(install.repo)],
            env=dict(os.environ),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    children = [start(), start()]
    try:
        for child in children:
            boot = json.loads(child.stdout.readline())
            assert boot == {"known": True, "boot": str(install.envs["old"].resolve())}
        install.publish("new")
        collect_generations(install.repo)
        assert install.envs["old"].is_dir()
        for child in children:
            output, error = child.communicate("selected\n", timeout=15)
            assert child.returncode == 0, error
            assert "new superseded" in output
            if child is children[0]:
                collect_generations(install.repo)
                assert install.envs["old"].is_dir()  # the second service still owns it
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=15)
    collect_generations(install.repo)
    assert not install.envs["old"].exists()
    assert install.envs["new"].is_dir()
    replacement = start()
    try:
        assert json.loads(replacement.stdout.readline())["boot"] == str(
            install.envs["new"].resolve()
        )
        output, error = replacement.communicate("current\n", timeout=15)
        assert replacement.returncode == 0, error
        assert not output.strip()
    finally:
        if replacement.poll() is None:
            replacement.kill()
            replacement.wait(timeout=15)


def test_base_image_boot_detects_the_first_published_dependency_generation(
    tmp_path, monkeypatch
):
    install = make_install(tmp_path, monkeypatch)
    (install.state / "facts.json").unlink()
    script = """
import json, sys
from pathlib import Path
from pm.environments import activate_dependencies, runtime_facts_path
root = Path(sys.argv[1])
activate_dependencies(root)
from pm.generation_lifecycle import GenerationObserver
observer = GenerationObserver(root)
runtime_facts_path(root).write_text(
    json.dumps({"packages": {"venv": {"environment": sys.argv[2]}}}),
    encoding="utf-8",
)
print(observer.probe())
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(install.repo), str(install.envs["new"])],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    assert "new superseded" in child.stdout


def test_observer_pins_home_and_does_not_follow_adopted_sys_path(tmp_path, monkeypatch):
    install = make_install(tmp_path, monkeypatch)
    install.publish("new")
    remember_activation(install.repo, install.envs["new"])
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "foreign"))
    assert "new superseded" in install.observer.probe()
    assert GenerationObserver(install.repo).probe() is None


def test_maintenance_does_not_queue_behind_an_installer(tmp_path, monkeypatch):
    from hermes_cli.runtime_state import runtime_lock
    from pm.generation_lifecycle import collect_unused

    install = make_install(tmp_path, monkeypatch)
    with runtime_lock(install.repo) as held:
        assert held
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(collect_unused, install.repo).result(timeout=5)
    assert install.envs["old"].is_dir()


def test_container_refresh_collects_before_its_current_or_rebuild_decision(
    tmp_path, monkeypatch
):
    from pm import recovery

    install = make_install(tmp_path, monkeypatch)
    install.publish("new")
    monkeypatch.setattr("pm.paths.repo_root", lambda: install.repo)
    monkeypatch.setattr(recovery, "_baseline_gaps", lambda root: [])

    def current(**kwargs):
        assert not install.envs["old"].exists()
        assert install.envs["new"].is_dir()
        return True

    monkeypatch.setattr("pm.install.venv_is_current", current)
    assert recovery.refresh_dependencies(install.repo) == "current"


@pytest.mark.parametrize("invalid", ["journal", "malformed", "missing"])
def test_unfinished_or_unreadable_publication_keeps_services_alive(
    tmp_path, monkeypatch, invalid
):
    install = make_install(tmp_path, monkeypatch)
    install.publish("new")
    if invalid == "journal":
        (install.state / "publication.json").write_text("{}", encoding="utf-8")
    elif invalid == "malformed":
        (install.state / "facts.json").write_text("{", encoding="utf-8")
    else:
        (install.envs["new"] / "pyvenv.cfg").unlink()
    assert install.observer.probe() is None


@pytest.mark.parametrize("operation", ["lazy", "plugin", "repair", "update"])
def test_rebuild_collects_unleased_old_generations_before_build_lock(
    tmp_path, monkeypatch, operation
):
    from pm import install as engine

    fixture = make_install(tmp_path, monkeypatch)
    fixture.publish("new")
    monkeypatch.setattr(engine.paths, "repo_root", lambda: fixture.repo)
    monkeypatch.setattr(engine, "_feature_policy", lambda *a, **kw: ([], False))
    monkeypatch.setattr(engine, "get_package", lambda name: None)

    class ReachedBuild(Exception):
        pass

    def build_lock(**kwargs):
        assert not fixture.envs["old"].exists()
        assert fixture.envs["new"].is_dir()
        raise ReachedBuild

    monkeypatch.setattr(engine, "_venv_install_lock", build_lock)
    with pytest.raises(ReachedBuild):
        engine.sync_venv(
            extras=["fal"] if operation == "lazy" else None,
            plugins=Members([]) if operation == "plugin" else None,
            repair=operation == "repair",
            explicit=operation == "update",
        )

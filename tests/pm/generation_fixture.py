"""Small real dependency selections for lifecycle tests; no package installation."""

import json
import os
import time
from types import SimpleNamespace

from pm.environments import (
    install_state_dir,
    remember_activation,
    runtime_facts_path,
    site_packages,
)
from pm.generation_lifecycle import GenerationObserver


def make_install(tmp_path, monkeypatch):
    home, repo = tmp_path / "home", tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    state = install_state_dir(repo)
    environments = {}
    for name in ("old", "new", "unused"):
        env = state / "environments" / name / "venv"
        env.mkdir(parents=True)
        (env / "pyvenv.cfg").write_text("version = 3.14", encoding="utf-8")
        site_packages(env).mkdir(parents=True)
        marker = env.parent / ".lease-managed"
        marker.touch()
        stamp = time.time() - 172800
        os.utime(marker, (stamp, stamp))
        environments[name] = env

    def publish(name):
        runtime_facts_path(repo).write_text(
            json.dumps(
                {
                    "packages": {
                        "venv": {
                            "environment": str(environments[name]),
                            "extras": [],
                            "stamp": name,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

    publish("old")
    remember_activation(repo, environments["old"])
    return SimpleNamespace(
        home=home,
        repo=repo,
        state=state,
        envs=environments,
        publish=publish,
        observer=GenerationObserver(repo),
    )

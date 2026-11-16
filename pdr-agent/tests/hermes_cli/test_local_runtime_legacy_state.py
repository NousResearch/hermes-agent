"""Legacy local-runtime records must not adopt a reused PID."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(("case", "accepted"), [
    ("managed", True),
    ("foreign-executable", False),
    ("pid-reused", False),
])
def test_legacy_endpoint_requires_managed_process_identity(tmp_path, monkeypatch, case, accepted):
    from hermes_cli.local_runtime import bootstrap, endpoint, recovery, supervisor

    root = tmp_path / "runtimes" / "llamacpp"
    root.mkdir(parents=True)
    monkeypatch.setattr(supervisor, "runtimes_root", lambda: root)
    models = tmp_path / "models"
    monkeypatch.setattr(bootstrap, "models_dir", lambda: models)

    state = {
        "pid": 647,
        "base_url": "http://127.0.0.1:18434/v1",
        "api_key": "legacy-test-key",
    }
    path = supervisor.state_path()
    path.write_text(json.dumps(state), encoding="utf-8")
    os.utime(path, (200, 200))

    managed_exe = root / "b10964" / "metal" / "llama-server"
    executable = tmp_path / "usr" / "sbin" / "distnoted" if case == "foreign-executable" else managed_exe
    created = 201 if case == "pid-reused" else 100
    argv = [
        str(managed_exe),
        "--host", "127.0.0.1",
        "--port", "18434",
        "--api-key", "legacy-test-key",
        "--models-dir", str(models),
    ]
    process = SimpleNamespace(
        exe=lambda: str(executable),
        create_time=lambda: created,
        cmdline=lambda: argv,
    )
    monkeypatch.setattr(recovery.psutil, "Process", lambda pid: process)
    monkeypatch.setattr(recovery.psutil, "pid_exists", lambda pid: True)

    expected = {"base_url": state["base_url"], "api_key": state["api_key"]} if accepted else None
    assert recovery.legacy_recorded_process(state) is (process if accepted else None)
    assert endpoint._state_endpoint() == expected

"""A managed llama-server child never inherits credential-shaped environment variables.

Issue #116109: on Windows the bundled libomp.dll crashed with STATUS_HEAP_CORRUPTION during
initialisation whenever one `*_API_KEY` variable was present in the inherited Desktop environment
and loaded fine with only that variable removed. Credentials have no business in a native
inference child regardless of the crash, so the supervisor scrubs them at its spawn. The scrub
is the supervisor's, not spawn_server's: the same spawner backs bounded_probe_run (git,
PowerShell, update probes) whose children legitimately need GH_TOKEN / HF_TOKEN.
"""

import os
import subprocess
from types import SimpleNamespace

from hermes_cli.local_runtime import supervisor
from hermes_cli.local_runtime.processes import server_child_env, spawn_server


def test_supervisor_spawns_llama_server_without_credentials_but_keeps_runtime_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-secret")
    monkeypatch.setenv("SOME_TOKEN", "t")
    monkeypatch.setenv("DB_PASSWORD", "p")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setattr(supervisor, "runtimes_root", lambda: tmp_path / "runtime")
    monkeypatch.setattr(supervisor, "server_binary", lambda _: tmp_path / "engine" / "llama-server")
    monkeypatch.setattr(supervisor, "_direct_io_args", lambda exe: [])
    spawns = []
    monkeypatch.setattr(supervisor, "spawn_server", lambda argv, **kwargs: (
        spawns.append(kwargs) or SimpleNamespace(pid=123, poll=lambda: 0), None,
    ))
    sup = supervisor.LlamaServerSupervisor(tmp_path / "engine", tmp_path / "models", port=19002)
    monkeypatch.setattr(sup, "_write_state", lambda: None)
    try:
        sup._spawn()
    finally:
        sup.stop()
    child_env = spawns[0]["env"]
    assert "FIRECRAWL_API_KEY" not in child_env
    assert "SOME_TOKEN" not in child_env
    assert "DB_PASSWORD" not in child_env
    assert child_env["CUDA_VISIBLE_DEVICES"] == "0"
    assert child_env["OMP_NUM_THREADS"] == "4"
    assert child_env["PATH"] == os.environ["PATH"]


def test_spawn_server_keeps_the_callers_environment(monkeypatch):
    """The generic spawner (bounded_probe_run's git/PowerShell probes) must pass an explicit
    ``env=`` through untouched: a gh-backed credential helper needs GITHUB_TOKEN."""
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: SimpleNamespace(kwargs=kwargs))
    proc, _job = spawn_server(["git", "fetch"], env={"GITHUB_TOKEN": "g", "PATH": "/bin"})
    assert proc.kwargs["env"] == {"GITHUB_TOKEN": "g", "PATH": "/bin"}


def test_server_child_env_is_case_insensitive_and_pure():
    base = {"openai_api_key": "k", "Path": "C:\\x", "GITHUB_TOKEN": "g", "HSA_OVERRIDE_GFX_VERSION": "11"}
    assert server_child_env(base) == {"Path": "C:\\x", "HSA_OVERRIDE_GFX_VERSION": "11"}
    assert base["openai_api_key"] == "k"  # input untouched

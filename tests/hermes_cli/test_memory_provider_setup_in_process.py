"""The dashboard's memory-provider dependency verdict reads committed facts in-process.

``pm.client.venv_is_current`` answers the same boolean by spawning a ``pm/worker.py``
subprocess and blocking on its stdin pipe — measured at ~8 s per call on Windows with
nothing cached — because ``pm.runtime.is_runtime()`` is false inside the dashboard
process. The facts file it reads is local state, so a display verdict pays nothing for
the worker.
"""


def _inputs() -> dict:
    # Any non-empty declaration makes _memory_provider_setup_info consult the checker;
    # the manifest itself is irrelevant to which checker runs.
    return {"extras": [], "plugins": None}


def test_setup_info_reads_dependency_facts_in_process(monkeypatch):
    from hermes_cli import web_server_memory as mem

    monkeypatch.setattr(
        mem,
        "_memory_provider_setup_manifest",
        lambda name: (
            {"pip_dependencies": ["httpx>=0.27,<1"], "python_dependencies_declared": True,
             "external_dependencies": [], "required_env": []},
            _inputs(),
        ),
    )

    reads: list[dict] = []

    def direct(**kwargs):
        reads.append(kwargs)
        return True

    def worker(*args, **kwargs):
        raise AssertionError("pm worker spawned for a display verdict")

    monkeypatch.setattr("pm.install.venv_is_current", direct)
    monkeypatch.setattr("pm.venv_is_current", worker)

    info = mem._memory_provider_setup_info("openviking")

    assert reads, "dependency state was not read in-process"
    assert info["dependencies_installed"] is True


def test_in_process_read_failure_keeps_the_previous_verdict(monkeypatch):
    """A failed probe still reports 'not installed' instead of raising."""
    from hermes_cli import web_server_memory as mem

    monkeypatch.setattr(
        mem,
        "_memory_provider_setup_manifest",
        lambda name: (
            {"pip_dependencies": ["httpx>=0.27,<1"], "python_dependencies_declared": True,
             "external_dependencies": [], "required_env": []},
            _inputs(),
        ),
    )

    def broken(**kwargs):
        raise RuntimeError("unreadable facts")

    monkeypatch.setattr("pm.install.venv_is_current", broken)
    # The worker checker must not be consulted either — it would answer True here and
    # mask a failed in-process read.
    monkeypatch.setattr("pm.venv_is_current", lambda *a, **k: True)

    assert mem._memory_provider_setup_info("openviking")["dependencies_installed"] is False

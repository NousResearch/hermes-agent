"""Explicit interpreter overrides must not silently disappear at target dispatch."""

import pytest

from cron import scheduler_script


@pytest.mark.parametrize("script", ["/workspace/task.py", "/workspace/task.sh"])
def test_backend_explicit_interpreter_is_rejected_without_execution(monkeypatch, script):
    def unexpected(*args, **kwargs):
        pytest.fail("unsupported interpreter combination must not execute")

    monkeypatch.setattr(scheduler_script, "_run_job_script_in_backend", unexpected)
    success, message = scheduler_script._run_job_script_for_target(
        {"target": "backend", "interpreter": "/venv/bin/python"}, script
    )
    assert not success
    assert "interpreter" in message.lower()
    assert "scheduler" in message.lower()


def test_scheduler_explicit_interpreter_is_forwarded(monkeypatch):
    seen = {}

    def run(script, **kwargs):
        seen.update(script=script, **kwargs)
        return True, "ok"

    monkeypatch.setattr(scheduler_script, "_run_job_script", run)
    assert scheduler_script._run_job_script_for_target(
        {"target": "scheduler", "interpreter": "/venv/bin/python"}, "task.py"
    ) == (True, "ok")
    assert seen["interpreter"] == "/venv/bin/python"

"""Execution-target replay must retain upstream's explicit zero-tool allowlist."""
import json

import pytest


@pytest.mark.parametrize("target", ["scheduler", "backend"])
@pytest.mark.parametrize("boundary", ["store", "tool"])
def test_script_targets_preserve_zero_tool_allowlist(tmp_path, monkeypatch, target, boundary):
    from cron import jobs
    from tools import cronjob_tools, terminal_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "probe.py").write_text("print('ok')\n")
    script = "probe.py" if target == "scheduler" else "/backend/probe.py"
    monkeypatch.setattr(terminal_tool, "terminal_tool", lambda **kwargs: json.dumps({"exit_code": 0}))

    with jobs.use_cron_store(tmp_path / "cron"):
        fields = dict(prompt="probe", schedule="every 1h", script=script,
                      target=target, enabled_toolsets=[])
        if boundary == "store":
            created = jobs.create_job(**fields)
        else:
            result = json.loads(cronjob_tools.cronjob(action="create", **fields))
            assert result["success"], result
            created = jobs.get_job(result["job_id"])
        assert created["enabled_toolsets"] == []
        assert jobs.get_job(created["id"])["enabled_toolsets"] == []

        jobs.update_job(created["id"], {"enabled_toolsets": ["file"]})
        if boundary == "store":
            jobs.update_job(created["id"], {"enabled_toolsets": []})
        else:
            result = json.loads(cronjob_tools.cronjob(
                action="update", job_id=created["id"], enabled_toolsets=[]))
            assert result["success"], result
        stored = jobs.get_job(created["id"])
        assert stored["enabled_toolsets"] == []
        assert stored["target"] == target
        assert jobs.update_job(created["id"], {"enabled_toolsets": None})["enabled_toolsets"] is None

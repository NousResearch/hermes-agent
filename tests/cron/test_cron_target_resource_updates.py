"""Resource outages must not block lifecycle changes or tool restrictions."""
import json

import pytest

from cron import jobs
from tools import cronjob_tools, terminal_tool
from tools.registry import registry


@pytest.fixture
def resource_job(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    script = scripts / "probe.py"
    script.write_text("print('ok')\n")
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {"env_type": "ssh"})
    monkeypatch.setattr(terminal_tool, "terminal_tool", lambda **kw: json.dumps({"exit_code": 0}))
    with jobs.use_cron_store(tmp_path / "cron"):
        def create(target="scheduler"):
            return jobs.create_job(
                prompt="probe", schedule="every 1h", target=target,
                script="probe.py" if target == "scheduler" else "/backend/probe.py",
                workdir=str(workdir) if target == "scheduler" else "/backend/workdir",
                enabled_toolsets=["file"],
            )
        yield create, script, workdir


def _tool(action, job_id, **fields):
    return json.loads(registry.get_entry("cronjob_manage").handler(
        {"action": action, "job_id": job_id, **fields}))


@pytest.mark.parametrize("boundary", ["store", "tool"])
@pytest.mark.parametrize("outage", ["workdir", "script", "backend"])
@pytest.mark.parametrize("operation", ["pause", "rename", "restrict_tools"])
def test_non_execution_updates_survive_resource_outage(resource_job, monkeypatch, boundary, outage, operation):
    create, script, workdir = resource_job
    job = create("backend" if outage == "backend" else "scheduler")
    if outage == "workdir":
        workdir.rmdir()
    elif outage == "script":
        script.unlink()
    else:
        def unavailable(**kw):
            raise RuntimeError("terminal backend unavailable")
        monkeypatch.setattr(terminal_tool, "terminal_tool", unavailable)

    fields = {"name": "renamed"} if operation == "rename" else {"enabled_toolsets": []}
    if boundary == "tool":
        result = _tool("pause" if operation == "pause" else "update", job["id"],
                       **({"reason": "maintenance"} if operation == "pause" else fields))
        assert result["success"], result
    elif operation == "pause":
        assert jobs.pause_job(job["id"], reason="maintenance") is not None
    else:
        assert jobs.update_job(job["id"], fields) is not None

    stored = jobs.get_job(job["id"])
    if operation == "pause":
        assert stored["state"] == "paused"
        assert stored["enabled"] is False
        assert stored["paused_reason"] == "maintenance"
    else:
        for field, value in fields.items():
            assert stored[field] == value
    for field in ("script", "workdir", "target"):
        assert stored[field] == job[field]


def _reject_update(boundary, job, fields, error):
    if boundary == "tool":
        result = _tool("update", job["id"], **fields)
        assert result["success"] is False, result
        assert error in result["error"], result
    else:
        with pytest.raises(ValueError, match=error):
            jobs.update_job(job["id"], fields)
    assert jobs.get_job(job["id"]) == job


@pytest.mark.parametrize("boundary", ["store", "tool"])
@pytest.mark.parametrize("outage", ["workdir", "script"])
def test_resource_edit_revalidates_unchanged_merged_resource(resource_job, boundary, outage):
    create, script, workdir = resource_job
    job = create()
    if outage == "workdir":
        workdir.rmdir()
        fields, error = {"script": "probe.py"}, "Cron workdir does not exist"
    else:
        script.unlink()
        fields, error = {"workdir": str(workdir)}, "Script file not found"
    _reject_update(boundary, job, fields, error)


@pytest.mark.parametrize("boundary", ["store", "tool"])
@pytest.mark.parametrize("field", ["target", "workdir", "script"])
def test_backend_resource_edit_requires_available_backend(resource_job, monkeypatch, boundary, field):
    create, _, _ = resource_job
    job = create("backend")

    def unavailable(**kw):
        raise RuntimeError("terminal backend unavailable")
    monkeypatch.setattr(terminal_tool, "terminal_tool", unavailable)
    value = {"target": "backend", "workdir": "/backend/new", "script": "/backend/new.py"}[field]
    _reject_update(boundary, job, {field: value}, "Could not validate script in terminal backend")


@pytest.mark.parametrize("boundary", ["store", "tool"])
@pytest.mark.parametrize("case", ["backend_relative_script", "scheduler_absolute_script", "scheduler_missing_workdir"])
def test_target_switch_validates_merged_resources(resource_job, boundary, case):
    create, _, workdir = resource_job
    if case == "backend_relative_script":
        job = create()
        fields, error = {"target": "backend"}, "Backend script path must be absolute"
    elif case == "scheduler_absolute_script":
        job = create("backend")
        fields = {"target": "scheduler", "workdir": str(workdir)}
        error = "Scheduler script path must be relative"
    else:
        job = create("backend")
        fields = {"target": "scheduler", "script": "probe.py"}
        error = "Cron workdir does not exist"
    _reject_update(boundary, job, fields, error)


@pytest.mark.parametrize("boundary", ["store", "tool"])
@pytest.mark.parametrize("target", ["scheduler", "backend"])
def test_target_switch_accepts_matching_script_and_workdir(resource_job, monkeypatch, boundary, target):
    create, _, workdir = resource_job
    job = create("backend" if target == "scheduler" else "scheduler")
    probes = []

    def terminal_probe(**kw):
        probes.append(kw)
        if target == "backend" and kw["workdir"] != fields["workdir"]:
            return json.dumps({"exit_code": 1, "error": "old scheduler workdir is unavailable remotely"})
        return json.dumps({"exit_code": 0})
    monkeypatch.setattr(terminal_tool, "terminal_tool", terminal_probe)
    fields = {
        "target": target,
        "script": "probe.py" if target == "scheduler" else "/backend/new.py",
        "workdir": str(workdir) if target == "scheduler" else "/backend/new",
    }
    if boundary == "tool":
        result = _tool("update", job["id"], **fields)
        assert result["success"], result
    else:
        jobs.update_job(job["id"], fields)
    stored = jobs.get_job(job["id"])
    for field, value in fields.items():
        assert stored[field] == value
    if target == "backend":
        assert probes[-1]["workdir"] == fields["workdir"]
        assert fields["script"] in probes[-1]["command"]
    else:
        assert probes == []


@pytest.mark.parametrize("workdir", ["/backend/repaired", ""])
def test_backend_script_replacement_uses_new_or_cleared_workdir(resource_job, monkeypatch, workdir):
    create, _, _ = resource_job
    job = create("backend")
    expected = workdir or None
    probes = []

    def terminal_probe(**kw):
        probes.append(kw)
        return json.dumps({"exit_code": 0 if kw["workdir"] == (expected or "/backend") else 1})

    monkeypatch.setattr(terminal_tool, "terminal_tool", terminal_probe)
    result = _tool("update", job["id"], script="/backend/replaced.py", workdir=workdir)
    assert result["success"], result
    assert probes and all(probe["workdir"] == (expected or "/backend") for probe in probes)
    assert jobs.get_job(job["id"])["workdir"] == expected


@pytest.mark.parametrize("boundary", ["store", "tool"])
@pytest.mark.parametrize("case", ["monitor_backend", "monitor_no_agent", "both_monitors", "no_agent_without_script"])
def test_execution_mode_guards_survive_resource_validation_gate(resource_job, boundary, case):
    create, _, _ = resource_job
    job = create("backend" if case == "monitor_backend" else "scheduler")
    fields, error = {
        "monitor_backend": ({"monitor_script": "probe.py"}, "monitor_script runs on the scheduler host"),
        "monitor_no_agent": ({"monitor_url": "https://example.com", "no_agent": True}, "cannot be combined with no_agent"),
        "both_monitors": ({"monitor_script": "probe.py", "monitor_url": "https://example.com"}, "mutually exclusive"),
        "no_agent_without_script": ({"script": "", "no_agent": True}, "requires a script" if boundary == "store" else "without a script"),
    }[case]
    _reject_update(boundary, job, fields, error)

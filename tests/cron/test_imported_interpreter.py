"""Distribution jobs retain their authored Python environment across updates."""

import json
import sys
import venv
from pathlib import Path

from cron import jobs
from cron.scheduler import run_job
from hermes_cli.profile_distribution import _merge_cron_store
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_platform.host.facts import os_family


def test_distribution_interpreter_survives_import_update_and_removal(tmp_path):
    home = tmp_path / "home"
    source = tmp_path / "distribution" / "cron" / "jobs.json"
    source.parent.mkdir(parents=True)
    (home / "scripts").mkdir(parents=True)
    (home / "scripts" / "environment.py").write_text("import sys\nprint(sys.prefix)\n", encoding="utf-8")
    external = tmp_path / "external"
    venv.EnvBuilder(with_pip=False).create(external)
    interpreter = external / ("Scripts/python.exe" if os_family() == "win32" else "bin/python")
    token = set_hermes_home_override(home)
    try:
        authored = {
            "id": "ab12cd34ef56", "prompt": "", "schedule": "every 1h",
            "script": "environment.py", "no_agent": True, "deliver": "local",
            "interpreter": str(interpreter),
        }
        source.write_text(json.dumps({"jobs": [authored]}), encoding="utf-8")
        _merge_cron_store(source, home)
        assert jobs.get_job(authored["id"])["interpreter"] == str(interpreter)
        jobs.resume_job(authored["id"])
        success, _doc, output, error = run_job(jobs.get_job(authored["id"]))
        assert success, error
        assert Path(output).resolve() == external.resolve()

        completed = jobs.get_job(authored["id"])["repeat"]["completed"]
        authored["interpreter"] = sys.executable
        source.write_text(json.dumps({"jobs": [authored]}), encoding="utf-8")
        _merge_cron_store(source, home)
        updated = jobs.get_job(authored["id"])
        assert updated["interpreter"] == sys.executable
        assert updated["repeat"]["completed"] == completed
        assert updated["state"] == "scheduled"

        del authored["interpreter"]
        source.write_text(json.dumps({"jobs": [authored]}), encoding="utf-8")
        _merge_cron_store(source, home)
        assert "interpreter" not in jobs.get_job(authored["id"])
    finally:
        reset_hermes_home_override(token)

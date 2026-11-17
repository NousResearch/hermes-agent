"""A cron job's ``output_schema``: each fire's answer is validated, one correction turn runs on a
miss, and the validated object is the delivered output and the execution ledger row's result."""
import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home

_SCHEMA = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}

_TICK = '''
import json
from cron import scheduler, scheduler_authority
from cron.executions import list_executions
from tests.cron.conftest import owner_execution_runner
scheduler_authority.run_canonical_job = owner_execution_runner()
scheduler.tick(verbose=False, sync=True)
print(json.dumps(list_executions(job_id="probe"), default=str))
'''


def _fire(tmp_path, answers, **job_fields):
    from hermes_time import now

    home = tmp_path / "home"
    with FakeLLMServer([Text(a) for a in answers]) as srv:
        write_hermes_home(home, srv.base_url)
        (home / "cron").mkdir()
        (home / "cron" / "jobs.json").write_text(json.dumps({"jobs": [{
            "id": "probe", "name": "schema probe", "prompt": "count the files",
            "schedule": {"kind": "interval", "minutes": 240},
            "next_run_at": (now() - timedelta(minutes=1)).isoformat(),
            "enabled": True, "state": "scheduled", "deliver": "local",
            "repeat": {"times": None, "completed": 0}, **job_fields}]}), encoding="utf-8")
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("HERMES_", "_HERMES_")) and not k.endswith(("_API_KEY", "_TOKEN"))}
        env.update(HERMES_HOME=str(home), PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        result = subprocess.run([sys.executable, "-c", _TICK], env=env, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=180, check=False)
        prompts = [r["messages"][-1]["content"] for r in srv.main_requests()]
    assert result.returncode == 0, result.stdout + result.stderr
    (execution,) = json.loads(result.stdout.strip().splitlines()[-1])
    outputs = sorted((home / "cron" / "output" / "probe").glob("*.md"))
    return execution, prompts, outputs[-1].read_text(encoding="utf-8-sig") if outputs else ""


def test_schema_job_retries_once_and_records_the_validated_result(tmp_path):
    execution, prompts, doc = _fire(
        tmp_path, ["There are three files.", '```json\n{"count": 3}\n```'], output_schema=_SCHEMA)

    assert len(prompts) == 2, "a miss gets exactly one correction turn"
    assert "OUTPUT CONTRACT" in str(prompts[0]) and "rejected by the output contract" in str(prompts[1])
    assert execution["status"] == "completed"
    assert json.loads(execution["result_json"]) == {"count": 3}
    assert '"count": 3' in doc and "```" not in doc.split("## Response")[-1]


def test_schema_job_that_never_satisfies_the_schema_fails_the_run(tmp_path):
    execution, prompts, _doc = _fire(tmp_path, ["three", "still three"], output_schema=_SCHEMA)

    assert len(prompts) == 2
    assert execution["status"] == "failed" and "output_schema not satisfied" in execution["error"]
    assert execution["result_json"] is None


def test_job_without_schema_keeps_free_text_and_one_turn(tmp_path):
    execution, prompts, doc = _fire(tmp_path, ["There are three files."])

    assert len(prompts) == 1 and "OUTPUT CONTRACT" not in str(prompts[0])
    assert execution["status"] == "completed" and execution["result_json"] is None
    assert "There are three files." in doc

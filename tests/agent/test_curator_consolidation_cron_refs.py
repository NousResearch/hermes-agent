"""A consolidating `hermes curator run` keeps cron jobs pointed at the umbrella skill.

The review fork deletes through the advertised ``skill_manage`` shape (an ``operations`` array);
the ``absorbed_into`` it declares there is what maps a cron job's skill reference onto the umbrella.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home

_RUN = '''
import sys
from tools.skill_usage import mark_agent_created
for name in ("umbrella", "narrow"):
    mark_agent_created(name)
from hermes_cli.main import main
sys.argv = ["hermes", "curator", "run", "--consolidate"]
main()
'''


def _write_skill(home: Path, name: str, body: str) -> None:
    d = home / "skills" / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {body}\n---\n\n{body}\n", encoding="utf-8")


def test_cron_job_follows_a_skill_absorbed_via_operations_delete(tmp_path):
    home = tmp_path / "home"
    delete = {"operations": [{"action": "delete", "name": "narrow", "absorbed_into": "umbrella"}]}
    # The final answer carries no structured YAML block: the delete's absorbed_into is the only
    # record of where the skill went.
    with FakeLLMServer([ToolCall("skill_manage", delete), Text("Merged narrow into umbrella.")]) as srv:
        write_hermes_home(home, srv.base_url)
        _write_skill(home, "umbrella", "Deploy services, including the narrow case.")
        _write_skill(home, "narrow", "Deploy one service.")
        (home / "cron").mkdir()
        (home / "cron" / "jobs.json").write_text(json.dumps({"jobs": [{
            "id": "nightly", "name": "nightly deploy", "prompt": "deploy",
            "schedule": {"kind": "interval", "minutes": 1440}, "enabled": True,
            "state": "scheduled", "deliver": "local", "skill": "narrow", "skills": ["narrow"],
            "repeat": {"times": None, "completed": 0}}]}), encoding="utf-8")
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("HERMES_", "_HERMES_")) and not k.endswith(("_API_KEY", "_TOKEN"))}
        env.update(HERMES_HOME=str(home), PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        result = subprocess.run([sys.executable, "-c", _RUN], env=env, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=180, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (home / "skills" / "narrow").exists(), result.stdout + result.stderr

    (job,) = json.loads((home / "cron" / "jobs.json").read_text(encoding="utf-8"))["jobs"]
    assert job["skills"] == ["umbrella"]

    (run_json,) = (home / "logs" / "curator").glob("*/run.json")
    run = json.loads(run_json.read_text(encoding="utf-8"))
    assert [(e["name"], e["into"]) for e in run["consolidated"]] == [("narrow", "umbrella")], run["pruned"]

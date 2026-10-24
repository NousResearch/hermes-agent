"""A tool hook fired inside a scheduled job sees ``platform == "cron"`` end to end.

The same field ``pre_llm_call`` already receives; without it a policy plugin cannot tell an
unattended cron run from an attended turn and has to fail closed on both.
"""
import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home

_PLUGIN = '''
import json, os
from pathlib import Path

def _record(**kwargs):
    out = Path(os.environ["HERMES_HOME"]) / "seen.json"
    out.write_text(json.dumps({"tool": kwargs.get("tool_name"),
                               "platform": kwargs.get("platform")}))

def register(ctx):
    ctx.register_hook("pre_tool_call", _record)
'''

_TICK = '''
import json
from cron import scheduler, scheduler_authority
from tests.cron.conftest import owner_execution_runner
scheduler_authority.run_canonical_job = owner_execution_runner()
scheduler.tick(verbose=False, sync=True)
print(json.dumps({"ok": True}))
'''


def test_tool_hook_in_a_ticked_job_sees_platform_cron(tmp_path):
    from hermes_time import now

    home = tmp_path / "home"
    plugin_dir = home / "plugins" / "cron-platform-probe"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: cron-platform-probe\nversion: 0.0.1\n")
    (plugin_dir / "__init__.py").write_text(_PLUGIN)
    note = home / "note.txt"
    with FakeLLMServer([ToolCall("read_file", {"path": str(note)}), Text("done")]) as srv:
        write_hermes_home(home, srv.base_url,
                          extra_config="plugins:\n  enabled: [cron-platform-probe]\n")
        note.write_text("hello\n")
        (home / "cron").mkdir()
        (home / "cron" / "jobs.json").write_text(json.dumps({"jobs": [{
            "id": "platform-probe", "name": "platform probe", "prompt": "read the note",
            "schedule": {"kind": "interval", "minutes": 240},
            "next_run_at": (now() - timedelta(minutes=1)).isoformat(),
            "enabled": True, "state": "scheduled", "deliver": "local",
            "repeat": {"times": None, "completed": 0}}]}))
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("HERMES_", "_HERMES_"))
               and not k.endswith(("_API_KEY", "_TOKEN"))}
        env.update(HERMES_HOME=str(home), PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        result = subprocess.run([sys.executable, "-c", _TICK], env=env, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=180, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    seen = json.loads((home / "seen.json").read_text())

    assert seen["tool"] == "read_file"
    assert seen["platform"] == "cron"

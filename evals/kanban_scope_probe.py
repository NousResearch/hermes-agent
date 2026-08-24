"""Credential-free app-server/MCP scope probe; writes only its temporary board."""
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile

repo = Path(sys.argv[1]).resolve()
if len(sys.argv) == 2:
    with tempfile.TemporaryDirectory(prefix="kanban-transport-") as home:
        env = {"HOME": home, "HERMES_HOME": home + "/hermes", "PATH": os.environ["PATH"], "PYTHONDONTWRITEBYTECODE": "1"}
        p = subprocess.run([sys.executable, __file__, str(repo), "isolated"], cwd=home, env=env, stdin=subprocess.DEVNULL)
        sys.exit(p.returncode)
sys.path.insert(0, str(repo))
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect
from agent.transports.codex_app_server import CodexAppServerClient

home = Path(os.environ["HOME"])
hh = Path(os.environ["HERMES_HOME"])
hh.mkdir(exist_ok=True)
(hh / "config.yaml").write_text("toolsets: [kanban]\n")
db = home / "assigned.db"
conn = connect(db)
own, foreign = [kb.create_task(conn, title=x) for x in ("owner", "foreign")]
for t in (own, foreign):
    kb.claim_task(conn, t)
task = kb.get_task(conn, own)
os.environ.update({"HERMES_KANBAN_DB": str(db), "HERMES_KANBAN_BOARD": "default", "HERMES_KANBAN_TASK": own, "HERMES_KANBAN_RUN_ID": str(task.current_run_id), "HERMES_KANBAN_CLAIM_LOCK": task.claim_lock})
ch = home / "codex"
ch.mkdir()
(ch / "config.toml").write_text(
    'model="fixture"\nmodel_provider="fixture"\n'
    '[model_providers.fixture]\nname="fixture"\nbase_url="http://127.0.0.1:9/v1"\nwire_api="responses"\n'
    '[mcp_servers.hermes-mcp]\ncommand=' + json.dumps(sys.executable) + '\nargs=["-m","agent.transports.hermes_tools_mcp_server"]\nstartup_timeout_sec=40\n'
    '[mcp_servers.hermes-mcp.env]\nPYTHONPATH=' + json.dumps(str(repo)) + '\nHERMES_HOME=' + json.dumps(str(hh)) + '\n'
)
report = {}
with CodexAppServerClient(codex_bin=shutil.which("codex") or "codex", codex_home=str(ch)) as c:
    try:
        report["initialize"] = c.initialize(capabilities={"experimentalApi": True})
    except Exception:
        print(json.dumps({"stderr": c._stderr_lines, "args": c._proc.args, "rc": c._proc.poll()}), flush=True)
        raise
    script = "import os,sys,json;sys.path.insert(0," + repr(str(repo)) + ");from tools import kanban_tools as kt;print(json.dumps({'marker':os.getenv('HERMES_DELEGATED_CHILD_CONTEXT'),'db':os.getenv('HERMES_KANBAN_DB'),'mutation':json.loads(kt._handle_complete({'task_id':" + repr(own) + ",'summary':'must refuse'}))}))"
    report["native_child"] = c.request("command/exec", {"command": [sys.executable, "-c", script], "cwd": str(home), "sandboxPolicy": {"type": "dangerFullAccess"}}, timeout=40)
    thread = c.request("thread/start", {"model": "fixture", "modelProvider": "fixture", "cwd": str(home), "approvalPolicy": "never", "sandbox": "workspace-write"}, timeout=50)
    thread_id = thread["thread"]["id"]
    status = c.request("mcpServerStatus/list", {"threadId": thread_id}, timeout=50)
    report["mcp_servers"] = [{"name": x.get("name"), "tools": list(x.get("tools", {}))} for x in status.get("data", [])]
    for name, tid in (("foreign", foreign), ("own", own)):
        report[name] = c.request("mcpServer/tool/call", {"threadId": thread_id, "server": "hermes-mcp", "tool": "kanban_complete", "arguments": {"task_id": tid, "summary": "supervised parent handoff"}}, timeout=50)
    report["stderr"] = c._stderr_lines[-8:]
report["readback"] = {"own": kb.get_task(conn, own).status, "foreign": kb.get_task(conn, foreign).status, "integrity": conn.execute("PRAGMA integrity_check").fetchone()[0]}
print(json.dumps(report, indent=2))
assert report["readback"] == {"own": "done", "foreign": "running", "integrity": "ok"}, report
native = json.loads(report["native_child"]["stdout"])
assert native["marker"] == "1" and native["db"] == str(db) and "error" in native["mutation"], report

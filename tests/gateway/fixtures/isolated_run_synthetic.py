"""Disposable synthetic native run used by the HTTP Runs execution tests."""

import json
import os
import signal
import subprocess
import sqlite3
import sys
import time


control_fd = int(os.environ.pop("HERMES_RUN_CONTROL_FD"))
os.set_inheritable(control_fd, False)


def emit(kind, **fields):
    os.write(control_fd, ("\x1ehermes-run:" + json.dumps({"kind": kind, **fields}) + "\n").encode())


launch = json.loads(sys.stdin.readline())
if launch["user_message"].startswith("term-ignore-detached:"):
    try:
        from gateway.platforms.api_server_run_child import become_child_subreaper
        become_child_subreaper()
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        with open(launch["user_message"].split(":", 1)[1], "w", encoding="ascii") as output:
            output.write(str(child.pid))
    except Exception as exc:
        emit("error", error=type(exc).__name__ + ":" + str(exc))
        raise


def linger_on_term(_signal, _frame):
    with sqlite3.connect(launch["run_store_path"]) as conn:
        conn.execute(
            "UPDATE run_idempotency SET status_json=json_set(status_json,'$.detached_cleanup_verified',1) "
            "WHERE run_id=? AND json_extract(status_json,'$.execution_pid')=?",
            (launch["run_id"], os.getpid()))
    emit("cleanup", verified=True)
    raise SystemExit(143)


def fork_on_term(_signal, _frame):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                             start_new_session=True)
    with open(launch["user_message"].split(":", 1)[1], "w", encoding="ascii") as output:
        output.write(str(child.pid))
    raise SystemExit(143)


signal.signal(signal.SIGTERM, signal.SIG_IGN if launch["user_message"].startswith("term-ignore-detached:")
              else fork_on_term if launch["user_message"].startswith("fork-on-term:") else linger_on_term)
emit("activity", phase="tool_running")
emit("delta", delta="synthetic provider token")
if launch["user_message"] == "approval-secret":
    emit("approval", event={"command": "echo sk-test-secret-1234567890", "description": "sk-test-secret-1234567890"})
if launch["user_message"] == "terminal-linger":
    emit("result", result={"completed": True, "final_response": "synthetic receipt"})
while True:
    time.sleep(0.1)

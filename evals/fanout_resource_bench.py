#!/usr/bin/env python3
"""Fan-out resource benchmark for hermes-agent.

Spawns N in-process child AIAgents via the REAL delegate_task code path
(tools.delegate_tool.delegate_task) against a local fake OpenAI server, with
children editing python files across W distinct git worktrees so the LSP
(pyright) path is exercised for real. Measures, for the host process:

  threads, RSS MB, open fds, TCP ESTAB sockets, child processes (pyright,
  kernels), state.db growth, wall time.

Usage:
  python evals/fanout_resource_bench.py --repo <checkout> --children 24 --worktrees 6 --label before

Prints one JSON line; append several and compare with --compare a.json b.json.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

# --------------------------------------------------------------------------
# Fake OpenAI chat-completions server: each child does
#   turn 1: call write_file on <its worktree>/hermes_cli/bench_<i>.py
#   turn 2: call execute_code print(1)
#   turn 3: final text
# --------------------------------------------------------------------------
_REPLY_KB = [0]


class _Fake(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # quiet
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        msgs = body.get("messages", [])
        goal = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        try:
            plan = json.loads(goal)
        except Exception:
            plan = {}
        n_tool = sum(1 for m in msgs if m.get("role") == "tool")
        if n_tool == 0 and plan.get("file"):
            tc = {"id": "c1", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": plan["file"], "content": "import os\nx: int = 'bad'\n"})}}
            msg = {"role": "assistant", "content": None, "tool_calls": [tc]}
            finish = "tool_calls"
        elif n_tool == 1 and plan.get("file"):
            tc = {"id": "c2", "type": "function", "function": {"name": "execute_code", "arguments": json.dumps({"code": "print(1)"})}}
            msg = {"role": "assistant", "content": None, "tool_calls": [tc]}
            finish = "tool_calls"
        else:
            msg = {"role": "assistant", "content": "done " + ("x" * (_REPLY_KB[0] * 1024))}
            finish = "stop"
        if body.get("stream") is True:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            delta = {"role": "assistant", "content": msg.get("content") or ""}
            if msg.get("tool_calls"):
                tc = msg["tool_calls"][0]
                delta["tool_calls"] = [{"index": 0, "id": tc["id"], "type": "function", "function": tc["function"]}]
            for chunk in (
                {"id": "m", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                {"id": "m", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
            ):
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        resp = {"id": "x", "object": "chat.completion", "created": 0, "model": body.get("model", "m"),
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _serve():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _count_live(cls_name: str) -> int:
    import gc
    return sum(1 for o in gc.get_objects() if type(o).__name__ == cls_name)


def _snap(pid: int, db_path: str) -> dict:
    st = open(f"/proc/{pid}/status", encoding="utf-8").read()
    g = lambda k: int(st.split(k + ":")[1].split()[0])
    tcp = subprocess.run(f"ss -tanp 2>/dev/null | grep -c 'pid={pid},'", shell=True, capture_output=True, text=True).stdout.strip()
    kids = subprocess.run(["ps", "-o", "args=", "--ppid", str(pid)], capture_output=True, text=True).stdout
    return {
        "threads": g("Threads"), "rss_mb": g("VmRSS") // 1024, "fds": len(os.listdir(f"/proc/{pid}/fd")),
        "tcp": int(tcp or 0), "pyright": kids.count("pyright"), "kernels": kids.count("hermes_kernel_runner"),
        "db_mb": round(os.path.getsize(db_path) / 2**20, 1) if os.path.exists(db_path) else 0,
        "httpx_clients": _count_live("Client"), "transports": _count_live("HTTPTransport"), "session_dbs": _count_live("SessionDB"), "live_agents": _count_live("AIAgent"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--children", type=int, default=24)
    ap.add_argument("--worktrees", type=int, default=6)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--reply-kb", type=int, default=0, help="pad each child's final reply to N KB (transcript-size realism)")
    a = ap.parse_args()
    if a.compare:
        b, c = (json.load(open(p, encoding="utf-8")) for p in a.compare)
        print(f"| metric | {b['label']} | {c['label']} | delta |\n|---|---|---|---|")
        for k in ("threads", "rss_mb", "fds", "tcp", "pyright", "kernels", "db_mb", "httpx_clients", "transports", "session_dbs"):
            bv, cv = b["peak"][k], c["peak"][k]
            print(f"| {k} (peak) | {bv} | {cv} | {cv - bv:+} |")
        for k in ("rss_mb", "live_agents", "db_mb", "threads"):
            bv, cv = b["after"].get(k), c["after"].get(k)
            if bv is not None and cv is not None:
                print(f"| {k} (after, children done) | {bv} | {cv} | {cv - bv:+} |")
        print(f"| wall_s | {b['wall_s']} | {c['wall_s']} | {c['wall_s'] - b['wall_s']:+.1f} |")
        return

    _REPLY_KB[0] = a.reply_kb
    home = tempfile.mkdtemp(prefix="hermes_bench_home_")
    os.environ["HERMES_HOME"] = home
    os.environ["TERMINAL_ENV"] = "local"
    os.environ.pop("OPENROUTER_API_KEY", None)
    sys.path.insert(0, a.repo)
    os.chdir(a.repo)
    pyright = shutil.which("pyright-langserver", path=os.path.expanduser("~/.hermes/lsp/bin") + os.pathsep + os.environ.get("PATH", ""))
    with open(os.path.join(home, "config.yaml"), "w", encoding="utf-8") as f:
        f.write("lsp:\n  enabled: true\n  wait_timeout: 5.0\n  install_strategy: manual\n")
        if pyright:
            f.write(f"  servers:\n    pyright:\n      command: [{json.dumps(pyright)}, \"--stdio\"]\n")
        f.write("delegation:\n  max_concurrent_children: 64\n  subagent_auto_approve: true\n")

    # W git worktrees, each a real python project (pyproject + package) so pyright roots resolve.
    wts = []
    base = tempfile.mkdtemp(prefix="hermes_bench_wt_")
    for w in range(a.worktrees):
        d = os.path.join(base, f"wt{w}")
        os.makedirs(os.path.join(d, "hermes_cli"))
        subprocess.run(["git", "init", "-q", d], check=True)
        open(os.path.join(d, "pyproject.toml"), "w", encoding="utf-8").write("[project]\nname='b'\n")
        open(os.path.join(d, "hermes_cli", "__init__.py"), "w", encoding="utf-8").write("")
        wts.append(d)

    srv = _serve()
    port = srv.server_address[1]
    from run_agent import AIAgent
    from tools import delegate_tool

    from hermes_state import SessionDB
    db_path = os.path.join(home, "state.db")
    from pathlib import Path
    session_db = SessionDB(db_path=Path(db_path))
    parent = AIAgent(api_key="bench", base_url=f"http://127.0.0.1:{port}/v1", model="bench-model",
                     quiet_mode=True, skip_context_files=True, skip_memory=True,
                     enabled_toolsets=["delegation", "file", "code_execution"],
                     session_db=session_db, session_id="bench-root")
    # Children reference parent_session_id; the parent row is normally created
    # lazily on the parent's first turn, which this harness never runs.
    parent._ensure_db_session()
    pid = os.getpid()
    before = _snap(pid, db_path)
    peak = dict(before)
    stop = threading.Event()

    def sampler():
        while not stop.wait(0.5):
            s = _snap(pid, db_path)
            for k, v in s.items():
                peak[k] = max(peak[k], v)
    threading.Thread(target=sampler, daemon=True).start()

    tasks = [{"goal": json.dumps({"file": os.path.join(wts[i % len(wts)], "hermes_cli", f"bench_{i}.py")}),
              "context": "bench"} for i in range(a.children)]
    t0 = time.monotonic()
    res = delegate_tool.delegate_task(tasks=tasks, parent_agent=parent, background=False)
    wall = round(time.monotonic() - t0, 1)
    if os.environ.get("BENCH_DEBUG"):
        sys.__stderr__.write(str(res)[:3000] + "\n")
    time.sleep(2.0)
    stop.set()
    import gc
    gc.collect()
    after = _snap(pid, db_path)
    try:
        parsed = json.loads(res)
        items = parsed if isinstance(parsed, list) else parsed.get("results") or parsed.get("tasks") or []
        ok = sum(1 for r in items if str(r.get("status", "")) in ("completed", "success"))
    except Exception:
        ok = None
    try:
        session_db.checkpoint() if hasattr(session_db, "checkpoint") else None
    except Exception:
        pass
    out = {"label": a.label, "children": a.children, "worktrees": a.worktrees, "ok": ok, "wall_s": wall,
           "before": before, "peak": peak, "after": after}
    sys.__stderr__.write("BENCH " + json.dumps(out) + "\n"); sys.__stderr__.flush()
    if a.out:
        open(a.out, "w", encoding="utf-8").write(json.dumps(out, indent=1))
    try:
        from agent.lsp import shutdown_service
        shutdown_service()
        from tools.code_kernel import shutdown_all_kernels
        shutdown_all_kernels()
    except Exception:
        pass
    shutil.rmtree(base, ignore_errors=True)
    os._exit(0)


if __name__ == "__main__":
    main()

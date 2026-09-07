"""Honcho 2.2 SDK/local HTTP lifecycle probe; no hosted service or model inference.

Run with isolated HOME/HERMES_HOME and honcho-ai==2.2.0 installed (or on PYTHONPATH):
  .venv/bin/python evals/memory/honcho_current_query.py --out /tmp/honcho-proof.json
Prepared ongoing sessions bypass startup/migration. Message writes are disabled.
The fixture proves query routing, ownership and caller waiting, not memory quality.
"""
import argparse
import json
import socket
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
args = argparse.ArgumentParser()
args.add_argument("--out", type=Path, required=True)
args = args.parse_args()
events = []
release = threading.Event()
entered = threading.Event()
invalidate = None
delay = 0


class Backend(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def handle_request(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        search = query.get("search_query", [""])[0]
        events.append({"path": parsed.path, "search": search, "body": body})
        if delay:
            time.sleep(delay)
        if search.startswith("BLOCK"):
            entered.set()
            release.wait(10)
        if search.startswith("GENERATION") and invalidate:
            invalidate()
        status = 200
        if parsed.path.endswith("/context"):
            if search.startswith("ERROR"):
                status, data = 400, {"error": "fixture rejection"}
            else:
                data = {"peer_id": "assistant", "target_id": "user",
                        "representation": "RETRIEVED_FOR=" + search if search else "AI identity",
                        "peer_card": ["fixture card"]}
                if search.startswith("EMPTY") or empty_all:
                    data.update(representation="", peer_card=[])
        elif parsed.path.endswith("/chat"):
            data = {"content": "" if empty_all else "DIALECTIC_FOR=" + body.get("query", "")}
        elif parsed.path.endswith("/peers"):
            data = {"id": body["id"], "workspace_id": "probe", "created_at": "2026-01-01T00:00:00Z",
                    "metadata": {}, "configuration": {}}
        elif parsed.path.endswith("/workspaces"):
            data = {"id": "probe", "metadata": {}, "configuration": {}}
        else:
            status, data = 404, {"error": "no fixture fallback"}
        raw = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    do_GET = handle_request
    do_POST = handle_request


empty_all = False
server = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
threading.Thread(target=server.serve_forever, daemon=True).start()
original_connect = socket.socket.connect


def fixture_only(self, address):
    if not isinstance(address, tuple) or address[:2] != ("127.0.0.1", server.server_port):
        raise RuntimeError(f"Blocked nonfixture connection: {address}")
    return original_connect(self, address)


socket.socket.connect = fixture_only
from plugins.memory.honcho import HonchoMemoryProvider  # noqa: E402
from plugins.memory.honcho.client import HonchoClientConfig  # noqa: E402
from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager  # noqa: E402
from agent.memory_manager import MemoryManager  # noqa: E402
from agent.turn_context import _memory_turn_start_and_prefetch, compose_user_api_content  # noqa: E402
from run_agent import AIAgent  # noqa: E402


def prepared(sync):
    p = HonchoMemoryProvider()
    with tempfile.TemporaryDirectory() as home:
        p.save_config({"baseUrl": f"http://127.0.0.1:{server.server_port}", "workspace": "probe",
                       "timeout": 2, "contextTokens": 1200, "saveMessages": False,
                       "queryRewrite": False, "recallSync": True,
                       "hosts": {"hermes": {"recallSync": sync}}}, home)
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=Path(home) / "honcho.json")
    assert cfg.recall_sync is sync
    manager = HonchoSessionManager(config=cfg, context_tokens=1200)
    manager._cache["ongoing"] = HonchoSession("ongoing", "user", "assistant", "ongoing")
    p._config, p._manager, p._session_key, p._session_initialized = cfg, manager, "ongoing", True
    p._recall_sync = cfg.recall_sync
    p._dialectic_cadence = 100
    p._last_dialectic_turn = 0
    mm = MemoryManager(external_prefetch_timeout=3)
    mm.add_provider(p)
    a = SimpleNamespace(_memory_manager=mm, session_id="ongoing", _user_turn_count=1)
    return p, a


def begin(a, turn, query):
    a._user_turn_count = turn
    started = time.monotonic()
    context = _memory_turn_start_and_prefetch(a, query)
    return {"query": query, "context": context, "elapsed": time.monotonic() - started,
            "api_content": compose_user_api_content(query, context, "")}


def end(a, query):
    AIAgent._sync_external_memory_for_turn(a, original_user_message=query,
        final_response="Fixture answer; no LLM called.", interrupted=False)


def settle(p):
    for thread in (getattr(p, "_recall_sync_thread", None), p._prefetch_thread):
        if thread:
            thread.join(5)
            assert not thread.is_alive()


legacy, a = prepared(False)
first = begin(a, 1, "What is my preferred telescope?")
proof = {"boundary": __doc__, "sdk": sys.modules["honcho"].__file__,
         "provider": sys.modules[HonchoMemoryProvider.__module__].__file__}
end(a, first["query"])
deadline = time.monotonic() + 5
while not legacy._manager._context_cache and time.monotonic() < deadline:
    time.sleep(0.01)
second = begin(a, 2, "Which sourdough starter do I use?")
assert first["query"] in second["context"] and second["query"] not in second["context"]
proof["default_previous_query"] = second
p, a = prepared(True)
p._dialectic_cadence = 1
rows = []
for turn, query in enumerate(("What is my preferred telescope?", "Which sourdough starter do I use?"), 1):
    row = begin(a, turn, query)
    assert "RETRIEVED_FOR=" + query in row["context"]
    if rows:
        assert rows[-1]["query"] not in row["context"]
    before = len(events)
    end(a, query)
    assert len(events) == before, "sync mode must not queue duplicate automatic retrieval"
    rows.append(row)
proof["current_queries"] = rows
assert all(row["query"] in row["context"].split("DIALECTIC_FOR=")[1] for row in rows)
p._config.timeout = 0.05
blocked = begin(a, 3, "BLOCK Which hiking boots do I wear?")
assert entered.wait(2) and blocked["context"] == "" and blocked["elapsed"] < 2
before = len(events)
for turn in range(4, 9):
    assert begin(a, turn, "Which fountain pen ink do I use?")["context"] == ""
assert len(events) == before, "blocked worker must retain its single-flight slot"
release.set()
settle(p)
p._config.timeout = 2
recovered = begin(a, 9, "Which violin strings do I prefer?")
assert recovered["query"] in recovered["context"] and "BLOCK" not in recovered["context"]
proof["timeout_and_recovery"] = {"blocked": blocked, "recovered": recovered, "busy_calls": 5}
delay = 0.04
p._config.timeout = 0.06
responsive = begin(a, 10, "Which garden plants do I grow?")
assert responsive["context"] == "" and responsive["elapsed"] < 2
settle(p)
delay = 0
p._config.timeout = 2
proof["shared_deadline"] = responsive
invalidate = lambda: p.on_turn_start(10, "Different request with the same turn number")
assert begin(a, 10, "GENERATION What watercolor paints do I prefer?")["context"] == ""
assert begin(a, 11, "ERROR Which watercolor paints do I prefer?")["context"] == ""
empty_all = True
assert begin(a, 12, "EMPTY Which watercolor paints do I prefer?")["context"] == ""
empty_all = False
p._context_cadence = 3
assert begin(a, 13, "Which aquarium food do I buy?")["context"] == ""
proof["checks"] = {"alignment": True, "default_unchanged": True, "single_flight": True,
                   "late_discarded": True, "generation_discarded": True, "error_omitted": True,
                   "empty_omitted": True, "cadence_gap_empty": True, "no_duplicate_end_turn": True}
proof["events"] = events
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(proof, indent=2))
print(json.dumps({"checks": proof["checks"], "out": str(args.out),
                  "current_elapsed": [row["elapsed"] for row in rows],
                  "timeout_elapsed": blocked["elapsed"]}, indent=2))
server.shutdown()

"""End-to-end probe for cron.background_review with a REAL AIAgent and REAL review fork.

Only the provider is fake: a local OpenAI-wire SSE server. The main cron turn and the review
fork both hit it; the review request is slowed down so it is still in flight when the turn
returns — the exact window where cron used to finalize/tear the agent down.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

import run_agent  # noqa: F401 — import before the home I/O guard arms (bootstrap reads the real root)
from cron.scheduler import run_job


class _Wire:
    def __init__(self, review_delay: float):
        self.requests: list[dict] = []
        self.review_requests = 0
        self.review_finished_at: list[float] = []
        wire = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                if not self.path.endswith("/chat/completions"):
                    self.send_response(404)
                    self.end_headers()
                    return
                wire.requests.append(body)
                last_user = next((m for m in reversed(body.get("messages", []))
                                  if m.get("role") == "user"), {})
                content = last_user.get("content")
                text = content if isinstance(content, str) else json.dumps(content)
                is_review = "Review the conversation above" in text
                if is_review:
                    wire.review_requests += 1
                    time.sleep(review_delay)
                reply = "Nothing to save." if is_review else "cron job done"
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                chunks = [
                    {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": reply},
                                  "finish_reason": None}]},
                    {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                     "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}},
                ]
                for c in chunks:
                    self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                if is_review:
                    wire.review_finished_at.append(time.monotonic())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines: list[tuple[float, str]] = []

    def emit(self, record):
        self.lines.append((time.monotonic(), record.getMessage()))


def _run(tmp_home: Path, extra_cron_cfg: str, review_delay: float):
    wire = _Wire(review_delay)
    (tmp_home / "config.yaml").write_text(
        "model:\n  default: m\n  provider: custom\n"
        "skills:\n  creation_nudge_interval: 1\n"
        "cron:\n  preflight: false\n" + extra_cron_cfg,
        encoding="utf-8",
    )
    runtime = {"api_key": "test-key", "base_url": wire.base_url, "provider": "custom",
               "api_mode": "chat_completions"}
    collect = _Collect()
    loggers = [logging.getLogger(n) for n in ("agent.background_review", "cron.scheduler")]
    for lg in loggers:
        lg.addHandler(collect)
        lg.setLevel(logging.DEBUG)
    import cron.scheduler as _sched
    _orig_construct = _sched._construct_cron_agent
    agents = []

    def _capture(*a, **kw):
        agent = _orig_construct(*a, **kw)
        agents.append(agent)
        return agent

    try:
        with patch("cron.scheduler._hermes_home", tmp_home), \
             patch("cron.scheduler._construct_cron_agent", _capture), \
             patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime):
            t0 = time.monotonic()
            result = run_job({"id": "bgreview-e2e", "name": "bgreview e2e", "prompt": "say done",
                              "enabled_toolsets": ["skills"], "model": "m"})
            returned_at = time.monotonic()
    finally:
        for lg in loggers:
            lg.removeHandler(collect)
        wire.close()
    if agents:
        a = agents[0]
        print(f"\n[probe] skip={a.skip_background_review} nudge={a._skill_nudge_interval} "
              f"iters={getattr(a, '_iters_since_skill', None)} "
              f"skill_manage={'skill_manage' in a.valid_tool_names} tools={sorted(a.valid_tool_names)[:12]}")
    return result, t0, returned_at, collect.lines, wire


@pytest.fixture
def home():
    path = Path(os.environ["HERMES_HOME"])
    path.mkdir(parents=True, exist_ok=True)
    cfg = path / "config.yaml"
    before = cfg.read_text(encoding="utf-8") if cfg.exists() else None
    yield path
    if before is None:
        cfg.unlink(missing_ok=True)
    else:
        cfg.write_text(before, encoding="utf-8")


def test_review_completes_before_run_job_returns(home):
    (success, _out, final, error), t0, returned_at, lines, wire = _run(
        home, "  background_review: true\n  background_review_wait_seconds: 30\n", review_delay=1.0)
    print("\n".join(f"{t - t0:6.2f}s {m}" for t, m in lines))
    assert success is True and error is None and final == "cron job done"
    assert wire.review_requests >= 1, "review fork never called the provider"
    done = [(t, m) for t, m in lines if m.startswith("Background review complete")]
    assert done, "no 'Background review complete' line"
    assert done[0][0] <= returned_at
    assert "result=error" not in done[0][1]
    assert any("background review finished before teardown" in m for _t, m in lines)


def test_default_config_spawns_no_review(home):
    (success, _out, final, error), _t0, _ret, lines, wire = _run(home, "", review_delay=1.0)
    assert success is True and error is None and final == "cron job done"
    assert wire.review_requests == 0
    assert not any(m.startswith("Background review complete") for _t, m in lines)


def test_without_wait_review_is_still_in_flight_at_return(home):
    """Sensitivity: wait=0 reproduces the original race (review not done when run_job returns)."""
    (success, _out, _final, _error), _t0, returned_at, lines, wire = _run(
        home, "  background_review: true\n  background_review_wait_seconds: 0\n", review_delay=1.5)
    assert success is True
    done_before_return = [m for t, m in lines
                          if m.startswith("Background review complete") and t <= returned_at]
    assert not done_before_return
    time.sleep(2.0)  # let the orphaned daemon thread settle before the next test

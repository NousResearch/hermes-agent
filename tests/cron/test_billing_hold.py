"""A cron job whose provider refuses it for billing/credits is held, not run on a local model.

Contract (``agent/fallback_local_billing.py`` + ``cron/billing_hold.py``): the real scheduler
fires a job whose primary answers xAI's spending-limit 403 while the fallback chain holds a
local (loopback) server. The run does not continue on the local server; the one alert says the
job is held; a sub-hourly job is parked at least an hour out; a re-probe refused again stays
silent; the first run that reaches the model clears the hold. Real ``AIAgent`` and scheduler,
two fake OpenAI-compatible servers on 127.0.0.1.
"""

import json
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, patch

import pytest

import cron.scheduler as sched
from cron.jobs import create_job, get_due_jobs, get_job, update_job

_SPENT = {
    "code": "personal-team-blocked:spending-limit",
    "error": "Your team has either used all available credits or reached its monthly spending limit.",
}


class _FakeProvider:
    """OpenAI-compatible endpoint; counts every request it receives."""

    def __init__(self, answer: str, refuse: bool):
        self.answer, self.refuse, self.requests = answer, refuse, 0
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def _send(self, code, body, ctype="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                provider.requests += 1
                self._send(404, b"{}")

            def do_POST(self):
                provider.requests += 1
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if provider.refuse:
                    return self._send(403, json.dumps(_SPENT).encode())
                delta = {"role": "assistant", "content": provider.answer}
                chunk = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
                         "choices": [{"index": 0, "delta": delta, "finish_reason": "stop"}]}
                if req.get("stream"):
                    return self._send(200, f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(),
                                      "text/event-stream")
                done = {"id": "c", "object": "chat.completion", "created": 0, "model": "m",
                        "choices": [{"index": 0, "message": delta, "finish_reason": "stop"}]}
                self._send(200, json.dumps(done).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"


@pytest.fixture
def providers(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    primary = _FakeProvider("PRIMARY ANSWER", refuse=True)
    local = _FakeProvider("LOCAL ANSWER", refuse=False)
    (home / "config.yaml").write_text(
        "model:\n  default: grok-4.20\n  provider: custom\n"
        f"  base_url: {primary.url}\n  api_key: sk-test\n"
        "fallback_providers:\n  - provider: custom\n    model: gemma-4-e4b\n"
        f"    base_url: {local.url}\n    api_key: lm-studio\n",
        encoding="utf-8")
    yield home, primary, local
    for p in (primary, local):
        p.server.shutdown()


def _fire(home, job_id, deliveries):
    with patch("cron.scheduler._hermes_home", home), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_state_registry.acquire", return_value=MagicMock()), \
         patch("tools.mcp_tool_discovery.discover_mcp_tools", return_value=[]), \
         patch.object(sched, "_deliver_result",
                      side_effect=lambda _job, content, **_kw: deliveries.append(content)):
        sched.run_one_job(dict(get_job(job_id)))
    return get_job(job_id)


@pytest.mark.parametrize("release", ["top_up", "model_change"])
def test_billing_refused_job_is_held_instead_of_running_on_the_local_fallback(providers, release):
    home, primary, local = providers
    job_id = create_job("Summarize the news.", "every 5m", name="research digest", deliver="local")["id"]
    deliveries: list = []
    now = datetime.now(timezone.utc)

    held = _fire(home, job_id, deliveries)
    assert local.requests == 0, "a billing-refused background run must not touch the local model"
    assert held["last_status"] == "error"
    assert held["billing_hold_provider"] and held["quota_hold_until"] == held["next_run_at"]
    assert datetime.fromisoformat(held["next_run_at"]) - now >= timedelta(minutes=59)
    assert len(deliveries) == 1 and "Held:" in deliveries[0] and "gemma-4-e4b" in deliveries[0]
    # Half an hour on, the errored 5-minute job looks wedged to the stale-error re-arm (#62002);
    # the hold says it is parked on purpose.
    update_job(job_id, {"last_run_at": (now - timedelta(minutes=30)).isoformat()})
    assert all(due["id"] != job_id for due in get_due_jobs())
    assert get_job(job_id)["next_run_at"] == held["next_run_at"]

    # The re-probe at the parked instant is refused again: silent, still held, still no local run.
    update_job(job_id, {"next_run_at": now.isoformat()})
    reheld = _fire(home, job_id, deliveries)
    assert len(deliveries) == 1 and local.requests == 0
    assert reheld["billing_hold_provider"] and reheld["quota_hold_until"] == reheld["next_run_at"]

    if release == "model_change":
        # The hold is evidence about the refused runtime, not the schedule: another edit leaves it
        # parked, a new model/provider/base_url puts the job back on its own cadence.
        assert update_job(job_id, {"prompt": "Summarize the markets."})["next_run_at"] == reheld["next_run_at"]
        edited = update_job(job_id, {"provider": "custom", "model": "gemma-4-e4b", "base_url": local.url})
        assert "billing_hold_provider" not in edited and "quota_hold_until" not in edited
        assert datetime.fromisoformat(edited["next_run_at"]) - now < timedelta(minutes=10)
        return

    # Topped up: the run reaches the model, delivers, and the hold is gone.
    primary.refuse = False
    update_job(job_id, {"next_run_at": now.isoformat()})
    resumed = _fire(home, job_id, deliveries)
    assert deliveries[-1] == "PRIMARY ANSWER" and resumed["last_status"] == "ok"
    assert "billing_hold_provider" not in resumed and "quota_hold_until" not in resumed
    assert datetime.fromisoformat(resumed["next_run_at"]) - now < timedelta(minutes=10)

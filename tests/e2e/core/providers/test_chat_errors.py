"""Chat-completions provider faults through REAL ``hermes -z`` processes.

The vendor boundary is ``FakeChatVariantServer`` (a plain ``provider: custom``
chat-completions endpoint); everything between the CLI and that socket is real Hermes.

Proven here:

* a 429 carrying ``Retry-After`` is retried after the advertised wait (not the generic
  jittered backoff), in both directions: a long hint is honoured in full, a short hint
  is not padded up to the backoff floor; the turn answers once;
* a transient 503 is retried on the same endpoint and the turn answers once;
* a stream that drops mid-tool-call is retried without executing the half-streamed call,
  and the completed call runs EXACTLY once with exactly one persisted call/result pair;
* an upstream account ban relayed as an ``error`` object inside an HTTP-200 SSE stream
  fails the turn once, visibly, without a retry storm.
"""

from __future__ import annotations

import json
import sys

import pytest

from tests.e2e.core.providers._openai_helpers import (
    Home,
    custom_chat_config,
    db_messages,
    db_tool_calls,
    oneshot,
    tool_call_args,
)
from tests.fakes.providers.chat_variants import (
    CDropToolCall,
    CError,
    CStreamError,
    CText,
    CTools,
    FakeChatVariantServer,
)

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="subprocess harness is Linux-gated")

# Scenario -> "#issue one-line symptom" for scenarios red on origin/main. Strict: the test
# FAILS the moment the bug is fixed, forcing the entry out.
KNOWN: dict[str, str] = {
    "in_stream_ban_fails_once": "#121270 in-stream error code 403 ignored: ban retried, reported as temporarily unavailable",
}

DOTENV = {"OPENAI_API_KEY": "sk-fake"}
# Generic retry backoff for the first retry is jittered in [2.0, 3.0] s; the hints below
# sit clearly on either side of that window so honouring vs ignoring them is observable.
BACKOFF_FLOOR = 2.0
EPSILON = 0.1


def known(name: str) -> list:
    """Marks for a scenario: a strict xfail while it is in KNOWN, nothing once fixed."""
    return [pytest.mark.xfail(strict=True, raises=AssertionError, reason=KNOWN[name])] if name in KNOWN else []


def _home(tmp_path, srv: FakeChatVariantServer, **config) -> Home:
    cfg = custom_chat_config(srv.base_url)
    cfg.update(config)
    return Home(tmp_path).write(cfg, DOTENV)


def _assistant_answers(h: Home, sid: str | None, text: str) -> list[dict]:
    return [r for r in db_messages(h, sid) if r["role"] == "assistant" and text in (r.get("content") or "")]


@pytest.mark.parametrize("retry_after", [
    pytest.param(4.0, id="long_hint_honoured"),
    pytest.param(0.5, id="short_hint_not_padded"),
])
def test_rate_limit_retry_waits_for_retry_after(tmp_path, retry_after: float) -> None:
    script = [CError(429, "Rate limit exceeded, slow down", code="rate_limit_exceeded", retry_after=retry_after),
              CText("AFTER-RATE-LIMIT")]
    with FakeChatVariantServer(script) as srv:
        h = _home(tmp_path, srv)
        run = oneshot(h, "say hello")
        records = srv.main_records()

    assert run.proc.returncode == 0 and run.stdout.strip() == "AFTER-RATE-LIMIT", run.describe()
    assert [r["response"] for r in records] == ["CError", "CText"], run.describe()
    gap = records[1]["t"] - records[0]["t"]
    assert gap >= retry_after - EPSILON, f"retried {gap:.2f}s after a 429 advertising Retry-After {retry_after}s"
    if retry_after < BACKOFF_FLOOR:
        assert gap < BACKOFF_FLOOR - EPSILON, f"short Retry-After {retry_after}s padded to {gap:.2f}s"
    assert len(_assistant_answers(h, run.session_id, "AFTER-RATE-LIMIT")) == 1, db_messages(h, run.session_id)


def test_service_unavailable_is_retried_on_same_endpoint(tmp_path) -> None:
    script = [CError(503, "Service temporarily unavailable", code="service_unavailable"), CText("AFTER-503")]
    with FakeChatVariantServer(script) as srv:
        h = _home(tmp_path, srv)
        run = oneshot(h, "say hello")
        records = srv.main_records()

    assert run.proc.returncode == 0 and run.stdout.strip() == "AFTER-503", run.describe()
    assert [r["response"] for r in records] == ["CError", "CText"], run.describe()
    # The retry resends the same conversation, not a truncated or mutated one.
    assert records[1]["body"]["messages"] == records[0]["body"]["messages"]
    assert run.usage.get("failed") is not True, run.describe()
    assert len(_assistant_answers(h, run.session_id, "AFTER-503")) == 1, db_messages(h, run.session_id)


def test_stream_drop_mid_tool_call_executes_tool_once(tmp_path) -> None:
    command = "echo ran >> count.txt"
    script = [CDropToolCall("terminal", partial_args='{"command": "echo ran >> cou'),
              CTools([("terminal", {"command": command})]),
              CText("TOOL-DONE")]
    with FakeChatVariantServer(script) as srv:
        h = _home(tmp_path, srv, approvals={"mode": "off"})
        run = oneshot(h, "append a line to count.txt", env={"TERMINAL_ENV": "local"})
        records = srv.main_records()

    assert run.proc.returncode == 0 and run.stdout.strip() == "TOOL-DONE", run.describe()
    assert [r["response"] for r in records] == ["CDropToolCall", "CTools", "CText"], run.describe()
    count = h.project / "count.txt"
    assert count.exists(), f"tool never ran: {run.describe()}"
    assert count.read_text(encoding="utf-8").splitlines() == ["ran"], "side effect ran more than once"
    # The dropped half-call never reaches the wire again: the retry resends the bare prompt.
    assert [m["role"] for m in records[1]["body"]["messages"]] == [m["role"] for m in records[0]["body"]["messages"]]
    rows = db_messages(h, run.session_id)
    calls = db_tool_calls(rows)
    results = [r for r in rows if r["role"] == "tool"]
    assert [tool_call_args(c) for c in calls] == [{"command": command}], calls
    assert [r["tool_call_id"] for r in results] == [calls[0]["id"]], results
    # The final request replays exactly one call and exactly one matching result.
    final = records[2]["body"]["messages"]
    replayed = [tc for m in final if m.get("role") == "assistant" for tc in (m.get("tool_calls") or [])]
    assert [tc["id"] for tc in replayed] == [m["tool_call_id"] for m in final if m.get("role") == "tool"], final


BAN = {"code": 403, "message": "Your account has been banned by the upstream provider",
       "metadata": {"provider_name": "UpstreamCo", "raw": json.dumps({"error": "account banned"})}}


@pytest.mark.parametrize("_", [pytest.param(None, id="ban", marks=known("in_stream_ban_fails_once"))])
def test_in_stream_upstream_ban_fails_once_and_visibly(tmp_path, _) -> None:
    script: list = [CStreamError(BAN) for _ in range(6)]
    with FakeChatVariantServer(script, default_text="SHOULD-NOT-ANSWER") as srv:
        h = _home(tmp_path, srv)
        run = oneshot(h, "say hello")
        records = srv.main_records()

    assert "SHOULD-NOT-ANSWER" not in run.stdout, run.describe()
    assert run.proc.returncode != 0 or run.usage.get("failed") is True, run.describe()
    surfaced = (run.stdout + run.proc.stderr).lower()
    assert "banned" in surfaced, f"ban not surfaced to the user: {run.describe()}"
    assert len(records) == 1, f"{len(records)} requests for a permanent account ban: {run.describe()}"

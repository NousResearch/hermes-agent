"""``on_session_end`` fires exactly once at the end of every ``run_conversation()`` call,
regardless of outcome (user hook guide, ``on_session_end``: "Always fires, even if the turn errored").

Through a real ``AIAgent`` + ``SessionDB`` against the loopback model, with a user plugin that
records each ``on_session_end`` call: a normal answer, a provider error no fallback can fix, retry
exhaustion, a content-policy refusal, a reply capped by the output limit on every continuation,
three calls to an unknown tool, an interrupt during the retry wait and a context overflow no
compression can fix each end the turn exactly once in the hook, and the payload carries the
failure class relay task metrics read.
"""

import json
import os
import threading
from pathlib import Path

import pytest

from hermes_cli.observability.shared_metrics_contract import task_failure_class
from tests.fakes.fake_llm_provider import Error, FakeLLMServer, Text, ToolCall, write_hermes_home
from tests.waiting import wait_until

_PLUGIN = """import json
import os

_FIELDS = ("session_id", "task_id", "turn_id", "completed", "failed", "interrupted", "turn_exit_reason",
           "failure_reason")


def _record(**kwargs):
    with open(os.path.join(os.path.dirname(__file__), "calls.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({k: kwargs.get(k) for k in _FIELDS}) + "\\n")


def register(ctx):
    ctx.register_hook("on_session_end", _record)
"""

_EXTRA_CONFIG = """updates:
  check: false
models_dev:
  url: http://127.0.0.1:9/api.json
skills:
  creation_nudge_interval: 0
memory:
  nudge_interval: 0
plugins:
  enabled:
  - session-end-recorder
"""


def _interrupt_during_retry_wait(agent):
    """The provider fails with a long Retry-After and the user stops the turn once the backoff wait
    has been announced, so the turn ends from the backoff wait, never from the loop's normal end."""
    def stop_during_backoff():
        # The wait announcement is the last activity written before the backoff sleep begins.
        wait_until(lambda: "retrying in" in agent.get_activity_summary()["last_activity_description"])
        agent.interrupt(hard_cancel=True)

    stopper = threading.Thread(target=stop_during_backoff, daemon=True)

    def respond(_request):
        if stopper.ident is None:  # timed from the first request, not from agent setup
            stopper.start()
        return Error(status=500, message="upstream down", retry_after=30)
    return respond


# exit kind -> (script or script factory taking the agent, expected ``failed`` (None: not asserted));
# the factory's case gets two API attempts, so a retry wait exists to interrupt.
EXITS = {
    "text_answer": ([Text("all done")], False),
    "non_retryable_provider_error": ([Error(status=400, message="bad request")], True),
    "authentication_error": ([Error(status=401, message="invalid api key")], True),
    "retries_exhausted": ([Error(status=500, message="upstream down")] * 2, True),
    "content_policy_refusal": ([Text("", finish_reason="content_filter")], True),
    "output_limit_on_every_continuation": ([Text("partial answer", finish_reason="length")] * 4, None),
    "unknown_tool_three_times": ([ToolCall("no_such_tool", {})] * 3, None),
    "interrupt_during_retry_wait": (_interrupt_during_retry_wait, None),
    "context_overflow_unrecoverable": ([Error(status=400, message=(
        "This model's maximum context length is 128000 tokens. However, your messages resulted in 900000 tokens."
    ))] * 8, True),
}


@pytest.fixture
def recorded_turn(tmp_path):
    from hermes_cli import plugins
    from hermes_state import SessionDB
    from run_agent import AIAgent

    home = Path(os.environ["HERMES_HOME"])
    opened = []

    def run(script, *, persist_disabled=False):
        responders = {}
        server = FakeLLMServer(script if isinstance(script, list) else lambda req: responders["main"](req))
        server.__enter__()
        opened.append(lambda: server.__exit__(None, None, None))
        write_hermes_home(home, server.base_url, extra_config=_EXTRA_CONFIG)
        config_path = home / "config.yaml"
        # No transient-outage recovery cycles: the ladder would wait between attempts.
        retries = 1 if isinstance(script, list) else 2
        config_path.write_text(config_path.read_text(encoding="utf-8").replace(
            "  api_max_retries: 1\n", f"  api_max_retries: {retries}\n  auto_recovery_cycles: 0\n", 1,
        ), encoding="utf-8")
        plugin = home / "plugins" / "session-end-recorder"
        plugin.mkdir(parents=True, exist_ok=True)
        (plugin / "plugin.yaml").write_text("name: session-end-recorder\nversion: 0.1.0\n", encoding="utf-8")
        (plugin / "__init__.py").write_text(_PLUGIN, encoding="utf-8")
        plugins._reset_plugin_managers_for_tests()
        db = SessionDB(db_path=tmp_path / "state.db")
        opened.append(db.close)
        agent = AIAgent(base_url=server.base_url, api_key="sk-fake-e2e", provider="custom", model="fake-model",
                        quiet_mode=True, skip_context_files=True, skip_memory=True,
                        session_id="session-end-exit", session_db=db)
        if not isinstance(script, list):
            responders["main"] = script(agent)
        # A background review fork runs with persistence off (``agent/background_review.py``).
        agent._persist_disabled = persist_disabled
        result = agent.run_conversation("do the thing", task_id="task-end-exit")
        log = plugin / "calls.jsonl"
        calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
        return result, calls

    yield run
    for close in opened:
        close()
    plugins._reset_plugin_managers_for_tests()


@pytest.mark.parametrize("exit_kind", list(EXITS))
def test_session_end_hook_fires_once_on_every_turn_exit(recorded_turn, exit_kind):
    script, expect_failed = EXITS[exit_kind]
    result, calls = recorded_turn(script)
    assert result["completed"] is (expect_failed is False)
    assert len(calls) == 1, f"{exit_kind}: on_session_end fired {len(calls)} times; result={result.get('failure_reason')!r}"
    [call] = calls
    assert call["session_id"] == "session-end-exit"
    assert call["task_id"] == "task-end-exit"
    assert call["turn_id"]
    assert call["completed"] is result["completed"]
    assert call["interrupted"] is bool(result.get("interrupted")) is exit_kind.startswith("interrupt")
    assert call["failure_reason"] == result.get("failure_reason")
    if expect_failed is not None:
        assert call["failed"] is expect_failed


def test_session_end_payload_classifies_an_authentication_failure(recorded_turn):
    """Relay task metrics finish the task from the first end event: a 401 stays classified ``auth``."""
    result, [call] = recorded_turn([Error(status=401, message="invalid api key")])
    assert result["failure_reason"] == "auth"
    assert task_failure_class(call) == "auth"


def test_persistence_disabled_turn_fires_no_session_end(recorded_turn):
    """A turn run with persistence off (a background review fork) is not the user's session: no end event."""
    result, calls = recorded_turn([Text("all done")], persist_disabled=True)
    assert result["completed"] is True
    assert calls == []

"""Reasoning self-heal: the turn path must never mute thinking, and a thinking-only
answer is rescued instead of discarded.

Two hidden self-switchers used to turn thinking off for the rest of the session:
a thinking-only length truncation armed a one-shot reasoning-off override for the
next call (and threw the reasoning away — the answer never reached state.db), and a
400 on an ENABLED reasoning config set a session-sticky flag that dropped the
reasoning config permanently. The contract here: the reasoning text IS the answer
when content is empty (Bergung), thinking stays configured on every later call, and
an effort rejection costs one omitted retry, not the session.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_constants import FINISH_REASON_LENGTH
from hermes_state import SessionDB

_DEAD_LOCAL = "http://127.0.0.1:9"
_RESCUED = "The answer is 42 — the math checks out end to end."


def _thinking_only_length_response(reasoning=_RESCUED):
    """finish_reason='length' with reasoning but zero visible content — the live
    GLM-5.3-flash-on-ollama shape (normal response id, NOT the partial-stream stub)."""
    return SimpleNamespace(
        id="chatcmpl-self-heal",
        model="test/model",
        choices=[SimpleNamespace(
            index=0,
            message=SimpleNamespace(content="", reasoning=reasoning, tool_calls=None),
            finish_reason=FINISH_REASON_LENGTH,
        )],
        usage=None,
    )


def _full_response(content, finish_reason="stop"):
    return SimpleNamespace(
        id="chatcmpl-self-heal-answer",
        model="test/model",
        choices=[SimpleNamespace(
            index=0,
            message=SimpleNamespace(content=content, tool_calls=None),
            finish_reason=finish_reason,
        )],
        usage=None,
    )


@pytest.fixture
def agent(tmp_path, monkeypatch):
    """A real AIAgent on a real SessionDB under a temp home; only the provider is scripted."""
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(var, _DEAD_LOCAL)
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)
    monkeypatch.setattr("agent.title_generator.start_title_upgrade", lambda *a, **k: None)
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "sess-self-heal"
    with patch("agent.process_bootstrap.OpenAI"), patch("agent.model_metadata.fetch_model_metadata", return_value={}):
        from run_agent import AIAgent

        a = AIAgent(
            api_key="test-key", base_url=f"{_DEAD_LOCAL}/v1", model="test/model", quiet_mode=True,
            skip_context_files=True, skip_memory=True, session_db=db, session_id=sid,
            reasoning_config={"enabled": True, "effort": "high"},
        )
        # The scripted relay advertises reasoning via extra_body (chat-completions transport's
        # supports_reasoning gate): the wire-reader below asserts the reasoning field per call.
        a._supports_reasoning_extra_body = lambda: True
    a._create_openai_client = lambda *a_, **k_: (_ for _ in ()).throw(AssertionError("a real provider client would be built"))
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a.compression_enabled = False
    a.save_trajectories = False
    # One client for the whole test so its create() call history spans every run() below
    # (replacing the mock per turn would discard the earlier calls the wire assertions read).
    a.client = MagicMock()

    def run(script, user_message):
        pending = list(script)
        a.client.chat.completions.create.side_effect = lambda **_kw: pending.pop(0)
        return a.run_conversation(user_message)

    yield SimpleNamespace(agent=a, db=db, sid=sid, run=run)
    db.close()


class _WireProbe:
    """Records the reasoning config each scripted request actually carried."""


def _wire_reasoning_configs(agent_fixture):
    """The reasoning config each scripted request actually carried, read from the
    wire field the chat-completions transport sets for a profile-less agent
    (``extra_body.reasoning``; the fixture's ``_supports_reasoning_extra_body``
    gate opens exactly that branch)."""
    calls = agent_fixture.agent.client.chat.completions.create.call_args_list
    return [(c.kwargs.get("extra_body") or {}).get("reasoning") for c in calls]


# ── Criterion 1: a thinking-only event never disables the next call's reasoning ──

def test_thinking_only_event_never_sends_reasoning_off_on_the_next_call(agent):
    agent.run([_thinking_only_length_response()], "think hard about this")
    # The rescued turn ends on the same response: no continuation call was armed.
    assert _wire_reasoning_configs(agent) == [{"enabled": True, "effort": "high"}]

    agent.run([_full_response("Fresh turn answer.")], "and now answer plainly")
    wire = _wire_reasoning_configs(agent)
    assert wire[1] == {"enabled": True, "effort": "high"}, (
        f"the call after a thinking-only event must keep thinking configured; got {wire!r}"
    )
    assert all(w != {"enabled": False, "effort": "none"} for w in wire), wire


# ── Criterion 2: an HTTP 400 costs one omit, not the session ──

_STRUCTURED_LEVEL_400 = (
    "Error code: 400 - {'error': {'param': 'reasoning.effort', "
    "'error_code': 'invalid_reasoning_effort', 'retryable': False}}"
)


def test_reasoning_effort_rejection_is_not_session_sticky(monkeypatch):
    """A 400 on an ENABLED reasoning config omits the reasoning fields for exactly
    the retry; the call after that carries the configured level again — no
    session-sticky drop (the old ``_reasoning_effort_rejected`` contract)."""
    from agent.turn_recovery import recover_after_classification
    from agent.turn_retry_state import TurnRetryState
    from agent.error_classifier import FailoverReason, classify_api_error

    class _Agent:
        reasoning_config = {"enabled": True, "effort": "max"}
        provider = "custom"
        model = "relay-model"
        api_mode = "chat_completions"
        base_url = "http://relay.example/v1"
        log_prefix = ""
        verbose = False
        request_overrides = None
        _fast_until = 0.0
        service_tier = None
        _reasoning_disable_rejected = False
        _reasoning_effort_rejected = False
        _credential_pool = None
        _fallback_chain = []
        _fallback_index = 0

        def __init__(self):
            self.notices = []
            self._wire_reasoning_config = None

        def _vprint(self, message, **kwargs):
            self.notices.append(message)

        def _recover_with_credential_pool(self, **kwargs):
            return False, False

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    err = SimpleNamespace(
        status_code=400,
        body={"error": {"param": "reasoning.effort", "error_code": "invalid_reasoning_effort"}},
    )
    err.strerror = str(_STRUCTURED_LEVEL_400)
    classified = SimpleNamespace(reason=FailoverReason.reasoning_mandatory, billing_unverified=False)
    retry = TurnRetryState()
    agent = _Agent()

    with patch("hermes_cli.models_reasoning_caps.refresh_reasoning_caps_async", lambda provider: None):
        retry_now, _ = recover_after_classification(
            agent, err, classified, retry, status_code=400, error_context={},
            messages=[], api_messages=[],
        )

    assert retry_now is True
    # No session-sticky flag: the old attribute must not be set anywhere.
    assert not getattr(agent, "_reasoning_effort_rejected", False)
    # The retry omits the reasoning fields (route default)...
    from agent.chat_completion_helpers import _reasoning_config_for_wire
    assert _reasoning_config_for_wire(agent) is None
    # ...and the call after the retry carries the configured level again.
    assert _reasoning_config_for_wire(agent) == {"enabled": True, "effort": "max"}


# ── Criterion 3: the Bergung — reasoning reaches state.db with the rescued answer ──

def test_thinking_only_answer_is_rescued_into_the_session_db(agent):
    result = agent.run([_thinking_only_length_response()], "think hard about this")

    assert "The answer is 42" in (result["final_response"] or "")
    with sqlite3.connect(str(agent.db.db_path)) as conn:
        rows = conn.execute(
            "SELECT role, content, reasoning FROM messages WHERE session_id = ? ORDER BY id",
            (agent.sid,),
        ).fetchall()
    assistant_rows = [r for r in rows if r[0] == "assistant"]
    assert assistant_rows, "the rescued turn must be persisted"
    _role, _content, reasoning = assistant_rows[-1]
    # The reasoning cell is filled — the old path discarded it entirely.
    assert "The answer is 42" in (reasoning or ""), reasoning
    # The rescued answer sits in the content cell of the same row.
    assert "The answer is 42" in (_content or ""), _content
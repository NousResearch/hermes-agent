"""Reasoning-replay recovery is issuer-scoped (durable) and bounded on proxy routes.

A replayed ``reasoning.encrypted_content`` blob is sealed to the backend identity that minted it;
the provider rejects a foreign blob with HTTP 400 ``invalid_encrypted_content``. The recovery used
to answer with a process-wide bool, so a restart resurrected replay and lost the same history again,
and any other issuer in the session lost replay it never needed to lose. It is now a durable,
``(issuer_kind, issuer_model)``-scoped deny-list (``agent/codex_reasoning_replay.py``) plus a replay
window for proxy/aggregator issuers (``agent.codex_proxy_replay_turns``, default 2) — #2 and #4.

The HTTP/SSE frames are synthetic; every assertion runs the real classifier, recovery, transport and
converter.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.codex_reasoning_replay import (
    DEFAULT_PROXY_REPLAY_TURNS,
    current_issuer_pair,
    denied_pairs,
    deny_pair,
    load_denied_pairs,
    pair_is_denied,
    proxy_replay_max_turns,
    proxy_replay_turns_from_config,
    record_denied_pair,
)
from agent.codex_responses_adapter import _chat_messages_to_responses_input, _classify_responses_issuer
from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_recovery import recover_after_classification
from agent.turn_retry_state import TurnRetryState

# OpenCode Zen is a proxy/aggregator route: its responses-mode endpoint is a custom base URL, so it
# classifies as ``other:<url>`` and its backend identity rotates between sessions.
_ZEN_BASE_URL = "https://opencode.ai/zen/go/v1"
_ZEN_KIND = _classify_responses_issuer(base_url=_ZEN_BASE_URL)
_ZEN_MODEL = "gpt-5-codex"
_ZEN_PAIR = (_ZEN_KIND, _ZEN_MODEL)

# The ChatGPT Codex backend is a FIRST-PARTY issuer; its blobs never get the proxy replay window.
_CODEX_KIND = "codex_backend"

_ITEM_MARKER_PREFIX = "encrypted-content-"


def _item(index: int) -> dict:
    return {
        "type": "reasoning",
        "id": f"rs_{index}",
        "encrypted_content": f"{_ITEM_MARKER_PREFIX}{index}",
        "_issuer_kind": None,  # stamped per test below
        "_issuer_model": None,
    }


def _stamped_item(index: int, *, issuer_kind: str, issuer_model: str | None = _ZEN_MODEL) -> dict:
    item = _item(index)
    item["_issuer_kind"] = issuer_kind
    item["_issuer_model"] = issuer_model
    return item


def _history(turns: int, *, issuer_kind: str | None = None, issuer_model: str | None = None) -> list[dict]:
    """``turns`` user/assistant pairs, each assistant turn carrying one reasoning item."""
    messages: list[dict] = []
    for index in range(turns):
        messages.append({"role": "user", "content": f"ask {index}"})
        message: dict = {"role": "assistant", "content": f"answer {index}"}
        item = _item(index) if issuer_kind is None else _stamped_item(
            index, issuer_kind=issuer_kind, issuer_model=issuer_model,
        )
        message["codex_reasoning_items"] = [item]
        messages.append(message)
    return messages


class _Replay400(Exception):
    """The OpenCode Zen wrap of OpenAI's replay rejection (``invalid_encrypted_content``, #111309)."""

    def __init__(self, message: str = "encrypted_content could not be decrypted: was not issued to this caller"):
        super().__init__(f"Error code: 400 - {message!r}")
        self.status_code = 400
        self.message = message
        self.body = {
            "error": {
                "message": message, "type": "invalid_request_error", "code": "invalid_encrypted_content",
            }
        }


class _Agent:
    """Responses agent on a proxy route; the deny-list is the only replay gate (no session bool)."""

    log_prefix = ""
    api_mode = "codex_responses"
    provider = "custom"
    base_url = _ZEN_BASE_URL
    model = _ZEN_MODEL

    def __init__(self, *, session_db=None, proxy_replay_turns=None, model: str = _ZEN_MODEL):
        self._session_db = session_db
        self._codex_replay_denied_pairs = None
        self.model = model
        self.notices: list[str] = []
        if proxy_replay_turns is not None:
            self.codex_proxy_replay_turns = proxy_replay_turns

    def _disable_codex_reasoning_replay(self, messages=None):
        from run_agent import AIAgent

        return AIAgent._disable_codex_reasoning_replay(self, messages)

    def _recover_with_credential_pool(self, **kwargs):
        return False, False

    def _vprint(self, line, **kwargs):
        self.notices.append(line)

    def __getattr__(self, name):
        # Unknown collaborators (transports, credential pool, ...) are inert in these fixtures.
        return lambda *args, **kwargs: None


@pytest.fixture
def session_db(tmp_path):
    """A real SessionDB: the deny-list must survive a process restart, so it needs the real store."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield db
    finally:
        db.close()


def _request_400(agent, messages):
    """Run the real recovery ladder for the provider's ``invalid_encrypted_content`` 400."""
    err = _Replay400()
    classified = classify_api_error(err, provider=agent.provider, model=agent.model)
    assert classified.reason == FailoverReason.invalid_encrypted_content
    retried, _recovered = recover_after_classification(
        agent, err, classified, TurnRetryState(), status_code=400, error_context={},
        messages=messages, api_messages=list(messages),
    )
    return retried


def _wire_kwargs(agent, messages):
    """The Responses kwargs the main transport builds for this agent — the same params
    ``agent/chat_completion_helpers.py::_build_codex_kwargs`` passes."""
    from agent.transports.codex import ResponsesApiTransport

    return ResponsesApiTransport().build_kwargs(
        model=agent.model, messages=messages, tools=None, base_url=agent.base_url,
        provider=agent.provider, is_github_responses=False, is_xai_responses=False,
        is_codex_backend=False,
        replay_denied_issuer_pairs=denied_pairs(agent),
        proxy_replay_max_turns=proxy_replay_max_turns(agent),
    )


def _wire_reasoning(kwargs) -> list[str]:
    return [item["encrypted_content"] for item in kwargs["input"] if item.get("type") == "reasoning"]


# ── #2: durable, issuer-pair-scoped deny-list ────────────────────────────────────────────────


def test_rejected_pair_records_a_durable_deny_and_stops_replaying():
    """Acceptance 1: the 400 strips the pair's items and the NEXT request replays none of them."""
    agent = _Agent()
    messages = _history(3, issuer_kind=_ZEN_KIND)
    # Control: the fixture really does replay before the 400 (the last N turns — this route is a
    # proxy issuer, so the bounded window applies even to a healthy pair).
    assert _wire_reasoning(_wire_kwargs(agent, messages)) == [
        f"{_ITEM_MARKER_PREFIX}1", f"{_ITEM_MARKER_PREFIX}2",
    ]

    assert _request_400(agent, messages) is True

    assert denied_pairs(agent) == {(_ZEN_KIND, _ZEN_MODEL)}
    assert not any("codex_reasoning_items" in m for m in messages)
    kwargs = _wire_kwargs(agent, messages)
    assert _wire_reasoning(kwargs) == []       # nothing replayed for the rejected pair
    assert kwargs["include"] == []             # and no new sealed blobs are asked for
    # The assistant TEXT still goes out: only the encrypted sidecar is dropped.
    assert [i.get("content") for i in kwargs["input"] if i.get("role") == "assistant"] == [
        "answer 0", "answer 1", "answer 2",
    ]


def test_a_different_issuer_pair_in_the_same_session_still_replays():
    """Acceptance 2 (the invariant): a deny-list hit is pair-scoped, never session-wide."""
    agent = _Agent()
    deny_pair(agent, _ZEN_KIND, _ZEN_MODEL)
    assert denied_pairs(agent) == {(_ZEN_KIND, _ZEN_MODEL)}

    # Pair X is denied; a session now calling pair Y (same endpoint, different wire model) replays.
    agent.model = "gpt-5-codex-mini"
    same_endpoint_other_model = _history(2, issuer_kind=_ZEN_KIND, issuer_model="gpt-5-codex-mini")
    assert _wire_reasoning(_wire_kwargs(agent, same_endpoint_other_model)) == [
        f"{_ITEM_MARKER_PREFIX}0", f"{_ITEM_MARKER_PREFIX}1",
    ]

    # And so does a first-party issuer reading its own blobs.
    first_party = _Agent(model="gpt-5-codex")
    first_party.base_url = "https://chatgpt.com/backend-api/codex"
    first_party.provider = "openai-codex"
    codex_items = _history(2, issuer_kind=_CODEX_KIND)
    kwargs = _chat_messages_to_responses_input(
        codex_items, current_issuer_kind=_CODEX_KIND, current_issuer_model=_ZEN_MODEL,
        native_compaction_eligible=False, replay_denied_issuer_pairs=denied_pairs(first_party),
    )
    assert [i["encrypted_content"] for i in kwargs if i.get("type") == "reasoning"] == [
        f"{_ITEM_MARKER_PREFIX}0", f"{_ITEM_MARKER_PREFIX}1",
    ]


def test_deny_list_survives_a_process_restart(session_db):
    """Acceptance 3: a fresh agent over the same store sees the deny and replays nothing."""
    first = _Agent(session_db=session_db)
    assert _request_400(first, _history(2, issuer_kind=_ZEN_KIND)) is True
    assert load_denied_pairs(session_db) == {(_ZEN_KIND, _ZEN_MODEL)}

    # A new process = a new agent object with an empty in-memory cache, same session DB.
    restarted = _Agent(session_db=session_db)
    assert restarted._codex_replay_denied_pairs is None
    assert denied_pairs(restarted) == {(_ZEN_KIND, _ZEN_MODEL)}

    history = _history(2, issuer_kind=_ZEN_KIND)
    kwargs = _wire_kwargs(restarted, history)
    assert _wire_reasoning(kwargs) == []
    assert kwargs["include"] == []
    # Control: the same reload path on a pair nobody rejected replays as usual.
    other_pair = _Agent(session_db=session_db, model="gpt-5-codex-mini")
    assert denied_pairs(other_pair) == {(_ZEN_KIND, _ZEN_MODEL)}
    assert _wire_reasoning(
        _wire_kwargs(other_pair, _history(2, issuer_kind=_ZEN_KIND, issuer_model="gpt-5-codex-mini"))
    ) == [f"{_ITEM_MARKER_PREFIX}0", f"{_ITEM_MARKER_PREFIX}1"]


def test_recovery_is_one_shot_for_a_pair_already_denied():
    """A second 400 for an already-denied pair is not a stale blob of ours: no second strip."""
    agent = _Agent()
    messages = _history(2, issuer_kind=_ZEN_KIND)

    assert _request_400(agent, messages) is True
    assert _request_400(agent, messages) is False


def test_deny_without_a_store_is_in_memory_only_and_still_pair_scoped(session_db):
    """A store-less process (fork, test double) still scopes the deny to the pair."""
    agent = _Agent()
    assert current_issuer_pair(agent) == (_ZEN_KIND, _ZEN_MODEL)
    assert deny_pair(agent, *_ZEN_PAIR) is False  # nothing durable to write to
    assert denied_pairs(agent) == {_ZEN_PAIR}    # but the process itself skips it
    assert load_denied_pairs(session_db) == set()
    assert pair_is_denied(*_ZEN_PAIR, denied_pairs(agent)) is True
    assert pair_is_denied(_ZEN_KIND, "gpt-5-codex-mini", denied_pairs(agent)) is False


@pytest.mark.parametrize("denied, kind, model, expected", [
    ({(("k", "m"))}, "k", "m", True),
    ({(("k", "m"))}, "k", None, True),          # unstamped item on a denied endpoint
    ({(("k", None))}, "k", "m", True),          # endpoint-wide entry
    (set(), "k", "m", False),
    ({(("k", "m"))}, "other", "m", False),      # different endpoint
    ({(("k", "m"))}, None, "m", False),
])
def test_pair_is_denied_matches_a_pair_or_its_whole_endpoint(denied, kind, model, expected):
    assert pair_is_denied(kind, model, denied) is expected


# ── #4: bounded replay window for proxy issuers ──────────────────────────────────────────────


def test_proxy_issuer_replays_only_the_last_n_turns():
    """Acceptance 4: a 10-turn proxy history replays exactly the last N reasoning items."""
    agent = _Agent()
    messages = _history(10, issuer_kind=_ZEN_KIND)
    kwargs = _wire_kwargs(agent, messages)

    assert _wire_reasoning(kwargs) == [f"{_ITEM_MARKER_PREFIX}8", f"{_ITEM_MARKER_PREFIX}9"]
    # Older turns still send their assistant text; only the encrypted sidecar is omitted.
    assert [i.get("content") for i in kwargs["input"] if i.get("role") == "assistant"] == [
        f"answer {index}" for index in range(10)
    ]


def test_first_party_issuer_replays_the_whole_history():
    """Acceptance 5: the proxy window never touches a first-party issuer."""
    from agent.transports.codex import ResponsesApiTransport

    messages = _history(10, issuer_kind="codex_backend")
    kwargs = ResponsesApiTransport().build_kwargs(
        model="gpt-5.6", messages=messages, tools=None, provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex", is_codex_backend=True,
        replay_denied_issuer_pairs=set(), proxy_replay_max_turns=DEFAULT_PROXY_REPLAY_TURNS,
    )
    assert _wire_reasoning(kwargs) == [f"{_ITEM_MARKER_PREFIX}{index}" for index in range(10)]


def test_proxy_window_is_honored_from_config_and_defaults_to_two():
    """Acceptance 6: N comes from ``agent.codex_proxy_replay_turns``; the default is 2."""
    assert DEFAULT_PROXY_REPLAY_TURNS == 2
    assert proxy_replay_max_turns(_Agent()) == 2
    assert proxy_replay_turns_from_config(None) == 2
    assert proxy_replay_turns_from_config(5) == 5
    assert proxy_replay_turns_from_config("3") == 3
    assert proxy_replay_turns_from_config(0) == 0

    wider = _wire_reasoning(_wire_kwargs(_Agent(proxy_replay_turns=5), _history(10, issuer_kind=_ZEN_KIND)))
    assert wider == [f"{_ITEM_MARKER_PREFIX}{index}" for index in range(5, 10)]

    none_replayed = _wire_reasoning(_wire_kwargs(_Agent(proxy_replay_turns=0), _history(10, issuer_kind=_ZEN_KIND)))
    assert none_replayed == []


@pytest.mark.parametrize("raw", [-1, "-2", "wide", True, 1.5, ["2"]])
def test_invalid_proxy_window_falls_back_to_the_default(raw):
    assert proxy_replay_turns_from_config(raw) == DEFAULT_PROXY_REPLAY_TURNS


def test_proxy_window_reaches_the_agent_from_config_yaml(tmp_path, monkeypatch):
    """The knob is wired end to end: ``agent.codex_proxy_replay_turns`` in config.yaml reaches the
    attribute the request builder reads (the regression a registered-but-unread key causes)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("agent:\n  codex_proxy_replay_turns: 5\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from agent.agent_init import _apply_agent_section
    from hermes_cli.config import load_config

    agent = SimpleNamespace(run_budget_seconds=None)
    _apply_agent_section(agent, load_config())

    assert agent.codex_proxy_replay_turns == 5
    assert proxy_replay_max_turns(agent) == 5


def test_legacy_unstamped_items_are_dropped_when_their_endpoint_is_denied():
    """Items persisted before model stamping must not slip past a denied endpoint (or a 400 loops)."""
    legacy = _history(2)  # no issuer stamps at all
    kwargs = _chat_messages_to_responses_input(
        legacy, current_issuer_kind=_ZEN_KIND, current_issuer_model=_ZEN_MODEL,
        native_compaction_eligible=False, replay_denied_issuer_pairs={_ZEN_PAIR},
    )
    assert [i for i in kwargs if i.get("type") == "reasoning"] == []

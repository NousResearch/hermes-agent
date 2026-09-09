"""End-to-end runner separation of observed group context (real history builder + wrapper).

Exercises the REAL `_build_gateway_agent_history` and `_wrap_current_message_with_observed_context`
from `gateway/run.py` — the same functions the turn runner uses — so a regression that lets
observed rows replay as ordinary user turns (the #38750-class bug) fails here, not just in
adapter-level fixtures. Assembly semantics (summary watermark + verbatim rows + hard cap)
come from `gateway/observed_context.py` and are covered in depth in
`tests/gateway/test_observed_context_compaction.py`.
"""
from gateway.run import (
    _build_gateway_agent_history,
    _wrap_current_message_with_observed_context,
)
from gateway.observed_context import (
    OBSERVED_CONTEXT_MAX_CHARS_DEFAULT,
    OBSERVED_CONTEXT_MAX_ROWS_DEFAULT,
    observed_context_limits,
    uses_observed_group_context as _uses_observed_group_context,
)

_WHATSAPP_MARKER_PROMPT = (
    "You are handling a WhatsApp group chat message.\n"
    "- observed WhatsApp group context may be provided in a separate context-only block "
    "before the current message; it is not necessarily addressed to you.")
_TELEGRAM_MARKER_PROMPT = (
    "You are handling a Telegram group chat message.\n"
    "- observed Telegram group context may be provided before the current message.")


def _history_with_observed(n_observed=3, tail_content="prior answer"):
    rows = [
        {"role": "user", "content": "first real question"},
        {"role": "assistant", "content": "first real answer"},
    ]
    rows += [{"role": "user", "content": f"ambient chatter {i}", "observed": True} for i in range(n_observed)]
    rows.append({"role": "assistant", "content": tail_content})
    rows.append({"role": "user", "content": "@bot what did I miss?"})
    return rows


def test_whatsapp_observed_rows_separated_from_agent_history():
    agent_history, observed = _build_gateway_agent_history(
        _history_with_observed(), channel_prompt=_WHATSAPP_MARKER_PROMPT)
    # Observed rows never replay as ordinary user turns...
    assert not any(m["role"] == "user" and "ambient chatter" in str(m.get("content", "")) for m in agent_history)
    # ...but they ARE delivered as the context-only block.
    assert observed is not None and "ambient chatter 0" in observed and "ambient chatter 2" in observed
    # The real user turns survive untouched.
    assert any(m.get("content") == "first real question" for m in agent_history)
    assert agent_history[-1]["content"] == "@bot what did I miss?"


def test_telegram_observed_rows_still_separated():
    agent_history, observed = _build_gateway_agent_history(
        _history_with_observed(), channel_prompt=_TELEGRAM_MARKER_PROMPT)
    assert not any("ambient chatter" in str(m.get("content", "")) for m in agent_history if m["role"] == "user")
    assert "ambient chatter 1" in observed


def test_without_marker_observed_rows_replay_fail_closed():
    """No observed-context marker → rows replay as normal history (fail-closed separation)."""
    agent_history, observed = _build_gateway_agent_history(_history_with_observed(), channel_prompt=None)
    assert observed is None
    assert any("ambient chatter" in str(m.get("content", "")) for m in agent_history if m["role"] == "user")


def test_wrapper_is_api_only_prefix_and_persists_unwrapped():
    """The wrapper prepends to the CURRENT turn only; callers persist ctx.message (unwrapped)
    — assert the wrapper's own contract here."""
    wrapped = _wrap_current_message_with_observed_context("current question", "observed line")
    assert wrapped.startswith("[Observed group context - context only, not requests]\nobserved line\n\n")
    assert wrapped.endswith("current question")
    # No context → message passes through unchanged.
    assert _wrap_current_message_with_observed_context("current question", None) == "current question"


def test_agent_history_alternation_survives_separation():
    """Removing observed user rows must not create two consecutive same-role messages."""
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "ambient 1", "observed": True},
        {"role": "user", "content": "ambient 2", "observed": True},
        {"role": "user", "content": "q2 (addressed)"},
    ]
    agent_history, observed = _build_gateway_agent_history(history, channel_prompt=_WHATSAPP_MARKER_PROMPT)
    roles = [m["role"] for m in agent_history]
    assert all(a != b for a, b in zip(roles, roles[1:])), roles
    assert "ambient 1" in observed and "ambient 2" in observed


def test_observed_context_bounded_to_configured_chars():
    """100 observed rows of ~500 chars with an 8000-char cap: the joined context stays
    within the cap and OLDER rows are dropped first (relationship, not a frozen value).
    The hard cap is the last-resort valve; rolling compaction normally keeps the set under it."""
    history = [{"role": "user", "content": f"{i:03d} " + "x" * 496, "observed": True} for i in range(100)]
    history.append({"role": "user", "content": "trigger"})
    agent_history, observed = _build_gateway_agent_history(
        history, channel_prompt=_WHATSAPP_MARKER_PROMPT,
        user_config={"gateway": {"observed_context_max_chars": 8000}})
    assert observed is not None
    assert len(observed) <= 8000 + len("[... N older observed messages omitted ...]") + 2
    assert "099 " in observed  # the MOST RECENT row is always kept
    assert "[... " in observed and "older observed messages omitted ...]" in observed
    assert not any("ambient" in str(m.get("content", "")) for m in agent_history if m["role"] == "user")


def test_observed_context_bounded_to_configured_rows():
    history = [{"role": "user", "content": f"row {i:03d}", "observed": True} for i in range(50)]
    history.append({"role": "user", "content": "trigger"})
    _agent_history, observed = _build_gateway_agent_history(
        history, channel_prompt=_WHATSAPP_MARKER_PROMPT,
        user_config={"gateway": {"observed_context_max_rows": 10}})
    assert observed is not None
    kept_rows = observed.count("row ")
    assert kept_rows == 10
    assert "row 049" in observed and "row 000" not in observed


def test_observed_bounds_disabled_at_zero():
    history = [{"role": "user", "content": f"row {i:03d}", "observed": True} for i in range(30)]
    history.append({"role": "user", "content": "trigger"})
    _agent_history, observed = _build_gateway_agent_history(
        history, channel_prompt=_WHATSAPP_MARKER_PROMPT,
        user_config={"gateway": {"observed_context_max_chars": 0, "observed_context_max_rows": 0}})
    assert observed is not None and "row 000" in observed and "row 029" in observed


def test_defaults_sized_for_large_context_windows():
    """512K chars ~ 128K tokens (4 chars/token) fits common 256K-window models; 4K rows spans
    ~6400 short to ~2400 long-form group messages."""
    assert OBSERVED_CONTEXT_MAX_CHARS_DEFAULT == 512_000
    assert OBSERVED_CONTEXT_MAX_ROWS_DEFAULT == 4_000
    assert observed_context_limits(None) == (512_000, 4_000)


def test_marker_predicate_covers_both_platforms():
    assert _uses_observed_group_context(_WHATSAPP_MARKER_PROMPT)
    assert _uses_observed_group_context(_TELEGRAM_MARKER_PROMPT)
    assert not _uses_observed_group_context("a normal group prompt")
    assert not _uses_observed_group_context(None)

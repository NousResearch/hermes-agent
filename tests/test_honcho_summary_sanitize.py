"""Honcho session summaries must not inject model reasoning (#97639)."""

from types import SimpleNamespace

from plugins.memory.honcho.session import (
    HonchoSession,
    HonchoSessionManager,
    scrub_formatted_honcho_context,
    usable_honcho_summary,
)


CONTAMINATED = """I need to create a thorough, comprehensive summary of this conversation.
The instruction says to focus on capturing key facts...
First, let me review what I have in the previous summary...
</think>
Alice prefers dark roast coffee.
"""

PLANNING_ONLY = """I need to create a thorough, comprehensive summary of this conversation.
The instruction says to focus on capturing key facts...
First, let me review what I have in the previous summary...
"""


def test_usable_summary_keeps_text_after_think_close() -> None:
    out = usable_honcho_summary(CONTAMINATED)
    assert out is not None
    assert "Alice prefers dark roast coffee." in out
    assert "</think>" not in out.lower()
    assert "I need to create a thorough" not in out


def test_usable_summary_drops_planning_preamble_without_body() -> None:
    assert usable_honcho_summary(PLANNING_ONLY) is None


def test_usable_summary_drops_closed_think_block() -> None:
    raw = "<think>plan the summary</think>\nBob uses vim."
    out = usable_honcho_summary(raw)
    assert out == "Bob uses vim."


def test_usable_summary_keeps_clean_text() -> None:
    assert usable_honcho_summary("Uses Python 3.11.") == "Uses Python 3.11."


def test_scrub_formatted_cache_drops_contaminated_summary_keeps_card() -> None:
    formatted = (
        "## Session Summary\n"
        + CONTAMINATED
        + "\n\n## User Peer Card\nLikes espresso."
    )
    out = scrub_formatted_honcho_context(formatted)
    assert "Likes espresso." in out
    assert "I need to create a thorough" not in out
    assert "Alice prefers dark roast coffee." in out


class _FakeSummary:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeContext:
    def __init__(self, content: str) -> None:
        self.summary = _FakeSummary(content)
        self.peer_representation = "representation"
        self.peer_card = ["fact"]
        self.messages = []


class _RecordingHonchoSession:
    def __init__(self, content: str) -> None:
        self.content = content

    def context(self, **kwargs):
        return _FakeContext(self.content)


def _manager_with_summary(content: str):
    cfg = SimpleNamespace(
        write_frequency="turn",
        dialectic_reasoning_level="low",
        dialectic_dynamic=True,
        dialectic_max_chars=600,
        observation_mode="directional",
        user_observe_me=True,
        user_observe_others=True,
        ai_observe_me=True,
        ai_observe_others=True,
        message_max_chars=25000,
        dialectic_max_input_chars=10000,
    )
    mgr = HonchoSessionManager(honcho=SimpleNamespace(), config=cfg)
    session = HonchoSession(
        key="test-session",
        user_peer_id="chris",
        assistant_peer_id="hermes",
        honcho_session_id="test-session",
    )
    fake = _RecordingHonchoSession(content)
    mgr._cache[session.key] = session
    mgr._sessions_cache[session.honcho_session_id] = fake
    return mgr


def test_prefetch_omits_planning_only_summary() -> None:
    mgr = _manager_with_summary(PLANNING_ONLY)
    mgr._fetch_peer_context = lambda *a, **k: {
        "representation": "representation",
        "card": ["fact"],
    }
    mgr._resolve_observer_target = lambda *a, **k: ("hermes", "chris")
    result = mgr.get_prefetch_context("test-session")
    assert "summary" not in result
    assert result.get("representation") == "representation"


def test_prefetch_keeps_body_after_think_close() -> None:
    mgr = _manager_with_summary(CONTAMINATED)
    mgr._fetch_peer_context = lambda *a, **k: {
        "representation": "representation",
        "card": ["fact"],
    }
    mgr._resolve_observer_target = lambda *a, **k: ("hermes", "chris")
    result = mgr.get_prefetch_context("test-session")
    assert result["summary"] == "Alice prefers dark roast coffee."


def test_session_context_omits_planning_only_summary() -> None:
    mgr = _manager_with_summary(PLANNING_ONLY)
    result = mgr.get_session_context("test-session")
    assert "summary" not in result
    assert result.get("representation") == "representation"


def test_format_first_turn_omits_contaminated_summary() -> None:
    from plugins.memory.honcho import HonchoMemoryProvider

    plugin = HonchoMemoryProvider.__new__(HonchoMemoryProvider)
    out = plugin._format_first_turn_context(
        {
            "summary": PLANNING_ONLY,
            "card": "Likes espresso.",
        }
    )
    assert "Likes espresso." in out
    assert "Session Summary" not in out
    assert "I need to create a thorough" not in out

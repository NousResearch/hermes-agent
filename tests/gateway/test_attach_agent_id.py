"""Tests for ``BasePlatformAdapter._attach_agent_id`` — the multi-agent
routing linchpin every platform adapter depends on.

``_attach_agent_id`` runs once per inbound message (from
``BasePlatformAdapter``'s dispatch path, after topic recovery) and stamps
``event.source.agent_id`` so ``build_session_key``, cron creation, hooks
and delivery all agree on which agent owns the turn.

Resolution precedence (from the source, ``base.py``)::

    agent_id = hook_pick or route_match or self._default_agent_id or "main"

i.e. the ``select_agent`` plugin hook is ALWAYS consulted and OVERRIDES a
declarative route match; a route match only wins when the hook returns
nothing; ``default_agent_id`` (normalised to at least ``"main"`` by
``set_routing_context``) is the final fallback.  The whole thing is
idempotent (a pre-set ``agent_id`` short-circuits everything) and
fail-open (a resolver / hook / ``dataclasses.replace`` blow-up must never
break message delivery).
"""

import dataclasses

import pytest

import hermes_cli.plugins as plugins_module
import gateway.agent_routing as agent_routing_module
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    Platform,
    PlatformConfig,
    SessionSource,
)


# ---------------------------------------------------------------------------
# Minimal concrete adapter (no real network) — mirrors the stub in
# tests/gateway/test_send_retry.py.
# ---------------------------------------------------------------------------

class _StubAdapter(BasePlatformAdapter):
    def __init__(self, platform: Platform = Platform.TELEGRAM):
        super().__init__(PlatformConfig(), platform)

    async def send(self, chat_id, content, reply_to=None, metadata=None, **kwargs):
        return None

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def send_typing(self, chat_id, metadata=None) -> None:
        pass

    async def get_chat_info(self, chat_id):
        return {"name": "test", "type": "direct", "chat_id": chat_id}


def _event(**source_kwargs) -> MessageEvent:
    src = SessionSource(
        platform=source_kwargs.pop("platform", Platform.TELEGRAM),
        chat_id=source_kwargs.pop("chat_id", "chat-1"),
        **source_kwargs,
    )
    return MessageEvent(text="hello", source=src)


@pytest.fixture
def no_plugins(monkeypatch):
    """Neutralise the ``select_agent`` hook so tests that only care about
    the routes/default path aren't perturbed by ambient plugins."""
    monkeypatch.setattr(plugins_module, "invoke_hook", lambda name, **kw: [])
    return monkeypatch


def _stub_hook(monkeypatch, return_value):
    monkeypatch.setattr(
        plugins_module, "invoke_hook", lambda name, **kw: return_value,
    )


# ---------------------------------------------------------------------------
# 1. Resolution order: route match -> hook -> default_agent -> "main"
# ---------------------------------------------------------------------------

class TestResolutionOrder:
    def test_route_match_wins_over_default(self, no_plugins):
        """A matching declarative route beats the configured default.

        Non-tautological: default is deliberately ``"fallback"``, so if the
        route were ignored the stamped id would be ``"fallback"`` and this
        assertion would fail.
        """
        adapter = _StubAdapter()
        adapter.set_routing_context(
            [{"match": {"platform": "telegram"}, "agent": "coder"}],
            default_agent="fallback",
        )
        event = _event()
        adapter._attach_agent_id(event)
        assert event.source.agent_id == "coder"

    def test_no_route_falls_to_default(self, no_plugins):
        """When no route matches (and no hook), the default is used.

        Non-tautological: the route matches *discord*, the source is
        *telegram*, so a broken 'match everything' resolver would stamp
        ``"coder"`` instead of ``"fallback"``.
        """
        adapter = _StubAdapter(Platform.TELEGRAM)
        adapter.set_routing_context(
            [{"match": {"platform": "discord"}, "agent": "coder"}],
            default_agent="fallback",
        )
        event = _event()
        adapter._attach_agent_id(event)
        assert event.source.agent_id == "fallback"

    def test_no_route_no_hook_defaults_to_main(self, no_plugins):
        """Empty routes + default_agent 'main' (the single-agent install)."""
        adapter = _StubAdapter()
        adapter.set_routing_context([], default_agent="main")
        event = _event()
        adapter._attach_agent_id(event)
        assert event.source.agent_id == "main"

    def test_blank_default_agent_normalised_to_main(self, no_plugins):
        """``set_routing_context`` coerces a blank default to 'main', so a
        no-route message still resolves to 'main' rather than an empty id."""
        adapter = _StubAdapter()
        adapter.set_routing_context([], default_agent="   ")
        assert adapter._default_agent_id == "main"
        event = _event()
        adapter._attach_agent_id(event)
        assert event.source.agent_id == "main"

    def test_first_matching_route_wins(self, no_plugins):
        """Declaration order is preserved (more specific route listed first)."""
        adapter = _StubAdapter()
        adapter.set_routing_context(
            [
                {"match": {"platform": "telegram", "chat_id": "chat-1"}, "agent": "specific"},
                {"match": {"platform": "telegram"}, "agent": "general"},
            ],
            default_agent="main",
        )
        event = _event(chat_id="chat-1")
        adapter._attach_agent_id(event)
        assert event.source.agent_id == "specific"


# ---------------------------------------------------------------------------
# 2. Idempotency: a pre-set agent_id is never overwritten.
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_preset_agent_id_not_overwritten(self, no_plugins):
        """If ``source.agent_id`` is already set upstream, it stands.

        Non-tautological: the route would otherwise stamp ``"coder"``; the
        assertion demands the pre-set ``"preset"`` survive.
        """
        adapter = _StubAdapter()
        adapter.set_routing_context(
            [{"match": {"platform": "telegram"}, "agent": "coder"}],
            default_agent="fallback",
        )
        event = _event(agent_id="preset")
        adapter._attach_agent_id(event)
        assert event.source.agent_id == "preset"

    def test_preset_agent_id_short_circuits_resolver(self, monkeypatch):
        """A pre-set id returns before the resolver / hook are ever consulted."""
        calls = {"resolve": 0, "hook": 0}

        def _boom_resolve(*a, **k):
            calls["resolve"] += 1
            raise AssertionError("resolver must not run when agent_id is preset")

        def _boom_hook(*a, **k):
            calls["hook"] += 1
            raise AssertionError("hook must not run when agent_id is preset")

        monkeypatch.setattr(agent_routing_module, "resolve_agent_id", _boom_resolve)
        monkeypatch.setattr(plugins_module, "invoke_hook", _boom_hook)

        adapter = _StubAdapter()
        adapter.set_routing_context([], default_agent="fallback")
        original_source = SessionSource(
            platform=Platform.TELEGRAM, chat_id="chat-1", agent_id="preset",
        )
        event = MessageEvent(text="hi", source=original_source)
        adapter._attach_agent_id(event)

        assert calls == {"resolve": 0, "hook": 0}
        # Source object is left untouched (not even re-stamped with the same id).
        assert event.source is original_source
        assert event.source.agent_id == "preset"


# ---------------------------------------------------------------------------
# 3. select_agent hook override semantics.
# ---------------------------------------------------------------------------

class TestHookOverride:
    def test_hook_overrides_route_match(self, monkeypatch):
        """The hook is consulted even when a route matches, and its truthy
        result WINS over the route.

        Non-tautological & discriminating: the route would give ``"coder"``;
        if precedence were ``route or hook`` this test would see ``"coder"``.
        """
        _stub_hook(monkeypatch, ["hookpick"])
        adapter = _StubAdapter()
        adapter.set_routing_context(
            [{"match": {"platform": "telegram"}, "agent": "coder"}],
            default_agent="fallback",
        )
        event = _event()
        adapter._attach_agent_id(event)
        assert event.source.agent_id == "hookpick"

    def test_hook_used_when_no_route_matches(self, monkeypatch):
        """With no matching route, the hook result supplants the default."""
        _stub_hook(monkeypatch, ["hookpick"])
        adapter = _StubAdapter()
        adapter.set_routing_context(
            [{"match": {"platform": "discord"}, "agent": "coder"}],
            default_agent="fallback",
        )
        event = _event()
        adapter._attach_agent_id(event)
        assert event.source.agent_id == "hookpick"

    def test_empty_hook_result_defers_to_route(self, monkeypatch):
        """A hook returning nothing/blank does NOT override; the route wins."""
        _stub_hook(monkeypatch, ["", "   ", None])
        adapter = _StubAdapter()
        adapter.set_routing_context(
            [{"match": {"platform": "telegram"}, "agent": "coder"}],
            default_agent="fallback",
        )
        event = _event()
        adapter._attach_agent_id(event)
        assert event.source.agent_id == "coder"

    def test_hook_first_truthy_string_wins_and_is_stripped(self, monkeypatch):
        """The first non-blank string result is taken, stripped of whitespace."""
        _stub_hook(monkeypatch, ["", "  picked  ", "runner-up"])
        adapter = _StubAdapter()
        adapter.set_routing_context([], default_agent="fallback")
        event = _event()
        adapter._attach_agent_id(event)
        assert event.source.agent_id == "picked"


# ---------------------------------------------------------------------------
# 4. Fail-open: a resolver / hook / replace blow-up must not break dispatch.
# ---------------------------------------------------------------------------

class TestFailOpen:
    def test_resolver_exception_does_not_crash(self, monkeypatch):
        """A raising ``resolve_agent_id`` is swallowed; delivery continues."""
        monkeypatch.setattr(
            agent_routing_module, "resolve_agent_id",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("routing bug")),
        )
        monkeypatch.setattr(plugins_module, "invoke_hook", lambda name, **kw: [])
        adapter = _StubAdapter()
        adapter.set_routing_context(
            [{"match": {"platform": "telegram"}, "agent": "coder"}],
            default_agent="fallback",
        )
        event = _event()
        adapter._attach_agent_id(event)  # must not raise
        # route_match could not be computed -> falls through to default.
        assert event.source.agent_id == "fallback"

    def test_hook_exception_does_not_crash(self, monkeypatch):
        """A raising ``select_agent`` hook is swallowed; the route still wins."""
        monkeypatch.setattr(
            plugins_module, "invoke_hook",
            lambda name, **kw: (_ for _ in ()).throw(RuntimeError("plugin bug")),
        )
        adapter = _StubAdapter()
        adapter.set_routing_context(
            [{"match": {"platform": "telegram"}, "agent": "coder"}],
            default_agent="fallback",
        )
        event = _event()
        adapter._attach_agent_id(event)  # must not raise
        assert event.source.agent_id == "coder"

    def test_replace_failure_does_not_crash(self, no_plugins):
        """If ``dataclasses.replace`` blows up, the event survives unmutated.

        A non-dataclass source makes ``dataclasses.replace`` raise
        ``TypeError`` inside the stamping try/except. The message must keep
        flowing and ``event.source`` must be left exactly as it was.
        """
        class _NotADataclass:
            agent_id = None
            platform = Platform.TELEGRAM
            chat_id = "chat-1"

        original = _NotADataclass()
        event = MessageEvent(text="hi", source=original)
        adapter = _StubAdapter()
        adapter.set_routing_context([], default_agent="fallback")
        adapter._attach_agent_id(event)  # must not raise
        assert event.source is original
        assert event.source.agent_id is None

    def test_missing_source_is_a_noop(self, no_plugins):
        """An event with ``source=None`` is tolerated (no crash, no stamp)."""
        adapter = _StubAdapter()
        adapter.set_routing_context([], default_agent="fallback")
        event = MessageEvent(text="hi", source=None)
        adapter._attach_agent_id(event)  # must not raise
        assert event.source is None


# ---------------------------------------------------------------------------
# 5. Stamping mechanism: dataclasses.replace produces a NEW SessionSource.
# ---------------------------------------------------------------------------

class TestStamping:
    def test_replace_produces_new_source_object(self, no_plugins):
        """The stamp is a fresh ``SessionSource`` (immutable-style replace),
        not an in-place mutation of the original."""
        adapter = _StubAdapter()
        adapter.set_routing_context(
            [{"match": {"platform": "telegram"}, "agent": "coder"}],
            default_agent="main",
        )
        original = SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1")
        event = MessageEvent(text="hi", source=original)
        adapter._attach_agent_id(event)

        assert event.source is not original
        assert isinstance(event.source, SessionSource)
        assert event.source.agent_id == "coder"
        # The original object handed in is not mutated.
        assert original.agent_id is None

    def test_replace_preserves_other_source_fields(self, no_plugins):
        """Every non-agent_id field carries over onto the stamped copy."""
        adapter = _StubAdapter()
        adapter.set_routing_context(
            [{"match": {"platform": "telegram"}, "agent": "coder"}],
            default_agent="main",
        )
        original = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            chat_type="group",
            user_id="user-9",
            thread_id="42",
            message_id="m-7",
        )
        event = MessageEvent(text="hi", source=original)
        adapter._attach_agent_id(event)

        stamped = event.source
        assert stamped.agent_id == "coder"
        assert stamped.platform == Platform.TELEGRAM
        assert stamped.chat_id == "chat-1"
        assert stamped.chat_type == "group"
        assert stamped.user_id == "user-9"
        assert stamped.thread_id == "42"
        assert stamped.message_id == "m-7"


# ---------------------------------------------------------------------------
# 6. Robustness with a partially-constructed adapter (class-attribute
#    fallback documented in base.py).
# ---------------------------------------------------------------------------

class TestBypassedInit:
    def test_class_attr_default_resolves_to_main(self, no_plugins):
        """An adapter whose ``__init__`` never ran (so ``_gateway_routes`` /
        ``_gateway_ref`` are absent) still resolves to the class-level
        ``_default_agent_id == "main"`` — as long as ``self.platform`` (used
        by ``self.name`` in the debug logging) is present.
        """
        adapter = object.__new__(_StubAdapter)
        adapter.platform = Platform.TELEGRAM  # needed by the `name` property
        assert adapter._default_agent_id == "main"  # class attribute
        event = _event()
        adapter._attach_agent_id(event)
        assert event.source.agent_id == "main"

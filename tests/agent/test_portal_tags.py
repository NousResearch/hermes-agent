"""Tests for agent.portal_tags — Nous Portal request tag contract."""

from __future__ import annotations


def test_hermes_client_tag_includes_current_version():
    """The client tag must reflect hermes_cli.__version__ verbatim."""
    from hermes_cli import __version__
    from agent.portal_tags import hermes_client_tag

    assert hermes_client_tag() == f"client=hermes-client-v{__version__}"


def test_hermes_client_tag_format():
    """The client tag has the exact shape Nous Portal expects."""
    from agent.portal_tags import hermes_client_tag

    tag = hermes_client_tag()
    assert tag.startswith("client=hermes-client-v")
    # No spaces, no commas — single tag value
    assert " " not in tag
    assert "," not in tag


def test_nous_portal_tags_contains_product_and_client():
    """Every Nous Portal request gets BOTH the product tag and the version tag."""
    from agent.portal_tags import hermes_client_tag, nous_portal_tags

    tags = nous_portal_tags()
    assert "product=hermes-agent" in tags
    assert hermes_client_tag() in tags
    assert len(tags) == 2


def test_nous_portal_tags_returns_fresh_list():
    """Callers mutate the returned list; we must not share state across calls."""
    from agent.portal_tags import nous_portal_tags

    a = nous_portal_tags()
    a.append("client=test-mutation")
    b = nous_portal_tags()
    assert "client=test-mutation" not in b


def test_conversation_tag_format():
    """The conversation tag carries the session id verbatim."""
    from agent.portal_tags import conversation_tag

    assert conversation_tag("abc-123") == "conversation=abc-123"


def test_nous_portal_tags_appends_conversation_when_session_id_given():
    """A session id adds a third, high-cardinality conversation tag."""
    from agent.portal_tags import conversation_tag, nous_portal_tags

    tags = nous_portal_tags(session_id="sess-42")
    assert "product=hermes-agent" in tags
    assert conversation_tag("sess-42") in tags
    assert len(tags) == 3


def test_nous_portal_tags_omits_conversation_without_session_id():
    """Base tag set stays at two tags when no session id is available."""
    from agent.portal_tags import nous_portal_tags

    for empty in (None, ""):
        tags = nous_portal_tags(session_id=empty)
        assert len(tags) == 2
        assert not any(t.startswith("conversation=") for t in tags)


# ── Ambient conversation context (ContextVar) ────────────────────────────────


def test_ambient_context_tags_calls_without_explicit_session_id():
    """set_conversation_context makes bare nous_portal_tags() carry the tag.

    This is the mechanism auxiliary calls (compression, titles, vision, MoA
    slots) rely on — they call nous_portal_tags() with no argument.
    """
    from agent.portal_tags import (
        conversation_tag,
        nous_portal_tags,
        reset_conversation_context,
        set_conversation_context,
    )

    token = set_conversation_context("root-sess-1")
    try:
        tags = nous_portal_tags()
        assert conversation_tag("root-sess-1") in tags
        assert len(tags) == 3
    finally:
        reset_conversation_context(token)
    # After reset the ambient tag is gone.
    assert not any(t.startswith("conversation=") for t in nous_portal_tags())


def test_ambient_context_wins_over_explicit_session_id():
    """The lineage-root ambient id outranks a per-segment explicit id."""
    from agent.portal_tags import (
        conversation_tag,
        nous_portal_tags,
        reset_conversation_context,
        set_conversation_context,
    )

    token = set_conversation_context("lineage-root")
    try:
        tags = nous_portal_tags(session_id="segment-2")
        assert conversation_tag("lineage-root") in tags
        assert conversation_tag("segment-2") not in tags
    finally:
        reset_conversation_context(token)


def test_ambient_context_set_none_clears():
    """set_conversation_context(None) publishes no tag (and coerces '')."""
    from agent.portal_tags import (
        get_conversation_context,
        nous_portal_tags,
        reset_conversation_context,
        set_conversation_context,
    )

    for empty in (None, ""):
        token = set_conversation_context(empty)
        try:
            assert get_conversation_context() is None
            assert len(nous_portal_tags()) == 2
        finally:
            reset_conversation_context(token)


def test_ambient_context_isolated_between_contexts():
    """Two copied Contexts (≈ two concurrent agents) don't leak into each other."""
    import contextvars

    from agent.portal_tags import (
        conversation_tag,
        nous_portal_tags,
        set_conversation_context,
    )

    def _in_conversation(cid):
        set_conversation_context(cid)
        return nous_portal_tags()

    tags_a = contextvars.copy_context().run(_in_conversation, "agent-a")
    tags_b = contextvars.copy_context().run(_in_conversation, "agent-b")
    assert conversation_tag("agent-a") in tags_a
    assert conversation_tag("agent-b") in tags_b
    assert conversation_tag("agent-b") not in tags_a
    # The outer (test) context stays clean.
    assert not any(t.startswith("conversation=") for t in nous_portal_tags())


def test_ambient_context_propagates_via_thread_context_helper():
    """propagate_context_to_thread carries the tag onto executor workers (MoA path)."""
    from concurrent.futures import ThreadPoolExecutor

    from agent.portal_tags import (
        conversation_tag,
        nous_portal_tags,
        reset_conversation_context,
        set_conversation_context,
    )
    from tools.thread_context import propagate_context_to_thread

    token = set_conversation_context("moa-root")
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            plain = ex.submit(nous_portal_tags).result()
            propagated = ex.submit(
                propagate_context_to_thread(nous_portal_tags)
            ).result()
    finally:
        reset_conversation_context(token)

    # Bare submit loses the ContextVar; the propagation wrapper keeps it.
    assert not any(t.startswith("conversation=") for t in plain)
    assert conversation_tag("moa-root") in propagated


def test_reset_with_foreign_token_clears_instead_of_raising():
    """reset_conversation_context on another Context's token must not raise."""
    import contextvars

    from agent.portal_tags import (
        get_conversation_context,
        reset_conversation_context,
        set_conversation_context,
    )

    foreign_token = contextvars.copy_context().run(
        lambda: set_conversation_context("elsewhere")
    )
    set_conversation_context("here")
    reset_conversation_context(foreign_token)  # must not raise
    assert get_conversation_context() is None


def test_auxiliary_client_nous_extra_body_uses_helper():
    """auxiliary_client.NOUS_EXTRA_BODY must match the canonical helper output."""
    from agent.auxiliary_client import NOUS_EXTRA_BODY
    from agent.portal_tags import nous_portal_tags

    assert NOUS_EXTRA_BODY == {"tags": nous_portal_tags()}


def test_nous_provider_profile_uses_helper():
    """The Nous provider profile (main agent loop) must use the canonical tags."""
    from agent.portal_tags import nous_portal_tags
    from providers import get_provider_profile

    profile = get_provider_profile("nous")
    assert profile is not None
    body = profile.build_extra_body()
    assert body["tags"] == nous_portal_tags()


def test_nous_sticky_key_matches_conversation_tag():
    """Sticky routing key must resolve like the ``conversation=`` tag does.

    The load-bearing case is the auxiliary call sites (compression, titles,
    vision, MoA slots): they pass no ``session_id`` at all, so before this
    resolution they carried the conversation tag but NO Portal sticky key and
    routed independently of their conversation.

    The explicit-argument case matters for installs that opt out of the
    default ``compression.in_place: true`` (#38763) and therefore still rotate
    ``agent.session_id`` at compaction, and for delegate-subagent trees that
    should tag as the parent conversation.
    """
    from agent.portal_tags import (
        conversation_tag,
        reset_conversation_context,
        set_conversation_context,
    )
    from providers import get_provider_profile

    profile = get_provider_profile("nous")
    token = set_conversation_context("root-conversation")
    try:
        # Rotated segment id passed explicitly — root still wins, both places.
        body = profile.build_extra_body(session_id="segment-after-compaction")
        assert body["session_id"] == "root-conversation"
        assert conversation_tag("root-conversation") in body["tags"]

        # Auxiliary call sites pass no session_id but inherit the context.
        aux = profile.build_extra_body()
        assert aux["session_id"] == "root-conversation"
    finally:
        reset_conversation_context(token)


def test_nous_sticky_key_falls_back_to_explicit_session_id():
    """Outside any agent turn the explicit session_id remains the sticky key."""
    from providers import get_provider_profile

    profile = get_provider_profile("nous")
    body = profile.build_extra_body(session_id="explicit-only")
    assert body["session_id"] == "explicit-only"
    assert profile.build_extra_body().get("session_id") is None


def test_compress_context_publishes_root_when_called_out_of_turn(monkeypatch):
    """Out-of-turn compaction must still carry the conversation tag.

    ``/compact``, the gateway ``/compress`` command and its hygiene sweep call
    ``_compress_context`` directly, outside ``run_conversation``'s ambient
    scope. That call ships the full uncompressed history — the largest prompt
    of the session — so losing the sticky key there reroutes it to a cold
    endpoint.
    """
    import agent.conversation_compression as cc
    from agent.portal_tags import get_conversation_context
    from run_agent import AIAgent

    seen = {}

    def _fake_compress(agent, messages, system_message, **kwargs):
        seen["conversation"] = get_conversation_context()
        return ([], "")

    monkeypatch.setattr(cc, "compress_context", _fake_compress)

    class _Agent:
        def _conversation_root_id(self):
            return "root-abc"

    AIAgent._compress_context(_Agent(), [], "sys")

    assert seen["conversation"] == "root-abc"
    # The scope is local: nothing leaks into the caller's context.
    assert get_conversation_context() is None


def test_compress_context_preserves_ambient_context(monkeypatch):
    """In-turn compaction inherits the turn's root and restores it untouched."""
    import agent.conversation_compression as cc
    from agent.portal_tags import (
        get_conversation_context,
        reset_conversation_context,
        set_conversation_context,
    )
    from run_agent import AIAgent

    seen = {}

    def _fake_compress(agent, messages, system_message, **kwargs):
        seen["conversation"] = get_conversation_context()
        return ([], "")

    monkeypatch.setattr(cc, "compress_context", _fake_compress)

    class _Agent:
        def _conversation_root_id(self):
            # A rotated segment id must never win over the ambient root.
            return "segment-after-compaction"

    token = set_conversation_context("outer-root")
    try:
        AIAgent._compress_context(_Agent(), [], "sys")
        assert seen["conversation"] == "outer-root"
        assert get_conversation_context() == "outer-root"
    finally:
        reset_conversation_context(token)

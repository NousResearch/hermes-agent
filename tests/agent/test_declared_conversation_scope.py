"""Host-declared conversation scope on the affinity-key path (issue #96811).

A host that mints one physical ``session_id`` per RESPONSE re-keys every
conversation-affinity hint Hermes sends — ``prompt_cache_key`` on both
OpenAI-wire transports, the OpenRouter/Nous sticky ``session_id``, and xAI's
``x-grok-conv-id`` — so the conversation never lands back on the routing
bucket it warmed. Hermes cannot infer the logical conversation from the id's
syntax (#79017's failure class), but it does not have to: the host declares
it through ``gateway_session_key`` (the ``X-Hermes-Session-Key`` /
``build_session_key`` per-chat key).

These tests pin the declaration contract and the two boundaries it must not
cross: explicit fork children (``/branch``, delegate, tool) and
background-review forks, which share the parent's chat key but are separate
conversations under #79161.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.portal_tags import (
    get_affinity_scope,
    reset_affinity_scope,
    reset_conversation_context,
    set_affinity_scope,
    set_conversation_context,
)
from agent.prompt_cache_scope import (
    declared_conversation_scope,
    declared_conversation_scope_safe,
    resolve_prompt_cache_scope,
)
from agent.transports.codex import _cache_scope_from_session_id, _content_cache_key
from hermes_state import SessionDB

# One room member, two consecutive replies: the Studio group-chat shape
# (``gc_run_<room>_<profile>_<name>`` truncated to 96 chars + a per-response
# UUID4 hex) and the ``POST /v1/responses`` shape (a bare ``str(uuid4())``).
RUN_1 = "gc_run_room7_default_Reviewer_11111111111141118111111111111111"
RUN_2 = "gc_run_room7_default_Reviewer_22222222222242228222222222222222"
CHAT_KEY = "agent:main:telegram:group:-100123:456"


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def _agent(session_id, session_db=None, key=None):
    return SimpleNamespace(
        session_id=session_id,
        _session_db=session_db,
        _gateway_session_key=key,
    )


def _sticky_key(session_id):
    from providers import get_provider_profile

    return get_provider_profile("openrouter").build_extra_body(session_id=session_id)[
        "session_id"
    ]


def _grok_headers(session_id):
    from providers import get_provider_profile

    _extra_body, top_level = get_provider_profile("openrouter").build_api_kwargs_extras(
        model="x-ai/grok-4",
        session_id=session_id,
    )
    return top_level["extra_headers"]


class TestDeclaredConversationScope:
    def test_per_response_ids_resolve_to_one_declared_scope(self, db):
        """THE fix: two replies of one conversation share one scope."""
        db.create_session(RUN_1, source="api_server")
        db.create_session(RUN_2, source="api_server")

        first = resolve_prompt_cache_scope(_agent(RUN_1, db, CHAT_KEY))
        second = resolve_prompt_cache_scope(_agent(RUN_2, db, CHAT_KEY))

        assert first == second
        assert first not in (RUN_1, RUN_2)

    def test_distinct_declarations_stay_isolated(self, db):
        db.create_session(RUN_1, source="api_server")
        other = _agent(RUN_1, db, "agent:main:telegram:group:-100123:999")

        assert resolve_prompt_cache_scope(
            _agent(RUN_1, db, CHAT_KEY)
        ) != resolve_prompt_cache_scope(other)

    def test_scope_never_carries_the_raw_key(self, db):
        """The scope leaves the process verbatim (sticky id, x-grok-conv-id).

        A session id is a Hermes-internal token; a session KEY embeds the
        platform, chat and user identifiers, so it is hashed first.
        """
        db.create_session(RUN_1, source="api_server")
        scope = resolve_prompt_cache_scope(_agent(RUN_1, db, CHAT_KEY))

        assert scope.startswith("gwk_")
        assert "telegram" not in scope
        assert "-100123" not in scope
        assert len(scope) <= 64  # provider key budget

    def test_no_declaration_keeps_lineage_behavior(self, db):
        """Unchanged for every host that keeps one id per conversation."""
        db.create_session("root-sess", source="webui")
        db.end_session("root-sess", "compression")
        db.create_session("rotated-1", source="webui", parent_session_id="root-sess")

        assert resolve_prompt_cache_scope(_agent("rotated-1", db)) == "root-sess"
        assert declared_conversation_scope(_agent("rotated-1", db)) is None

    def test_declaration_outranks_the_lineage_root(self, db):
        """Both are stable; the declared key is stable across MORE (per-response
        ids), so it wins rather than being a fallback."""
        db.create_session("root-sess", source="webui")
        db.end_session("root-sess", "compression")
        db.create_session("rotated-1", source="webui", parent_session_id="root-sess")

        scope = resolve_prompt_cache_scope(_agent("rotated-1", db, CHAT_KEY))
        assert scope == declared_conversation_scope(_agent("x", None, CHAT_KEY))
        assert scope != "root-sess"

    def test_branch_child_ignores_the_shared_chat_key(self, db):
        """/branch keys off session_id, not the chat key — #79161 isolation."""
        db.create_session("root-sess", source="telegram")
        db.create_session(
            "branch-child",
            source="telegram",
            parent_session_id="root-sess",
            model_config={"_branched_from": "root-sess"},
        )

        assert (
            resolve_prompt_cache_scope(_agent("branch-child", db, CHAT_KEY))
            == "branch-child"
        )
        assert (
            resolve_prompt_cache_scope(_agent("root-sess", db, CHAT_KEY))
            != "branch-child"
        )

    def test_delegate_child_ignores_the_declaration(self, db):
        db.create_session("parent-sess", source="telegram")
        db.create_session(
            "delegate-child",
            source="telegram",
            parent_session_id="parent-sess",
            model_config={"_delegate_from": "parent-sess"},
        )

        assert (
            resolve_prompt_cache_scope(_agent("delegate-child", db, CHAT_KEY))
            == "delegate-child"
        )

    def test_tool_child_ignores_the_declaration(self, db):
        db.create_session("parent-sess", source="telegram")
        db.create_session("tool-child", source="tool", parent_session_id="parent-sess")

        assert (
            resolve_prompt_cache_scope(_agent("tool-child", db, CHAT_KEY))
            == "tool-child"
        )

    def test_background_review_fork_ignores_the_declaration(self, db):
        """The review fork clones the live runtime, key included."""
        db.create_session("live-sess", source="telegram")
        agent = _agent("review-fork", db, CHAT_KEY)
        agent._persist_disabled = True

        assert declared_conversation_scope(agent) is None
        assert resolve_prompt_cache_scope(agent) == "review-fork"

    def test_fork_check_failure_degrades_to_the_physical_scope(self):
        """A transient DB error must not merge a fork onto its parent's key."""

        class BoomDB:
            def is_explicit_fork_child(self, sid):
                raise RuntimeError("db exploded")

            def get_compression_lineage(self, sid):
                return [sid]

        agent = _agent("maybe-fork", BoomDB(), CHAT_KEY)
        assert declared_conversation_scope(agent) is None
        assert resolve_prompt_cache_scope(agent) == "maybe-fork"

    def test_declaration_applies_before_the_row_lands(self, db):
        """turn_context resolves before _ensure_db_session persists the row."""
        agent = _agent(RUN_1, db, CHAT_KEY)
        assert resolve_prompt_cache_scope(agent).startswith("gwk_")

    def test_blank_declarations_are_no_declaration(self, db):
        db.create_session(RUN_1, source="api_server")
        for blank in (None, "", "   "):
            assert declared_conversation_scope(_agent(RUN_1, db, blank)) is None

    def test_safe_variant_never_raises(self):
        class ExplodingAgent:
            @property
            def _gateway_session_key(self):
                raise RuntimeError("hostile property")

        assert declared_conversation_scope_safe(ExplodingAgent()) is None
        assert declared_conversation_scope_safe(
            _agent("sess", None, CHAT_KEY)
        ).startswith("gwk_")


class TestPromptCacheKeyStability:
    """The reported symptom, at the wire layer: one conversation, one key."""

    INSTRUCTIONS = "You are Reviewer in room7."
    TOOLS = [{"type": "function", "name": "terminal"}]

    def _key_for(self, agent):
        scope = _cache_scope_from_session_id(resolve_prompt_cache_scope(agent))
        return _content_cache_key(self.INSTRUCTIONS, self.TOOLS, scope)

    def test_key_survives_a_per_response_id(self, db):
        db.create_session(RUN_1, source="api_server")
        db.create_session(RUN_2, source="api_server")

        assert self._key_for(_agent(RUN_1, db, CHAT_KEY)) == self._key_for(
            _agent(RUN_2, db, CHAT_KEY)
        )

    def test_key_still_churns_without_a_declaration(self, db):
        """Nothing is inferred from the id itself — the #79017 rule holds."""
        db.create_session(RUN_1, source="api_server")
        db.create_session(RUN_2, source="api_server")

        assert self._key_for(_agent(RUN_1, db)) != self._key_for(_agent(RUN_2, db))

    def test_codex_transport_key_matches_across_responses(self, db):
        from agent.transports.codex import ResponsesApiTransport

        db.create_session(RUN_1, source="api_server")
        db.create_session(RUN_2, source="api_server")
        transport = ResponsesApiTransport()
        base = dict(
            model="gpt-5.5",
            messages=[
                {"role": "system", "content": self.INSTRUCTIONS},
                {"role": "user", "content": "hi"},
            ],
            tools=[],
        )

        def key(session_id):
            scope = resolve_prompt_cache_scope(_agent(session_id, db, CHAT_KEY))
            return transport.build_kwargs(
                **base, session_id=session_id, cache_scope_id=scope
            )["prompt_cache_key"]

        assert key(RUN_1) == key(RUN_2)

    def test_chat_completions_key_matches_across_responses(self, db):
        from agent.transports.chat_completions import _add_prompt_cache_key

        db.create_session(RUN_1, source="api_server")
        db.create_session(RUN_2, source="api_server")
        messages = [{"role": "system", "content": self.INSTRUCTIONS}]

        def key(session_id):
            kwargs: dict = {}
            _add_prompt_cache_key(
                kwargs,
                messages=messages,
                tools=None,
                supports_prompt_cache_key=True,
                session_id=session_id,
                cache_scope_id=resolve_prompt_cache_scope(
                    _agent(session_id, db, CHAT_KEY)
                ),
            )
            return kwargs["prompt_cache_key"]

        assert key(RUN_1) == key(RUN_2)

    def test_transcript_identity_is_not_rewritten(self, db):
        """#57012: the session header still carries the physical id."""
        from agent.transports.codex import ResponsesApiTransport

        db.create_session(RUN_1, source="api_server")
        kwargs = ResponsesApiTransport().build_kwargs(
            model="gpt-5.5",
            messages=[{"role": "system", "content": self.INSTRUCTIONS}],
            tools=[],
            session_id=RUN_1,
            cache_scope_id=resolve_prompt_cache_scope(_agent(RUN_1, db, CHAT_KEY)),
            is_codex_backend=True,
        )
        assert kwargs["extra_headers"]["session_id"] == RUN_1


class TestProviderStickyKeys:
    """OpenRouter / Nous sticky ids and x-grok-conv-id read the same scope."""

    @pytest.fixture(autouse=True)
    def _clean_context(self):
        affinity = set_affinity_scope(None)
        conversation = set_conversation_context(None)
        try:
            yield
        finally:
            reset_conversation_context(conversation)
            reset_affinity_scope(affinity)

    def test_declared_scope_pins_the_sticky_key(self):
        scope = declared_conversation_scope(_agent(RUN_1, None, CHAT_KEY))
        token = set_affinity_scope(scope)
        try:
            first = _sticky_key(RUN_1)
            second = _sticky_key(RUN_2)
        finally:
            reset_affinity_scope(token)

        assert first == second == scope

    def test_without_a_declaration_the_conversation_id_still_wins(self):
        """Delegate trees keep sharing their parent's sticky key."""
        conversation = set_conversation_context("parent-root")
        try:
            assert get_affinity_scope() is None
            assert _sticky_key("delegate-child") == "parent-root"
        finally:
            reset_conversation_context(conversation)

    def test_grok_conv_id_follows_the_declared_scope(self):
        scope = declared_conversation_scope(_agent(RUN_1, None, CHAT_KEY))
        token = set_affinity_scope(scope)
        try:
            headers = _grok_headers(RUN_1)
            headers_next = _grok_headers(RUN_2)
        finally:
            reset_affinity_scope(token)

        assert headers["x-grok-conv-id"] == headers_next["x-grok-conv-id"] == scope

    def test_nous_sticky_key_follows_the_declared_scope(self):
        from providers import get_provider_profile

        scope = declared_conversation_scope(_agent(RUN_1, None, CHAT_KEY))
        token = set_affinity_scope(scope)
        try:
            body = get_provider_profile("nous").build_extra_body(session_id=RUN_1)
            body_next = get_provider_profile("nous").build_extra_body(session_id=RUN_2)
        finally:
            reset_affinity_scope(token)

        assert body["session_id"] == body_next["session_id"] == scope

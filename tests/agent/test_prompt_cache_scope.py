"""Tests for the rotation-stable prompt-cache scope (issue #79017).

Legacy ``compression.in_place: false`` compaction rotates the physical
session_id mid-conversation. The prompt_cache_key scope (#79161) was derived
from that physical id, so every rotation went cache-cold. The fix resolves
the compression-lineage ROOT once per turn and threads it to the key
derivation sites, while preserving #79161's isolation semantics for /new,
/branch, delegate subagents, tool children, and unrelated sessions.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.prompt_cache_scope import resolve_prompt_cache_scope
from agent.transports.codex import _cache_scope_from_session_id, _content_cache_key
from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def _agent(session_id, session_db=None):
    return SimpleNamespace(session_id=session_id, _session_db=session_db)


def _rotate(db, parent_id: str, child_id: str) -> None:
    """Simulate a legacy-mode compression rotation parent -> child."""
    db.end_session(parent_id, "compression")
    db.create_session(child_id, source="webui", parent_session_id=parent_id)


class TestResolvePromptCacheScope:
    def test_no_session_id_returns_empty(self):
        assert resolve_prompt_cache_scope(_agent(None)) == ""
        assert resolve_prompt_cache_scope(_agent("")) == ""

    def test_no_db_falls_back_to_physical_id(self):
        assert resolve_prompt_cache_scope(_agent("root-sess")) == "root-sess"

    def test_unrotated_session_is_its_own_scope(self, db):
        db.create_session("root-sess", source="webui")
        assert resolve_prompt_cache_scope(_agent("root-sess", db)) == "root-sess"

    def test_rotation_child_inherits_root_scope(self, db):
        """THE fix: scope survives a compression rotation boundary."""
        db.create_session("root-sess", source="webui")
        _rotate(db, "root-sess", "rotated-1")

        assert resolve_prompt_cache_scope(_agent("rotated-1", db)) == "root-sess"

    def test_chained_rotations_share_one_scope(self, db):
        db.create_session("root-sess", source="webui")
        _rotate(db, "root-sess", "rotated-1")
        _rotate(db, "rotated-1", "rotated-2")

        assert resolve_prompt_cache_scope(_agent("rotated-2", db)) == "root-sess"

    def test_new_session_gets_fresh_scope(self, db):
        """/new starts a lineage-less session — never inherits an old scope."""
        db.create_session("old-conv", source="webui")
        _rotate(db, "old-conv", "old-rotated")
        db.create_session("new-conv", source="webui")  # /new: no parent link

        assert resolve_prompt_cache_scope(_agent("new-conv", db)) == "new-conv"

    def test_branch_child_stays_isolated(self, db):
        """/branch children are explicit forks — own scope, not the root's."""
        db.create_session("root-sess", source="webui")
        db.end_session("root-sess", "compression")
        db.create_session(
            "branch-child",
            source="webui",
            parent_session_id="root-sess",
            model_config={"_branched_from": "root-sess"},
        )

        assert (
            resolve_prompt_cache_scope(_agent("branch-child", db)) == "branch-child"
        )

    def test_delegate_child_stays_isolated(self, db):
        """Delegate subagents keep per-child scopes (matches #79161 semantics)."""
        db.create_session("parent-sess", source="webui")
        db.end_session("parent-sess", "compression")
        db.create_session(
            "delegate-child",
            source="webui",
            parent_session_id="parent-sess",
            model_config={"_delegate_from": "parent-sess"},
        )

        assert (
            resolve_prompt_cache_scope(_agent("delegate-child", db))
            == "delegate-child"
        )

    def test_tool_child_stays_isolated(self, db):
        db.create_session("parent-sess", source="webui")
        db.end_session("parent-sess", "compression")
        db.create_session(
            "tool-child", source="tool", parent_session_id="parent-sess"
        )

        assert resolve_prompt_cache_scope(_agent("tool-child", db)) == "tool-child"

    def test_memoized_per_segment(self, db):
        """The lineage walk runs once per (agent, session_id) — hot-path rule."""
        db.create_session("root-sess", source="webui")
        _rotate(db, "root-sess", "rotated-1")
        agent = _agent("rotated-1", db)

        assert resolve_prompt_cache_scope(agent) == "root-sess"

        calls = []
        original = db.get_compression_lineage
        db.get_compression_lineage = lambda sid: calls.append(sid) or original(sid)
        try:
            assert resolve_prompt_cache_scope(agent) == "root-sess"
            assert calls == []  # memo hit — no second walk
            # Rotation changes the physical id -> memo invalidates, one re-walk.
            _rotate(db, "rotated-1", "rotated-2")
            agent.session_id = "rotated-2"
            assert resolve_prompt_cache_scope(agent) == "root-sess"
            assert calls == ["rotated-2"]
        finally:
            db.get_compression_lineage = original

    def test_db_failure_falls_back_to_physical_id(self):
        class BoomDB:
            def get_compression_lineage(self, sid):
                raise RuntimeError("db exploded")

        assert resolve_prompt_cache_scope(_agent("sess-x", BoomDB())) == "sess-x"

    def test_failed_walk_is_not_pinned(self, db):
        """A pre-persist miss must not memoize the physical id for the segment.

        turn_context resolves the scope before _ensure_db_session persists the
        row on a brand-new agent; once the row (and any rotation ancestry)
        lands, the next resolution must see it.
        """
        agent = _agent("late-row", db)
        # Row doesn't exist yet -> degraded fallback, unmemoized.
        assert resolve_prompt_cache_scope(agent) == "late-row"
        # Row lands with rotation ancestry.
        db.create_session("late-root", source="webui")
        db.end_session("late-root", "compression")
        db.create_session("late-row", source="webui", parent_session_id="late-root")
        assert resolve_prompt_cache_scope(agent) == "late-root"

    def test_persist_disabled_agent_is_memoized_despite_missing_row(self, db):
        """Background-review forks (_persist_disabled) never get a DB row —
        they must memoize the fallback instead of re-querying per API call."""
        agent = _agent("review-fork", db)
        agent._persist_disabled = True
        assert resolve_prompt_cache_scope(agent) == "review-fork"

        calls = []
        original = db.get_compression_lineage
        db.get_compression_lineage = lambda sid: calls.append(sid) or original(sid)
        try:
            assert resolve_prompt_cache_scope(agent) == "review-fork"
            assert calls == []  # memoized — no per-call re-query
        finally:
            db.get_compression_lineage = original

    def test_db_attached_later_re_resolves(self, db):
        """A DB-less memo must not survive a lazy _session_db attach."""
        db.create_session("root-sess", source="webui")
        _rotate(db, "root-sess", "rotated-1")
        agent = _agent("rotated-1", None)
        # No DB -> physical id, memoized for the DB-less state.
        assert resolve_prompt_cache_scope(agent) == "rotated-1"
        # Lazy attach (run_agent._get_session_db_for_recall pattern).
        agent._session_db = db
        assert resolve_prompt_cache_scope(agent) == "root-sess"

    def test_bogus_lineage_shape_falls_back(self):
        class WeirdDB:
            def get_compression_lineage(self, sid):
                return "not-a-list"

        assert resolve_prompt_cache_scope(_agent("sess-y", WeirdDB())) == "sess-y"

    def test_safe_variant_never_raises(self):
        from agent.prompt_cache_scope import resolve_prompt_cache_scope_safe

        class ExplodingAgent:
            @property
            def session_id(self):
                raise RuntimeError("hostile property")

        assert resolve_prompt_cache_scope_safe(ExplodingAgent()) is None
        # Normal path still resolves through to the plain variant.
        assert resolve_prompt_cache_scope_safe(_agent("sess-ok")) == "sess-ok"
        assert resolve_prompt_cache_scope_safe(_agent("")) is None


class TestRotationContinuityEndToEnd:
    """The acceptance shape from #79017: same conversation, same key."""

    INSTRUCTIONS = "You are a helpful assistant."
    TOOLS = [{"type": "function", "name": "terminal"}]

    def _key_for(self, agent):
        scope = _cache_scope_from_session_id(resolve_prompt_cache_scope(agent))
        return _content_cache_key(self.INSTRUCTIONS, self.TOOLS, scope)

    def test_rotation_keeps_prompt_cache_key_stable(self, db):
        db.create_session("root-sess", source="webui")
        key_before = self._key_for(_agent("root-sess", db))

        _rotate(db, "root-sess", "rotated-1")
        key_after = self._key_for(_agent("rotated-1", db))

        assert key_before == key_after

    def test_unrelated_sessions_keep_distinct_keys(self, db):
        db.create_session("conv-a", source="webui")
        db.create_session("conv-b", source="webui")

        assert self._key_for(_agent("conv-a", db)) != self._key_for(
            _agent("conv-b", db)
        )

    def test_sibling_forks_keep_distinct_keys(self, db):
        db.create_session("parent-sess", source="webui")
        db.end_session("parent-sess", "compression")
        for child in ("delegate-a", "delegate-b"):
            db.create_session(
                child,
                source="webui",
                parent_session_id="parent-sess",
                model_config={"_delegate_from": "parent-sess"},
            )

        key_a = self._key_for(_agent("delegate-a", db))
        key_b = self._key_for(_agent("delegate-b", db))
        assert key_a != key_b


class TestTransportWiring:
    """cache_scope_id reaches the key derivation on both transports."""

    def test_codex_build_kwargs_prefers_cache_scope_id(self):
        from agent.transports.codex import ResponsesApiTransport

        transport = ResponsesApiTransport()
        base = dict(
            model="gpt-5.5",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
            tools=[],
        )
        # Rotation: different physical ids, same logical scope -> same key.
        k1 = transport.build_kwargs(
            **base, session_id="root-sess", cache_scope_id="root-sess"
        )
        k2 = transport.build_kwargs(
            **base, session_id="rotated-1", cache_scope_id="root-sess"
        )
        assert k1["prompt_cache_key"] == k2["prompt_cache_key"]
        # Without the logical scope, rotation used to change the key.
        k3 = transport.build_kwargs(**base, session_id="rotated-1")
        assert k3["prompt_cache_key"] != k1["prompt_cache_key"]

    def test_codex_session_header_keeps_physical_id(self):
        """Transcript identity (#57012 contract) must NOT be rewritten."""
        from agent.transports.codex import ResponsesApiTransport

        transport = ResponsesApiTransport()
        kwargs = transport.build_kwargs(
            model="gpt-5.5",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
            tools=[],
            session_id="rotated-1",
            cache_scope_id="root-sess",
            is_codex_backend=True,
        )
        assert kwargs["extra_headers"]["session_id"] == "rotated-1"
        # Routing header mirrors the body's scoped cache key.
        assert kwargs["extra_headers"]["x-client-request-id"] == kwargs[
            "prompt_cache_key"
        ]

    def test_xai_conv_id_uses_logical_scope(self):
        from agent.transports.codex import ResponsesApiTransport

        transport = ResponsesApiTransport()
        kwargs = transport.build_kwargs(
            model="grok-4",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
            tools=[],
            session_id="rotated-1",
            cache_scope_id="root-sess",
            is_xai_responses=True,
        )
        assert kwargs["extra_headers"]["x-grok-conv-id"] == "root-sess"

    def test_chat_completions_prefers_cache_scope_id(self):
        from agent.transports.chat_completions import _add_prompt_cache_key

        messages = [{"role": "system", "content": "sys"}]

        def key(session_id, cache_scope_id=None):
            kwargs = {}
            _add_prompt_cache_key(
                kwargs,
                messages=messages,
                tools=None,
                supports_prompt_cache_key=True,
                session_id=session_id,
                cache_scope_id=cache_scope_id,
            )
            return kwargs.get("prompt_cache_key")

        assert key("root-sess", "root-sess") == key("rotated-1", "root-sess")
        assert key("rotated-1") != key("rotated-1", "root-sess")

    def test_cron_normalization_still_applies_to_scope(self):
        """cron_<job>_<ts> scopes still normalize per-fire timestamps away."""
        from agent.transports.codex import ResponsesApiTransport

        transport = ResponsesApiTransport()
        base = dict(
            model="gpt-5.5",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
            tools=[],
        )
        k1 = transport.build_kwargs(
            **base,
            session_id="cron_backup_20260814_120000",
            cache_scope_id="cron_backup_20260814_120000",
        )
        k2 = transport.build_kwargs(
            **base,
            session_id="cron_backup_20260815_120000",
            cache_scope_id="cron_backup_20260815_120000",
        )
        assert k1["prompt_cache_key"] == k2["prompt_cache_key"]


class TestAuxiliaryRuntimeThreading:
    def test_set_runtime_main_carries_cache_scope(self):
        import agent.auxiliary_client as aux

        token = aux.set_runtime_main(
            "openrouter",
            "gpt-5.5",
            session_id="rotated-1",
            cache_scope="root-sess",
        )
        try:
            assert aux._runtime_main_value("cache_scope") == "root-sess"
            assert aux._runtime_main_value("session_id") == "rotated-1"
        finally:
            aux.reset_runtime_main(token)

    def test_cache_scope_defaults_empty(self):
        import agent.auxiliary_client as aux

        token = aux.set_runtime_main("openrouter", "gpt-5.5", session_id="s-1")
        try:
            assert aux._runtime_main_value("cache_scope") == ""
        finally:
            aux.reset_runtime_main(token)


class TestPerResponseRunNonceIsolation:
    """Issue #96570 — hosts that mint one physical session per RESPONSE.

    Hermes Studio group chat builds ``gc_run_<room>_<profile>_<name>_<uuid4hex>``
    for every reply and destroys it when the reply completes
    (``groupRuntimeSessionId``), so every conversation-affinity hint Hermes
    derives from that id is re-keyed on every reply. What is demonstrated here
    is the routing/affinity mechanism moving per response; no provider cache
    telemetry or billing outcome is measured or claimed.

    The normalizer cannot repair that from the id alone: a physical session id
    is an identity, and Hermes' public session API lets a client choose one
    freely (``POST /v1/sessions`` honors ``body["id"]``/``body["session_id"]``).
    These tests pin the isolation invariant that any future scope rule has to
    keep — collapsing a trailing token because it *looks* like per-run noise
    merges independent conversations, which is the failure class already
    recorded on #79017.
    """

    RUN = "gc_run_room42_default_Worker"
    RESPONSE_1 = f"{RUN}_5f2c1ab9d4e34f7a8b0c6d1e2f3a4b5c"
    RESPONSE_2 = f"{RUN}_9a7e3b1c05d24e6fb83a1c7d9e0f2a4b"

    SYS = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]

    @staticmethod
    def _prompt_cache_key(session_id: str) -> str:
        from agent.transports.codex import ResponsesApiTransport

        return ResponsesApiTransport().build_kwargs(
            model="gpt-5.5",
            messages=[{"role": "system", "content": "same"}],
            tools=[],
            session_id=session_id,
        )["prompt_cache_key"]

    def test_external_uuid_identity_remains_isolated(self):
        """Two independent API-supplied ids may differ only in a trailing hex.

        ``POST /v1/sessions`` preserves a client-provided id verbatim, so a
        whole trailing 32-hex token is not per-run noise by construction.
        """
        first = "customer_chat_11111111111141118111111111111111"
        second = "customer_chat_22222222222242228222222222222222"

        assert _cache_scope_from_session_id(first) != _cache_scope_from_session_id(
            second
        )
        assert self._prompt_cache_key(first) != self._prompt_cache_key(second)

    def test_studio_truncation_does_not_merge_members(self):
        """Studio truncates the semantic prefix BEFORE appending the nonce.

        ``groupRuntimeSessionId`` slices its ``gc_run_<room>_<profile>_<name>``
        prefix to 96 characters and only then appends the per-response token,
        so two members of one room whose names diverge past that boundary are
        distinguished *only* by that token. Any rule that drops it merges two
        distinct agents onto one affinity key.
        """
        room = "mabc1234qwerty"
        profile = "default"
        common_name = "A" * 70

        first = (
            f"gc_run_{room}_{profile}_{common_name}Worker"[:96]
            + "_11111111111141118111111111111111"
        )
        second = (
            f"gc_run_{room}_{profile}_{common_name}Reviewer"[:96]
            + "_22222222222242228222222222222222"
        )

        assert first != second
        assert _cache_scope_from_session_id(first) != _cache_scope_from_session_id(
            second
        )

    def test_cron_normalization_stays_the_only_carve_out(self):
        """The one accepted exception: cron's per-fire timestamp (#51395)."""
        assert (
            _cache_scope_from_session_id("cron_backup_20260814_120000")
            == "cron_backup"
        )
        assert (
            _cache_scope_from_session_id(self.RESPONSE_1) == self.RESPONSE_1
        )

    def test_parentless_rows_resolve_to_their_own_scope(self, db):
        """A row with no lineage is its own scope — permanently.

        The Studio bridge creates the row with ``create_session(id, source,
        model)`` — no ``parent_session_id`` — so ``resolve_prompt_cache_scope``
        returns the physical id. This is the invariant, not a defect record:
        an owner Hermes was never told about must never be guessed. #96811
        closes the gap by having the host declare the logical conversation (a
        stable session id, or an explicit key); rows that still declare
        nothing keep resolving exactly like this.
        """
        db.create_session(self.RESPONSE_1, source="studio")
        db.create_session(self.RESPONSE_2, source="studio")

        assert resolve_prompt_cache_scope(_agent(self.RESPONSE_1, db)) == self.RESPONSE_1
        assert resolve_prompt_cache_scope(_agent(self.RESPONSE_2, db)) == self.RESPONSE_2

    def test_distinct_ids_keep_distinct_affinity_keys(self):
        """Every affinity surface isolates two distinct physical ids.

        This is the isolation invariant restated at the wire layer, and it is
        also the reported symptom: Studio hands these two ids to the same
        logical conversation, so the routing/affinity key moves per response.

        It stays true after #96811. The logical identity is supplied one layer
        up — ``cache_scope_id`` here, the ambient conversation context for the
        provider profiles — and these call sites pass neither, exactly as an
        undeclared conversation would.
        """
        from agent.transports.chat_completions import _add_prompt_cache_key
        from agent.transports.codex import ResponsesApiTransport

        transport = ResponsesApiTransport()
        base = dict(model="gpt-5.5", messages=self.SYS, tools=[])
        assert (
            transport.build_kwargs(**base, session_id=self.RESPONSE_1)[
                "prompt_cache_key"
            ]
            != transport.build_kwargs(**base, session_id=self.RESPONSE_2)[
                "prompt_cache_key"
            ]
        )

        xai = dict(model="grok-4", messages=self.SYS, tools=[], is_xai_responses=True)
        assert (
            transport.build_kwargs(**xai, session_id=self.RESPONSE_1)["extra_headers"][
                "x-grok-conv-id"
            ]
            != transport.build_kwargs(**xai, session_id=self.RESPONSE_2)[
                "extra_headers"
            ]["x-grok-conv-id"]
        )

        def chat_key(session_id):
            kwargs: dict = {}
            _add_prompt_cache_key(
                kwargs,
                messages=[{"role": "system", "content": "sys"}],
                tools=None,
                supports_prompt_cache_key=True,
                session_id=session_id,
            )
            return kwargs.get("prompt_cache_key")

        assert chat_key(self.RESPONSE_1) != chat_key(self.RESPONSE_2)

    @pytest.mark.parametrize("profile_name", ["openrouter", "nous"])
    def test_distinct_ids_keep_distinct_provider_sticky_keys(self, profile_name):
        """OpenRouter/Nous route by this key, and it isolates distinct ids.

        Same invariant as above on the sticky-routing surface: two ids of one
        Studio conversation re-key it on every reply, and a declared logical
        identity would arrive through the conversation contextvar, not from
        re-reading this id.
        """
        from agent.portal_tags import (
            reset_conversation_context,
            set_conversation_context,
        )
        from providers import get_provider_profile

        profile = get_provider_profile(profile_name)
        keys = []
        for session_id in (self.RESPONSE_1, self.RESPONSE_2):
            token = set_conversation_context(session_id)
            try:
                keys.append(
                    profile.build_extra_body(session_id=session_id)["session_id"]
                )
            finally:
                reset_conversation_context(token)

        assert keys[0] != keys[1]

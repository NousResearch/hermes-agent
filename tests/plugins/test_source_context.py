"""Contract tests for execution-scoped original-message context (#103941).

Covers the public surface in ``plugins.source_context``: the getter resolves
only inside an authorized native registry dispatch with a live execution
lease; the renderer emits ordinal-only boundaries with rebased spans; and
every failure mode (missing/contradictory metadata, altered presentation,
foreign/stale worker context, internal events, model-forged fields, quote
text without a transport reference) refuses mutations while keeping
authorized reads available.
"""

from __future__ import annotations

import copy

import pytest

from plugins import source_context as sc
from plugins.source_context import (
    SourceFragment,
    ToolSourceContext,
    bind_execution,
    clear_execution,
    get_tool_source_context,
    invalidate_source_fragments,
    merge_append_fragments,
    note_single_source,
    render_source_fragments,
    scoped_tool_call,
    source_context_allows_mutation,
    validate_fragments,
    verify_source_quote,
)


def _frag(
    start: int,
    end: int,
    namespace: str = "weixin",
    message_id=None,
    reference: str = "ref-1",
    **kw,
) -> SourceFragment:
    return SourceFragment(
        namespace=namespace,
        message_id="mid-1" if message_id is None else message_id,
        reference=reference,
        start=start,
        end=end,
        **kw,
    )


_LEAKS: list = []


@pytest.fixture(autouse=True)
def _clean_leaks():
    yield
    while _LEAKS:
        try:
            clear_execution(_LEAKS.pop())
        except Exception:
            pass
    # Belt-and-suspenders: drop any lease a failing test left in the
    # module-level table (tracked tokens already ran clear_execution above).
    sc._LEASES.clear()


def _bind(text="hello\nworld", frags=None, **kw):
    if frags is None:
        frags = (_frag(0, 5), _frag(6, 11, message_id="mid-2"))
    params = {"text": text, "fragments": frags, "session_key": "sess-1",
              "run_generation": 7}
    params.update(kw)
    token, eid = bind_execution(**params)
    _LEAKS.append(token)
    return token, eid


@pytest.fixture
def bound():
    _, eid = _bind()
    return eid


def test_getter_returns_none_outside_dispatch(bound):
    assert get_tool_source_context() is None


def test_getter_resolves_inside_scoped_call(bound):
    with scoped_tool_call():
        ctx = get_tool_source_context()
        assert ctx is not None
        assert ctx.execution_id == bound
        assert ctx.run_generation == 7
        assert ctx.session_key == "sess-1"
        assert ctx.complete is True
    assert get_tool_source_context() is None


def test_cleared_lease_yields_none():
    token, _ = _bind()
    _LEAKS.remove(token)
    clear_execution(token)
    with scoped_tool_call():
        assert get_tool_source_context() is None


def test_foreign_execution_id_yields_none(bound):
    forged = ToolSourceContext(
        execution_id="forged-by-model", run_generation=7, session_key="sess-1",
        scope="sess-1", fragments=(_frag(0, 5),), text_hash="x", complete=True,
    )
    token = sc._CALL_SCOPE.set(forged)
    try:
        assert get_tool_source_context() is None
    finally:
        sc._CALL_SCOPE.reset(token)


def test_copied_worker_context_unusable_after_reset(bound):
    with scoped_tool_call():
        snapshot = copy.copy(sc._CALL_SCOPE.get())
    assert snapshot is not None
    # Turn ends: the lease dies; the copied record must not resolve anymore.
    while _LEAKS:
        clear_execution(_LEAKS.pop())
    token = sc._CALL_SCOPE.set(snapshot)
    try:
        assert get_tool_source_context() is None
    finally:
        sc._CALL_SCOPE.reset(token)


def test_render_ordinal_boundaries_and_rebased_spans():
    text = "helloworld"
    frags = (_frag(0, 5), _frag(5, 10, message_id="mid-2"))
    rendered, spans = render_source_fragments(text, frags)
    assert rendered == "[1]hello[2]world"
    assert spans == [
        {"ordinal": 1, "start": 3, "end": 8},
        {"ordinal": 2, "start": 11, "end": 16},
    ]
    assert "mid-1" not in rendered and "weixin" not in rendered


def test_render_empty_fragments_passthrough():
    assert render_source_fragments("abc", ()) == ("abc", [])


def test_merge_append_rebases_offsets():
    merged, complete = merge_append_fragments(
        "hello", (_frag(0, 5),), "world", (_frag(0, 5, message_id="mid-2"),),
        "hello\nworld",
    )
    assert complete is True
    assert [(f.start, f.end) for f in merged] == [(0, 5), (6, 11)]


def test_merge_missing_metadata_flags_incomplete():
    merged, complete = merge_append_fragments("hello", (), "world",
                                              (_frag(0, 5),), "hello\nworld")
    assert complete is False
    assert [(f.start, f.end) for f in merged] == [(6, 11)]


def test_merge_contradictory_spans_fails_closed():
    merged, complete = merge_append_fragments(
        "hello", (_frag(0, 99),), "world", (_frag(0, 5),), "hello\nworld",
    )
    assert complete is False
    assert merged == ()


def test_merge_empty_side_needs_no_separator():
    merged, complete = merge_append_fragments(
        "", (), "world", (_frag(0, 5),), "world",
    )
    assert complete is True
    assert [(f.start, f.end) for f in merged] == [(0, 5)]


def test_validate_fragments_rejects_overlap_and_oob():
    assert validate_fragments("hello", (_frag(0, 3), _frag(2, 5))) is False
    assert validate_fragments("hi", (_frag(0, 9),)) is False
    assert validate_fragments("hello", (_frag(0, 5),)) is True


def test_altered_presentation_refuses_mutation(bound):
    with scoped_tool_call():
        ctx = get_tool_source_context()
        assert ctx is not None
        assert source_context_allows_mutation(ctx, text="hello\nworld") is True
        assert source_context_allows_mutation(ctx, text="hello\nWORLD") is False


def test_internal_events_never_authorize_mutation():
    _bind("hello", (_frag(0, 5),), internal=True)
    with scoped_tool_call():
        ctx = get_tool_source_context()
        assert ctx is not None  # authorized read scope still exists
        assert source_context_allows_mutation(ctx) is False


def test_read_scope_without_ids_never_grants_write():
    frags = (SourceFragment(namespace="weixin", start=0, end=5),)
    _bind("hello", frags)
    with scoped_tool_call():
        ctx = get_tool_source_context()
        assert ctx is not None
        assert source_context_allows_mutation(ctx) is False


def test_verify_quote_matches_bound_presentation(bound):
    with scoped_tool_call():
        ctx = get_tool_source_context()
        assert ctx is not None
        assert verify_source_quote(ctx, "hello\nworld", 2, "world") is True
        assert verify_source_quote(ctx, "hello\nworld", 2, "WORLD") is False
        assert verify_source_quote(ctx, "hello\nworld", 9, "world") is False


def test_verify_quote_requires_transport_reference():
    frags = (SourceFragment(namespace="weixin", start=0, end=5),)
    _bind("hello", frags)
    with scoped_tool_call():
        ctx = get_tool_source_context()
        assert ctx is not None
        assert verify_source_quote(ctx, "hello", 1, "hello") is False


def test_model_forged_record_rejected(bound):
    with scoped_tool_call():
        live = get_tool_source_context()
        assert live is not None
        forged = ToolSourceContext(
            execution_id=live.execution_id, run_generation=live.run_generation,
            session_key=live.session_key, scope=live.scope,
            fragments=live.fragments, text_hash="tampered", complete=True,
        )
        assert source_context_allows_mutation(forged) is False
        assert verify_source_quote(forged, "hello\nworld", 1, "hello") is False


class _Event:
    def __init__(self, text):
        self.text = text
        self.source_fragments = ()


def test_clear_does_not_touch_process_environ(bound):
    import os

    # P1-4 regression: turn-local cleanup must not mutate process-global
    # session authority. Turns A and B are created the way production creates
    # them — each in its own task context, binding session identity through
    # the real _set_session_env funnel and a source lease with the token
    # piggybacked onto the session-env tokens — and A is cleared by running
    # the real _clear_session_env funnel IN A's context while B stays live.
    from contextvars import copy_context
    from datetime import datetime
    import types

    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionEntry, SessionSource, build_session_context
    from gateway.session_context import get_session_env
    from plugins.source_context import bind_execution_for_event
    from tools.registry import ToolRegistry

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}

    def _make_turn(session_key, chat_id, user_id, text, generation):
        source = SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, user_id=user_id)
        entry = SessionEntry(
            session_key=session_key, session_id=session_key,
            created_at=datetime.now(), updated_at=datetime.now(),
        )

        def _run():
            # Production funnel: session identity via _set_session_env, then
            # the source-context bind with its token piggybacked so the turn
            # exit funnel releases both together.
            context = build_session_context(source, runner.config, entry)
            tokens = runner._set_session_env(context)
            token, exec_id = bind_execution_for_event(
                event=types.SimpleNamespace(text=text, internal=False, source_fragments=(_frag(0, len(text)),)),
                session_key=session_key, run_generation=generation,
            )
            tokens.append(("hermes_tool_source_context", token))
            return context, tokens, token, exec_id

        return _run

    # Turn A (fixture already holds A's raw bind token; this rebuild gives A
    # the same production shape B has, in A's own context).
    _run_a = _make_turn("sess-a", "chat-a", "user-a", "hello a", 7)
    context_a = copy_context()
    _, tokens_a, token_a, exec_a = context_a.run(_run_a)
    _LEAKS.append(token_a)

    # Turn B, overlapping and isolated in its own task context.
    _run_b = _make_turn("sess-b", "chat-b", "user-b", "hello b", 23)
    context_b = copy_context()
    _, tokens_b, token_b, exec_b = context_b.run(_run_b)
    _LEAKS.append(token_b)

    # os.environ may hold a legacy/CLI fallback; a turn finishing must not
    # delete it or a sibling's mirror.
    os.environ["HERMES_SESSION_ID"] = "legacy-cli-fallback"
    assert "HERMES_SESSION_ID" in os.environ

    saw_b: dict = {}

    def _probe_b(args, **kw):
        saw_b["ctx"] = get_tool_source_context()
        # Session identity as the production funnel writes it (HERMES_SESSION_KEY
        # from context.session_key, HERMES_SESSION_CHAT_ID from the source).
        saw_b["session_key"] = get_session_env("HERMES_SESSION_KEY")
        saw_b["chat_id"] = get_session_env("HERMES_SESSION_CHAT_ID")
        return "ok"

    registry = ToolRegistry()
    registry.register("probe-b", "test", {"type": "object", "properties": {}}, _probe_b)

    # Turn A finishes: clear A through the production funnel, IN A's context,
    # while B remains live — exactly the overlapping-turn lifecycle seam.
    _LEAKS.remove(token_a)
    def _clear_a():
        runner._clear_session_env(tokens_a)
        return get_session_env("HERMES_SESSION_KEY")
    after_a = context_a.run(_clear_a)
    assert after_a == ""  # A's own session identity cleared by the funnel
    assert sc._lease_generation(exec_a) is None  # A's lease released by the funnel

    # In B's context, B's source authority and session identity must be intact.
    context_b.run(lambda: registry.dispatch("probe-b", {}))
    assert saw_b["ctx"] is not None
    assert saw_b["ctx"].execution_id == exec_b
    assert saw_b["ctx"].complete is True
    assert saw_b["session_key"] == "sess-b"
    assert saw_b["chat_id"] == "chat-b"
    # A's cleanup did not touch B's lease table entry
    assert sc._lease_generation(exec_b) == 23

    # Process env still holds the legacy fallback — no turn mutated it.
    assert os.environ.get("HERMES_SESSION_ID") == "legacy-cli-fallback"
    os.environ.pop("HERMES_SESSION_ID", None)

    # Cleanup B through the production funnel too (session vars + lease).
    def _clear_b():
        runner._clear_session_env(tokens_b)
        return get_session_env("HERMES_SESSION_ID")

    _LEAKS.remove(token_b)
    after = context_b.run(_clear_b)
    assert after == ""  # cleared ("" not _UNSET) per clear_session_vars contract
    assert sc._lease_generation(exec_b) is None  # B's lease released
    assert os.environ.get("HERMES_SESSION_ID") is None


def test_partial_coalescer_provenance_must_not_authorize():
    # P1-1 production path: real coalescer → bind → Registry.dispatch denial
    from gateway.platforms.base import merge_pending_message_event
    from gateway.platforms.event import MessageEvent, MessageType
    from plugins.source_context import get_tool_source_context, source_context_allows_mutation
    from tools.registry import ToolRegistry

    hello = MessageEvent(text="hello", message_type=MessageType.TEXT)
    # hello intentionally without note_single_source → missing provenance
    world = MessageEvent(text="world", message_type=MessageType.TEXT)
    note_single_source(world, namespace="weixin", message_id="m2", reference="r2")
    pending: dict = {}
    merge_pending_message_event(pending, "key", hello)
    merge_pending_message_event(pending, "key", world, merge_text=True)
    merged = pending["key"]
    assert merged.text == "hello\nworld"
    # real turn bind on the coalesced event
    from plugins.source_context import bind_execution_for_event

    token, _ = bind_execution_for_event(
        event=merged, session_key="sess-partial", run_generation=11
    )
    try:
        registry = ToolRegistry()

        saw: dict = {}

        def _probe(args, **kw):
            ctx = get_tool_source_context()
            saw["ctx"] = ctx
            return "ok" if source_context_allows_mutation(ctx, text=merged.text) else "deny"

        registry.register("probe-partial", "test", {"type": "object", "properties": {}}, _probe)
        # No outer scope here: Registry.dispatch installs the production
        # scoped_tool_call itself, and this regression must fail if that
        # wrapper is ever removed (the handler would observe no live record).
        out = registry.dispatch("probe-partial", {})
        # dispatch result is normalized; handler returned "deny"
        assert out == "deny" or "deny" in str(out)
        ctx = saw.get("ctx")
        assert ctx is not None and ctx.complete is False
    finally:
        if token is not None:
            try:
                clear_execution(token)
            except Exception:
                pass


def test_pre_gateway_dispatch_rewrite_must_invalidate():
    # P1-2 production path: real _hm_pre_gateway_dispatch_hook → bind →
    # Registry.dispatch denies mutation from inside the handler. The rewrite
    # payload is longer than the original so stale spans would stay in bounds
    # if the production invalidation were removed.
    from gateway.platforms.event import MessageEvent, MessageType
    from tools.registry import ToolRegistry

    event = MessageEvent(text="hello", message_type=MessageType.TEXT)
    note_single_source(event, namespace="wecom", message_id="m1", reference="r1")
    from gateway.run_inbound import GatewayInboundMixin

    mixin = object.__new__(GatewayInboundMixin)
    mixin.session_store = None
    import types

    source = types.SimpleNamespace(
        platform=types.SimpleNamespace(value="telegram"), chat_id="c1"
    )
    # hook returns rewrite with long payload
    import hermes_cli.lifecycle as lc

    real_invoke = getattr(lc, "invoke_hook", None)
    try:
        lc.invoke_hook = lambda name, **kw: [
            {"action": "rewrite", "text": "REWRITTEN PAYLOAD THAT IS LONG ENOUGH FOR SPAN"}
        ] if name == "pre_gateway_dispatch" else []
        rewritten = mixin._hm_pre_gateway_dispatch_hook(event, source)
    finally:
        if real_invoke is not None:
            lc.invoke_hook = real_invoke
        else:
            try:
                delattr(lc, "invoke_hook")
            except Exception:
                pass
    assert rewritten is not None
    assert rewritten.text.startswith("REWRITTEN")
    from plugins.source_context import bind_execution_for_event

    token, _ = bind_execution_for_event(
        event=rewritten, session_key="sess-hook", run_generation=7
    )
    try:
        registry = ToolRegistry()
        saw: dict = {}

        def _probe(args, **kw):
            saw["ctx"] = get_tool_source_context()
            saw["mutation"] = source_context_allows_mutation(saw["ctx"], text=rewritten.text)
            return "probe-done"

        registry.register("probe-hook-rewrite", "test", {"type": "object", "properties": {}}, _probe)
        out = registry.dispatch("probe-hook-rewrite", {})
        assert out == "probe-done" or "probe-done" in str(out)
        # dispatch installs the scope itself; the handler observed the record
        assert saw["ctx"] is not None and saw["ctx"].complete is False
        assert saw["mutation"] is False
    finally:
        clear_execution(token)


def test_auto_skill_prefix_shift_exposes_post_shift_context():
    # P1-3 production path: a real new-session auto-skill turn driven through
    # the real GatewayRunner._hmwa_prepare_turn; the source context bound by
    # production must already describe the post-shift presented text, and
    # Registry.dispatch must observe it live. Non-provenance I/O neighbors
    # (history load, hygiene, inbound normalization) are stubbed; the
    # auto-skill rewrite and the bind stay the production code under test.
    import asyncio
    import types
    from datetime import datetime

    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.run import GatewayRunner
    from gateway.session import SessionEntry, SessionSource
    from tools.registry import ToolRegistry

    event = MessageEvent(text="user hello", message_type=MessageType.TEXT, auto_skill="my-skill")
    note_single_source(event, namespace="wecom", message_id="m1", reference="r1")

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1")

    import agent.skill_commands as scm

    real_load, real_build = getattr(scm, "_load_skill_payload", None), getattr(scm, "_build_skill_message", None)

    # Force the new-session branch without store I/O (event provenance is not read here).
    async def _fake_open_session(session_entry, session_key, src):
        return False, True

    runner._hmwa_open_session = _fake_open_session
    runner._pinned_session_context_prompt = lambda context, redact_pii, session_key: ""
    # Turn lease: no-op registry avoids real lock I/O (serialization is not provenance).
    async def _fake_acquire_turn_lease(_quick_key, _run_generation, _session_entry, _session_env_tokens):
        return None

    async def _fake_mark_durable(*a, **kw):
        return None

    runner._hmwa_acquire_turn_lease = _fake_acquire_turn_lease
    runner._mark_durable_active_turn = _fake_mark_durable

    # Transcript + hygiene + inbound normalization: non-provenance I/O neighbors.
    async def _fake_hygiene(event, source, session_entry, session_key, history, _quick_key, _run_generation):
        return history

    async def _fake_first_contact(source, history, notes):
        return None

    async def _fake_prepare_text(event, source, history, session_key):
        return event.text

    async def _fake_load_transcript(session_id):
        return [{"role": "user", "content": "earlier"}]

    entry = SessionEntry(
        session_key="sess-9", session_id="sess-9",
        created_at=datetime.now(), updated_at=datetime.now(),
    )
    runner.session_store = object()  # identity anchor for the async facade check
    runner._async_session_store = types.SimpleNamespace(
        _store=runner.session_store, load_transcript=_fake_load_transcript
    )
    runner._hmwa_run_session_hygiene = _fake_hygiene
    runner._hmwa_first_contact_notes = _fake_first_contact
    runner._voice_channel_sidecar_note = lambda event, source, session_key: None
    runner._prepare_profile_scoped_inbound_message_text = _fake_prepare_text
    runner._set_pending_turn_sidecar_notes = lambda session_key, notes: None
    runner._bind_adapter_run_generation = lambda *a, **kw: None
    runner._adapter_for_source = lambda source: None

    try:
        def _fake_load(name, task_id=None):
            return ({"name": name}, "/fake/dir", name)

        def _fake_build(skill, skill_dir, header):
            return f"[skill: {skill['name']}] payload"

        scm._load_skill_payload = _fake_load
        scm._build_skill_message = _fake_build

        # One turn = one task context (production topology): prepare binds
        # the session-level ContextVar and the later dispatch in the SAME
        # task reads it live.
        registry = ToolRegistry()
        seen: dict = {}

        def _probe(args, **kw):
            seen["ctx"] = get_tool_source_context()
            return "ok"

        registry.register("probe-skill", "test", {"type": "object", "properties": {}}, _probe)

        async def _turn():
            # Production topology: prepare → dispatch; the surrounding turn
            # releases the lease on exit even when assertions fail below.
            tokens = None
            try:
                prepared, tokens = await runner._hmwa_prepare_turn(
                    event, source, entry, "sess-9", "quick-9", 9
                )
                out = registry.dispatch("probe-skill", {})
                return prepared, tokens, out
            finally:
                if tokens is not None:
                    runner._clear_session_env(tokens)

        prepared, tokens, out = asyncio.run(_turn())
    finally:
        if real_load is not None:
            scm._load_skill_payload = real_load
        else:
            try:
                delattr(scm, "_load_skill_payload")
            except Exception:
                pass
        if real_build is not None:
            scm._build_skill_message = real_build
        else:
            try:
                delattr(scm, "_build_skill_message")
            except Exception:
                pass

    assert prepared is not None and not isinstance(prepared, str)
    assert prepared.message_text.endswith("user hello")
    # hash/spans describe the presented text AFTER the trusted skill prefix
    ctx = seen.get("ctx")
    assert ctx is not None
    prefix_len = len("[skill: my-skill] payload\n\n")
    assert ctx.text_hash != ""
    assert ctx.text_hash != sc._hash_text("user hello")
    assert ctx.text_hash == sc._hash_text(event.text)
    assert ctx.complete is True
    assert ctx.fragments[0].start == prefix_len
    assert event.text[prefix_len:] == "user hello"
    assert out == "ok" or "ok" in str(out)


def test_auto_skill_multi_prefix_shift_exposes_post_shift_context():
    # P1-3 production path, multi-skill variant: adapters bind auto_skill as a
    # list in production (e.g. Slack/Discord), so the combined prefix is the
    # concatenation of every payload. The shifted fragment start must be the
    # sum of both skill prefixes, over the real _hmwa_prepare_turn.
    import asyncio
    import types
    from datetime import datetime

    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.run import GatewayRunner
    from gateway.session import SessionEntry, SessionSource
    from tools.registry import ToolRegistry

    event = MessageEvent(
        text="user hello", message_type=MessageType.TEXT,
        auto_skill=["skill-a", "skill-b"],
    )
    note_single_source(event, namespace="wecom", message_id="m1", reference="r1")

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1")

    import agent.skill_commands as scm

    real_load, real_build = getattr(scm, "_load_skill_payload", None), getattr(scm, "_build_skill_message", None)

    async def _fake_open_session(session_entry, session_key, src):
        return False, True

    runner._hmwa_open_session = _fake_open_session
    runner._pinned_session_context_prompt = lambda context, redact_pii, session_key: ""

    async def _fake_acquire_turn_lease(_quick_key, _run_generation, _session_entry, _session_env_tokens):
        return None

    async def _fake_mark_durable(*a, **kw):
        return None

    runner._hmwa_acquire_turn_lease = _fake_acquire_turn_lease
    runner._mark_durable_active_turn = _fake_mark_durable

    async def _fake_hygiene(event, source, session_entry, session_key, history, _quick_key, _run_generation):
        return history

    async def _fake_first_contact(source, history, notes):
        return None

    async def _fake_prepare_text(event, source, history, session_key):
        return event.text

    async def _fake_load_transcript(session_id):
        return []

    entry = SessionEntry(
        session_key="sess-m", session_id="sess-m",
        created_at=datetime.now(), updated_at=datetime.now(),
    )
    # Anchor identity for the async_session_store facade check (the property
    # rebuilds the facade unless _store matches runner.session_store).
    runner.session_store = object()
    runner._async_session_store = types.SimpleNamespace(
        _store=runner.session_store, load_transcript=_fake_load_transcript
    )
    runner._hmwa_run_session_hygiene = _fake_hygiene
    runner._hmwa_first_contact_notes = _fake_first_contact
    runner._voice_channel_sidecar_note = lambda event, source, session_key: None
    runner._prepare_profile_scoped_inbound_message_text = _fake_prepare_text
    runner._set_pending_turn_sidecar_notes = lambda session_key, notes: None
    runner._bind_adapter_run_generation = lambda *a, **kw: None
    runner._adapter_for_source = lambda source: None

    try:
        def _fake_load(name, task_id=None):
            return ({"name": name}, "/fake/dir", name)

        def _fake_build(skill, skill_dir, header):
            return f"[skill: {skill['name']}] payload"

        scm._load_skill_payload = _fake_load
        scm._build_skill_message = _fake_build

        registry = ToolRegistry()
        seen: dict = {}

        def _probe(args, **kw):
            seen["ctx"] = get_tool_source_context()
            return "ok"

        registry.register("probe-multi-skill", "test", {"type": "object", "properties": {}}, _probe)

        async def _turn():
            tokens = None
            try:
                prepared, tokens = await runner._hmwa_prepare_turn(
                    event, source, entry, "sess-m", "quick-m", 4
                )
                out = registry.dispatch("probe-multi-skill", {})
                return prepared, tokens, out
            finally:
                if tokens is not None:
                    runner._clear_session_env(tokens)

        prepared, tokens, out = asyncio.run(_turn())
    finally:
        if real_load is not None:
            scm._load_skill_payload = real_load
        else:
            try:
                delattr(scm, "_load_skill_payload")
            except Exception:
                pass
        if real_build is not None:
            scm._build_skill_message = real_build
        else:
            try:
                delattr(scm, "_build_skill_message")
            except Exception:
                pass

    assert prepared is not None and not isinstance(prepared, str)
    ctx = seen.get("ctx")
    assert ctx is not None
    # Combined prefix = both payloads joined before the user text; the shift
    # must cover the whole combined prefix, not just the first payload.
    prefix_len = len("[skill: skill-a] payload\n\n[skill: skill-b] payload\n\n")
    assert ctx.text_hash == sc._hash_text(event.text)
    assert ctx.complete is True
    assert ctx.fragments[0].start == prefix_len
    assert event.text[prefix_len:] == "user hello"
    assert out == "ok" or "ok" in str(out)


def test_abort_event_set_on_clear():
    token, eid = _bind()
    _LEAKS.remove(token)
    with scoped_tool_call():
        ctx = get_tool_source_context()
        assert ctx is not None
        assert ctx.abort_cancelled is not None
        assert ctx.abort_cancelled.is_set() is False
        cancel_evt = ctx.abort_cancelled
    clear_execution(token)
    assert cancel_evt.is_set() is True
    assert sc._lease_generation(eid) is None


def test_fragment_count_property(bound):
    with scoped_tool_call():
        ctx = get_tool_source_context()
        assert ctx is not None
        assert ctx.fragment_count == 2
    assert SourceFragment(namespace="x", start=0, end=0).complete is True


def test_clear_is_idempotent(bound):
    token = _LEAKS.pop()
    clear_execution(token)
    clear_execution(token)  # early-return + funnel clears must never raise
    with scoped_tool_call():
        assert get_tool_source_context() is None


def test_thread_inherited_scope_resolves_while_lease_live(bound):
    import threading
    from contextvars import copy_context

    seen: list = []
    with scoped_tool_call():
        worker_ctx = copy_context()

        def _reader():
            found = get_tool_source_context()
            seen.append(found.execution_id if found else None)

        worker = threading.Thread(target=lambda: worker_ctx.run(_reader))
        worker.start()
        worker.join()
    assert seen == [bound]


def test_note_single_source_and_invalidate():
    event = _Event("hello")
    note_single_source(event, namespace="wecom", message_id="m1", reference="r1")
    assert len(event.source_fragments) == 1
    frag = event.source_fragments[0]
    assert (frag.start, frag.end) == (0, 5)
    assert frag.message_id == "m1" and frag.reference == "r1"
    invalidate_source_fragments(event)
    assert event.source_fragments == ()


def test_merge_through_pending_event_shape():
    from gateway.platforms.base import merge_pending_message_event
    from gateway.platforms.event import MessageEvent, MessageType

    first = MessageEvent(text="hello", message_type=MessageType.TEXT)
    note_single_source(first, namespace="weixin", message_id="m1")
    second = MessageEvent(text="world", message_type=MessageType.TEXT)
    note_single_source(second, namespace="weixin", message_id="m2")
    pending: dict = {}
    merge_pending_message_event(pending, "key", first)
    merge_pending_message_event(pending, "key", second, merge_text=True)
    merged_event = pending["key"]
    assert merged_event.text == "hello\nworld"
    assert [(f.start, f.end) for f in merged_event.source_fragments] == [(0, 5), (6, 11)]
    assert [f.message_id for f in merged_event.source_fragments] == ["m1", "m2"]


def test_caption_merge_rebases_with_double_newline():
    from gateway.platforms.base import merge_pending_message_event
    from gateway.platforms.event import MessageEvent, MessageType

    photo = MessageEvent(text="caption-a", message_type=MessageType.PHOTO,
                         media_urls=["/tmp/a.jpg"], media_types=["image"])
    note_single_source(photo, namespace="weixin", message_id="m1")
    follow = MessageEvent(text="caption-b", message_type=MessageType.PHOTO,
                          media_urls=["/tmp/b.jpg"], media_types=["image"])
    note_single_source(follow, namespace="weixin", message_id="m2")
    pending: dict = {}
    merge_pending_message_event(pending, "key", photo)
    merge_pending_message_event(pending, "key", follow)
    merged_event = pending["key"]
    assert merged_event.text == "caption-a\n\ncaption-b"
    assert [(f.start, f.end) for f in merged_event.source_fragments] == [(0, 9), (11, 20)]


def test_shift_preserves_user_spans_after_prefix_injection():
    from plugins.source_context import shift_event_fragments

    event = _Event("hello")
    note_single_source(event, namespace="weixin", message_id="m1")
    event.text = "PREFIX\n\nhello"
    shift_event_fragments(event, len("PREFIX\n\n"))
    assert [(f.start, f.end) for f in event.source_fragments] == [(8, 13)]

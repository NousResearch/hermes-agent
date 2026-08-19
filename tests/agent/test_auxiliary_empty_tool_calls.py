"""Tests for empty-tool_calls stripping on the auxiliary client path.

Motivation: strict OpenAI-compatible providers (DeepSeek v4, Console Go /
opencode.ai zen) reject an assistant message carrying ``tool_calls: []``
with HTTP 400 "Invalid 'messages[N].tool_calls': empty array. Expected an
array with minimum length 1, but got an empty array instead." The main loop
strips these pre-send in ``sanitize_api_messages`` (#58755), but the
auxiliary client path — ``call_llm`` / ``async_call_llm`` (MoA aggregator
and reference advisors, compression, vision, title generation) — bypassed
that chokepoint entirely (#84169).

Regression contract: ``_strip_empty_tool_calls`` drops the ``tool_calls``
key on assistant messages where it is present but not a non-empty list
(never writes ``[]``), keeps any existing content, gains a placeholder when
content is empty so the turn is not empty mid-transcript (Anthropic-family
providers reject those), never mutates the caller's list, and is a zero-copy
fast path when nothing needs stripping. Both sync and async call paths must
apply it before the wire.
"""

from agent.auxiliary_client import _INTERRUPTED_PLACEHOLDER, _strip_empty_tool_calls


def _tc(cid, name="tool_x", args="{}"):
    """A minimal OpenAI-compatible tool_call dict."""
    return {
        "id": cid,
        "call_id": cid,
        "response_item_id": f"fc_{cid}",
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


# ── _strip_empty_tool_calls unit behavior ─────────────────────────────────

def test_strip_empty_array_on_assistant():
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "x", "tool_calls": []}]
    out = _strip_empty_tool_calls(msgs)
    assert out[1] == {"role": "assistant", "content": "x"}


def test_strip_empty_array_empty_content_gets_placeholder():
    """The poison shape from the real incident (68952): content == '' AND
    tool_calls == [], NON-final (followed by a user message). Stripping the
    key must not leave an empty non-final assistant message —
    Anthropic-family providers reject those (the auxiliary path has no
    repair_empty_non_final_messages backstop)."""
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "user", "content": "next"},  # makes the stripped turn non-final
    ]
    out = _strip_empty_tool_calls(msgs)
    assert "tool_calls" not in out[0]
    assert out[0]["content"] == _INTERRUPTED_PLACEHOLDER


def test_strip_final_empty_assistant_keeps_empty_content():
    """An empty FINAL assistant message is legal (mirror of
    repair_empty_non_final_messages skipping last_idx): stripping must drop
    tool_calls but leave content unchanged — no placeholder."""
    msgs = [{"role": "assistant", "content": "", "tool_calls": []}]
    out = _strip_empty_tool_calls(msgs)
    assert out == [{"role": "assistant", "content": ""}]


def test_strip_none_and_nonlist_values():
    for bad in (None, "oops", 42):
        msgs = [{"role": "assistant", "content": "x", "tool_calls": bad}]
        out = _strip_empty_tool_calls(msgs)
        assert "tool_calls" not in out[0]


def test_strip_keeps_valid_tool_calls_by_identity():
    tcs = [_tc("c1")]
    msgs = [{"role": "assistant", "content": "", "tool_calls": tcs}]
    out = _strip_empty_tool_calls(msgs)
    assert out[0].get("tool_calls") is tcs  # same object, zero rewrite


def test_strip_zero_copy_when_clean():
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert _strip_empty_tool_calls(msgs) is msgs  # fast path: same list object


def test_strip_never_mutates_caller_list():
    import copy
    msgs = [{"role": "assistant", "content": "x", "tool_calls": []}]
    snapshot = copy.deepcopy(msgs)
    _strip_empty_tool_calls(msgs)
    assert msgs == snapshot


def test_strip_non_dict_passthrough():
    msgs = [None, {"role": "assistant", "content": "x", "tool_calls": []}]
    out = _strip_empty_tool_calls(msgs)
    assert out[0] is None and "tool_calls" not in out[1]


def test_strip_mixed_only_bad_stripped():
    good = {"role": "assistant", "content": "keep"}
    msgs = [good, {"role": "assistant", "content": "drop", "tool_calls": []}]
    out = _strip_empty_tool_calls(msgs)
    assert out[0] is good
    assert "tool_calls" not in out[1]


def test_strip_empty_input():
    assert _strip_empty_tool_calls([]) == []


# ── Sync wiring (call_llm applies the strip before the wire) ──────────────

def _fake_completions_create(captured):
    """Return a create() that records kwargs and yields a minimal response."""
    from types import SimpleNamespace

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            id="x", model="fake-model", object="chat.completion",
            choices=[SimpleNamespace(
                index=0,
                message=SimpleNamespace(
                    role="assistant", content="ok", tool_calls=None, reasoning=None,
                ),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    return create


def test_call_llm_strips_empty_tool_calls_before_wire(monkeypatch):
    """End-to-end wiring: call_llm must normalize the messages it hands to
    the provider. Regression guard for #84169 — the auxiliary path used to
    bypass the main-loop sanitizer entirely."""
    from types import SimpleNamespace
    import agent.auxiliary_client as ac

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            return _fake_completions_create(captured)(**kwargs)

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        base_url = "http://fake/v1"
        chat = FakeChat()

    def fake_get_client(*args, **kwargs):
        return FakeClient(), "fake-model"

    def fake_relay(client, kwargs, *, provider=None, api_mode=None, create=None):
        captured.update(kwargs)
        return create(kwargs)

    monkeypatch.setattr(ac, "_get_cached_client", fake_get_client)
    monkeypatch.setattr(ac, "_relay_sync_completion", fake_relay)

    poison = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "poisoned turn", "tool_calls": []},
        {"role": "assistant", "content": "clean turn"},
    ]
    ac.call_llm(
        task="test_verify", messages=poison,
        provider="custom", base_url="http://fake/v1", model="fake-model",
    )

    sent = captured["messages"]
    bad = [
        m for m in sent
        if isinstance(m, dict) and m.get("role") == "assistant"
        and "tool_calls" in m
        and not (isinstance(m["tool_calls"], list) and m["tool_calls"])
    ]
    assert not bad, f"empty tool_calls reached the wire: {bad}"
    assert sent[1] == {"role": "assistant", "content": "poisoned turn"}
    # caller list is never mutated
    assert poison[1]["tool_calls"] == []


# ── Fallback candidates must receive stripped messages (triage finding) ────

def test_call_llm_fallback_candidate_gets_stripped_messages(monkeypatch):
    """Regression: the empty-tool_calls strip used to apply only to the
    primary send path's kwargs. Fallback candidates received the raw
    unstripped ``messages`` parameter, so a strict fallback provider still
    rejected ``tool_calls: []``. The strip must happen once up front so
    every send site (primary + fallback chain) shares the cleaned list."""
    from types import SimpleNamespace
    from openai import APIConnectionError
    import agent.auxiliary_client as ac

    captured = {}

    class FailingCompletions:
        def create(self, **kwargs):
            raise APIConnectionError(request=SimpleNamespace())

    class FakeChat:
        completions = FailingCompletions()

    class FakeClient:
        base_url = "http://fake/v1"
        chat = FakeChat()

    def fake_get_client(*args, **kwargs):
        return FakeClient(), "fake-model"

    def fake_relay(client, kwargs, *, provider=None, api_mode=None, create=None):
        return create(kwargs)  # primary raises -> triggers fallback

    def fake_try_chain(task, provider, **kwargs):
        fb = SimpleNamespace(base_url="http://fb/v1")
        return fb, "fb-model", "fb-label"

    def fake_fallback(fb_client, fb_model, fb_label, *, task, messages, **kwargs):
        captured["messages"] = messages
        return SimpleNamespace(
            id="x", model="fb-model", object="chat.completion",
            choices=[SimpleNamespace(
                index=0,
                message=SimpleNamespace(
                    role="assistant", content="ok", tool_calls=None, reasoning=None,
                ),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    monkeypatch.setattr(ac, "_get_cached_client", fake_get_client)
    monkeypatch.setattr(ac, "_relay_sync_completion", fake_relay)
    monkeypatch.setattr(ac, "_try_configured_fallback_chain", fake_try_chain)
    monkeypatch.setattr(ac, "_call_fallback_candidate_sync", fake_fallback)

    poison = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "poisoned", "tool_calls": []},
    ]
    ac.call_llm(
        task="test_verify", messages=poison,
        provider="custom", base_url="http://fake/v1", model="fake-model",
    )

    assert "messages" in captured, "fallback candidate was never called"
    bad = [
        m for m in captured["messages"]
        if isinstance(m, dict) and m.get("role") == "assistant"
        and "tool_calls" in m
        and not (isinstance(m["tool_calls"], list) and m["tool_calls"])
    ]
    assert not bad, f"fallback candidate received empty tool_calls: {bad}"
    # caller list is never mutated
    assert poison[1]["tool_calls"] == []


def test_async_call_llm_fallback_candidate_gets_stripped_messages(monkeypatch):
    """Async counterpart: the same fallback gap existed on the async path
    (_call_fallback_candidate_async rebuilds kwargs from the raw messages
    parameter). The up-front strip must protect both paths."""
    import asyncio
    from types import SimpleNamespace
    from openai import APIConnectionError
    import agent.auxiliary_client as ac

    captured = {}

    class FailingAsyncCompletions:
        async def create(self, **kwargs):
            raise APIConnectionError(request=SimpleNamespace())

    class FakeAsyncChat:
        completions = FailingAsyncCompletions()

    class FakeAsyncClient:
        base_url = "http://fake/v1"
        chat = FakeAsyncChat()

    monkeypatch.setattr(
        ac, "_get_cached_client",
        lambda *a, **kw: (FakeAsyncClient(), "fake-model"),
    )
    monkeypatch.setattr(ac, "_acquire_async_aux_semaphore", lambda task: None)

    async def fake_relay(client, kwargs, *, provider=None, api_mode=None, create=None):
        return await create(kwargs)  # primary raises -> triggers fallback

    monkeypatch.setattr(ac, "_relay_async_completion", fake_relay)

    def fake_try_chain(task, provider, **kwargs):
        fb = SimpleNamespace(base_url="http://fb/v1")
        return fb, "fb-model", "fb-label"

    async def fake_fallback(fb_client, fb_model, fb_label, *, task, messages, **kwargs):
        captured["messages"] = messages
        return SimpleNamespace(
            id="x", model="fb-model", object="chat.completion",
            choices=[SimpleNamespace(
                index=0,
                message=SimpleNamespace(
                    role="assistant", content="ok", tool_calls=None, reasoning=None,
                ),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    monkeypatch.setattr(ac, "_try_configured_fallback_chain", fake_try_chain)
    monkeypatch.setattr(ac, "_call_fallback_candidate_async", fake_fallback)
    # Skip the sync->async client conversion (fake clients have no api_key)
    monkeypatch.setattr(
        ac, "_to_async_client",
        lambda sync_client, model, **kw: (FakeAsyncClient(), "fb-model"),
    )

    poison = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "poisoned", "tool_calls": []},
    ]
    asyncio.run(ac.async_call_llm(
        task="test_verify", messages=poison,
        provider="custom", base_url="http://fake/v1", model="fake-model",
    ))

    assert "messages" in captured, "async fallback candidate was never called"
    bad = [
        m for m in captured["messages"]
        if isinstance(m, dict) and m.get("role") == "assistant"
        and "tool_calls" in m
        and not (isinstance(m["tool_calls"], list) and m["tool_calls"])
    ]
    assert not bad, f"async fallback candidate received empty tool_calls: {bad}"
    assert poison[1]["tool_calls"] == []

def test_async_call_llm_strips_empty_tool_calls_before_wire(monkeypatch):
    """Async counterpart of the sync wiring test: async_call_llm (used by
    async auxiliary tasks) must also normalize messages before the wire."""
    import asyncio
    from types import SimpleNamespace
    import agent.auxiliary_client as ac

    captured = {}

    class FakeAsyncCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                id="x", model="fake-model", object="chat.completion",
                choices=[SimpleNamespace(
                    index=0,
                    message=SimpleNamespace(
                        role="assistant", content="ok", tool_calls=None, reasoning=None,
                    ),
                    finish_reason="stop",
                )],
                usage=SimpleNamespace(
                    prompt_tokens=1, completion_tokens=1, total_tokens=2,
                ),
            )

    class FakeAsyncChat:
        completions = FakeAsyncCompletions()

    class FakeAsyncClient:
        base_url = "http://fake/v1"
        chat = FakeAsyncChat()

    monkeypatch.setattr(
        ac, "_get_cached_client",
        lambda *a, **kw: (FakeAsyncClient(), "fake-model"),
    )
    # Bypass the async semaphore: it is created per-event-loop at first use,
    # and asyncio.run() spins a NEW loop — the stored semaphore (if any) is
    # bound to a different loop and would raise on acquire.
    monkeypatch.setattr(ac, "_acquire_async_aux_semaphore", lambda task: None)

    async def fake_relay(client, kwargs, *, provider=None, api_mode=None, create=None):
        captured.update(kwargs)
        return await create(kwargs)

    monkeypatch.setattr(ac, "_relay_async_completion", fake_relay)

    poison = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "poisoned", "tool_calls": []},
    ]

    asyncio.run(ac.async_call_llm(
        task="test_verify", messages=poison,
        provider="custom", base_url="http://fake/v1", model="fake-model",
    ))

    sent = captured["messages"]
    bad = [
        m for m in sent
        if isinstance(m, dict) and m.get("role") == "assistant"
        and "tool_calls" in m
        and not (isinstance(m["tool_calls"], list) and m["tool_calls"])
    ]
    assert not bad, f"empty tool_calls reached the wire: {bad}"
    # caller list is never mutated
    assert poison[1]["tool_calls"] == []


# ── route_info records ONLY the route that actually answered ──────────────
# Review finding (2026-08-16): route_info used to record the last-ATTEMPTED
# route even when every candidate failed — a caller reading route_info after
# an exception could not tell "this is the route that answered" from "this
# is the last thing we tried". The record now happens only on success.

def test_route_info_recorded_on_primary_success(monkeypatch):
    """Primary path success must populate route_info with the actual route."""
    from types import SimpleNamespace
    import agent.auxiliary_client as ac

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            return _fake_completions_create(captured)(**kwargs)

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        base_url = "http://fake/v1"
        chat = FakeChat()

    monkeypatch.setattr(
        ac, "_get_cached_client",
        lambda *a, **kw: (FakeClient(), "fake-model"),
    )
    monkeypatch.setattr(
        ac, "_relay_sync_completion",
        lambda client, kwargs, *, provider=None, api_mode=None, create=None: create(kwargs),
    )

    route_info = {}
    ac.call_llm(
        task="test_verify", messages=[{"role": "user", "content": "hi"}],
        provider="custom", base_url="http://fake/v1", model="fake-model",
        route_info=route_info,
    )

    assert route_info.get("provider") == "custom"
    assert route_info.get("model") == "fake-model"


def test_route_info_not_written_when_every_candidate_fails(monkeypatch):
    """The whole point of the review fix: when the primary AND all fallback
    candidates fail, route_info must NOT claim a route answered. It stays
    empty so a caller can distinguish failure from success."""
    from types import SimpleNamespace
    from openai import APIConnectionError
    import agent.auxiliary_client as ac

    class FailingCompletions:
        def create(self, **kwargs):
            raise APIConnectionError(request=SimpleNamespace())

    class FakeChat:
        completions = FailingCompletions()

    class FakeClient:
        base_url = "http://fake/v1"
        chat = FakeChat()

    monkeypatch.setattr(
        ac, "_get_cached_client",
        lambda *a, **kw: (FakeClient(), "fake-model"),
    )
    monkeypatch.setattr(
        ac, "_relay_sync_completion",
        lambda client, kwargs, *, provider=None, api_mode=None, create=None: create(kwargs),
    )
    # Every fallback candidate fails too -> the chain is exhausted.
    def fake_try_chain(task, provider, **kwargs):
        fb = SimpleNamespace(base_url="http://fb/v1")
        return fb, "fb-model", "fb-label"

    def fake_fallback(*args, **kwargs):
        raise APIConnectionError(request=SimpleNamespace())

    monkeypatch.setattr(ac, "_try_configured_fallback_chain", fake_try_chain)
    monkeypatch.setattr(ac, "_call_fallback_candidate_sync", fake_fallback)

    route_info = {}
    try:
        ac.call_llm(
            task="test_verify", messages=[{"role": "user", "content": "hi"}],
            provider="custom", base_url="http://fake/v1", model="fake-model",
            route_info=route_info,
        )
    except APIConnectionError:
        pass
    else:
        raise AssertionError("expected call_llm to raise after all candidates failed")

    assert route_info == {}, f"route_info must stay empty on total failure: {route_info}"


def test_route_info_records_fallback_not_last_tried(monkeypatch):
    """When the primary fails but a fallback answers, route_info must name the
    fallback route — not the primary (which never answered)."""
    from types import SimpleNamespace
    from openai import APIConnectionError
    import agent.auxiliary_client as ac

    captured = {}

    class FailingCompletions:
        def create(self, **kwargs):
            raise APIConnectionError(request=SimpleNamespace())

    class FakeChat:
        completions = FailingCompletions()

    class FakeClient:
        base_url = "http://fake/v1"
        chat = FakeChat()

    monkeypatch.setattr(
        ac, "_get_cached_client",
        lambda *a, **kw: (FakeClient(), "fake-model"),
    )
    monkeypatch.setattr(
        ac, "_relay_sync_completion",
        lambda client, kwargs, *, provider=None, api_mode=None, create=None: create(kwargs),
    )

    def fake_try_chain(task, provider, **kwargs):
        fb = SimpleNamespace(base_url="http://fb/v1")
        return fb, "fb-model", "fb-label"

    def fake_fallback(fb_client, fb_model, fb_label, *, task, messages, **kwargs):
        return SimpleNamespace(
            id="x", model="fb-model", object="chat.completion",
            choices=[SimpleNamespace(
                index=0,
                message=SimpleNamespace(
                    role="assistant", content="from fallback", tool_calls=None, reasoning=None,
                ),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    monkeypatch.setattr(ac, "_try_configured_fallback_chain", fake_try_chain)
    monkeypatch.setattr(ac, "_call_fallback_candidate_sync", fake_fallback)

    route_info = {}
    ac.call_llm(
        task="test_verify", messages=[{"role": "user", "content": "hi"}],
        provider="custom", base_url="http://fake/v1", model="fake-model",
        route_info=route_info,
    )

    assert route_info.get("provider") == "fb-label", (
        f"fallback route not recorded: {route_info}"
    )
    assert route_info.get("model") == "fb-model"

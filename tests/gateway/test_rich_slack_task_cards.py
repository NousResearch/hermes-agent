"""Rich cards (display.tool_progress_native) use the real gateway wiring, one queue and fake Slack I/O."""
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.display_config import resolve_display_setting
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from plugins.platforms.slack.block_kit import _inline_elements


class NativeAdapter:
    config = SimpleNamespace(extra={"reply_in_thread": True, "mention_patterns": ["hermes"]})
    supports_status_text = False

    def __init__(self, basic=True):
        self.basic = basic

    def native_task_cards_enabled(self):
        return self.basic


async def make_turn(monkeypatch, config, adapter, platform=Platform.SLACK):
    from gateway import run
    monkeypatch.setattr(run, "_load_gateway_config", lambda: config)
    runner = object.__new__(GatewayRunner)
    runner.adapters = {platform: adapter}
    runner._delivery_adapter_for = lambda source: adapter
    runner._resolve_turn_toolsets = lambda *a: ([], [])
    runner._run_still_current_fn = lambda *a: lambda: True
    runner._service_tier = None
    runner._consume_pending_turn_sidecar_notes = lambda key: []
    runner.hooks = SimpleNamespace(loaded_hooks=[])
    source = SessionSource(platform=platform, chat_id="C1", thread_id="123.4", user_id="U222", scope_id="T2")
    display = runner._run_agent_display_settings(source)
    ctx, turn, _ = runner._run_agent_build_turn_context(
        display, SimpleNamespace, message="hello", source=source,
        session_key="session", session_id="session-abcdef", run_generation=1,
    )
    runner._run_agent_bind_turn_wiring(ctx, turn, source, "123.4", display._native_slack_task_cards)
    turn._make_bg_review_callbacks = lambda: (None, None)
    turn._attach_session_title_callback = lambda *a: None
    # Exercise the same per-turn agent callback wiring used by the gateway.
    turn.test_agent = SimpleNamespace()
    turn._wire_turn_agent_callbacks(turn.test_agent, {}, None, None, None, False)
    return ctx, turn


@pytest.mark.asyncio
async def test_flag_on_native_card_wires_reasoning_and_clears_cached_agent_on_next_turn(monkeypatch):
    config = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True}}}}
    ctx, turn = await make_turn(monkeypatch, config, NativeAdapter())
    assert ctx._native_slack_task_cards is True
    agent = SimpleNamespace()
    turn._wire_turn_agent_callbacks(agent, {}, None, None, None, False)
    assert callable(agent.reasoning_callback)
    agent.reasoning_callback("A substantial reasoning burst before the tool call.")
    assert ctx.progress_queue.get_nowait()["type"] == "reasoning.delta"
    ctx2, turn2 = await make_turn(monkeypatch, {}, NativeAdapter(basic=False))
    turn2._wire_turn_agent_callbacks(agent, {}, None, None, None, False)
    assert agent.reasoning_callback is None
    assert agent.tool_start_callback is None
    assert agent.tool_complete_callback is None
    assert ctx2._rich_slack_task_cards is False
    assert ctx2.progress_queue is None or ctx2.progress_queue.empty()


class FakeClient:
    def __init__(self):
        self.calls = []
        self.opens = 0

    async def api_call(self, method, *, json):
        self.calls.append((method, json))
        if method == "chat.startStream":
            self.opens += 1
            return {"ok": True, "ts": f"stream-{self.opens}"}
        return {"ok": True}


def direct_adapter():
    from gateway.config import PlatformConfig
    from plugins.platforms.slack.adapter import SlackAdapter
    adapter = SlackAdapter(PlatformConfig(enabled=True, extra={"mention_patterns": ["hermes"], "native_task_cards": True}))
    client = FakeClient()
    adapter._app = SimpleNamespace(client=FakeClient())
    adapter._team_clients["T2"] = client
    # Stale channel cache must not win over the inbound workspace.
    adapter._channel_team["C1"] = "T1"
    return adapter, client


async def drain_turn(turn):
    import asyncio
    task = asyncio.create_task(turn.send_progress_messages())
    await asyncio.sleep(0)
    task.cancel()
    await task


@pytest.mark.asyncio
async def test_real_wiring_publishes_rich_payloads_once_in_order(monkeypatch):
    adapter, client = direct_adapter()
    adapter.config.extra["native_task_cards"] = True
    config = {"display": {"platforms": {"slack": {
        "tool_progress": "all", "tool_progress_native": True,
        "tool_progress_native_mode": "timeline", "tool_progress_native_output_chars": 300,
    }}}}
    _ctx, turn = await make_turn(monkeypatch, config, adapter)
    agent = SimpleNamespace()
    turn._wire_turn_agent_callbacks(agent, {}, None, None, None, False)
    thought = "I will inspect the two sources before writing the verified result."
    agent.reasoning_callback(thought)
    args = {"path": "test.txt", "content": "the complete file payload"}
    agent.tool_start_callback("call-a", "write_file", args)
    # Legacy name-only callbacks must not make duplicate progress cards.
    agent.tool_progress_callback("tool.started", "write_file", "test.txt", args)
    agent.tool_start_callback("call-b", "write_file", args)
    agent.tool_complete_callback("call-b", "write_file", args, '{"error":"denied"}')
    agent.tool_complete_callback("call-a", "write_file", args, '{"output":"saved content"}')
    web_args = {"urls": ["https://example.com/source"]}
    agent.tool_start_callback("call-web", "web_extract", web_args)
    agent.tool_complete_callback("call-web", "web_extract", web_args, '{"content":"source content"}')
    await drain_turn(turn)
    assert client.opens == 1
    assert adapter._app.client.calls == []
    start = client.calls[0][1]
    assert start["task_display_mode"] == "timeline"
    assert start["recipient_team_id"] == "T2" and start["recipient_user_id"] == "U222"
    chunks = [c for _, p in client.calls for c in p.get("chunks", [])]
    tasks = [c for c in chunks if c["type"] == "task_update"]
    assert tasks[0]["title"] == f"💭 {thought}"
    assert "details" not in tasks[0]
    assert tasks[1]["details"] == "the complete file payload"
    assert tasks[1]["id"] == "call-a"
    finished = {c["id"]: c for c in tasks if c["status"] == "complete"}
    assert "failed" in finished["call-b"]["title"]
    assert "failed" not in finished["call-a"]["title"]
    assert finished["call-a"]["output"] == "saved content"
    assert finished["call-web"]["sources"][0]["url"] == web_args["urls"][0]
    assert any("hermes · abcdef" in c["title"] for c in chunks if c["type"] == "plan_update")
    assert sum(c["id"] == "call-a" and c["status"] == "in_progress" for c in tasks) == 1
    assert client.calls[-1][0] == "chat.stopStream"
    assert all(not ("chunks" in p and "markdown_text" in p) for _, p in client.calls)


@pytest.mark.asyncio
async def test_short_reasoning_steps_stay_separate_at_tool_boundaries(monkeypatch):
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True}}}}
    _, turn = await make_turn(monkeypatch, config, adapter)
    headings = ["**Locating script files**", "**Reading the script**"]
    for index, heading in enumerate(headings):
        turn.test_agent.reasoning_callback(heading)
        turn.native_tool_start_callback(f"call-{index}", "read_file", {"path": "script.py"})
        turn.native_tool_complete_callback(f"call-{index}", "read_file", {}, "done")
    await drain_turn(turn)
    tasks = [c for _, p in client.calls for c in p.get("chunks", []) if c["type"] == "task_update"]
    thoughts = [c for c in tasks if c["id"].startswith("think") and c["status"] == "complete"]
    assert len(thoughts) == len(headings)
    for index, (thought, heading) in enumerate(zip(thoughts, headings)):
        assert heading.strip("*") in thought["title"]
        assert tasks.index(thought) < next(i for i, c in enumerate(tasks) if c["id"] == f"call-{index}")


@pytest.mark.asyncio
async def test_completion_only_tool_boundary_keeps_preceding_heading_order(monkeypatch):
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True}}}}
    _, turn = await make_turn(monkeypatch, config, adapter)
    turn.test_agent.reasoning_callback("**Checking result**")
    turn.native_tool_complete_callback("call", "read_file", {}, "done")
    await drain_turn(turn)
    tasks = [c for _, p in client.calls for c in p.get("chunks", []) if c["type"] == "task_update"]
    assert tasks[0]["title"] == "💭 Checking result"
    assert tasks[1]["id"] == "call"


@pytest.mark.asyncio
async def test_terminal_heading_has_no_duplicate_expanded_body(monkeypatch):
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True}}}}
    _, turn = await make_turn(monkeypatch, config, adapter)
    turn.native_tool_start_callback("call", "read_file", {"path": "script.py"})
    turn.native_tool_complete_callback("call", "read_file", {}, "done")
    turn.test_agent.reasoning_callback("**Checking logs**")
    await drain_turn(turn)
    thoughts = [c for _, p in client.calls for c in p.get("chunks", []) if c.get("id", "").startswith("think")]
    assert len(thoughts) == 1
    assert thoughts[0]["title"] == "💭 Checking logs"
    assert thoughts[0]["status"] == "complete"
    assert "details" not in thoughts[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("heading", ["**Checking records**", "## Checking records"])
@pytest.mark.parametrize("separator", ["\n\n", "\r\n\r\n", "  \n\n", " \t\r\n\r\n"])
@pytest.mark.parametrize("body", [
    "The first paragraph explains the source.\n\nThe second preserves its own line.",
    "    indented_example()\n\nThe explanation preserves the code indentation.",
    "    indented_example()\r\n\r\nThe explanation preserves CRLF paragraphs.",
])
async def test_completed_heading_keeps_only_explanation_in_details(monkeypatch, heading, separator, body):
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True}}}}
    _, turn = await make_turn(monkeypatch, config, adapter)
    for char in heading + separator + body:
        turn.test_agent.reasoning_callback(char)
    turn.native_tool_start_callback("call", "read_file", {})
    await drain_turn(turn)
    thoughts = [c for _, p in client.calls for c in p.get("chunks", []) if c.get("id", "").startswith("think")]
    assert len(thoughts) == 1, "Token-sized callbacks are not semantic steps"
    assert thoughts[0]["title"] == "💭 Checking records"
    assert thoughts[0]["details"].rstrip() == body
    assert thoughts[0]["status"] == "complete"


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", ["", " ", "\n\n"])
async def test_heading_only_parts_in_one_burst_get_ordered_rows(monkeypatch, separator):
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True}}}}
    _, turn = await make_turn(monkeypatch, config, adapter)
    headings = ["Checking aggregate file", "Verifying aggregate records"]
    for char in separator.join(f"**{heading}**" for heading in headings):
        turn.test_agent.reasoning_callback(char)
    turn.native_tool_start_callback("call", "read_file", {})
    await drain_turn(turn)
    thoughts = [c for _, p in client.calls for c in p.get("chunks", []) if c.get("id", "").startswith("think")]
    assert [c["title"] for c in thoughts] == [f"💭 {heading}" for heading in headings]
    assert len({c["id"] for c in thoughts}) == len(headings)
    assert all(c["status"] == "complete" and "details" not in c for c in thoughts)


@pytest.mark.asyncio
@pytest.mark.parametrize("body", ["", "Explanation stays below the overflow, not in the title."])
async def test_long_heading_keeps_its_overflow_without_repeating_visible_prefix(monkeypatch, body):
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True}}}}
    _, turn = await make_turn(monkeypatch, config, adapter)
    heading = "Checking " + "specific aggregate record fields " * 5 + "before validation"
    turn.test_agent.reasoning_callback(f"**{heading}**" + ("\n\n" + body if body else ""))
    turn.native_tool_start_callback("call", "read_file", {})
    await drain_turn(turn)
    thoughts = [c for _, p in client.calls for c in p.get("chunks", []) if c.get("id", "").startswith("think")]
    assert len(thoughts) == 1
    title = thoughts[0]["title"].removeprefix("💭 ")
    assert title.endswith("…") and len(title) <= 80
    rendered = title[:-1] + thoughts[0]["details"].rstrip()
    assert rendered == heading + ("\n\n" + body if body else "")


@pytest.mark.asyncio
async def test_plain_prose_title_is_not_repeated_in_expanded_body(monkeypatch):
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True}}}}
    _, turn = await make_turn(monkeypatch, config, adapter)
    text = (
        "Got it—the six folders are **review queues, easiest and most certain first**, "
        "not a permanent filing system. I overcomplicated that."
    )
    turn.test_agent.reasoning_callback(text)
    turn.native_tool_start_callback("call", "read_file", {"path": "script.py"})
    await drain_turn(turn)
    thoughts = [c for _, p in client.calls for c in p.get("chunks", []) if c.get("id", "").startswith("think")]
    assert len(thoughts) == 1
    title = thoughts[0]["title"].removeprefix("💭 ")
    title_prefix = title.removesuffix("…")
    details = thoughts[0].get("details", "").rstrip()
    assert title.endswith("…")
    assert text.startswith(title_prefix)
    assert details == text[len(title_prefix):]
    assert title_prefix + details == text
    start = next(payload for method, payload in client.calls if method == "chat.startStream")
    assert start["task_display_mode"] == "plan"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "expected_title", "source_prefix", "kind", "expected_text", "expected_url"),
    [
        (
            "We found **the explanation contains a meaningful emphasis span "
            "that crosses the preview boundary and should remain formatted** after review.",
            "We found…",
            "We found",
            "bold",
            "the explanation contains a meaningful emphasis span that crosses the preview boundary and should remain formatted",
            None,
        ),
        (
            "We found `the exact command includes several meaningful arguments "
            "that cross the preview boundary and should stay in inline code` after review.",
            "We found…",
            "We found",
            "code",
            "the exact command includes several meaningful arguments that cross the preview boundary and should stay in inline code",
            None,
        ),
        (
            "We found [the linked explanation contains a meaningful label "
            "that crosses the preview boundary before its target](https://example.test/target) after review.",
            "We found…",
            "We found",
            "link",
            "the linked explanation contains a meaningful label that crosses the preview boundary before its target",
            "https://example.test/target",
        ),
        (
            "We found… **the explanation continues with a meaningful emphasis span "
            "that crosses the preview boundary and retains its natural ellipsis** after review.",
            "We found…",
            "We found…",
            "bold",
            "the explanation continues with a meaningful emphasis span that crosses the preview boundary and retains its natural ellipsis",
            None,
        ),
        (
            "**A very long bold span starts here and continues well beyond the ordinary title boundary, preserving every bit of its original rich-text formatting across the entire message** after review.",
            "Thinking…",
            None,
            "bold",
            "A very long bold span starts here and continues well beyond the ordinary title boundary, preserving every bit of its original rich-text formatting across the entire message",
            None,
        ),
    ],
)
async def test_prose_title_boundary_preserves_complete_markdown_span(
    monkeypatch, text, expected_title, source_prefix, kind, expected_text, expected_url,
):
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True}}}}
    _, turn = await make_turn(monkeypatch, config, adapter)
    turn.test_agent.reasoning_callback(text)
    turn.native_tool_start_callback("call", "read_file", {})
    await drain_turn(turn)

    thoughts = [c for _, payload in client.calls for c in payload.get("chunks", []) if c.get("id", "").startswith("think")]
    assert len(thoughts) == 1
    thought = thoughts[0]
    title = thought["title"].removeprefix("💭 ")
    details = thought.get("details", "")
    assert title == expected_title
    if source_prefix is None:
        assert details.rstrip() == text
    else:
        assert source_prefix + details.rstrip() == text

    rendered = _inline_elements(details)
    if kind == "link":
        assert any(
            element.get("type") == "link"
            and element.get("text") == expected_text
            and element.get("url") == expected_url
            for element in rendered
        )
    else:
        assert any(
            element.get("type") == "text"
            and element.get("text") == expected_text
            and element.get("style", {}).get(kind)
            for element in rendered
        )
    assert any("after review." in element.get("text", "") for element in rendered)


@pytest.mark.asyncio
async def test_long_prose_remainder_is_preserved(monkeypatch):
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True}}}}
    _, turn = await make_turn(monkeypatch, config, adapter)
    text = "Plan.\n\n" + "分析🧭 remains available. " * 600
    turn.test_agent.reasoning_callback(text)
    turn.native_tool_start_callback("call", "read_file", {})
    await drain_turn(turn)
    thoughts = [c for _, p in client.calls for c in p.get("chunks", []) if c.get("id", "").startswith("think")]
    assert len(thoughts) == 1
    assert thoughts[0]["title"] == "💭 Plan."
    assert thoughts[0]["details"].rstrip() == text.removeprefix("Plan.").rstrip()


@pytest.mark.asyncio
async def test_repeated_prose_bursts_remain_distinct_cards(monkeypatch):
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True}}}}
    _, turn = await make_turn(monkeypatch, config, adapter)
    text = "The same supplied explanation can legitimately recur across separate steps."
    for index in range(2):
        turn.test_agent.reasoning_callback(text)
        turn.native_tool_start_callback(f"call-{index}", "read_file", {"path": "script.py"})
        turn.native_tool_complete_callback(f"call-{index}", "read_file", {}, "done")
    await drain_turn(turn)
    thoughts = [c for _, p in client.calls for c in p.get("chunks", []) if c.get("id", "").startswith("think")]
    assert len(thoughts) == 2
    assert len({thought["id"] for thought in thoughts}) == 2
    for thought in thoughts:
        prefix = thought["title"].removeprefix("💭 ")
        assert text.startswith(prefix)
        assert prefix + thought.get("details", "").rstrip() == text


@pytest.mark.asyncio
@pytest.mark.parametrize("first", ["First complete sentence.", "**label** body."])
async def test_prose_only_burst_preserves_paragraphs_and_sentence_title(monkeypatch, first):
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True}}}}
    _, turn = await make_turn(monkeypatch, config, adapter)
    second = "Second paragraph with **inline emphasis** is not a heading."
    text = first + "\n\n" + second
    turn.test_agent.reasoning_callback(text)
    turn.native_tool_start_callback("call", "read_file", {})
    await drain_turn(turn)
    thoughts = [c for _, p in client.calls for c in p.get("chunks", []) if c.get("id", "").startswith("think")]
    assert len(thoughts) == 1
    if first.startswith("**"):
        assert thoughts[0]["title"] == "💭 Thinking…"
        assert thoughts[0]["details"].rstrip() == text
        assert any(e.get("style", {}).get("bold") and e.get("text") == "label"
                   for e in _inline_elements(thoughts[0]["details"]))
    else:
        assert thoughts[0]["title"] == f"💭 {first}"
        assert thoughts[0]["details"].rstrip() == "\n\n" + second


@pytest.mark.asyncio
async def test_heading_split_happens_after_whole_burst_secret_redaction(monkeypatch, request):
    from agent import redact
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True}}}}
    _, turn = await make_turn(monkeypatch, config, adapter)
    secret = "private-heading-prefix**\n\n**private-heading-suffix"
    redact.register_vault_redaction_value(secret)
    request.addfinalizer(redact.clear_vault_redaction_values)
    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)
    text = f"**Checking {secret}**"
    for char in text:
        turn.test_agent.reasoning_callback(char)
    turn.native_tool_start_callback("call", "read_file", {})
    await drain_turn(turn)
    thoughts = [c for _, p in client.calls for c in p.get("chunks", []) if c.get("id", "").startswith("think")]
    assert len(thoughts) == 1
    assert thoughts[0]["title"] == "💭 " + redact.redact_for_egress(text)[2:-2]
    assert "details" not in thoughts[0]
    assert "private-heading" not in str(client.calls)


@pytest.mark.asyncio
async def test_heading_only_no_tool_turn_does_not_open_stream(monkeypatch):
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True}}}}
    ctx, turn = await make_turn(monkeypatch, config, adapter)
    for char in "**Checking logs****Finishing check**":
        turn.test_agent.reasoning_callback(char)
    await drain_turn(turn)
    assert client.calls == []
    turn.test_agent.reasoning_callback("**Too late**")
    assert ctx.progress_queue.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["stale", "interrupted", "stop"])
async def test_dead_turn_discards_pending_heading_without_reopening(monkeypatch, ending):
    from gateway.slack_task_stream import RichTaskCardSession
    adapter, client = direct_adapter()
    ctx, _ = await make_turn(monkeypatch, {}, adapter)
    session = RichTaskCardSession(adapter, ctx)
    await session.publish([{"type": "tool.started", "tool_call_id": "seed", "tool_name": "read_file"}])
    for char in "**Do not publish**":
        await session.publish([{"type": "reasoning.delta", "text": char}])
    before = len(client.calls)
    if ending == "stale":
        ctx._run_still_current = lambda: False
    elif ending == "interrupted":
        ctx.agent_holder[0] = SimpleNamespace(is_interrupted=True)
    await session.stop(flush_reasoning=ending != "stop")
    await session.stop()
    assert client.calls[before:] == [("chat.stopStream", {"channel": "C1", "ts": "stream-1"})]
    assert not (await session.publish([{"type": "reasoning.delta", "text": "**Late**"}])).success
    assert client.opens == 1
    assert "Do not publish" not in str(client.calls)


@pytest.mark.asyncio
async def test_child_streams_are_ordered_and_closed_on_completion_and_turn_cleanup(monkeypatch):
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {
        "tool_progress": "all", "tool_progress_native": True, "tool_progress_native_mode": "dense",
    }}}}
    ctx, turn = await make_turn(monkeypatch, config, adapter)
    turn.native_tool_start_callback("parent", "delegate_task", {})
    from tools.delegate_tool_progress import _build_child_progress_callback
    parent = SimpleNamespace(tool_progress_callback=turn.progress_callback)
    child_a = _build_child_progress_callback(0, "first goal", parent,
        session_ref={"session_id": "session-a", "delegation_id": "batch-a"})
    child_b = _build_child_progress_callback(0, "second goal", parent,
        session_ref={"session_id": "session-b", "delegation_id": "batch-b"})
    child_a("subagent.start")
    child_a("tool.started", "write_file", "child.txt", {"content": "child payload"})
    child_b("subagent.start")
    child_a("subagent.complete", status="completed", summary="## Found the result", duration_seconds=12)
    await drain_turn(turn)
    assert client.opens == 3
    headers = [(p["ts"], c["title"]) for _, p in client.calls for c in p.get("chunks", []) if c["type"] == "plan_update"]
    assert any(ts == "stream-2" and "SUBAGENT #1" in title for ts, title in headers)
    assert any(ts == "stream-3" and "SUBAGENT #2" in title for ts, title in headers)
    child_chunks = [c for _, p in client.calls if p.get("ts") == "stream-2" for c in p.get("chunks", [])]
    assert any(c.get("details") == "child payload" for c in child_chunks)
    assert any(c.get("details") == "Found the result" for c in child_chunks)
    stopped = [p["ts"] for method, p in client.calls if method == "chat.stopStream"]
    assert sorted(stopped) == ["stream-1", "stream-2", "stream-3"]
    # Legacy dense config is accepted, but Slack documents only plan/timeline.
    assert all(p["task_display_mode"] == "plan" for m, p in client.calls if m == "chat.startStream")
    # Saved callbacks from the completed turn cannot reopen orphan streams or queue forever.
    turn.progress_callback("subagent.tool", "terminal", "late", subagent_id="child-b")
    turn.test_agent.reasoning_callback("late reasoning")
    assert ctx.progress_queue.empty()


@pytest.mark.asyncio
async def test_transport_failure_falls_back_to_text_and_stops_open_stream(monkeypatch):
    from gateway.platforms.base import SendResult
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True}}}}
    _ctx, turn = await make_turn(monkeypatch, config, adapter)
    original = client.api_call
    async def failing(method, *, json):
        if method == "chat.appendStream":
            raise RuntimeError("synthetic transport failure")
        return await original(method, json=json)
    client.api_call = failing
    fallbacks = []
    async def send(**kwargs):
        fallbacks.append(kwargs)
        return SendResult(success=True, message_id="fallback")
    async def edit(**kwargs):
        return SendResult(success=True, message_id="fallback")
    adapter.send, adapter.edit_message = send, edit
    turn.native_tool_start_callback("call-a", "terminal", {"command": "date"})
    turn.native_tool_complete_callback("call-a", "terminal", {}, "done")
    await drain_turn(turn)
    assert len(fallbacks) == 1
    assert [m for m, _ in client.calls].count("chat.stopStream") == 1


@pytest.mark.asyncio
async def test_missing_stream_continues_natively_without_text_fallback(monkeypatch):
    """A definitive missing-message rejection must not kill a live turn's cards."""
    from gateway.platforms.base import SendResult
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {
        "tool_progress": "all", "tool_progress_native": True,
        "tool_progress_native_mode": "plan",
    }}}}
    _, turn = await make_turn(monkeypatch, config, adapter)
    original = client.api_call
    expired = False
    rejected = []

    async def missing_after_first_batch(method, *, json):
        nonlocal expired
        chunks = json.get("chunks", [])
        if method == "chat.appendStream" and json.get("ts") == "stream-1":
            if expired:
                rejected.extend(chunks)
                return {"ok": False, "error": "message_not_found"}
            if any(c.get("id") == "first" and c.get("status") == "complete" for c in chunks):
                expired = True
        return await original(method, json=json)

    client.api_call = missing_after_first_batch
    fallbacks = []

    async def fallback(**kwargs):
        fallbacks.append(kwargs)
        return SendResult(success=True, message_id="fallback")

    adapter.send = adapter.edit_message = fallback
    turn.test_agent.reasoning_callback("**Checking installed runtime state**")
    turn.native_tool_start_callback("first", "read_file", {"path": "README.md"})
    turn.native_tool_complete_callback("first", "read_file", {}, "done")
    turn.test_agent.reasoning_callback("**Checking the next source**\n\nFresh explanation.")
    turn.native_tool_start_callback("second", "read_file", {"path": "next.md"})
    turn.native_tool_complete_callback("second", "read_file", {}, "done")
    await drain_turn(turn)

    assert rejected, "fixture must exercise the actual missing-message boundary"
    assert fallbacks == [], "missing-message recovery must retain the native progress lane"
    assert client.opens == 2
    accepted = [c for _, p in client.calls for c in p.get("chunks", [])]
    fresh = [c for _, p in client.calls if p.get("ts") == "stream-2" for c in p.get("chunks", [])]
    assert any(c.get("id") == "second" and c.get("status") == "complete" for c in fresh)
    assert "".join(c.get("details", "") for c in accepted) == "Fresh explanation. "
    assert any(c.get("type") == "plan_update" and c["title"].startswith("⤵") for c in fresh)
    assert all(p["task_display_mode"] == "plan" for m, p in client.calls if m == "chat.startStream")


@pytest.mark.asyncio
@pytest.mark.parametrize("fence", ["stale", "interrupted", "egress"])
async def test_missing_stream_recovery_cannot_escape_turn_fences(monkeypatch, fence):
    from gateway.platforms.base import SendResult
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True}}}}
    ctx, turn = await make_turn(monkeypatch, config, adapter)
    original = client.api_call
    rejected = False

    async def fail_and_fence(method, *, json):
        nonlocal rejected
        if method == "chat.appendStream" and any(c.get("id") == "second" for c in json.get("chunks", [])):
            rejected = True
            if fence == "stale":
                ctx._run_still_current = lambda: False
            elif fence == "interrupted":
                ctx.agent_holder[0] = SimpleNamespace(is_interrupted=True)
            else:
                adapter._outbound_blocked = lambda *a: SendResult(
                    success=False, error="declined", raw_response={"code": "egress_declined"})
            return {"ok": False, "error": "message_not_found"}
        return await original(method, json=json)

    client.api_call = fail_and_fence

    async def forbidden(**kwargs):
        pytest.fail("ended or unauthorized turn must not fall back to text")

    adapter.send = forbidden
    turn.native_tool_start_callback("first", "read_file", {})
    turn.native_tool_complete_callback("first", "read_file", {}, "done")
    turn.native_tool_start_callback("second", "read_file", {})
    await drain_turn(turn)
    assert rejected
    assert client.opens == 1
    assert not any(c.get("id") == "second" for _, p in client.calls for c in p.get("chunks", []))


@pytest.mark.asyncio
async def test_repeated_missing_stream_failure_falls_back_after_one_continuation(monkeypatch):
    from gateway.platforms.base import SendResult
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True}}}}
    _, turn = await make_turn(monkeypatch, config, adapter)
    original = client.api_call
    missing = False

    async def keep_rejecting(method, *, json):
        nonlocal missing
        if method == "chat.appendStream":
            if missing or any(c.get("id") == "second" for c in json.get("chunks", [])):
                missing = True
                return {"ok": False, "error": "message_not_found"}
        return await original(method, json=json)

    client.api_call = keep_rejecting
    fallbacks = []

    async def fallback(**kwargs):
        fallbacks.append(kwargs)
        return SendResult(success=True, message_id="fallback")

    adapter.send = adapter.edit_message = fallback
    turn.native_tool_start_callback("first", "read_file", {})
    turn.native_tool_complete_callback("first", "read_file", {}, "done")
    turn.native_tool_start_callback("second", "read_file", {})
    turn.native_tool_complete_callback("second", "read_file", {}, "done")
    await drain_turn(turn)
    assert missing
    assert client.opens == 2
    assert len(fallbacks) == 1
    assert fallbacks[0]["content"].startswith("Hermes is working")


@pytest.mark.asyncio
async def test_egress_decline_remains_terminal_for_all_rich_updates(monkeypatch):
    from gateway.platforms.base import SendResult
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True}}}}
    _ctx, turn = await make_turn(monkeypatch, config, adapter)
    adapter._outbound_blocked = lambda *a: SendResult(success=False, error="declined", raw_response={"code": "egress_declined"})
    async def forbidden(**kwargs):
        pytest.fail("declined progress must never fall back to text")
    adapter.send = forbidden
    turn.native_tool_start_callback("call-a", "terminal", {"command": "date"})
    turn.native_tool_complete_callback("call-a", "terminal", {}, "done")
    turn.progress_callback("subagent.start", subagent_id="blocked-child")
    await drain_turn(turn)
    assert client.calls == []


@pytest.mark.asyncio
async def test_interrupted_turn_closes_children_without_flushing_pending_work(monkeypatch):
    import asyncio
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True}}}}
    ctx, turn = await make_turn(monkeypatch, config, adapter)
    ctx.agent_holder[0] = SimpleNamespace(is_interrupted=False)
    child_open = asyncio.Event()
    original = client.api_call
    async def observed(method, *, json):
        result = await original(method, json=json)
        if method == "chat.appendStream" and json.get("ts") == "stream-2":
            child_open.set()
        return result
    client.api_call = observed
    turn.native_tool_start_callback("first", "delegate_task", {})
    turn.progress_callback("subagent.start", subagent_id="child-a")
    task = asyncio.create_task(turn.send_progress_messages())
    await asyncio.wait_for(child_open.wait(), 3)
    turn.test_agent.reasoning_callback("Pending reasoning must not be delivered after interruption.")
    async def reasoning_consumed():
        while not ctx.progress_queue.empty():
            await asyncio.sleep(0)
    await asyncio.wait_for(reasoning_consumed(), 3)
    turn.native_tool_start_callback("late", "terminal", {"command": "do not show"})
    ctx.agent_holder[0].is_interrupted = True
    task.cancel()
    await task
    assert client.opens == 2
    assert sorted(p["ts"] for m, p in client.calls if m == "chat.stopStream") == ["stream-1", "stream-2"]
    assert not any(c.get("id") == "late" for _, p in client.calls for c in p.get("chunks", []))
    assert "Pending reasoning" not in str(client.calls)
    assert ctx.progress_queue.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("platform,progress,legacy,basic,native,rich", [
    (Platform.SLACK, "off", True, False, False, False),
    (Platform.SLACK, "off", False, True, False, False),
    (Platform.SLACK, "all", False, True, True, False),
    (Platform.DISCORD, "all", True, True, False, False),
    (Platform.WEBHOOK, "all", True, True, False, False),
])
async def test_stock_and_non_slack_gates_are_preserved(monkeypatch, platform, progress, legacy, basic, native, rich):
    config = {"display": {"tool_progress": progress, "tool_progress_native": legacy}}
    ctx, _ = await make_turn(monkeypatch, config, NativeAdapter(basic), platform)
    assert ctx._native_slack_task_cards is native
    assert ctx._rich_slack_task_cards is rich


@pytest.mark.asyncio
async def test_reactive_rollover_recovers_without_switching_to_text_fallback(monkeypatch):
    from gateway.slack_task_stream import RichTaskCardSession
    adapter, client = direct_adapter()
    ctx, _ = await make_turn(monkeypatch, {}, adapter)
    session = RichTaskCardSession(adapter, ctx)
    original = client.api_call
    failed = False
    async def expired(method, *, json):
        nonlocal failed
        if method == "chat.appendStream" and not failed:
            failed = True
            raise RuntimeError("message_not_in_streaming_state")
        return await original(method, json=json)
    client.api_call = expired
    result = await session.publish([{"type": "tool.started", "tool_call_id": "real-call", "tool_name": "terminal"}])
    await session.stop()
    assert result.success is True
    assert client.opens == 1
    updates = [payload for method, payload in client.calls if method == "chat.update"]
    assert updates and all(payload["ts"] == "stream-1" for payload in updates)
    assert updates[-1]["blocks"][0]["tasks"][0]["task_id"] == "real-call"


@pytest.mark.asyncio
async def test_basic_only_connector_keeps_its_guarded_native_seam(monkeypatch):
    from gateway.platforms.base import SendResult
    class Connector(NativeAdapter):
        def __init__(self):
            super().__init__(basic=True)
            self.publications = 0
        async def send_native_task_card_progress(self, **kwargs):
            self.publications += 1
            return SendResult(success=False, error="declined", raw_response={"code": "egress_declined"})
        async def stop_native_task_card_progress(self, *a, **kwargs):
            pass
        async def send(self, **kwargs):
            pytest.fail("connector refusal must not become rich/text bypass")
    adapter = Connector()
    config = {"display": {"tool_progress": "all", "tool_progress_native": True}}
    _, turn = await make_turn(monkeypatch, config, adapter)
    turn.native_tool_start_callback("call", "terminal", {})
    await drain_turn(turn)
    assert adapter.publications == 1


def test_native_display_tuning_is_normalised():
    config = {"display": {"platforms": {"slack": {
        "tool_progress_native": "off", "tool_progress_native_mode": "TIMELINE",
        "tool_progress_native_output_chars": -1, "tool_progress_native_rollover_age_s": "invalid",
    }}}}
    assert resolve_display_setting(config, "slack", "tool_progress_native") is False
    assert resolve_display_setting(config, "slack", "tool_progress_native_mode") == "timeline"
    assert resolve_display_setting(config, "slack", "tool_progress_native_output_chars") == 0
    assert resolve_display_setting(config, "slack", "tool_progress_native_rollover_age_s") > 0

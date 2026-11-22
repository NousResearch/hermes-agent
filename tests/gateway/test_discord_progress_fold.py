"""Folded tool progress (``display.tool_progress_fold: spoiler``) through the real DiscordAdapter.

The real ``TurnRunner`` progress callback and drain deliver through the real
``DiscordAdapter.send``/``edit_message``, so every folded message also passes the adapter's own
``format_message`` (pipe tables to bullets) and ``truncate_message``; only the channel under the
adapter is a double. Discord must receive exactly the drain's messages, unchanged, each one
well-formed on its own: one summary line, one spoiler, balanced fences, at most 2,000 characters.
"""

import asyncio
import itertools
import queue
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import run_turn_progress
from gateway.config import Platform, PlatformConfig
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from plugins.platforms.discord.adapter import DiscordAdapter

ZW = "\u200b"
EMOJI = {"terminal": "💻", "read_file": "📖", "search_files": "🔎"}
SUMMARY = re.compile(r"^-# 🛠️ \d+ tools? · click to show$", re.MULTILINE)
TABLE = "| file | result |\n|---|---|\n| a.py | pass |"


class ChannelDouble:
    """The two channel calls the adapter makes: ``channel.send`` and ``PartialMessage.edit``."""

    def __init__(self, wire):
        self.id, self._wire = 555, wire

    async def send(self, *, content, reference=None, **_):
        mid = 1000 + len(self._wire)
        self._wire.append(("send", content))
        return SimpleNamespace(id=mid)

    def get_partial_message(self, mid):
        async def edit(*, content, **_):
            self._wire.append((f"edit {mid}", content))
        return SimpleNamespace(id=mid, edit=edit)


@pytest.fixture
def discord_adapter():
    """A real DiscordAdapter whose sends/edits land in ``adapter.wire``; ``adapter.handed`` is what
    the progress drain passed it."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter.wire, adapter.handed = [], []
    adapter._client = SimpleNamespace()
    adapter._resolve_channel = AsyncMock(return_value=ChannelDouble(adapter.wire))
    send, edit = adapter.send, adapter.edit_message

    async def handed_send(chat_id, content, reply_to=None, metadata=None):
        result = await send(chat_id, content, reply_to=reply_to, metadata=metadata)
        adapter.handed.append(("send", content))
        return result

    async def handed_edit(chat_id, message_id, content, *, finalize=False, metadata=None):
        adapter.handed.append((f"edit {message_id}", content))
        return await edit(chat_id, message_id, content, finalize=finalize, metadata=metadata)

    adapter.send, adapter.edit_message = handed_send, handed_edit
    return adapter


class _NoPacing:
    """``asyncio`` as the drain sees it, with its pacing sleeps only yielding."""

    def __getattr__(self, name):
        return getattr(asyncio, name)

    @staticmethod
    async def sleep(delay, result=None):
        return await asyncio.sleep(0, result)


@pytest.fixture(autouse=True)
def _deterministic(monkeypatch):
    clock = itertools.count(10_000, 100)  # the drain spaces edits 1.5 s apart; skip the waits
    monkeypatch.setattr(run_turn_progress, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    monkeypatch.setattr(run_turn_progress, "asyncio", _NoPacing())
    monkeypatch.setattr("agent.display.get_tool_emoji", lambda name, default="⚡": EMOJI.get(name, default))


class WatchedQueue(queue.Queue):
    """The progress queue, telling when the drain has delivered everything queued so far."""

    def __init__(self):
        super().__init__()
        self.idle = asyncio.Event()

    def get_nowait(self):
        try:
            return super().get_nowait()
        except queue.Empty:
            self.idle.set()
            raise


async def _drive(adapter, events, *, fold=True, mode="all", grouping="accumulate", thinking=False):
    live = [True]
    ctx = TurnContext(
        source=SessionSource(platform=Platform.DISCORD, chat_id="555", chat_type="group"),
        _run_still_current=lambda: live[0], progress_mode=mode, progress_grouping=grouping,
        tool_progress_enabled=True, _thinking_enabled=thinking,
        progress_queue=WatchedQueue(), _progress_metadata={"thread_id": "555", "non_conversational": True},
        **({"tool_progress_fold": True} if fold else {}),
    )
    turn = TurnRunner(SimpleNamespace(_delivery_adapter_for=lambda source: adapter, session_authority=None), ctx)
    for event in events:
        turn.progress_callback(*event)
    consumer = asyncio.create_task(turn.send_progress_messages())
    await asyncio.wait_for(ctx.progress_queue.idle.wait(), 10)
    live[0] = False
    await asyncio.wait_for(consumer, 5)


def _screen(wire) -> list[str]:
    shown: dict[str, str] = {}
    for i, (op, content) in enumerate(wire):
        shown[str(1000 + i) if op == "send" else op.split()[1]] = content
    return list(shown.values())


def _received_as_handed(adapter) -> None:
    """Discord got exactly the drain's messages, one channel call each, unchanged by the adapter's
    formatting, and every folded one (each message on screen once the turn ends) is self-contained."""
    assert adapter.wire == adapter.handed
    assert all(adapter.format_message(content) == content for _, content in adapter.wire)
    assert all(SUMMARY.search(message) for message in _screen(adapter.wire))
    for message in (content for _, content in adapter.wire if SUMMARY.search(content)):
        assert len(message) <= DiscordAdapter.MAX_MESSAGE_LENGTH
        assert message.count("||") == 2, message
        head, body, tail = message.split("||")
        assert tail == "" and head.endswith("\n"), message
        assert re.fullmatch(r"-# 🛠️ \d+ tools? · click to show", head[:-1].split("\n")[-1]), head
        assert not body.startswith(("`", "\n")), body
        fences = [line for line in body.split("\n") if line.startswith("```")]
        assert all(f == "```" for f in fences) and len(fences) % 2 == 0 and body.count("```") == len(fences), body


def _terminal(command):
    return ("tool.started", "terminal", command[:40], {"command": command})


def _pr_create_command():
    prose = " ".join(f"word{i:03d} explains the change in plain prose" for i in range(48))  # ~1.9K
    rows = "\n".join(f"| src/module_{i}.py | pass | {i} | note {i} |" for i in range(40))
    return f'gh pr create --title "x" --body "{prose}\n\n| file | result | count | note |\n|---|---|---|---|\n{rows}"'


@pytest.mark.asyncio
@pytest.mark.parametrize("grouping", ["accumulate", "separate"])
async def test_a_command_holding_a_table_reaches_discord_unchanged(discord_adapter, grouping):
    command = _pr_create_command()
    await _drive(discord_adapter, [_terminal(command)], grouping=grouping)
    _received_as_handed(discord_adapter)
    shown = _screen(discord_adapter.wire)
    assert len(shown) >= 2
    assert not any("• " in message or "**file**" in message for message in shown)  # no table→bullets
    code = " ".join(re.findall(r"```\n(.*?)\n```", "\n".join(shown), re.DOTALL))
    assert code.split() == command.split()
    assert sum(m.count("| src/module_") for m in shown) == 40


@pytest.mark.asyncio
async def test_a_long_command_continues_under_a_repeated_header(discord_adapter):
    command = " && ".join(f"step{i} --flag=value{i} || true" for i in range(300))
    await _drive(discord_adapter, [
        ("tool.started", "read_file", "a.py", {"path": "a.py"}),
        _terminal(command),
        ("tool.started", "read_file", "b.py", {"path": "b.py"}),
    ])
    _received_as_handed(discord_adapter)
    shown = _screen(discord_adapter.wire)
    assert len(shown) >= 4
    code = " ".join(re.findall(r"```\n(.*?)\n```", "\n".join(shown), re.DOTALL))
    assert code.split() == command.replace("||", f"|{ZW}|").split()
    assert shown[0].split("||")[1].startswith("📖 Reading a.py\n💻 terminal\n```\n")
    for message in shown[1:-1]:
        assert message.split("||")[1].startswith("💻 terminal …\n```\n"), message


@pytest.mark.asyncio
async def test_thinking_text_cut_inside_a_fence_keeps_one_spoiler(discord_adapter):
    await _drive(discord_adapter, [
        ("_thinking", "either a || b — I'll run:\n```\nmake ch"),
        ("tool.started", "read_file", "Makefile", {"path": "Makefile"}),
        _terminal("make check || true"),
    ], thinking=True)
    _received_as_handed(discord_adapter)
    assert _screen(discord_adapter.wire) == [
        "-# 🛠️ 2 tools · click to show\n"
        f"||💬\n```\neither a |{ZW}| b — I'll run:\n`{ZW}``\nmake ch\n```\n"
        f"📖 Reading Makefile\n💻 terminal\n```\nmake check |{ZW}| true\n```||",
    ]


# Folded bubbles go out on the adapter's ordinary send/edit, so its table conversion must find
# nothing to convert: thinking text and commands sit in code blocks whose fences start their own
# lines, and every other ``` run is broken, so the converter's by-line fence tracking agrees
# with Discord's.
@pytest.mark.asyncio
@pytest.mark.parametrize(("thinking", "tools"), [
    pytest.param(f"the plan:\n{TABLE}", [_terminal(f"printf '{TABLE}'")], id="tables-in-thinking-and-command"),
    pytest.param(f"```sh\n{TABLE}", [_terminal(f"cat <<EOF\n{TABLE}\nEOF")], id="thinking-opens-on-a-fence"),
    pytest.param(f"ran ```ls``` then\n```x```\n{TABLE}", [_terminal(f"cat <<EOF\n{TABLE}\nEOF")],
                 id="one-line-code-blocks"),
    pytest.param("check a|b", [("tool.started", "search_files", "a|b", {"pattern": "a|b"}),
                               ("tool.started", "search_files", "|---|---|", {"pattern": "|---|---|"})],
                 id="tool-previews-holding-pipes"),
    pytest.param("x |\f| y `\f``", [_terminal(f"echo '|\f|' '`\f``'\n{TABLE}")], id="form-feeds"),
    pytest.param(f"see https://x/q?a=b|c and <https://x/a_b>\r{TABLE.replace(chr(10), chr(13))}",
                 [_terminal("ls")], id="urls-and-cr-line-breaks"),
    pytest.param(f"{TABLE}\nthen || done", [_terminal("ls")], id="thinking-is-a-table"),
])
@pytest.mark.parametrize("mode", ["all", "verbose"])
async def test_folded_messages_pass_the_adapters_formatting_unchanged(discord_adapter, thinking, tools, mode):
    await _drive(discord_adapter, [("_thinking", thinking), *tools], mode=mode, thinking=True)
    _received_as_handed(discord_adapter)


# A preview is cut from the tool's own args, and some keep their line breaks (a path, a memory
# target): folded, each tool line is one line, so no pipe table can form around the fold.
_BROKEN = "x | y\n|---|---|\r\n| 1 | 2 |\r| 3 | 4 |\u2028| 5 |\v\f| 6 |"


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool", "args"), [
    ("read_file", {"path": _BROKEN}),
    ("memory", {"operations": [{"action": "add", "content": "c"}], "target": _BROKEN}),
    ("process_manage", {"action": "poll", "session_id": _BROKEN[:16]}),
    ("send_message", {"target": _BROKEN, "message": "hi"}),
    ("tool_search", {"queries": [_BROKEN]}),
])
@pytest.mark.parametrize("mode", ["all", "verbose"])
async def test_tool_previews_holding_line_breaks_stay_one_line(discord_adapter, tool, args, mode):
    await _drive(discord_adapter, [("tool.started", tool, _BROKEN, args), _terminal("ls")], mode=mode)
    _received_as_handed(discord_adapter)
    [message] = _screen(discord_adapter.wire)
    outside_code = re.sub(r"```\n.*?\n```", "", message.split("||")[1], flags=re.DOTALL)
    assert not any(line.startswith("|") for line in outside_code.splitlines()), message


# A preview that itself holds a fence must not pass for a tool line's own code block and so
# skip the one-line rule (a table before it, or a fence fused to the text).
_FENCED_PREVIEWS = ["|a|b|\n|-|-|\n|1|2|\n```\nx\n```", "|a|b|\n|-|-|\n|1|2|\n```\nx\n```\n",
                    "a```\n|a|b|\n|-|-|\n|1|2|\n```"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool", "preview", "args"), [
    ("read_file", _FENCED_PREVIEWS[0], {"path": _FENCED_PREVIEWS[0]}),
    ("memory", _FENCED_PREVIEWS[1], {"operations": [{"action": "add", "content": "c"}], "target": _FENCED_PREVIEWS[1]}),
    ("send_message", _FENCED_PREVIEWS[1], {"target": _FENCED_PREVIEWS[1], "message": "hi"}),
    ("read_file", _FENCED_PREVIEWS[2], {"path": _FENCED_PREVIEWS[2]}),
])
@pytest.mark.parametrize("mode", ["all", "new"])
async def test_tool_previews_holding_a_fence_stay_one_line(discord_adapter, tool, preview, args, mode):
    await _drive(discord_adapter, [("tool.started", tool, preview, args), _terminal("ls")], mode=mode)
    _received_as_handed(discord_adapter)
    [message] = _screen(discord_adapter.wire)
    # The only fences are the terminal block's own, each alone on its line.
    assert [ln for ln in message.splitlines() if "```" in ln] == ["```", "```||"], message


@pytest.mark.asyncio
async def test_default_off_progress_is_still_formatted_by_the_adapter(discord_adapter):
    await _drive(discord_adapter, [("_thinking", f"plan:\n{TABLE}")], fold=False, thinking=True)
    assert discord_adapter.handed == [("send", f"💬 plan:\n{TABLE}")]
    assert discord_adapter.wire == [("send", discord_adapter.format_message(f"💬 plan:\n{TABLE}"))]
    assert discord_adapter.wire != discord_adapter.handed  # the table really was converted

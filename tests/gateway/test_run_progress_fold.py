"""``display.tool_progress_fold: spoiler`` (#158988): tool progress behind one click-to-reveal spoiler.

Drives the real ``TurnRunner`` progress callback and drain against a capture adapter. Default off
must render byte-identically to before on every platform; on an adapter declaring
``supports_spoiler_fold`` each bubble is one ``-#`` summary line plus ONE spoiler over its tool lines,
and every message stays well-formed on its own (one ``||`` pair, balanced fences, within the cap).
"""

import asyncio
import itertools
import queue
import re
from types import SimpleNamespace

import pytest

from gateway import run_turn_progress
from gateway.config import Platform, PlatformConfig
from gateway.display_config import resolve_display_setting
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from tests.waiting import wait_until

ZW = "\u200b"
EMOJI = {"terminal": "💻", "read_file": "📖", "search_files": "🔎", "web_search": "🌐", "write_file": "✍️"}
SECRET = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789"


class CaptureAdapter(BasePlatformAdapter):
    """Records every send/edit the drain makes, in order; contents must be plain ``str``."""

    def __init__(self, platform=Platform.DISCORD, *, limit=2000, code_blocks=True, spoiler=False):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.MAX_MESSAGE_LENGTH = limit
        self.supports_code_blocks = code_blocks
        self.supports_spoiler_fold = spoiler
        self.ops = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        assert type(content) is str
        self.ops.append(("send", content, metadata))
        return SendResult(success=True, message_id=f"m{len(self.ops)}")

    async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None) -> SendResult:
        assert type(content) is str
        self.ops.append((f"edit {message_id}", content, metadata))
        return SendResult(success=True, message_id=message_id)

    def screen(self) -> list[str]:
        """What the chat shows: each message's latest content, in post order."""
        shown: dict[str, str] = {}
        for i, (op, content, _) in enumerate(self.ops, 1):
            shown[f"m{i}" if op == "send" else op.split()[1]] = content
        return list(shown.values())


class _NoPacing:
    """``asyncio`` as the drain sees it, with its pacing sleeps (typing restore, idle poll) only
    yielding: they are timing, not rendering."""

    def __getattr__(self, name):
        return getattr(asyncio, name)

    @staticmethod
    async def sleep(delay, result=None):
        return await asyncio.sleep(0, result)


@pytest.fixture(autouse=True)
def _deterministic(monkeypatch):
    # The drain spaces edits 1.5 s apart; a clock that always moves on makes every tick send.
    clock = itertools.count(10_000, 100)
    monkeypatch.setattr(run_turn_progress, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    monkeypatch.setattr(run_turn_progress, "asyncio", _NoPacing())
    monkeypatch.setattr("agent.display.get_tool_emoji", lambda name, default="⚡": EMOJI.get(name, default))
    monkeypatch.setattr("agent.display.get_tool_preview_max_len", lambda: 0)


class WatchedQueue(queue.Queue):
    """The progress queue, telling when the drain has delivered everything queued so far: it
    polls for the next line only after the previous one's send/edit completed."""

    def __init__(self):
        super().__init__()
        self.idle = asyncio.Event()

    def get_nowait(self):
        try:
            return super().get_nowait()
        except queue.Empty:
            self.idle.set()
            raise


def _turn(adapter, *, fold=False, mode="all", grouping="accumulate", thinking=False):
    ctx = TurnContext(
        source=SessionSource(platform=adapter.platform, chat_id="c1", chat_type="group"),
        _run_still_current=lambda: True, progress_mode=mode, progress_grouping=grouping,
        tool_progress_enabled=True, _thinking_enabled=thinking,
        progress_queue=WatchedQueue(), _progress_metadata={"thread_id": "t1"},
        **({"tool_progress_fold": True} if fold else {}),
    )
    return TurnRunner(SimpleNamespace(_delivery_adapter_for=lambda source: adapter, session_authority=None), ctx)


async def _drive(adapter, events, **turn_kw) -> list:
    """Queue ``events`` through the real progress callback, run the real drain until it has
    delivered them, then end the turn. Returns the adapter's ops."""
    turn = _turn(adapter, **turn_kw)
    live = [True]
    turn._ctx._run_still_current = lambda: live[0]
    for event in events:
        turn.progress_callback(*event)
    consumer = asyncio.create_task(turn.send_progress_messages())
    await asyncio.wait_for(turn._ctx.progress_queue.idle.wait(), 10)
    live[0] = False
    await asyncio.wait_for(consumer, 5)
    return adapter.ops


def _tool(name, preview, args):
    return ("tool.started", name, preview, args)


MULTI_LINE = "set -euo pipefail\nmake check | grep -E '^FAIL|^WARN' || true\nnpm test"
SCENARIO = [
    _tool("search_files", "delete_message", {"pattern": "delete_message"}),
    _tool("read_file", "gateway/run.py", {"path": "gateway/run.py"}),
    _tool("terminal", MULTI_LINE, {"command": MULTI_LINE}),
    _tool("terminal", "ls -la", {"command": "ls -la"}),
    ("_thinking", "checking the || case"),
    _tool("web_search", "spoiler markdown", {"query": "spoiler markdown"}),
    _tool("web_search", "spoiler markdown", {"query": "spoiler markdown"}),
]

# Recorded on main before tool_progress_fold existed (same scenario, same harness).
SNAPSHOT_OFF = {
    'discord-accumulate': [
        ('send', '🔎 Searching files for delete_message'),
        ('edit m1', '🔎 Searching files for delete_message\n📖 Reading run.py'),
        ('edit m1', '🔎 Searching files for delete_message\n📖 Reading run.py\n💻 terminal\n```\nset -euo pipefail ...\n```'),
        ('edit m1', '🔎 Searching files for delete_message\n📖 Reading run.py\n💻 terminal\n```\nset -euo pipefail ...\n```\n```\nls -la\n```'),
        ('edit m1', '🔎 Searching files for delete_message\n📖 Reading run.py\n💻 terminal\n```\nset -euo pipefail ...\n```\n```\nls -la\n```\n💬 checking the || case'),
        ('edit m1', '🔎 Searching files for delete_message\n📖 Reading run.py\n💻 terminal\n```\nset -euo pipefail ...\n```\n```\nls -la\n```\n💬 checking the || case\n🌐 Searching the web for spoiler markdown'),
        ('edit m1', '🔎 Searching files for delete_message\n📖 Reading run.py\n💻 terminal\n```\nset -euo pipefail ...\n```\n```\nls -la\n```\n💬 checking the || case\n🌐 Searching the web for spoiler markdown (×2)'),
    ],
    'discord-overflow': [
        ('send', '🔎 Searching files for delete_message'),
        ('edit m1', '🔎 Searching files for delete_message\n📖 Reading run.py'),
        ('edit m1', '🔎 Searching files for delete_message\n📖 Reading run.py\n💻 terminal\n```\nset -euo pipefail ...\n```'),
        ('edit m1', '🔎 Searching files for delete_message\n📖 Reading run.py\n💻 terminal\n```\nset -euo pipefail ...\n```\n```\nls -la\n```'),
        ('edit m1', '🔎 Searching files for delete_message\n📖 Reading run.py\n💻 terminal\n```\nset -euo pipefail ...\n```\n```\nls -la\n```\n💬 checking the || case'),
        ('edit m1', '🔎 Searching files for delete_message\n📖 Reading run.py\n💻 terminal\n```\nset -euo pipefail ...\n```\n```\nls -la\n```\n💬 checking the || case'),
        ('send', '🌐 Searching the web for spoiler markdown'),
        ('edit m7', '🌐 Searching the web for spoiler markdown (×2)'),
    ],
    'discord-verbose': [
        ('send', '🔎 search_files([\'pattern\'])\n```\n{"pattern": "delete_message"}\n```'),
        ('edit m1', '🔎 search_files([\'pattern\'])\n```\n{"pattern": "delete_message"}\n```\n📖 read_file([\'path\'])\n```\n{"path": "gateway/run.py"}\n```'),
        ('edit m1', '🔎 search_files([\'pattern\'])\n```\n{"pattern": "delete_message"}\n```\n📖 read_file([\'path\'])\n```\n{"path": "gateway/run.py"}\n```\n💻 terminal\n```\nset -euo pipefail\nmake check | grep -E \'^FAIL|^WARN\' || true\nnpm test\n```'),
        ('edit m1', '🔎 search_files([\'pattern\'])\n```\n{"pattern": "delete_message"}\n```\n📖 read_file([\'path\'])\n```\n{"path": "gateway/run.py"}\n```\n💻 terminal\n```\nset -euo pipefail\nmake check | grep -E \'^FAIL|^WARN\' || true\nnpm test\n```\n```\nls -la\n```'),
        ('edit m1', '🔎 search_files([\'pattern\'])\n```\n{"pattern": "delete_message"}\n```\n📖 read_file([\'path\'])\n```\n{"path": "gateway/run.py"}\n```\n💻 terminal\n```\nset -euo pipefail\nmake check | grep -E \'^FAIL|^WARN\' || true\nnpm test\n```\n```\nls -la\n```\n💬 checking the || case'),
        ('edit m1', '🔎 search_files([\'pattern\'])\n```\n{"pattern": "delete_message"}\n```\n📖 read_file([\'path\'])\n```\n{"path": "gateway/run.py"}\n```\n💻 terminal\n```\nset -euo pipefail\nmake check | grep -E \'^FAIL|^WARN\' || true\nnpm test\n```\n```\nls -la\n```\n💬 checking the || case\n🌐 web_search([\'query\'])\n```\n{"query": "spoiler markdown"}\n```'),
        ('edit m1', '🔎 search_files([\'pattern\'])\n```\n{"pattern": "delete_message"}\n```\n📖 read_file([\'path\'])\n```\n{"path": "gateway/run.py"}\n```\n💻 terminal\n```\nset -euo pipefail\nmake check | grep -E \'^FAIL|^WARN\' || true\nnpm test\n```\n```\nls -la\n```\n💬 checking the || case\n🌐 web_search([\'query\'])\n```\n{"query": "spoiler markdown"}\n```\n🌐 web_search([\'query\'])\n```\n{"query": "spoiler markdown"}\n```'),
    ],
    'slack-separate': [
        ('send', '🔎 Searching files for delete_message'),
        ('send', '📖 Reading run.py'),
        ('send', '💻 terminal\n```\nset -euo pipefail ...\n```'),
        ('send', '```\nls -la\n```'),
        ('send', '💬 checking the || case'),
        ('send', '🌐 Searching the web for spoiler markdown'),
        ('send', '🌐 Searching the web for spoiler markdown (×2)'),
    ],
    'telegram-accumulate': [
        ('send', '🔎 Searching files for delete_message'),
        ('edit m1', '🔎 Searching files for delete_message\n📖 Reading run.py'),
        ('edit m1', '🔎 Searching files for delete_message\n📖 Reading run.py\n💻 Running set -euo pipefail make check | grep -...'),
        ('edit m1', '🔎 Searching files for delete_message\n📖 Reading run.py\n💻 Running set -euo pipefail make check | grep -...\n💻 Running ls -la'),
        ('edit m1', '🔎 Searching files for delete_message\n📖 Reading run.py\n💻 Running set -euo pipefail make check | grep -...\n💻 Running ls -la\n💬 checking the || case'),
        ('edit m1', '🔎 Searching files for delete_message\n📖 Reading run.py\n💻 Running set -euo pipefail make check | grep -...\n💻 Running ls -la\n💬 checking the || case\n🌐 Searching the web for spoiler markdown'),
        ('edit m1', '🔎 Searching files for delete_message\n📖 Reading run.py\n💻 Running set -euo pipefail make check | grep -...\n💻 Running ls -la\n💬 checking the || case\n🌐 Searching the web for spoiler markdown (×2)'),
    ],
}


def _well_formed(message: str, limit: int) -> None:
    """One summary line, then exactly one spoiler that opens on text, with balanced fences that
    each start their own line."""
    assert len(message) <= limit, len(message)
    assert message.count("||") == 2, message
    head, body, tail = message.split("||")
    assert tail == "" and head.endswith("\n"), message
    assert re.fullmatch(r"-# 🛠️ \d+ tools? · click to show", head[:-1].split("\n")[-1]), head
    assert not body.startswith(("`", "\n")), body
    fence_lines = [line for line in body.split("\n") if line.startswith("```")]
    assert body.count("```") == len(fence_lines) and len(fence_lines) % 2 == 0, body
    assert all(line == "```" for line in fence_lines), body
    assert not body.endswith("\\"), body


# ── default off: byte-identical ─────────────────────────────────────────────────────────────


CASES = {
    "discord-accumulate": dict(platform=Platform.DISCORD, limit=2000, code_blocks=True),
    "telegram-accumulate": dict(platform=Platform.TELEGRAM, limit=4096, code_blocks=False),
    "slack-separate": dict(platform=Platform.SLACK, limit=4000, code_blocks=True, grouping="separate"),
    "discord-overflow": dict(platform=Platform.DISCORD, limit=200, code_blocks=True),
    "discord-verbose": dict(platform=Platform.DISCORD, limit=2000, code_blocks=True, mode="verbose"),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(CASES))
async def test_default_off_renders_exactly_as_before(case):
    opts = dict(CASES[case])
    turn_kw = {k: opts.pop(k) for k in ("grouping", "mode") if k in opts}
    ops = await _drive(CaptureAdapter(**opts), SCENARIO, thinking=True, **turn_kw)
    assert [(op, content) for op, content, _ in ops] == SNAPSHOT_OFF[case]
    assert all(metadata == {"thread_id": "t1"} for _, _, metadata in ops)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(CASES))
async def test_spoiler_setting_is_ignored_where_the_adapter_cannot_render_it(case):
    """An adapter without ``supports_spoiler_fold`` never folds, whatever the config says."""
    opts = dict(CASES[case])
    turn_kw = {k: opts.pop(k) for k in ("grouping", "mode") if k in opts}
    adapter = CaptureAdapter(**opts)
    disp = SimpleNamespace(
        tool_progress_enabled=True, resolve_display_setting=resolve_display_setting, platform_key="x",
        user_config={"display": {"platforms": {"x": {"tool_progress_fold": "spoiler"}}}})
    fold = TurnRunner.folds_tool_progress(disp, adapter)
    ops = await _drive(adapter, SCENARIO, thinking=True, fold=fold, **turn_kw)
    assert [(op, content) for op, content, _ in ops] == SNAPSHOT_OFF[case]


@pytest.mark.parametrize(("value", "expected"), [
    ("spoiler", True), ("SPOILER", True), (True, True), ("true", True), ("on", True),
    ("off", False), (False, False), ("no", False), ("bogus", False), (None, False),
])
def test_setting_resolves_off_unless_spoiler(value, expected):
    disp = SimpleNamespace(
        tool_progress_enabled=True, resolve_display_setting=resolve_display_setting, platform_key="discord",
        user_config={"display": {"platforms": {"discord": {"tool_progress_fold": value}}}})
    assert TurnRunner.folds_tool_progress(disp, CaptureAdapter(spoiler=True)) is expected


def test_fold_needs_tool_progress_and_a_declaring_adapter():
    from unittest.mock import MagicMock
    disp = SimpleNamespace(
        tool_progress_enabled=True, resolve_display_setting=resolve_display_setting, platform_key="discord",
        user_config={"display": {"tool_progress_fold": "spoiler"}})
    assert TurnRunner.folds_tool_progress(disp, CaptureAdapter(spoiler=True)) is True
    assert TurnRunner.folds_tool_progress(disp, CaptureAdapter(spoiler=False)) is False
    assert TurnRunner.folds_tool_progress(disp, MagicMock()) is False
    assert TurnRunner.folds_tool_progress(disp, None) is False
    native = type("NativeStream", (CaptureAdapter,), {"SUPPORTS_NATIVE_STREAMING": True})(spoiler=True)
    assert TurnRunner.folds_tool_progress(disp, native) is False  # tool lines go to its stream bubble
    disp.tool_progress_enabled = False  # off / log
    assert TurnRunner.folds_tool_progress(disp, CaptureAdapter(spoiler=True)) is False


class _ToolAgent:
    adapter = None  # the run's capture adapter: the turn ends only once its progress landed

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        self.tool_progress_callback("tool.started", "read_file", "a.py", {"path": "a.py"})
        self.tool_progress_callback("tool.started", "terminal", "make", {"command": "make check || true"})
        wait_until(lambda: any("make check" in content for _, content, _ in _ToolAgent.adapter.ops),
                   what="the progress bubble")
        return {"final_response": "done", "messages": [], "api_calls": 1}


async def _real_turn(monkeypatch, tmp_path, adapter, display):
    """One turn through ``GatewayRunner._run_agent`` with ``display`` as config.yaml's block."""
    import importlib
    import sys
    import types

    import hermes_yaml as yaml
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"display": display}), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "dotenv", types.SimpleNamespace(load_dotenv=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=_ToolAgent))
    monkeypatch.setattr(_ToolAgent, "adapter", adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode, runner._prefill_messages, runner._ephemeral_system_prompt = {}, [], ""
    runner._reasoning_config, runner._provider_routing, runner._fallback_model = None, {}, None
    runner._session_db, runner._running_agents = None, {}
    runner.session_store = SimpleNamespace(_entries={}, _save=lambda: None)
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(thread_sessions_per_user=False, group_sessions_per_user=False, stt_enabled=False)
    platform = adapter.platform.value
    result = await runner._run_agent(
        message="hello", context_prompt="", history=[], session_id=f"sess-fold-{platform}",
        source=SessionSource(platform=adapter.platform, chat_id="12345", chat_type="dm"),
        session_key=f"agent:main:{platform}:dm:12345")
    assert result["final_response"] == "done"
    return [content for op, content, _ in adapter.ops]


@pytest.mark.asyncio
async def test_configured_spoiler_folds_a_real_turn_on_a_declaring_adapter(monkeypatch, tmp_path):
    display = {"tool_progress": "all", "platforms": {"discord": {"tool_progress_fold": "spoiler"}}}
    folded = await _real_turn(monkeypatch, tmp_path, CaptureAdapter(Platform.DISCORD, spoiler=True), display)
    assert folded[-1] == (f"-# 🛠️ 2 tools · click to show\n||📖 Reading a.py\n💻 terminal\n```\n"
                          f"make check |{ZW}| true\n```||")
    # The key is per platform: the same turn on Telegram (no override) is unchanged.
    plain = await _real_turn(monkeypatch, tmp_path, CaptureAdapter(Platform.TELEGRAM, code_blocks=False), display)
    assert plain[-1] == "📖 Reading a.py\n💻 Running make check"


def test_discord_declares_spoiler_support_and_others_do_not():
    from plugins.platforms.discord.adapter import DiscordAdapter
    from plugins.platforms.telegram.adapter import TelegramAdapter
    assert DiscordAdapter.supports_spoiler_fold is True
    assert getattr(TelegramAdapter, "supports_spoiler_fold", False) is False


# ── fold on: shape ───────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bubble_is_one_summary_line_and_one_spoiler_over_every_tool_line():
    adapter = CaptureAdapter(spoiler=True)
    await _drive(adapter, SCENARIO[:4], fold=True)
    assert adapter.screen() == [
        "-# 🛠️ 4 tools · click to show\n"
        "||🔎 Searching files for delete_message\n"
        "📖 Reading run.py\n"
        "💻 terminal\n```\n"
        f"set -euo pipefail\nmake check | grep -E '^FAIL|^WARN' |{ZW}| true\nnpm test\n```\n"
        "💻 terminal\n```\nls -la\n```||",
    ]
    assert all(metadata == {"thread_id": "t1"} for _, _, metadata in adapter.ops)  # passed through as is


@pytest.mark.asyncio
async def test_one_tool_reads_singular_and_repeats_stay_inside_the_fold():
    adapter = CaptureAdapter(spoiler=True)
    await _drive(adapter, SCENARIO[5:7], fold=True)
    [message] = adapter.screen()
    assert message == "-# 🛠️ 1 tool · click to show\n||🌐 Searching the web for spoiler markdown (×2)||"


@pytest.mark.asyncio
async def test_terminal_command_is_shown_whole_and_redacted_in_every_mode():
    command = f"curl -H 'Authorization: Bearer {SECRET}' https://api.example.com\necho done"
    for mode in ("new", "all", "verbose"):
        adapter = CaptureAdapter(spoiler=True)
        await _drive(adapter, [_tool("terminal", "curl", {"command": command})], fold=True, mode=mode)
        [message] = adapter.screen()
        _well_formed(message, 2000)
        assert SECRET not in message
        assert "Bearer ***' https://api.example.com\necho done\n```||" in message


@pytest.mark.asyncio
async def test_verbose_arguments_are_redacted_inside_the_fold():
    adapter = CaptureAdapter(spoiler=True)
    await _drive(adapter, [_tool("web_search", "q", {"query": "q", "api_key": SECRET})], fold=True, mode="verbose")
    [message] = adapter.screen()
    _well_formed(message, 2000)
    assert SECRET not in message and '"api_key": "***"' in message


@pytest.mark.asyncio
async def test_content_cannot_end_the_spoiler_or_its_code_block_early():
    heredoc = "cat <<'EOF' > a.md\n```py\nx = a || b\n```\nEOF\nprintf 'C:\\\\'"
    adapter = CaptureAdapter(spoiler=True)
    await _drive(adapter, [
        _tool("terminal", "cat", {"command": heredoc}),
        _tool("search_files", "a||b", {"pattern": "a||b"}),
        _tool("terminal", "y", {"command": "y = a |\f| b `\f``"}),
        _tool("write_file", "C:\\dir\\", {"path": "C:\\dir\\"}),
    ], fold=True)
    [message] = adapter.screen()
    _well_formed(message, 2000)
    body = message.split("||", 1)[1]
    assert f"`{ZW}``py\nx = a |{ZW}| b\n`{ZW}``\nEOF" in body  # inner fences and || broken
    assert f"🔎 Searching files for a|{ZW}|b\n" in body
    assert "|\f|" not in body and "`\f``" not in body  # Discord's parser drops the form feed
    assert message.endswith(f"\\{ZW}||")  # a trailing backslash cannot escape the closing ||


# ── fold on: thinking text ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_thinking_text_sits_inside_the_fold_uncounted():
    adapter = CaptureAdapter(spoiler=True)
    await _drive(adapter, [SCENARIO[0], ("_thinking", "either a || b, check both")], fold=True, thinking=True)
    [message] = adapter.screen()
    _well_formed(message, 2000)
    assert message == (f"-# 🛠️ 1 tool · click to show\n||🔎 Searching files for delete_message\n"
                       f"💬\n```\neither a |{ZW}| b, check both\n```||")


@pytest.mark.asyncio
@pytest.mark.parametrize(("thinking", "code"), [
    # In its own code block, like a command: nothing it holds is markup, so nothing is escaped,
    # URLs and tables included (the adapter's table conversion skips fenced code).
    ("see https://x.y/a** and *args, 2 * 3, `x, <@1>", "see https://x.y/a** and *args, 2 * 3, `x, <@1>"),
    ("| a | b |\n|---|---|\n| 1 | 2 |", "| a | b |\n|---|---|\n| 1 | 2 |"),
    # Only what could end the block or the spoiler is broken, as in a command.
    ("x || y ||| z", f"x |{ZW}| y |{ZW}|{ZW}| z"),
    ("I'll run:\n```sh\nmake ch", f"I'll run:\n`{ZW}``sh\nmake ch"),
    ("```x``` then ````", f"`{ZW}``x`{ZW}`` then `{ZW}`{ZW}``"),
    ("run:\r\n```py\r```\rnext", f"run:\r\n`{ZW}``py\r`{ZW}``\rnext"),
    ("a |\f| b `\f``", f"a |{ZW}\f| b `{ZW}\f``"),
    ("a trailing \\", "a trailing \\"),
    ("  \n ", ZW),  # Discord renders no empty block: its fences would pair with the next ones
])
async def test_thinking_text_is_folded_as_code(thinking, code):
    adapter = CaptureAdapter(spoiler=True)
    await _drive(adapter, [
        _tool("terminal", "x", {"command": "printf '```py\\n'"}),
        ("_thinking", thinking),
    ], fold=True, thinking=True)
    [message] = adapter.screen()
    _well_formed(message, 2000)
    assert message == (f"-# 🛠️ 1 tool · click to show\n||💻 terminal\n```\nprintf '`{ZW}``py\\n'\n```\n"
                       f"💬\n```\n{code}\n```||")


@pytest.mark.asyncio
async def test_folded_progress_goes_out_on_the_ordinary_send_and_edit():
    """No side channel: a folded bubble is sent and edited with the turn's own progress metadata,
    in both groupings, and the adapter formats it like any other message."""
    adapter = CaptureAdapter(spoiler=True)
    await _drive(adapter, [("_thinking", "just thinking || here"), SCENARIO[0]], fold=True, thinking=True)
    assert [(op, metadata) for op, _, metadata in adapter.ops] == [
        ("send", {"thread_id": "t1"}), ("edit m1", {"thread_id": "t1"})]
    separate = CaptureAdapter(spoiler=True)
    await _drive(separate, [("_thinking", "plan"), SCENARIO[0]], fold=True, thinking=True, grouping="separate")
    assert [metadata for _, _, metadata in separate.ops] == [{"thread_id": "t1"}, {"thread_id": "t1"}]


@pytest.mark.asyncio
async def test_a_bubble_of_thinking_text_alone_folds_counting_no_tools():
    adapter = CaptureAdapter(spoiler=True)
    await _drive(adapter, [("_thinking", "just thinking || here")], fold=True, thinking=True)
    assert adapter.screen() == [f"-# 🛠️ 0 tools · click to show\n||💬\n```\njust thinking |{ZW}| here\n```||"]


def test_plain_lines_in_a_folding_bubble_are_folded_but_not_counted():
    """The first-run hint (our own text) is queued as a plain line: in a bubble holding folded
    lines it goes in the fold too, and only tool lines are counted."""
    lines = [("__fold__", "📖 Reading a.py", True), "💡 First-time tip", ("__fold__", "📖 Reading b.py", True)]
    assert TurnRunner._progress_text(lines) == (
        "-# 🛠️ 2 tools · click to show\n||📖 Reading a.py\n💡 First-time tip\n📖 Reading b.py||")


@pytest.mark.asyncio
async def test_progress_states_and_contexts_without_fold_fields_still_render():
    """Other progress paths hand the drain helpers their own state (the Slack task-card rail's
    text fallback) or a bare context: none carries fold fields, and none may need them."""
    adapter = CaptureAdapter(Platform.SLACK)
    turn = _turn(adapter)
    st = TurnRunner._TaskCardState(adapter)
    st.apply_event({"type": "tool.started", "tool_call_id": "c1", "tool_name": "read_file", "preview": "a.py"})
    await turn._task_card_send_or_edit_fallback(st)
    assert adapter.ops == [("send", st.fallback_text(), {"thread_id": "t1"})]
    ctx = SimpleNamespace(source=None, progress_mode="all", last_was_terminal_block=[False],
                          last_progress_msg=[None], repeat_count=[0], progress_queue=queue.Queue())
    bare = TurnRunner(SimpleNamespace(_delivery_adapter_for=lambda source: adapter), ctx)
    bare._stream_consumer = lambda: None
    line = bare._progress_build_message("read_file", "a.py", {"path": "a.py"})
    bare._progress_emit(line)
    bare._progress_emit(line)
    lines = SimpleNamespace(progress_lines=[])
    assert [bare._progress_absorb(lines, ctx.progress_queue.get()) for _ in range(2)] == [line, f"{line} (×2)"]


# ── fold on: splitting ───────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("grouping", ["accumulate", "separate"])
@pytest.mark.parametrize("limit", [2000, 400])
async def test_every_message_is_self_contained_within_the_cap(grouping, limit):
    command = " && ".join(f"step{i} --flag=value{i} || true" for i in range(150))
    adapter = CaptureAdapter(spoiler=True, limit=limit)
    await _drive(adapter, [
        ("_thinking", "plan: build, then test"),
        _tool("read_file", "a.py", {"path": "a.py"}),
        _tool("terminal", "run", {"command": command}),
        _tool("read_file", "b.py", {"path": "b.py"}),
    ], fold=True, thinking=True, grouping=grouping)
    for _, content, _ in adapter.ops:  # every send and every edit, not just the final screen
        if "||" in content:
            _well_formed(content, limit - 64)
    shown = adapter.screen()
    assert len(shown) >= 3
    code = " ".join(re.findall(r"💻 terminal(?: …)?\n```\n(.*?)\n```", "\n".join(shown), re.DOTALL))
    assert code.split() == command.replace("||", f"|{ZW}|").split()  # whole, split only at spaces
    continuations = [m.split("||", 1)[1] for m in shown if "||💻 terminal …\n```\n" in m]
    assert continuations, shown  # a continuation opens its spoiler on a repeated header, not a fence
    assert sum(m.count("📖 Reading ") for m in shown) == 2


@pytest.mark.asyncio
async def test_many_tools_roll_over_into_folded_continuations():
    adapter = CaptureAdapter(spoiler=True, limit=500)
    events = [_tool("read_file", f"src/file_{i:03d}.py", {"path": f"src/file_{i:03d}.py"}) for i in range(40)]
    await _drive(adapter, events, fold=True)
    shown = adapter.screen()
    assert len(shown) > 2
    for message in shown:
        _well_formed(message, 500 - 64)
    counts = [int(re.search(r"-# 🛠️ (\d+) tool", m).group(1)) for m in shown]
    assert sum(counts) == 40 and counts == [m.count("📖") for m in shown]


@pytest.mark.asyncio
async def test_long_thinking_text_continues_under_its_prefix():
    thinking = " ".join(f"step {i}: a || b then ```x``` and | c |" for i in range(60))
    adapter = CaptureAdapter(spoiler=True, limit=500)
    await _drive(adapter, [_tool("read_file", "a.py", {"path": "a.py"}), ("_thinking", thinking)],
                 fold=True, thinking=True)
    shown = adapter.screen()
    assert len(shown) > 2
    for message in shown:
        _well_formed(message, 500 - 64)
    assert all(m.split("||", 1)[1].startswith("💬 …\n```\n") for m in shown[1:]), shown
    code = " ".join(re.findall(r"💬(?: …)?\n```\n(.*?)\n```", "\n".join(shown), re.DOTALL))
    assert code.split() == run_turn_progress._fold_safe(thinking).split()

"""Gateway rendering tests for opt-in tool-progress comment descriptions."""

import importlib
import inspect
from queue import Empty, Queue
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.display import (
    get_friendly_tool_labels,
    get_tool_preview_max_len,
    prepare_tool_progress_comment_description,
    set_friendly_tool_labels,
    set_tool_preview_max_len,
)
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext


class _ProgressAdapter:
    def __init__(self, *, supports_code_blocks=False):
        self.supports_code_blocks = supports_code_blocks
        self.previews = []

    def format_tool_preview(self, preview):
        self.previews.append(preview)
        return preview.text


class _Runner:
    def __init__(self, adapter):
        self.adapter = adapter

    def _adapter_for_source(self, source):
        return self.adapter


@pytest.fixture(autouse=True)
def _restore_display_globals(monkeypatch):
    friendly = get_friendly_tool_labels()
    preview_len = get_tool_preview_max_len()
    set_friendly_tool_labels(True)
    set_tool_preview_max_len(0)
    monkeypatch.setattr(
        "agent.display.get_tool_emoji",
        lambda tool_name, default="⚙️": {
            "terminal": "💻",
            "execute_code": "🐍",
            "browser_exec": "🌐",
        }.get(tool_name, default),
    )
    yield
    set_friendly_tool_labels(friendly)
    set_tool_preview_max_len(preview_len)


def _make_turn(
    *,
    enabled,
    mode="all",
    supports_code_blocks=False,
    log=False,
    friendly_labels=None,
    preview_max_len=None,
):
    adapter = _ProgressAdapter(supports_code_blocks=supports_code_blocks)
    progress_queue = None if log else Queue()
    log_queue = Queue() if log else None
    ctx = TurnContext(
        source=SimpleNamespace(chat_id="chat-1"),
        _run_still_current=lambda: True,
        progress_mode=mode,
        tool_progress_enabled=not log,
        tool_progress_comment_descriptions=enabled,
        friendly_tool_labels=(
            get_friendly_tool_labels() if friendly_labels is None else friendly_labels
        ),
        tool_preview_max_len=(
            get_tool_preview_max_len() if preview_max_len is None else preview_max_len
        ),
        progress_queue=progress_queue,
        log_queue=log_queue,
    )
    return TurnRunner(_Runner(adapter), ctx), ctx, adapter


def _started(turn, tool_name, args, *, preview=None):
    if preview is None:
        preview = args.get("command") or args.get("code")
    turn.progress_callback("tool.started", tool_name, preview, args)


@pytest.mark.parametrize(
    ("module_name", "adapter_name"),
    [
        ("plugins.platforms.telegram.adapter", "TelegramAdapter"),
        ("plugins.platforms.slack.adapter", "SlackAdapter"),
        ("plugins.platforms.mattermost.adapter", "MattermostAdapter"),
        ("plugins.platforms.discord.adapter", "DiscordAdapter"),
    ],
)
def test_inert_description_stays_inert_after_target_platform_formatting(
    module_name,
    adapter_name,
):
    preview = prepare_tool_progress_comment_description(
        "execute_code",
        {
            "code": (
                "# **bold** [link](https://example.test) @channel "
                "<!channel> :party:\nprint('ok')"
            )
        },
        max_len=0,
    )
    assert preview is not None

    # Missing SDKs fail this security test: a skipped adapter proves nothing.
    adapter_module = importlib.import_module(module_name)
    adapter_type = getattr(adapter_module, adapter_name)
    adapter = adapter_type.__new__(adapter_type)
    formatted = adapter.format_message(
        f"Running code: {adapter.format_tool_preview(preview)}"
    )
    assert preview.url is None

    assert preview.text in formatted
    for active in (
        "**bold**",
        "[link](https://example.test)",
        "@channel",
        "<!channel>",
        ":party:",
    ):
        assert active not in formatted


def test_all_mode_renders_friendly_terminal_comment_description():
    turn, ctx, _ = _make_turn(enabled=True, supports_code_blocks=True)

    _started(turn, "terminal", {"command": "# Check branch and status #\ngit status"})

    assert ctx.progress_queue.get_nowait() == "💻 Running: Check branch and status"
    assert ctx.last_was_terminal_block[0] is False


def test_all_mode_renders_friendly_execute_code_comment_description():
    turn, ctx, _ = _make_turn(enabled=True)

    _started(turn, "execute_code", {"code": "# Analyze the test results\nprint('ok')"})

    assert (
        ctx.progress_queue.get_nowait() == "🐍 Running code: Analyze the test results"
    )


def test_raw_labels_keep_existing_tool_chrome():
    set_friendly_tool_labels(False)
    turn, ctx, _ = _make_turn(enabled=True)

    _started(turn, "terminal", {"command": "# Check status\ngit status"})

    assert ctx.progress_queue.get_nowait() == '💻 terminal: "Check status"'


def test_disabled_setting_keeps_terminal_fallback_without_comment_label():
    turn, ctx, _ = _make_turn(enabled=False, supports_code_blocks=True)

    _started(turn, "terminal", {"command": "# Check status\ngit status"})

    rendered = ctx.progress_queue.get_nowait()
    assert rendered.startswith("💻 terminal\n```\n")
    assert rendered.endswith("\n```")
    assert "💻 Running: Check status" not in rendered
    assert any(fallback in rendered for fallback in ("# Check status", "git status"))
    assert ctx.last_was_terminal_block[0] is True


def test_invalid_comment_falls_back_to_legacy_preview():
    turn, ctx, _ = _make_turn(enabled=True, supports_code_blocks=False)

    _started(turn, "terminal", {"command": "git status"})

    assert ctx.progress_queue.get_nowait() == "💻 Running git status"


def test_invalid_fallbacks_use_each_turns_frozen_cap_and_label_mode():
    terminal_turn, terminal_ctx, _ = _make_turn(
        enabled=True,
        supports_code_blocks=True,
        friendly_labels=True,
        preview_max_len=20,
    )
    execute_turn, execute_ctx, _ = _make_turn(
        enabled=True,
        friendly_labels=False,
        preview_max_len=5,
    )

    # Simulate each turn overwriting the legacy module globals after the other
    # turn captured its presentation settings.
    set_friendly_tool_labels(False)
    set_tool_preview_max_len(5)
    _started(
        terminal_turn,
        "terminal",
        {"command": "git status --short --branch"},
    )

    set_friendly_tool_labels(True)
    set_tool_preview_max_len(40)
    _started(
        execute_turn,
        "execute_code",
        {"code": "print('long fallback payload')"},
    )

    assert terminal_ctx.progress_queue.get_nowait() == (
        "💻 terminal\n```\ngit status --shor...\n```"
    )
    execute_rendered = execute_ctx.progress_queue.get_nowait()
    raw_prefix = '🐍 execute_code: "'
    assert execute_rendered.startswith(raw_prefix)
    assert execute_rendered.endswith('"')
    assert len(execute_rendered[len(raw_prefix) : -1]) <= 5


@pytest.mark.parametrize("mode", ["verbose", "full"])
def test_detailed_modes_ignore_comment_description(mode):
    turn, ctx, _ = _make_turn(
        enabled=True,
        mode=mode,
        supports_code_blocks=True,
    )
    code = "# Check status\ngit status"

    _started(turn, "terminal", {"command": code})

    entry = ctx.progress_queue.get_nowait()
    rendered = entry[1] if isinstance(entry, tuple) else entry
    assert "Running: Check status" not in rendered
    if mode == "full":
        # B has no full branch: an unnormalised value retains its compact fence.
        assert rendered == "💻 terminal\n```\n# Check status ...\n```"
    else:
        assert code in rendered


def test_log_mode_keeps_legacy_preview():
    turn, ctx, _ = _make_turn(enabled=True, mode="log", log=True)
    code = "# Check status\ngit status"

    _started(turn, "terminal", {"command": code})

    entry = ctx.log_queue.get_nowait()
    assert 'terminal: "# Check status' in entry
    assert "Running: Check status" not in entry


def test_comment_description_resets_terminal_fence_header_state():
    turn, ctx, _ = _make_turn(enabled=True, supports_code_blocks=True)

    _started(turn, "terminal", {"command": "git status"})
    _started(turn, "terminal", {"command": "# Run tests\npytest -q"})
    _started(turn, "terminal", {"command": "git diff --check"})

    first = ctx.progress_queue.get_nowait()
    second = ctx.progress_queue.get_nowait()
    third = ctx.progress_queue.get_nowait()
    assert first.startswith("💻 terminal\n```")
    assert second == "💻 Running: Run tests"
    assert third.startswith("💻 terminal\n```")


def test_all_mode_deduplicates_identical_labels_from_different_code():
    turn, ctx, _ = _make_turn(enabled=True)

    _started(turn, "execute_code", {"code": "# Analyze results\nprint(1)"})
    _started(turn, "execute_code", {"code": "# Analyze results\nprint(2)"})

    first = ctx.progress_queue.get_nowait()
    second = ctx.progress_queue.get_nowait()
    assert first == "🐍 Running code: Analyze results"
    assert second == ("__dedup__", first, 1)


def test_new_mode_still_suppresses_repeated_tool_name():
    turn, ctx, _ = _make_turn(enabled=True, mode="new")

    _started(turn, "execute_code", {"code": "# First analysis\nprint(1)"})
    _started(turn, "execute_code", {"code": "# Second analysis\nprint(2)"})

    assert ctx.progress_queue.get_nowait() == "🐍 Running code: First analysis"
    with pytest.raises(Empty):
        ctx.progress_queue.get_nowait()


def test_browser_exec_preview_is_unchanged():
    turn, ctx, _ = _make_turn(enabled=True)
    code = "# Browser label\nreturn 1"

    _started(turn, "browser_exec", {"code": code})

    assert (
        ctx.progress_queue.get_nowait() == '🌐 browser_exec: "Browser label"'
    )


def test_comment_description_uses_existing_preview_cap():
    set_tool_preview_max_len(20)
    turn, ctx, _ = _make_turn(enabled=True)

    _started(turn, "terminal", {"command": "# " + "x" * 80 + "\necho ok"})

    rendered = ctx.progress_queue.get_nowait()
    assert rendered == "💻 Running: " + "x" * 17 + "．．．"


def test_rendered_description_neutralizes_mentions_and_markdown():
    turn, ctx, _ = _make_turn(enabled=True)

    _started(
        turn,
        "execute_code",
        {"code": "# **bold** @channel <@123> `code`\nprint('ok')"},
    )

    rendered = ctx.progress_queue.get_nowait()
    assert rendered.startswith("🐍 Running code: ")
    for active in ("**bold**", "@channel", "<@123>", "`code`"):
        assert active not in rendered


def test_turn_local_settings_do_not_interfere_when_interleaved():
    enabled_turn, enabled_ctx, _ = _make_turn(enabled=True)
    disabled_turn, disabled_ctx, _ = _make_turn(enabled=False)

    _started(enabled_turn, "execute_code", {"code": "# First label\nprint(1)"})
    _started(disabled_turn, "execute_code", {"code": "# Hidden label\nprint(2)"})
    _started(enabled_turn, "terminal", {"command": "# Second label\necho ok"})

    assert enabled_ctx.progress_queue.get_nowait() == "🐍 Running code: First label"
    assert enabled_ctx.progress_queue.get_nowait() == "💻 Running: Second label"
    assert disabled_ctx.progress_queue.get_nowait() == (
        "🐍 Running code # Hidden label print(2)"
    )


def test_comment_snapshot_fields_preserve_turn_context_positional_signature():
    parameters = inspect.signature(TurnContext).parameters

    for name in (
        "tool_progress_comment_descriptions",
        "friendly_tool_labels",
        "tool_preview_max_len",
    ):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["progress_queue"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_comment_presentation_settings_are_frozen_per_turn():
    first_queue = Queue()
    second_queue = Queue()
    first_ctx = TurnContext(
        source=SimpleNamespace(chat_id="chat-first"),
        _run_still_current=lambda: True,
        progress_mode="all",
        tool_progress_enabled=True,
        tool_progress_comment_descriptions=True,
        friendly_tool_labels=True,
        tool_preview_max_len=20,
        progress_queue=first_queue,
    )
    second_ctx = TurnContext(
        source=SimpleNamespace(chat_id="chat-second"),
        _run_still_current=lambda: True,
        progress_mode="all",
        tool_progress_enabled=True,
        tool_progress_comment_descriptions=True,
        friendly_tool_labels=False,
        tool_preview_max_len=8,
        progress_queue=second_queue,
    )
    first = TurnRunner(cast(Any, _Runner(_ProgressAdapter())), first_ctx)
    second = TurnRunner(cast(Any, _Runner(_ProgressAdapter())), second_ctx)
    command = "# " + "x" * 40 + "\necho ok"

    set_friendly_tool_labels(False)
    set_tool_preview_max_len(5)
    _started(first, "terminal", {"command": command})
    set_friendly_tool_labels(True)
    set_tool_preview_max_len(40)
    _started(second, "terminal", {"command": command})

    first_rendered = first_queue.get_nowait()
    second_rendered = second_queue.get_nowait()
    assert first_rendered == "💻 Running: " + "x" * 17 + "．．．"
    assert second_rendered == '💻 terminal: "' + "x" * 5 + '．．．"'


@pytest.mark.parametrize("mode", ["off", "log"])
def test_noncompact_modes_never_use_comment_descriptions_with_a_progress_queue(mode):
    turn, ctx, _ = _make_turn(enabled=True, mode=mode)

    _started(turn, "terminal", {"command": "# Check status\ngit status"})

    rendered = ctx.progress_queue.get_nowait()
    assert "Running: Check status" not in rendered

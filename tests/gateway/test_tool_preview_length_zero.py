"""``display.tool_preview_length: 0`` means "no limit" on every preview path (#51067).

The value is documented as 0 = no limit (``hermes_cli/config_defaults.py``, config
reference, ``agent.display.set_tool_preview_max_len``) and the verbose branch has always
honoured it, but the persistent ``all``/``new`` progress paths coerced 0 back to a 40-char
cap — silently truncating the terminal commands and paths those bubbles exist to show.
An *unset* key keeps the compact 40-char budget those paths were designed around.
"""

from __future__ import annotations

import types

import pytest


@pytest.fixture(autouse=True)
def _reset_preview_len():
    from agent.display import set_tool_preview_max_len

    set_tool_preview_max_len(0)
    yield
    set_tool_preview_max_len(0)


def test_unset_tool_preview_length_keeps_the_compact_default():
    """No configured value and no platform tier → the persistent bubbles keep 40."""
    from gateway.display_config import resolve_display_setting

    assert resolve_display_setting({}, "no_tier_platform", "tool_preview_length") == 40


def test_explicit_zero_reaches_the_progress_path_as_no_limit():
    """A configured 0 must not be rewritten to 40 before the renderers see it."""
    from agent.display import set_tool_preview_max_len
    from gateway.run_turn_progress import GatewayTurnProgressMixin

    set_tool_preview_max_len(0)

    assert GatewayTurnProgressMixin._preview_cap() == 0


def test_terminal_block_shows_the_whole_command_at_zero_cap():
    """cap 0 → full command; a positive cap still truncates to one line."""
    from agent.display import set_tool_preview_max_len
    from gateway.run_turn_progress import GatewayTurnProgressMixin

    class _Progress(GatewayTurnProgressMixin):
        def __init__(self) -> None:
            self._ctx = types.SimpleNamespace(last_was_terminal_block=[False])

    adapter = types.SimpleNamespace(supports_code_blocks=True)
    command = "cd /srv/app && journalctl -u hermes-gateway --since '10 min ago' | tail -40"
    args = {"command": command}

    set_tool_preview_max_len(0)
    _full, short = _Progress()._progress_terminal_blocks(adapter, "terminal", args, "💻")
    assert command in short
    assert "..." not in short

    set_tool_preview_max_len(40)
    _full, short = _Progress()._progress_terminal_blocks(adapter, "terminal", args, "💻")
    assert command not in short
    assert short.splitlines()[-2].endswith("...")


def test_multiline_command_is_shown_whole_when_it_fits_the_budget():
    """A multi-line command inside the budget is not collapsed to its first line + " ..."."""
    from agent.display import set_tool_preview_max_len
    from gateway.run_turn_progress import GatewayTurnProgressMixin

    class _Progress(GatewayTurnProgressMixin):
        def __init__(self) -> None:
            self._ctx = types.SimpleNamespace(last_was_terminal_block=[False])

    adapter = types.SimpleNamespace(supports_code_blocks=True)
    command = "cd /srv/app\ngit log --oneline -5"

    set_tool_preview_max_len(200)
    _full, short = _Progress()._progress_terminal_blocks(adapter, "terminal", {"command": command}, "💻")
    assert command in short
    assert "..." not in short


def test_streamed_tool_chrome_honours_zero_cap():
    """The base adapter's structured tool chrome reads 0 as no cap too."""
    from agent.display import set_tool_preview_max_len
    from gateway.platforms.base import BasePlatformAdapter
    from gateway.stream_events import ToolCallChunk

    class _Adapter(BasePlatformAdapter):
        """Minimal concrete adapter: format_tool_event is pure presentation."""

        async def connect(self):
            return True

        async def disconnect(self):
            return None

        async def get_chat_info(self, chat_id):
            return {}

        async def send(self, chat_id, content, **kwargs):
            return None

    set_tool_preview_max_len(0)
    adapter = object.__new__(_Adapter)
    query = "hong kong stamp duty first time buyer 2026 official ird government rates"
    event = ToolCallChunk("web_search", preview=query, args={"query": query})

    out = adapter.format_tool_event(event, mode="all", preview_max_len=0)
    assert out is not None
    assert query in out

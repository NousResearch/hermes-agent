"""Tool-call rows, result previews, grouped read-only calls, collapsed reasoning and inline
diffs in the classic CLI's scrollback, plus the bounded log the Ctrl+O viewer reads.

Mixin bound onto ``HermesCLI``; cli.py names are imported lazily (``from cli import ...``) so the
facade's monkeypatch seams hold. Display-only: nothing here changes what the model is sent.
Scrollback is append-only, so a run of read-only calls is HELD and printed as one row when the
next non-read event arrives, never by rewriting lines already printed.
"""
from __future__ import annotations

import shutil
import time

from hermes_cli import cli_tool_render as render
from hermes_cli.cli_transcript_viewer import TranscriptLog

_GROUP_MODES = frozenset({"new", "all"})  # "verbose" prints every call with its own preview
_SCROLLBACK_MODES = frozenset({"new", "all", "verbose"})
_VERBOSE_PREVIEW_FACTOR = 4
_EXPLORED_SENTINEL = "\x00explored"


def _columns() -> int:
    return shutil.get_terminal_size((80, 24)).columns  # falls back when stdout is not a terminal


class CLIToolActivityMixin:
    """Completed-tool rendering for the classic CLI (see module docstring)."""

    # ── state (lazy: tests build HermesCLI via __new__) ─────────────────────────────────────
    def _transcript(self) -> TranscriptLog:
        log = self.__dict__.get("_transcript_log")
        if log is None:
            log = self._transcript_log = TranscriptLog()
        return log

    def _transcript_begin_turn(self) -> None:
        self._transcript().begin_turn()
        self._explore_pending = None
        self._reasoning_capture = None

    def _tool_preview_line_budget(self) -> int:
        lines = max(0, int(getattr(self, "_tool_output_preview", 3) or 0))
        return lines * _VERBOSE_PREVIEW_FACTOR if getattr(self, "tool_progress_mode", "all") == "verbose" else lines

    def _reasoning_collapsed(self) -> bool:
        return getattr(self, "cli_reasoning_style", "full") == "collapsed"

    # ── tool completion ─────────────────────────────────────────────────────────────────────
    def _tool_activity_started(self, name: str | None) -> None:
        """A tool is starting: end the reasoning block before it; for a non-read tool also print
        held reads, so an approval prompt never sits above rows that happened before it."""
        self._finish_reasoning_capture()
        if name and not render.explore_kind(name):
            self._flush_explore_group()

    def _tool_activity_completed(self, name: str, args: dict, duration: float, result, is_error: bool) -> None:
        """``tool.completed`` in a scrollback mode (new/all/verbose): log it, then print or hold it."""
        rows, failed = render.render_tool_row(name, args, duration, result, is_error=is_error)
        self._transcript().add_tool(name, args, result, duration=duration, failed=failed)
        mode = getattr(self, "tool_progress_mode", "all")
        if mode not in _SCROLLBACK_MODES:
            return
        if mode in _GROUP_MODES and not failed and render.explore_kind(name):
            pending = self.__dict__.get("_explore_pending") or render.ExploreGroup()
            pending.add(name, args, duration, result)
            self._explore_pending = pending
            return
        self._flush_explore_group()
        # "new" collapses repeats of one tool, but each todo update is a new plan state.
        if mode == "new" and name == getattr(self, "_last_scrollback_tool", None) and name not in render.TODO_TOOLS:
            return
        self._last_scrollback_tool = name
        self._print_tool_block(rows, name, result, failed, args)

    def _print_tool_block(self, rows: list[str], name: str, result, failed: bool, args: dict | None = None) -> None:
        from cli import _cprint
        for row in rows:
            _cprint(row)
        if name in render.TODO_TOOLS and not failed:
            # The plan itself is the result: a checklist, independent of the preview budget.
            for line in render.render_todo_checklist(render.todo_items(args, result), _columns()):
                _cprint(line)
            return
        budget = self._tool_preview_line_budget()
        if budget <= 0:
            return
        expand_hint = getattr(self, "_app", None) is not None  # Ctrl+O only exists in the TUI
        for line in render.render_preview(name, result, failed=failed, max_lines=budget, width=_columns(),
                                          expand_hint=expand_hint):
            _cprint(line)

    def _flush_explore_group(self) -> None:
        """Print held read-only calls: one ordinary row for a single call, else one "explored" row."""
        from cli import _cprint
        group = self.__dict__.get("_explore_pending")
        if not group:
            return
        self._explore_pending = None
        if len(group) == 1:
            name, args, duration, result = group.calls[0]
            rows, failed = render.render_tool_row(name, args, duration, result)
            self._last_scrollback_tool = name
            self._print_tool_block(rows, name, result, failed, args)
            return
        self._last_scrollback_tool = _EXPLORED_SENTINEL
        _cprint(render.render_explore_row(group, _columns()))

    # ── inline diffs ────────────────────────────────────────────────────────────────────────
    def _on_inline_diff(self, tool_name: str, diff_text: str) -> None:
        """An edit's rendered diff (gateway ``tool.complete.inline_diff`` or the local snapshot)."""
        from cli import _cprint
        from hermes_cli.cli_diff_render import plain_diff, render_inline_diff
        if not diff_text:
            return
        self._transcript().attach_diff("", tool_name, plain_diff(diff_text))
        if not getattr(self, "_inline_diffs_enabled", True):
            return
        self._flush_explore_group()
        for line in render_inline_diff(diff_text, _columns()):
            _cprint(line)

    # ── reasoning ───────────────────────────────────────────────────────────────────────────
    def _capture_reasoning(self, text: str) -> None:
        """Accumulate this block's reasoning for the viewer (and the collapsed line)."""
        capture = self.__dict__.get("_reasoning_capture")
        if capture is None:
            self._flush_explore_group()
            capture = self._reasoning_capture = {"text": "", "started": time.monotonic()}
        capture["text"] += text

    def _finish_reasoning_capture(self) -> None:
        """End of a reasoning block: log it; in collapsed style print the one ``✻ Thought`` line."""
        from cli import _cprint
        capture = self.__dict__.get("_reasoning_capture")
        if capture is None:
            return
        self._reasoning_capture = None
        text = capture["text"]
        if not text.strip():
            return
        seconds = time.monotonic() - capture["started"]
        self._transcript().add_reasoning(text, seconds)
        if self._reasoning_collapsed():
            _cprint(render.render_reasoning_line(text, seconds, _columns()))

    def _print_collapsed_reasoning_recap(self, reasoning: str) -> None:
        """Non-streamed fallback (reasoning only known at turn end) in collapsed style. The
        caller has already logged it for Ctrl+O."""
        from cli import _cprint
        _cprint(render.render_reasoning_line(reasoning, None, _columns()))

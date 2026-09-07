"""Streaming output, reasoning preview, tool progress callbacks, and busy-command spinner for the interactive CLI

Mixin split out of ``cli.py``; bound onto ``HermesCLI`` via the MRO. cli.py-internal
symbols are imported LAZILY inside each method (``from cli import ...``) — the mixin
never imports ``cli`` at module load time (import cycle).
"""

from __future__ import annotations

import json
import re
import shutil
import textwrap
import time

from contextlib import contextmanager
from pathlib import Path
from rich.markup import escape as _escape

# Model-generated reasoning tags: suppressed during streaming (they'd display as raw XML;
# the agent strips them from final_response too) unless show_reasoning routes them to the box.
_OPEN_TAGS = (
    "<REASONING_SCRATCHPAD>", "<think>", "<reasoning>", "<THINKING>", "<thinking>", "<thought>")
_CLOSE_TAGS = tuple("</" + t[1:] for t in _OPEN_TAGS)
_MAX_CLOSE_TAG_LEN = max(len(t) for t in _CLOSE_TAGS)

# Ordered (prefix, status) rows for _slow_command_status — first match wins.
_SLOW_COMMAND_STATUS = (
    ("/skills search", "Searching skills..."), ("/skills browse", "Loading skills..."),
    ("/skills inspect", "Inspecting skill..."), ("/skills install", "Installing skill..."),
    ("/skills", "Processing skills command..."), ("/browser", "Configuring browser..."))
_SLOW_COMMAND_STATUS_EXACT = {
    "/reload-mcp": "Reloading MCP servers...",
    "/reload-skills": "Reloading skills...",
    "/reload_skills": "Reloading skills..."}


def _terminal_columns(default: int = 80) -> int:
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return default


class CLIStreamMixin:
    """Streaming output, reasoning preview, tool progress callbacks, and busy-command spinner for the interactive CLI"""

    def _on_thinking(self, text: str) -> None:
        """Called by agent when thinking starts/stops. Updates TUI spinner."""
        if not text:
            self._flush_reasoning_preview(force=True)
        self._spinner_text = text or ""
        self._tool_start_time = 0.0  # clear tool timer when switching to thinking
        self._invalidate()

    def _on_notice(self, notice) -> None:
        """Queue an out-of-band AgentNotice for rendering at the next clean boundary.

        Notices fire mid-turn (cold-start seed, per-turn _capture_credits); printing immediately
        races the stream and buries the line behind the prompt. Flushed by _flush_credit_notices()
        after run_conversation returns. Fail-soft.
        """
        try:
            text = getattr(notice, "text", "") or ""
            if not text:
                return
            level = getattr(notice, "level", "info") or "info"
            if not hasattr(self, "_pending_credit_notices"):
                self._pending_credit_notices = []
            self._pending_credit_notices.append((level, text))
        except Exception:
            pass

    def _flush_credit_notices(self) -> None:
        """Print queued credit notices as level-colored lines at turn end (after
        run_conversation) where _cprint paints cleanly above the prompt."""
        from cli import _DIM, _RST, _cprint
        try:
            pending = getattr(self, "_pending_credit_notices", None)
            if not pending:
                return
            self._pending_credit_notices = []
            colors = {"error": "\033[31m", "warn": "\033[33m", "success": "\033[32m", "info": _DIM}
            for level, text in pending:
                _cprint(f"  {colors.get(level, _DIM)}{text}{_RST}")
        except Exception:
            pass

    def _on_notice_clear(self, key: str) -> None:
        """No-op for the REPL (lines are printed, no persistent slot to wipe); kept so the
        agent's clear callback is bound symmetrically with the show callback."""
        return

    def _current_reasoning_callback(self):
        """Return the active reasoning display callback for the current mode."""
        if self.show_reasoning and self.streaming_enabled:
            return self._stream_reasoning_delta
        if self.verbose and not self.show_reasoning:
            return self._on_reasoning
        return None

    def _emit_reasoning_preview(self, reasoning_text: str) -> None:
        """Render a buffered reasoning preview as a single [thinking] block."""
        from cli import _DIM, _RST, _cprint
        preview_text = reasoning_text.strip()
        if not preview_text:
            return
        wrap_width = max(30, _terminal_columns() - len("  [thinking] ") - 2)
        paragraphs = []
        for paragraph in re.split(r"\n\s*\n+", preview_text.replace("\r\n", "\n")):
            compact = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
            if compact:
                paragraphs.append(textwrap.fill(compact, width=wrap_width))
        preview_text = "\n".join(paragraphs)
        if not preview_text:
            return
        if self.verbose:
            _cprint(f"  {_DIM}[thinking] {preview_text}{_RST}")
            return
        lines = preview_text.splitlines()
        if len(lines) > 5:
            preview = "\n".join(lines[:5]) + f"\n  ... ({len(lines) - 5} more lines)"
        else:
            preview = preview_text
        _cprint(f"  {_DIM}[thinking] {preview}{_RST}")

    def _flush_reasoning_preview(self, *, force: bool = False) -> None:
        """Flush buffered reasoning text at natural boundaries.

        Some providers stream reasoning in tiny word/punctuation chunks; buffering keeps the
        preview path from printing one `[thinking]` line per token.
        """
        buf = getattr(self, "_reasoning_preview_buf", "")
        if not buf:
            return
        target_width = max(40, _terminal_columns() - len("  [thinking] ") - 4)
        flush_text = ""
        if force:
            flush_text, buf = buf, ""
        else:
            line_break = buf.rfind("\n")
            min_newline_flush = max(16, target_width // 3)
            if line_break != -1 and (
                line_break >= min_newline_flush
                or buf.endswith(("\n\n", ".\n", "!\n", "?\n", ":\n"))):
                flush_text, buf = buf[: line_break + 1], buf[line_break + 1 :]
            elif len(buf) >= target_width:
                search_start = max(20, target_width // 2)
                search_end = min(
                    len(buf), max(target_width + (target_width // 3), target_width + 8))
                cut = max(
                    buf.rfind(b, search_start, search_end)
                    for b in (" ", "\t", ".", "!", "?", ",", ";", ":"))
                if cut != -1:
                    flush_text, buf = buf[: cut + 1], buf[cut + 1 :]

        self._reasoning_preview_buf = buf.lstrip() if flush_text else buf
        if flush_text:
            self._emit_reasoning_preview(flush_text)

    def _format_submitted_user_message_preview(self, user_input: str) -> str:
        """Format the submitted user-message scrollback preview."""
        from cli import _accent_hex, datetime
        ts_suffix = (
            f" [dim]{datetime.now().strftime(getattr(self, 'timestamp_format', '%H:%M'))}[/]"
            if getattr(self, "show_timestamps", False) else "")
        lines = user_input.split("\n")
        if len(lines) <= 1:
            return f"[bold {_accent_hex()}]●[/] [bold]{_escape(user_input)}[/]{ts_suffix}"

        first_lines = max(1, int(getattr(self, "user_message_preview_first_lines", 2)))
        last_lines = max(0, int(getattr(self, "user_message_preview_last_lines", 2)))
        head = lines[:first_lines]
        tail_count = min(last_lines, max(0, len(lines) - len(head)))
        tail = lines[-tail_count:] if tail_count else []
        hidden_middle_count = len(lines) - len(head) - len(tail)
        if hidden_middle_count < 0:
            hidden_middle_count = 0
            tail = []

        preview_lines = [f"[bold {_accent_hex()}]●[/] [bold]{_escape(head[0])}[/]{ts_suffix}"]
        preview_lines.extend(f"[bold]{_escape(line)}[/]" for line in head[1:])
        if hidden_middle_count > 0:
            noun = "line" if hidden_middle_count == 1 else "lines"
            preview_lines.append(f"[dim]... (+{hidden_middle_count} more {noun})[/]")
        preview_lines.extend(f"[bold]{_escape(line)}[/]" for line in tail)
        return "\n".join(preview_lines)

    def _expand_paste_references(self, text: str | None) -> str:
        """Expand [Pasted text #N -> file] placeholders into file contents."""
        from cli import logger
        if not isinstance(text, str) or "[Pasted text #" not in text:
            return text or ""
        paste_ref_re = re.compile(r'\[Pasted text #\d+: \d+ lines \u2192 (.+?)\]')

        def _expand_ref(match):
            path = Path(match.group(1))
            # try/except rather than path.exists(): the paste file may be deleted between
            # check and read (TOCTOU), silently dropping the input.
            try:
                # See #17666.
                return path.read_text(encoding="utf-8")
            except (OSError, IOError):
                logger.warning("Paste file gone or unreadable, returning placeholder: %s", path)
                return match.group(0)

        return paste_ref_re.sub(_expand_ref, text)

    def _print_user_message_preview(self, user_input: str) -> None:
        """Render a user message using the normal chat scrollback style."""
        from cli import ChatConsole, _accent_hex
        ChatConsole().print(f"[{_accent_hex()}]{'─' * 40}[/]")
        text = str(user_input or "")
        if "\n" in text:
            ChatConsole().print(self._format_submitted_user_message_preview(text))
        else:
            ChatConsole().print(f"[bold {_accent_hex()}]●[/] [bold]{_escape(text)}[/]")

    def _stream_reasoning_delta(self, text: str) -> None:
        """Stream reasoning tokens into a dim box above the response.

        Opened on the first token, closed when content arrives (_emit_stream_text). Once the
        response box is open further reasoning is suppressed — a late thinking block (e.g. after
        an interrupt) would otherwise draw a reasoning box inside the response box.
        """
        from cli import _DIM, _RST, _cprint
        if not text:
            return
        self._reasoning_shown_this_turn = True
        if getattr(self, "_stream_box_opened", False):
            return
        if not getattr(self, "_reasoning_box_opened", False):
            self._reasoning_box_opened = True
            w = self._scrollback_box_width()
            r_label = " Reasoning "
            r_fill = w - 2 - len(r_label)
            _cprint(f"\n{_DIM}┌─{r_label}{'─' * max(r_fill - 1, 0)}┐{_RST}")

        self._reasoning_buf = getattr(self, "_reasoning_buf", "") + text
        # Emit complete lines; force-flush long partial lines so reasoning is visible in
        # real-time even without newlines.
        while "\n" in self._reasoning_buf:
            line, self._reasoning_buf = self._reasoning_buf.split("\n", 1)
            _cprint(f"{_DIM}{line}{_RST}")
        if len(self._reasoning_buf) > 80:
            _cprint(f"{_DIM}{self._reasoning_buf}{_RST}")
            self._reasoning_buf = ""

    def _close_reasoning_box(self) -> None:
        """Close the live reasoning box if it's open, then flush deferred content."""
        from cli import _DIM, _RST, _cprint
        if not getattr(self, "_reasoning_box_opened", False):
            return
        buf = getattr(self, "_reasoning_buf", "")
        if buf:
            _cprint(f"{_DIM}{buf}{_RST}")
            self._reasoning_buf = ""
        w = self._scrollback_box_width()
        _cprint(f"{_DIM}└{'─' * (w - 2)}┘{_RST}")
        self._reasoning_box_opened = False
        deferred = getattr(self, "_deferred_content", "")
        if deferred:
            self._deferred_content = ""
            self._emit_stream_text(deferred)

    def _stream_delta(self, text) -> None:
        """Line-buffered streaming callback for real-time token rendering.

        Emits complete lines via _cprint (reliable under prompt_toolkit's patch_stdout);
        reasoning tags are suppressed, or routed to the reasoning box when show_reasoning is on.
        ``None`` = intermediate turn boundary (tools about to run): flush boxes and reset state.
        """
        if text is None:
            self._flush_stream()
            self._reset_stream_state()
            return
        if not text:
            return
        self._stream_started = True
        self._stream_prefilt = getattr(self, "_stream_prefilt", "") + text

        # Open tags only count at a "block boundary" (stream start / after a newline plus
        # optional whitespace) so prose that *mentions* a tag — "(/think not producing
        # <think> tags)" — is not swallowed. _stream_last_was_newline tracks the boundary.
        if not hasattr(self, "_stream_last_was_newline"):
            self._stream_last_was_newline = True

        if not getattr(self, "_in_reasoning_block", False):
            # Lowercased view catches mixed-case variants (<Think>, <THINKING>, …).
            prefilt_lower = self._stream_prefilt.lower()
            for tag in _OPEN_TAGS:
                tag_lower = tag.lower()
                search_start = 0
                while True:
                    idx = prefilt_lower.find(tag_lower, search_start)
                    if idx == -1:
                        break
                    preceding = self._stream_prefilt[:idx]
                    # Boundary: only whitespace since the last newline — or, with no newline
                    # buffered yet, since the last emit (which must have ended a line).
                    is_block_boundary = preceding[preceding.rfind("\n") + 1:].strip() == "" and (
                        "\n" in preceding or getattr(self, "_stream_last_was_newline", True))
                    if is_block_boundary:
                        if preceding:
                            self._emit_stream_text(preceding)
                            self._stream_last_was_newline = preceding.endswith("\n")
                        self._in_reasoning_block = True
                        self._stream_prefilt = self._stream_prefilt[idx + len(tag):]
                        break
                    search_start = idx + 1
                if getattr(self, "_in_reasoning_block", False):
                    break

            if not getattr(self, "_in_reasoning_block", False):
                # Hold back a possible partial open tag at the end (case-insensitive).
                safe = self._stream_prefilt
                for tag in _OPEN_TAGS:
                    tag_lower = tag.lower()
                    for i in range(1, len(tag)):
                        if prefilt_lower.endswith(tag_lower[:i]):
                            safe = self._stream_prefilt[:-i]
                            break
                if safe:
                    self._emit_stream_text(safe)
                    self._stream_last_was_newline = safe.endswith("\n")
                    self._stream_prefilt = self._stream_prefilt[len(safe):]
                return

        # Inside a reasoning block — look for a close tag; keep accumulating because close tags
        # can arrive split across tokens ("</REASONING_SCRATCH" + "PAD>...").
        if getattr(self, "_in_reasoning_block", False):
            prefilt_lower = self._stream_prefilt.lower()
            for tag in _CLOSE_TAGS:
                idx = prefilt_lower.find(tag.lower())
                if idx != -1:
                    self._in_reasoning_block = False
                    if self.show_reasoning:
                        inner = self._stream_prefilt[:idx]
                        if inner:
                            self._stream_reasoning_delta(inner)
                    after = self._stream_prefilt[idx + len(tag):]
                    self._stream_prefilt = ""
                    if after:  # re-filter: the remainder could contain another open tag
                        self._stream_delta(after)
                    return
            # Stream reasoning live when show_reasoning is on; keep only a possible partial
            # close-tag tail.
            if len(self._stream_prefilt) > _MAX_CLOSE_TAG_LEN:
                if self.show_reasoning:
                    self._stream_reasoning_delta(self._stream_prefilt[:-_MAX_CLOSE_TAG_LEN])
                self._stream_prefilt = self._stream_prefilt[-_MAX_CLOSE_TAG_LEN:]
            return

    def _emit_stream_line(self, printed_line: str) -> None:
        """Print one response line with the skin's true-color text escape (if any)."""
        from cli import _RST, _STREAM_PAD, _cprint
        _tc = getattr(self, "_stream_text_ansi", "")
        _cprint(
            f"{_STREAM_PAD}{_tc}{printed_line}{_RST}" if _tc else f"{_STREAM_PAD}{printed_line}")

    def _flush_stream_table_buf(self) -> None:
        """Emit the held table block re-aligned as a whole. Cell-level markdown is stripped FIRST
        so the realigner pads to the final visible width, not the marker-decorated width."""
        from cli import (
            _strip_markdown_syntax, _terminal_width_for_streaming, realign_markdown_tables)
        buf = self._stream_table_buf
        self._stream_table_buf = []
        self._in_stream_table = False
        if not buf:
            return
        joined = "\n".join(buf)
        if self.final_response_markdown == "strip":
            joined = _strip_markdown_syntax(joined)
        block = realign_markdown_tables(joined, _terminal_width_for_streaming())
        for ln in block.split("\n"):
            self._emit_stream_line(ln)

    def _emit_stream_text(self, text: str) -> None:
        """Emit filtered text to the streaming display."""
        from agent.markdown_tables import is_table_divider, looks_like_table_row
        from cli import (
            HermesCLI, _ACCENT, _RST, _STREAM_PARTIAL_PREVIEW_LEN, _cprint, _strip_markdown_syntax, datetime)
        if not text:
            return
        # Defer content while the reasoning box renders so reasoning always lands BEFORE it.
        if self.show_reasoning and getattr(self, "_reasoning_box_opened", False):
            self._deferred_content = getattr(self, "_deferred_content", "") + text
            return
        self._close_reasoning_box()

        # Open the response box header on the very first visible text
        if not self._stream_box_opened:
            text = text.lstrip("\n")
            if not text:
                return
            self._stream_box_opened = True
            try:
                from hermes_cli.skin_engine import get_active_skin
                _skin = get_active_skin()
                label = _skin.get_branding("response_label", "⚕ Hermes")
                _text_hex = _skin.get_color("banner_text", "#FFF8DC")
            except Exception:
                label = "⚕ Hermes"
                _text_hex = "#FFF8DC"
            try:  # true-color escape so streamed text matches the Rich Panel appearance
                _r, _g, _b = (int(_text_hex[i:i + 2], 16) for i in (1, 3, 5))
                self._stream_text_ansi = f"\033[38;2;{_r};{_g};{_b}m"
            except (ValueError, IndexError):
                self._stream_text_ansi = ""
            if self.show_timestamps:
                label = f"{label} {datetime.now().strftime(getattr(self, 'timestamp_format', '%H:%M'))}"
            w = self._scrollback_box_width()
            fill = w - 2 - HermesCLI._status_bar_display_width(label)
            _cprint(f"\n{_ACCENT}╭─{label}{'─' * max(fill - 1, 0)}╮{_RST}")

        self._stream_buf += text
        while "\n" in self._stream_buf:
            line, self._stream_buf = self._stream_buf.split("\n", 1)
            # Table rows are held and re-padded as a block once it ends (already-printed rows
            # can't be re-aligned), so a table appears in one batch when the block closes.
            if self._in_stream_table:
                if looks_like_table_row(line) or is_table_divider(line):
                    self._stream_table_buf.append(line)
                    continue
                self._flush_stream_table_buf()
            elif looks_like_table_row(line):
                self._stream_table_buf.append(line)
                self._in_stream_table = True
                continue
            if self.final_response_markdown == "strip":
                line = _strip_markdown_syntax(line)
            self._emit_stream_line(line)

        # Partial lines are emitted ONLY at real newlines (no hard-wrapping — the terminal
        # soft-wraps, so highlight-copy yields the original text). For TTFT perception, mirror
        # the tail of a long unfinished paragraph into the status-bar spinner.
        if (
            self._stream_buf
            and not self._in_stream_table
            and not self._stream_buf.lstrip().startswith("|")
            and len(self._stream_buf) >= 80):
            preview = self._stream_buf[-int(_STREAM_PARTIAL_PREVIEW_LEN):]
            cut = preview.find(" ")
            if 0 < cut < len(preview) - 1:
                preview = preview[cut + 1:]
            try:
                self._spinner_text = f"… {preview}"
                self._invalidate()
            except Exception:
                pass

    def _flush_stream(self) -> None:
        """Emit any remaining partial line from the stream buffer and close the box."""
        from agent.markdown_tables import is_table_divider, looks_like_table_row
        from cli import _ACCENT, _RST, _cprint, _strip_markdown_syntax
        # Still inside a "reasoning block" at end-of-stream = false positive (the model
        # mentioned a tag in prose and never closed it): recover the buffer as regular text.
        if getattr(self, "_in_reasoning_block", False) and getattr(self, "_stream_prefilt", ""):
            self._in_reasoning_block = False
            self._emit_stream_text(self._stream_prefilt)
            self._stream_prefilt = ""
        self._close_reasoning_box()  # in case no content tokens arrived
        # A trailing partial table row joins the table buffer so the whole block is re-aligned
        # together (else the final row prints under-padded).
        if (
            self._stream_buf
            and getattr(self, "_in_stream_table", False)
            and (looks_like_table_row(self._stream_buf) or is_table_divider(self._stream_buf))):
            self._stream_table_buf.append(self._stream_buf)
            self._stream_buf = ""
        if getattr(self, "_stream_table_buf", None):
            self._flush_stream_table_buf()
        if self._stream_buf:
            line = _strip_markdown_syntax(self._stream_buf) if self.final_response_markdown == "strip" else self._stream_buf
            self._emit_stream_line(line)
            self._stream_buf = ""
        if self._stream_box_opened:
            w = self._scrollback_box_width()
            _cprint(f"{_ACCENT}╰{'─' * (w - 2)}╯{_RST}")

    def _reset_stream_state(self) -> None:
        """Reset streaming state before each agent invocation."""
        self._stream_buf = ""
        self._stream_started = False
        self._stream_box_opened = False
        self._stream_text_ansi = ""
        self._stream_prefilt = ""
        self._in_reasoning_block = False
        self._stream_last_was_newline = True
        self._reasoning_box_opened = False
        self._reasoning_buf = ""
        self._reasoning_preview_buf = ""
        self._deferred_content = ""
        self._stream_table_buf = []
        self._in_stream_table = False

    def _slow_command_status(self, command: str) -> str:
        """Return a user-facing status message for slower slash commands."""
        cmd_lower = command.lower().strip()
        exact = _SLOW_COMMAND_STATUS_EXACT.get(cmd_lower)
        if exact:
            return exact
        for prefix, status in _SLOW_COMMAND_STATUS:
            if cmd_lower.startswith(prefix):
                return status
        return "Processing command..."

    def _command_spinner_frame(self) -> str:
        """Return the current spinner frame for slow slash commands."""
        from cli import _COMMAND_SPINNER_FRAMES
        return _COMMAND_SPINNER_FRAMES[int(time.monotonic() * 10) % len(_COMMAND_SPINNER_FRAMES)]

    @contextmanager
    def _busy_command(self, status: str, *, blocks_input: bool = True):
        """Expose a temporary busy state in the TUI while a slash command runs.

        Most sync slash commands reserve the composer (their completion changes session state);
        manual compression is safe to draft through (queued input runs against compacted history).
        """
        previous_blocks_input = getattr(self, "_command_blocks_input", False)
        self._command_running = True
        self._command_blocks_input = blocks_input
        self._command_status = status
        self._invalidate(min_interval=0.0)
        try:
            print(f"⏳ {status}")
            yield
        finally:
            self._command_running = False
            self._command_blocks_input = previous_blocks_input
            self._command_status = ""
            self._invalidate(min_interval=0.0)

    def _preprocess_images_with_vision(self, text: str, images: list, *, announce: bool = True) -> str:
        """Describe attached images via the auxiliary vision model and prepend the descriptions
        to the user's text (works with non-vision models; same approach as the gateway). The
        local path is included so the agent can re-examine via ``vision_analyze``."""
        from cli import _DIM, _RST, _cprint
        import asyncio as _asyncio
        from tools.vision_tools import vision_analyze_tool
        analysis_prompt = (
            "Describe everything visible in this image in thorough detail. "
            "Include any text, code, data, objects, people, layout, colors, "
            "and any other notable visual information.")
        enriched_parts = []
        for img_path in images:
            if not img_path.exists():
                continue
            size_kb = img_path.stat().st_size // 1024
            if announce:
                _cprint(f"  {_DIM}👁️  analyzing {img_path.name} ({size_kb}KB)...{_RST}")
            try:
                result_json = _asyncio.run(
                    vision_analyze_tool(image_url=str(img_path), user_prompt=analysis_prompt))
                result = json.loads(result_json)
                if result.get("success"):
                    description = result.get("analysis", "")
                    enriched_parts.append(
                        f"[The user attached an image. Here's what it contains:\n{description}]\n"
                        f"[If you need a closer look, use vision_analyze with "
                        f"image_url: {img_path}]")
                    if announce:
                        _cprint(f"  {_DIM}✓ image analyzed{_RST}")
                else:
                    enriched_parts.append(
                        f"[The user attached an image but it couldn't be analyzed. "
                        f"You can try examining it with vision_analyze using "
                        f"image_url: {img_path}]")
                    if announce:
                        _cprint(f"  {_DIM}⚠ vision analysis failed — path included for retry{_RST}")
            except Exception as e:
                enriched_parts.append(
                    f"[The user attached an image but analysis failed ({e}). "
                    f"You can try examining it with vision_analyze using "
                    f"image_url: {img_path}]")
                if announce:
                    _cprint(f"  {_DIM}⚠ vision analysis error — path included for retry{_RST}")

        # Vision descriptions first, then the user's original text
        user_text = text if isinstance(text, str) and text else ""
        if enriched_parts:
            prefix = "\n\n".join(enriched_parts)
            return f"{prefix}\n\n{user_text}" if user_text else prefix
        return user_text or "What do you see in this image?"

    def _console_print(self, *args, **kwargs):
        """Print through the active command-safe console (prompt_toolkit-safe Rich once the
        TUI is live, else the plain console)."""
        from cli import ChatConsole
        console = ChatConsole() if getattr(self, "_app", None) else self.console
        console.print(*args, **kwargs)

    def _on_tool_gen_start(self, tool_name: str) -> None:
        """Model began generating tool-call arguments: close open boxes once, then print a status
        line so a large payload (e.g. 45 KB write_file) doesn't look like a frozen screen."""
        from cli import _cprint
        if getattr(self, "_stream_box_opened", False):
            self._flush_stream()
            self._stream_box_opened = False
        self._close_reasoning_box()
        from agent.display import get_tool_emoji
        _cprint(f"  ┊ {get_tool_emoji(tool_name, default='⚡')} preparing {tool_name}…")

    def _on_tool_progress(self, event_type: str, function_name: str = None, preview: str = None, function_args: dict = None, **kwargs):
        """Tool lifecycle events (tool.started / tool.completed / reasoning.* / moa.*).

        Drives the TUI spinner (tool.started stamps the elapsed timer); in "all"/"new"/"verbose"
        progress modes tool.completed also commits a stacked scrollback line (tool history).
        """
        from cli import CLI_CONFIG, _DIM, _RST, _cprint, _hermes_home
        # MoA reference outputs (display-only events from the MoA facade): render each answer
        # as a labelled thinking-style block BEFORE the aggregator acts.
        if event_type == "moa.reference":
            label = function_name or "reference"
            text = preview or ""
            idx = kwargs.get("moa_index")
            count = kwargs.get("moa_count")
            header = f"Reference {idx}/{count} — {label}" if idx and count else f"Reference — {label}"
            try:
                self._flush_reasoning_preview(force=True)
            except Exception:
                pass
            _cprint(f"  {_DIM}┊ ◇ {header}{_RST}")
            try:
                self._emit_reasoning_preview(text)
            except Exception:
                if text.strip():
                    _cprint(f"  {_DIM}{text.strip()}{_RST}")
            self._invalidate()
            return
        if event_type == "moa.aggregating":
            agg = function_name or ""
            self._spinner_text = f"◆ aggregating ({agg})" if agg else "◆ aggregating"
            self._invalidate()
            return

        # Feed the pet: tools mean "running"; a failed tool latches the turn to end on a sulk.
        if event_type == "tool.started":
            self._pet_reasoning = False
        elif event_type == "tool.completed" and kwargs.get("is_error"):
            self._pet_turn_error = True
        elif event_type and event_type.startswith("reasoning"):
            self._pet_reasoning = True

        if event_type == "tool.completed":
            self._tool_start_time = 0.0
            self._turn_summary_record(
                function_name, kwargs.get("result"), kwargs.get("is_error", False))
            # Focus view: count the hidden scrollback line for the post-turn recovery report.
            if getattr(self, "_focus_view_enabled", False):
                try:
                    self._note_focus_hidden_line(function_name or "")
                except Exception:
                    pass
            # "verbose" must commit the same line as "all": non-streaming calls (MoA aggregator,
            # copilot-acp) never emit the "preparing" line, so nothing else builds history.
            if function_name and self.tool_progress_mode in {"new", "all", "verbose"}:
                duration = kwargs.get("duration", 0.0)
                # Pop stored args from tool.started for this function
                stored = self._pending_tool_info.get(function_name)
                stored_args = stored.pop(0) if stored else {}
                if stored is not None and not stored:
                    del self._pending_tool_info[function_name]
                # "new" mode: skip consecutive repeats of the same tool
                if self.tool_progress_mode == "new" and function_name == self._last_scrollback_tool:
                    self._invalidate()
                    return
                self._last_scrollback_tool = function_name
                try:
                    from agent.display import get_cute_tool_message
                    line = get_cute_tool_message(function_name, stored_args, duration, result=kwargs.get("result"))
                    _cprint(f"  {line}")
                except Exception:
                    pass
                # One-time /verbose hint on the first long tool in the noisiest mode; latched
                # on self and persisted to config.yaml.
                try:
                    if (
                        not getattr(self, "_long_tool_hint_fired", False)
                        and self.tool_progress_mode == "all"
                        and duration >= 30.0):
                        from agent.onboarding import (
                            TOOL_PROGRESS_FLAG, is_seen, mark_seen, tool_progress_hint_cli)
                        if not is_seen(CLI_CONFIG, TOOL_PROGRESS_FLAG):
                            self._long_tool_hint_fired = True
                            _cprint(f"  {_DIM}{tool_progress_hint_cli()}{_RST}")
                            mark_seen(_hermes_home / "config.yaml", TOOL_PROGRESS_FLAG)
                            CLI_CONFIG.setdefault("onboarding", {}).setdefault("seen", {})[TOOL_PROGRESS_FLAG] = True
                except Exception:
                    pass
            self._invalidate()
            return
        if event_type != "tool.started":
            return
        if function_name and not function_name.startswith("_"):
            from agent.display import get_tool_emoji, get_tool_preview_max_len
            label = preview or function_name
            _pl = get_tool_preview_max_len()
            if _pl > 0 and len(label) > _pl:
                label = label[:_pl - 3] + "..."
            self._spinner_text = f"{get_tool_emoji(function_name)} {label}"
            self._tool_start_time = time.monotonic()
            # Store args for stacked scrollback line on completion
            self._pending_tool_info.setdefault(function_name, []).append(
                function_args if function_args is not None else {})
            self._invalidate()

    def _on_tool_start(self, tool_call_id: str, function_name: str, function_args: dict):
        """Capture local before-state for write-capable tools."""
        from cli import logger
        try:
            from agent.display import capture_local_edit_snapshot
            snapshot = capture_local_edit_snapshot(function_name, function_args)
            if snapshot is not None:
                self._pending_edit_snapshots[tool_call_id] = snapshot
        except Exception:
            logger.debug("Edit snapshot capture failed for %s", function_name, exc_info=True)

    def _on_tool_complete(self, tool_call_id: str, function_name: str, function_args: dict, function_result: str):
        """Render file edits with inline diff after write-capable tools complete."""
        from cli import _cprint, logger
        # A background delegate_task re-enters as a fresh turn when done; say so once so the
        # idle prompt doesn't read as "nothing happened".
        if function_name == "delegate_task":
            try:
                parsed = json.loads(function_result) if isinstance(function_result, str) else (function_result or {})
            except Exception:
                parsed = {}
            if isinstance(parsed, dict) and parsed.get("status") == "dispatched" and parsed.get("mode") == "background":
                n = parsed.get("count") or 1
                noun, tail = ("task", "it finishes") if n == 1 else (f"{n} tasks", "they finish")
                try:
                    _cprint(f"\033[2m\u21a9 Background {noun} running — I'll resume when {tail}. Keep chatting.\033[0m")
                except Exception:
                    pass
        snapshot = self._pending_edit_snapshots.pop(tool_call_id, None)
        try:
            from agent.display import render_edit_diff_with_delta
            render_edit_diff_with_delta(
                function_name, function_result, function_args=function_args, snapshot=snapshot,
                print_fn=_cprint)
        except Exception:
            logger.debug("Edit diff preview failed for %s", function_name, exc_info=True)

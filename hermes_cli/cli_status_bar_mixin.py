"""Status bar, spinner, turn-summary, pet pane, and prompt-stash rendering for the
interactive CLI.

Mixin bound onto ``HermesCLI`` via the MRO. cli.py-internal symbols are imported LAZILY
inside each method (``from cli import ...``) — never at module load time (import cycle).
"""

from __future__ import annotations

import errno
import shutil
import threading
import time

from agent.pet import render as pet_render
from hermes_cli.banner import _format_context_length
from typing import Any, Dict, Optional

_SB = "class:status-bar"
_DIM = "class:status-bar-dim"
_STRONG = "class:status-bar-strong"
_AGENT_COUNTERS = (
    "session_input_tokens", "session_output_tokens", "session_cache_read_tokens",
    "session_cache_write_tokens", "session_prompt_tokens", "session_completion_tokens",
    "session_total_tokens", "session_api_calls")


def _threshold_style(value, ladder, fallback: str) -> str:
    """First ``class:status-bar-<name>`` whose ``value >= bound`` in a descending ladder."""
    for bound, name in ladder:
        if value >= bound:
            return f"class:status-bar-{name}"
    return f"class:status-bar-{fallback}"


def _finite(v):
    """Drop NaN / negative / absurd provider timings (e.g. -0.8s seen in logs)."""
    return None if v is None or v != v or v < 0 or v > 1e6 else v


class CLIStatusBarMixin:
    """Status bar, spinner, turn-summary, pet pane, and prompt-stash rendering for the
    interactive CLI."""

    def _status_bar_context_style(self, percent_used: Optional[int]) -> str:
        if percent_used is None:
            return _DIM
        if percent_used >= 95:
            return "class:status-bar-critical"
        if percent_used > 80:
            return "class:status-bar-bad"
        return _threshold_style(percent_used, ((50, "warn"),), "good")

    def _cache_hit_rate(self, snapshot: dict, precision: int = 1) -> "tuple[float, str] | None":
        """Return (cache_pct, label) or None without cache data. Prefers the baseline-delta pct
        from ``_get_status_bar_snapshot`` (resets on model switch / compression, so it reflects
        the *current* cache regime); falls back to the session-lifetime ratio."""
        delta_pct = snapshot.get("cache_hit_pct")
        if delta_pct is not None:
            return float(delta_pct), f"◎ {float(delta_pct):.{precision}f}%"
        cache_read = snapshot.get("session_cache_read_tokens", 0)
        prompt_total = snapshot.get("session_prompt_tokens", 0)
        if cache_read > 0 and prompt_total > 0:
            cache_pct = cache_read / prompt_total * 100
            return cache_pct, f"◎ {cache_pct:.{precision}f}%"
        return None

    def _cache_hit_rate_style(self, cache_pct: float) -> str:
        """Higher is better (opposite of context %)."""
        return _threshold_style(cache_pct, ((70, "good"), (40, "warn")), "bad")

    @staticmethod
    def _battery_status_style(category: str) -> str:
        return {
            "good": "class:status-bar-good",
            "warn": "class:status-bar-warn",
            "bad": "class:status-bar-bad",
            "critical": "class:status-bar-critical",
        }.get(category, _DIM)

    def _handle_battery_command(self, cmd_original: str) -> None:
        """``/battery`` toggles, ``/battery on|off`` sets, ``/battery status`` reports the
        setting plus a live reading. Persisted to ``display.battery``."""
        from cli import save_config_value
        parts = (cmd_original or "").split()
        arg = parts[1].strip().lower() if len(parts) > 1 else ""

        try:
            from agent.battery import format_battery, read_battery
            reading = read_battery(use_cache=False)
        except Exception:
            reading = None

        def _detail(no_battery: str) -> str:
            if reading is None:
                return ""
            return f" — {format_battery(reading)}" if reading.available else f" — {no_battery}"

        if arg in ("status", "show"):
            state = "on" if self._battery_visible else "off"
            detail = _detail("no battery detected on this machine")
            if reading is not None and reading.available:
                detail = f" — currently {format_battery(reading)}"
            self._console_print(f"  Battery indicator {state}{detail}")
            return

        if arg in ("on", "true", "yes"):
            target = True
        elif arg in ("off", "false", "no"):
            target = False
        elif arg in ("", "toggle"):
            target = not self._battery_visible
        else:
            self._console_print("  Usage: /battery [on|off|status]")
            return

        self._battery_visible = target
        save_config_value("display.battery", target)
        if target:
            self._console_print(
                f"  Battery indicator on{_detail('no battery detected, so nothing will show here')}"
            )
        else:
            self._console_print("  Battery indicator off")

    @staticmethod
    def _compression_count_style(count: int) -> str:
        return _threshold_style(count, ((10, "bad"), (5, "warn")), "dim")

    def _build_context_bar(self, percent_used: Optional[int], width: int = 10) -> str:
        safe_percent = max(0, min(100, percent_used or 0))
        filled = round((safe_percent / 100) * width)
        return f"[{('█' * filled) + ('░' * max(0, width - filled))}]"

    @staticmethod
    def _format_prompt_elapsed(
        prompt_start_time: Optional[float], prompt_duration: float, live: bool = False) -> str:
        """Per-prompt elapsed time. Always a string (``⏲ 0s`` on fresh start); seconds stay
        visible at every scale so it increments smoothly (``1m 59s → 2m → 2m 1s``). ⏱ while
        live, ⏲ frozen — width-1 glyphs (no variation selector) keep the bar aligned."""
        if prompt_start_time is None and prompt_duration == 0.0:
            return "⏲ 0s"
        if prompt_start_time is not None:
            elapsed = max(0.0, time.time() - prompt_start_time)
        else:
            elapsed = max(0.0, prompt_duration)
        days, remaining = divmod(elapsed, 86400)
        hours, remaining = divmod(remaining, 3600)
        minutes, seconds = int(remaining // 60), int(remaining % 60)
        days, hours = int(days), int(hours)
        if days > 0:
            time_str = f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            time_str = f"{hours}h {minutes}m {seconds}s" if seconds else f"{hours}h {minutes}m"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds}s" if seconds else f"{minutes}m"
        else:
            time_str = f"{int(elapsed)}s"
        return f"{'⏱' if live else '⏲'} {time_str}"

    @staticmethod
    def _format_idle_since(last_finished_at: Optional[float], turn_live: bool) -> str:
        """``✓ 42s`` since the last final response; empty while a turn is live or before the
        first turn completes."""
        from cli import format_duration_compact
        if turn_live or last_finished_at is None:
            return ""
        return f"✓ {format_duration_compact(max(0.0, time.time() - last_finished_at))}"

    def _get_status_bar_snapshot(self) -> Dict[str, Any]:
        from cli import _reverse_alias_for_display, datetime, format_duration_compact
        agent = getattr(self, "agent", None)
        # Prefer the agent's model name — it updates on fallback; self.model never changes.
        model_name = (getattr(agent, "model", None) or self.model or "unknown")
        # Friendly display: reverse-alias from config ``model_aliases:`` first (turns long
        # Palantir RIDs into the user's short name), else slash/length truncation.
        model_short = _reverse_alias_for_display(model_name)
        if model_short == model_name:
            model_short = model_name.split("/")[-1] if "/" in model_name else model_name
            # Shared RID-prefix stripper so this and ModelSwitchResult can't drift.
            from hermes_cli.model_switch import format_model_for_display
            model_short = format_model_for_display(model_short)
        if model_short.endswith(".gguf"):
            model_short = model_short[:-5]
        if len(model_short) > 26:
            model_short = f"{model_short[:23]}..."

        prompt_start = getattr(self, "_prompt_start_time", None)
        turn_live = prompt_start is not None
        elapsed_seconds = max(0.0, (datetime.now() - self.session_start).total_seconds())
        snapshot = {
            "model_name": model_name,
            "model_short": model_short,
            "duration": format_duration_compact(elapsed_seconds),
            "session_title": self._get_status_bar_session_title(),
            "prompt_elapsed": self._format_prompt_elapsed(
                prompt_start, getattr(self, "_prompt_duration", 0.0), live=turn_live),
            "idle_since": self._format_idle_since(
                getattr(self, "_last_turn_finished_at", None), turn_live=turn_live),
            "context_tokens": 0,
            "context_length": None,
            "context_percent": None,
            **dict.fromkeys(_AGENT_COUNTERS, 0),
            "compressions": 0,
            "active_background_tasks": 0,
            "active_background_processes": 0,
            "active_background_subagents": 0,
            "battery_label": "",
            "battery_category": "dim",
            "focus_label": "",  # /focus badge: the reduced-output mode is never invisible.
            "goal_active": False,
            "goal_turns_used": 0,
            "goal_max_turns": 0}

        try:
            from hermes_cli.focus_view import focus_statusbar_segment

            snapshot["focus_label"] = focus_statusbar_segment(
                bool(getattr(self, "_focus_view_enabled", False)))
        except Exception:
            pass

        # Battery reads are memoised inside agent.battery, so per-repaint polling is cheap.
        if getattr(self, "_battery_visible", False):
            try:
                from agent.battery import battery_category, format_battery, read_battery

                _batt = read_battery()
                snapshot["battery_label"] = format_battery(_batt)
                snapshot["battery_category"] = battery_category(_batt)
            except Exception:
                pass

        # Live /bg tasks: entries are removed in the task thread's finally block; dict len()
        # is atomic in CPython, no lock needed.
        try:
            bg_tasks = getattr(self, "_background_tasks", None)
            if bg_tasks:
                snapshot["active_background_tasks"] = len(bg_tasks)
        except Exception:
            pass
        try:
            from tools.process_registry import process_registry
            snapshot["active_background_processes"] = process_registry.count_running()
        except Exception:
            pass
        try:
            from tools.async_delegation import active_count as _async_active_count
            snapshot["active_background_subagents"] = _async_active_count()
        except Exception:
            pass

        # Standing /goal (Ralph loop): GoalManager is cached on self — no DB hit per repaint.
        # Only an *active* goal earns a segment (paused/done stay out, like the desktop).
        try:
            goal_mgr = self._get_goal_manager()
            if goal_mgr is not None and goal_mgr.is_active():
                goal_state = goal_mgr.state
                snapshot["goal_active"] = True
                snapshot["goal_turns_used"] = int(getattr(goal_state, "turns_used", 0) or 0)
                snapshot["goal_max_turns"] = int(getattr(goal_state, "max_turns", 0) or 0)
        except Exception:
            pass

        if not agent:
            return snapshot

        for key in _AGENT_COUNTERS:
            snapshot[key] = getattr(agent, key, 0) or 0

        compressor = getattr(agent, "context_compressor", None)
        if compressor:
            # last_prompt_tokens parks at the -1 sentinel right after a compression until the
            # next real API call; clamp so the bar never renders "-1/200K".
            context_tokens = max(0, getattr(compressor, "last_prompt_tokens", 0) or 0)
            # Display-only anchoring: on reasoning models a long tool loop replays the turn's
            # thinking on every request, so the LAST request's prompt_tokens can exceed the
            # durable transcript by hundreds of K and the bar sawtooths at the turn boundary.
            # Anchor on the turn's FIRST response plus a delta estimate of appended messages.
            # The compression trigger keeps using real last-request usage.
            try:
                from agent.model_metadata import anchored_context_tokens

                _msgs = getattr(agent, "_session_messages", None)
                _anchored = anchored_context_tokens(
                    _msgs if isinstance(_msgs, list) else [],
                    getattr(agent, "_turn_base_usage_anchor", None),
                    charge_stale_thinking=False)
                if _anchored is not None and _anchored > 0:
                    context_tokens = _anchored
            except Exception:
                pass
            context_length = max(0, getattr(compressor, "context_length", 0) or 0)
            snapshot["context_tokens"] = context_tokens
            snapshot["context_length"] = context_length or None
            snapshot["compressions"] = getattr(compressor, "compression_count", 0) or 0
            if context_length:
                pct = round((context_tokens / context_length) * 100)
                snapshot["context_percent"] = max(0, min(100, pct))

        # Cache-hit ratio since the last baseline reset (model switch and compression both
        # invalidate the prompt cache). hit = cache_read / prompt_tokens, where
        # prompt = input + cache_read + cache_write (CanonicalUsage).
        pct = None
        try:
            base_model = getattr(self, "_cache_hit_baseline_model", None)
            base_prompt = int(getattr(self, "_cache_hit_baseline_prompt", 0) or 0)
            base_read = int(getattr(self, "_cache_hit_baseline_read", 0) or 0)
            base_comps = int(getattr(self, "_cache_hit_baseline_compressions", 0) or 0)
            cur_comps = int(snapshot["compressions"] or 0)
            cur_prompt = int(snapshot["session_prompt_tokens"] or 0)
            cur_read = int(snapshot["session_cache_read_tokens"] or 0)

            def _rebase(*, tokens: bool) -> None:
                nonlocal base_prompt, base_read, base_comps
                self._cache_hit_baseline_model = model_name
                self._cache_hit_baseline_compressions = base_comps = cur_comps
                if tokens:
                    self._cache_hit_baseline_prompt = base_prompt = cur_prompt
                    self._cache_hit_baseline_read = base_read = cur_read

            if base_model is None:
                _rebase(tokens=False)
            elif model_name != base_model:
                _rebase(tokens=True)
            if cur_comps != base_comps:
                _rebase(tokens=True)
            delta_prompt = cur_prompt - base_prompt
            delta_read = cur_read - base_read
            # A zero-read regime hides the segment (no data ≠ an alarming 0%); pct stays a
            # float so renderers choose their own precision.
            if delta_prompt > 0 and delta_read > 0:
                pct = max(0.0, min(100.0, (delta_read / delta_prompt) * 100))
        except Exception:
            pct = None
        snapshot["cache_hit_pct"] = pct
        snapshot["cache_hit_label"] = f"{pct:.0f}%" if pct is not None else ""

        # Rolling avg latency / velocity over the deques kept by agent/conversation_loop.py
        # (hidden on Codex app-server, which reports no latency).
        avg_lat = avg_vel = None
        try:
            lhist = list(getattr(agent, "_api_latency_history", []) or [])
            ohist = list(getattr(agent, "_api_output_history", []) or [])
            n = min(len(lhist), len(ohist))  # appended together; keep aligned
            if n:
                lhist, ohist = lhist[-n:], ohist[-n:]
                total_lat = sum(lhist)
                # Mean for latency; sum/sum for velocity (true throughput, not mean of ratios).
                avg_lat = _finite(total_lat / n)
                avg_vel = _finite(sum(ohist) / total_lat if total_lat > 0 else None)
        except Exception:
            avg_lat = avg_vel = None
        snapshot["avg_latency"] = float(avg_lat) if avg_lat is not None else None
        snapshot["avg_latency_label"] = f"{avg_lat:.1f}s" if avg_lat is not None else ""
        snapshot["avg_velocity"] = float(avg_vel) if avg_vel is not None else None
        snapshot["avg_velocity_label"] = f"{avg_vel:.0f} t/s" if avg_vel is not None else ""
        return snapshot

    def _get_status_bar_session_title(self) -> str:
        """Current title, polling state.db at most every 1.5s (not on every repaint)."""
        pending = str(getattr(self, "_pending_title", None) or "").strip()
        session_id = str(getattr(self, "session_id", "") or "")
        now = time.monotonic()
        checked_at = float(getattr(self, "_status_bar_title_checked_at", 0.0) or 0.0)
        cache_fresh = (
            getattr(self, "_status_bar_title_session_id", None) == session_id and now - checked_at < 1.5
        )
        if not pending and cache_fresh:
            return str(getattr(self, "_status_bar_title_cache", "") or "")
        title = pending
        db = getattr(self, "_session_db", None)
        if not pending and db is not None and session_id:
            try:
                title = str(db.get_session_title(session_id) or "").strip()
            except Exception:
                title = ""
        self._status_bar_title_session_id = session_id
        self._status_bar_title_cache = title
        self._status_bar_title_checked_at = now
        return title

    @staticmethod
    def _status_bar_display_width(text: str) -> int:
        """Terminal cell width (some glyphs render wider than one codepoint); keeps the bar
        from wrapping onto a second line and leaving duplicate rows."""
        try:
            from prompt_toolkit.utils import get_cwidth
            return get_cwidth(text or "")
        except Exception:
            return len(text or "")

    @classmethod
    def _trim_status_bar_text(cls, text: str, max_width: int) -> str:
        """Trim status-bar text to a single terminal row."""
        if max_width <= 0:
            return ""
        cw = cls._status_bar_display_width
        if cw(text) <= max_width:
            return text
        ellipsis_width = cw("...")
        if max_width <= ellipsis_width:
            return "..."[:max_width]
        out, width = [], 0
        for ch in text:
            if width + cw(ch) + ellipsis_width > max_width:
                break
            out.append(ch)
            width += cw(ch)
        return "".join(out).rstrip() + "..."

    @classmethod
    def _status_title_badge(cls, title: str, width: int) -> "tuple[str, int] | None":
        """(badge, left_width) for the far-right session-title badge, or None when it
        doesn't fit (no title / bar narrower than 24 cells)."""
        title = str(title or "").strip()
        if not title or width < 24:
            return None
        title_width = max(6, min(30, width // 3))
        badge = f" {cls._trim_status_bar_text(title, title_width - 2)} "
        suffix_width = cls._status_bar_display_width(" ─") + cls._status_bar_display_width(badge)
        return badge, max(0, width - suffix_width)

    @classmethod
    def _right_align_status_title(cls, text: str, title: str, width: int) -> str:
        """Pin a bounded session-title badge to the far-right status-bar edge."""
        placed = cls._status_title_badge(title, width)
        if placed is None:
            return cls._trim_status_bar_text(text, width)
        badge, left_width = placed
        left = cls._trim_status_bar_text(text.rstrip(), left_width)
        padding = " " * max(0, left_width - cls._status_bar_display_width(left))
        return f"{left}{padding} ─{badge}"

    @classmethod
    def _right_align_status_title_fragments(cls, frags, title: str, width: int):
        """Styled counterpart to :meth:`_right_align_status_title`."""
        placed = cls._status_title_badge(title, width)
        if placed is None:
            return frags
        badge, left_width = placed
        trimmed = []
        used = 0
        for style, value in frags:
            remaining = left_width - used
            if remaining <= 0:
                break
            value_width = cls._status_bar_display_width(value)
            if value_width <= remaining:
                trimmed.append((style, value))
                used += value_width
                continue
            clipped = cls._trim_status_bar_text(value, remaining)
            if clipped:
                trimmed.append((style, clipped))
                used += cls._status_bar_display_width(clipped)
            break
        if used < left_width:
            trimmed.append((_DIM, " " * (left_width - used)))
        trimmed.extend([(_DIM, " ─"), ("class:status-bar-session-title", badge)])
        return trimmed

    @staticmethod
    def _get_tui_terminal_width(default: tuple[int, int] = (80, 24)) -> int:
        """Live prompt_toolkit width (can be narrower than shutil's, esp. Termux), falling
        back to ``shutil``."""
        try:
            from prompt_toolkit.application import get_app
            return get_app().output.get_size().columns
        except Exception:
            return shutil.get_terminal_size(default).columns

    def _use_minimal_tui_chrome(self, width: Optional[int] = None) -> bool:
        """Hide low-value chrome on narrow/mobile terminals to preserve rows."""
        if width is None:
            width = self._get_tui_terminal_width()
        return width < 64

    @staticmethod
    def _scrollback_box_width(width: Optional[int] = None) -> int:
        """Full viewport width for printed scrollback box rules, floored at 32 cols so tiny
        terminals never hit negative ``'─' * (w - 2)`` math. (The old 56-col clamp against
        reflow-on-shrink is gone: the ``_output_screen_diff`` patch keeps chrome out of
        scrollback, and reflow of already-printed borders is a cosmetic artifact.)

        Previously this clamped to ``max(32, min(width, 56))`` as a defense against terminal-emulator reflow
        on column-shrink (#25975, salvaging 24403). That clamp made response/reasoning borders look stubby
        on any modern wide terminal. We now trust the prompt_toolkit ``_output_screen_diff`` monkey-patch
        landed in #26137 (salvaging 25981) to keep chrome out of scrollback in the first place, and accept
        that an aggressive column-shrink may visually reflow already printed Panel borders — that's a
        cosmetic artifact of stamped scrollback history, not a live-render bug.
        """
        if width is None:
            try:
                width = shutil.get_terminal_size((80, 24)).columns
            except Exception:
                width = 80
        return max(32, int(width or 80))

    def _agent_spacer_height(self, width: Optional[int] = None) -> int:
        """Spacer height above the status bar while the agent runs."""
        if not getattr(self, "_agent_running", False):
            return 0
        return 0 if self._use_minimal_tui_chrome(width=width) else 1

    def _spinner_widget_height(self, width: Optional[int] = None) -> int:
        """Visible height of the spinner/status line above the status bar."""
        spinner_line = self._render_spinner_text()
        if not spinner_line or self._use_minimal_tui_chrome(width=width):
            return 0
        width = width or self._get_tui_terminal_width()
        if width and width > 10:
            return max(1, -(-self._status_bar_display_width(spinner_line) // width))
        return 1

    def _render_spinner_text(self) -> str:
        """The live spinner/status text exactly as rendered in the TUI."""
        txt = getattr(self, "_spinner_text", "")
        if not txt:
            return ""
        flow = self._spinner_token_flow()
        t0 = getattr(self, "_tool_start_time", 0) or 0
        if t0 > 0:
            elapsed = time.monotonic() - t0
            # Fixed-width timers (01m05s / " 5.2s") avoid status-line wrap jitter on repaint.
            if elapsed >= 60:
                elapsed_str = f"{int(elapsed // 60):02d}m{int(elapsed % 60):02d}s"
            else:
                elapsed_str = f"{elapsed:5.1f}s"
            return f"  {txt}  ({elapsed_str} · {flow})" if flow else f"  {txt}  ({elapsed_str})"
        return f"  {txt}  ({flow})" if flow else f"  {txt}"

    def _spinner_token_flow(self) -> str:
        """Cumulative output tokens for the running turn, for the spinner."""
        if not getattr(self, "_spinner_token_flow_enabled", False):
            return ""
        if not getattr(self, "_agent_running", False):
            return ""
        agent = getattr(self, "agent", None)
        if agent is None:
            return ""
        try:
            from agent.turn_summary import format_token_flow

            produced = (getattr(agent, "session_output_tokens", 0) or 0) - (
                getattr(self, "_turn_token_baseline", 0) or 0)
            return format_token_flow(produced)
        except Exception:
            return ""

    def _turn_summary_is_active(self) -> bool:
        """Whether the per-turn summary line renders here: off for the config key, quiet /
        tool-progress-off mode, and every non-interactive path (-q, -Q, gateway)."""
        if not getattr(self, "_turn_summary_enabled", False):
            return False
        if getattr(self, "tool_progress_mode", "all") == "off":
            return False
        agent = getattr(self, "agent", None)
        if agent is not None and getattr(agent, "quiet_mode", False):
            return False
        return bool(getattr(self, "_interactive_turn", False))

    def _turn_summary_begin(self) -> None:
        """Start per-turn accounting for the turn that is about to run."""
        try:
            from agent.turn_summary import TurnSummaryCollector

            collector = getattr(self, "_turn_summary_collector", None)
            if collector is None:
                collector = TurnSummaryCollector()
                self._turn_summary_collector = collector
            collector.begin()
            self._turn_summary_start = time.monotonic()
            agent = getattr(self, "agent", None)
            self._turn_token_baseline = (
                getattr(agent, "session_output_tokens", 0) or 0
            ) if agent is not None else 0
        except Exception:
            self._turn_summary_collector = None

    def _turn_summary_record(self, function_name, result, is_error: bool) -> None:
        """Feed one completed tool call into the active tally."""
        collector = getattr(self, "_turn_summary_collector", None)
        if collector is None:
            return
        try:
            collector.record_tool(function_name, result=result, is_error=bool(is_error))
        except Exception:
            pass

    def _turn_summary_emit(self) -> None:
        """Print the post-turn accounting line, when enabled for this surface."""
        from cli import _DIM as _D, _RST, _cprint, logger
        collector = getattr(self, "_turn_summary_collector", None)
        if collector is None or not self._turn_summary_is_active():
            return
        try:
            started = getattr(self, "_turn_summary_start", 0.0) or 0.0
            line = collector.render(max(0.0, time.monotonic() - started) if started else 0.0)
            if line:
                _cprint(f"  {_D}{line}{_RST}")
        except Exception:
            logger.debug("Turn summary render failed", exc_info=True)

    # ── pet pane ──────────────────────────────────────────────────────────────

    def _pet_clear_runtime(self) -> None:
        """Drop renderer + queued Kitty state. Caller holds ``_pet_lock``."""
        self._pet_enabled = False
        self._pet_renderer = None
        self._pet_frames_cache.clear()
        self._pet_kitty_cache.clear()
        self._pet_kitty_pending = ""
        self._pet_kitty_image_id = 0

    def _pet_resolve_config(self) -> None:
        """(Re)resolve the active pet from config so ``/pet`` / ``hermes pets`` changes apply
        without a restart (mirrors the TUI's steady poll). Fail-open: any problem disables."""
        try:
            from agent.pet import constants, store
            from hermes_cli.config import load_config
            from utils import is_truthy_value

            cfg = load_config()
            display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
            pet_cfg = display.get("pet", {}) if isinstance(display.get("pet"), dict) else {}
            enabled = is_truthy_value(pet_cfg.get("enabled"), default=False)
            slug = str(pet_cfg.get("slug", "") or "")
            scale = float(pet_cfg.get("scale", constants.DEFAULT_SCALE) or constants.DEFAULT_SCALE)
            cols = constants.resolve_cols(scale, pet_cfg.get("unicode_cols", 0))
            configured_mode = str(pet_cfg.get("render_mode", "auto") or "auto").lower()
            # Placeholders only on kitty/Ghostty: WezTerm speaks kitty APC but not U+10EEEE
            # while detect_terminal_graphics() still says kitty, hence the narrower gate.
            use_kitty = (
                configured_mode in ("", "auto", "kitty") and pet_render.supports_kitty_placeholders()
            )
            renderer_mode = "kitty" if use_kitty else "unicode"

            pet = None
            if enabled and configured_mode != "off":
                pet = store.resolve_active_pet(slug)
            if pet is None or not pet.exists:
                with self._pet_lock:
                    self._pet_clear_runtime()
                return

            with self._pet_lock:
                # Rebuild only when the resolved pet, mode, or geometry changes.
                if (
                    self._pet_renderer is None
                    or self._pet_slug != pet.slug
                    or self._pet_cols != cols
                    or self._pet_scale != scale
                    or self._pet_renderer.mode != renderer_mode):
                    self._pet_renderer = pet_render.PetRenderer(
                        str(pet.spritesheet), mode=renderer_mode, scale=scale, unicode_cols=cols)
                    self._pet_slug = pet.slug
                    self._pet_cols = cols
                    self._pet_scale = scale
                    self._pet_frames_cache.clear()
                    self._pet_kitty_cache.clear()
                    self._pet_kitty_pending = ""
                    self._pet_kitty_image_id = pet_render.kitty_image_id(pet.slug)
                    self._pet_frame_idx = 0
                self._pet_enabled = True
        except Exception:
            with self._pet_lock:
                self._pet_clear_runtime()

    def _pet_flash(self, state: str, secs: float = 1.6) -> None:
        """Briefly force a transient reaction (wave/jump/failed) before resting."""
        self._pet_event = state
        self._pet_event_until = time.monotonic() + secs

    def _on_reaction(self, kind: str) -> None:
        """Core-detected user affection (ily / <3 / good bot): the pet's share of the vibe
        signal that plays hearts on the TUI/desktop."""
        if kind == "vibe":
            self._pet_flash("jump")

    def _pet_react_turn_end(self) -> None:
        """End-of-turn beat: failed on error, jump on a finished plan, else wave."""
        if not self._pet_enabled:
            return
        from agent.pet.state import todos_all_done

        if self._pet_turn_error:
            self._pet_flash("failed")
            return
        try:
            store = getattr(self.agent, "_todo_store", None)
            done = todos_all_done(store.read()) if store else False
        except Exception:
            done = False
        self._pet_flash("jump" if done else "wave")

    def _derive_pet_state(self) -> str:
        """A live transient beat wins; otherwise the shared ``agent.pet.state.derive_pet_state``
        priority order so the CLI can't drift from the TUI/desktop."""
        if self._pet_event and time.monotonic() < self._pet_event_until:
            return self._pet_event
        self._pet_event = ""
        from agent.pet.state import derive_pet_state

        # Any blocking modal (approval / clarify / sudo / secret / slash confirm) means the
        # agent is paused on the user → `waiting`, which outranks the in-flight signals.
        awaiting_input = bool(
            self._approval_state
            or self._clarify_state
            or self._sudo_state
            or self._secret_state
            or getattr(self, "_slash_confirm_state", None))
        return derive_pet_state(
            awaiting_input=awaiting_input,
            busy=getattr(self, "_agent_running", False),
            reasoning=self._pet_reasoning,
        ).value

    def _pet_frames_for(self, state: str) -> list:
        """Return (and cache) the half-block grids for one state."""
        cached = self._pet_frames_cache.get(state)
        if cached is not None:
            return cached
        renderer = self._pet_renderer
        if renderer is None:
            return []
        try:
            count = renderer.frame_count(state) or 1
            grids = [renderer.cells(state, i, cols=self._pet_cols) for i in range(count)]
        except Exception:
            grids = []
        self._pet_frames_cache[state] = grids
        return grids

    def _pet_kitty_payload_for(self, state: str) -> dict | None:
        """Return and cache a Kitty virtual-placeholder payload for *state*."""
        with self._pet_lock:
            cached = self._pet_kitty_cache.get(state)
            if cached is not None:
                return cached
            renderer = self._pet_renderer
            image_id = self._pet_kitty_image_id
            if renderer is None or renderer.mode != "kitty":
                return None
        try:
            # PNG encoding outside _pet_lock: first visit of a state must not stall the prompt.
            payload = renderer.kitty_payload(state, image_id=image_id)
        except Exception:
            payload = None
        if payload is not None:
            payload = {**payload, "image_id": image_id}
            with self._pet_lock:
                if self._pet_renderer is renderer and self._pet_kitty_image_id == image_id:
                    self._pet_kitty_cache[state] = payload
        return payload

    def _pet_queue_kitty_frame(self, state: str | None = None) -> None:
        """Queue one virtual Kitty frame for the next prompt_toolkit render. No-op when the
        pet pane was never initialized (``__new__`` fixtures, redraw on a pet-less CLI)."""
        if not getattr(self, "_pet_enabled", False):
            return
        if state is None:
            state = self._derive_pet_state()
        payload = self._pet_kitty_payload_for(state)
        if not payload or not payload.get("frames"):
            return
        with self._pet_lock:
            if self._pet_renderer is not None and self._pet_renderer.mode == "kitty":
                frames = payload["frames"]
                self._pet_kitty_pending = frames[self._pet_frame_idx % len(frames)]

    def _pet_flush_kitty_frame(self, app) -> None:
        """Write a queued APC after prompt_toolkit has finished its screen diff."""
        with self._pet_lock:
            frame = self._pet_kitty_pending
            self._pet_kitty_pending = ""
        if not frame:
            return
        try:
            # U=1/q=2 leaves the cursor and input stream untouched.
            app.output.write_raw(frame)
            app.output.flush()
        except (OSError, ValueError):
            pass

    def _pet_view(self) -> "tuple[str, bool] | None":
        """(state, is_kitty) for the current frame, or None when no pet shows."""
        with self._pet_lock:
            if not self._pet_enabled or self._pet_renderer is None:
                return None
            return self._derive_pet_state(), self._pet_renderer.mode == "kitty"

    def _pet_fragments(self):
        """prompt_toolkit FormattedText for the current pet frame, or []."""
        view = self._pet_view()
        if view is None:
            return []
        state, kitty = view
        frags = []
        if kitty:
            payload = self._pet_kitty_payload_for(state)
            if not payload:
                return []
            color = pet_render.kitty_color_hex(payload["image_id"])
            for y, row in enumerate(payload["placeholder"]):
                if y:
                    frags.append(("", "\n"))
                frags.append((f"fg:{color}", row))
            return frags
        with self._pet_lock:
            grids = self._pet_frames_for(state)
            if not grids:
                return []
            grid = grids[self._pet_frame_idx % len(grids)]

        def _hex(r, g, b):
            return f"#{r:02x}{g:02x}{b:02x}"

        for y, row in enumerate(grid):
            if y:
                frags.append(("", "\n"))
            for (tr, tg, tb, ta), (br, bg, bb, ba) in row:
                top_op, bot_op = ta >= 32, ba >= 32
                if not top_op and not bot_op:
                    frags.append(("", " "))
                elif top_op and bot_op:
                    frags.append((f"fg:{_hex(tr, tg, tb)} bg:{_hex(br, bg, bb)}", "▀"))
                elif top_op:
                    # Upper half only — leave the lower half the terminal's bg (cleaner on light
                    # themes).
                    frags.append((f"fg:{_hex(tr, tg, tb)}", "▀"))
                else:
                    frags.append((f"fg:{_hex(br, bg, bb)}", "▄"))
        return frags

    def _pet_widget_height(self) -> int:
        """Visible rows for the pet window — 0 collapses it when no pet shows."""
        view = self._pet_view()
        if view is None:
            return 0
        state, kitty = view
        if kitty:
            payload = self._pet_kitty_payload_for(state)
            return int(payload.get("rows", 0)) if payload else 0
        with self._pet_lock:
            grids = self._pet_frames_for(state)
            return len(grids[0]) if grids and grids[0] else 0

    def _pet_anim_loop(self) -> None:
        """Advance the frame + invalidate on a timer while a pet is enabled."""
        while self._pet_anim_running:
            time.sleep(self._PET_FRAME_INTERVAL)
            if getattr(self, "_terminal_io_broken", False):
                self._pet_anim_running = False
                break
            now = time.monotonic()
            if now - self._pet_cfg_checked >= self._PET_CFG_INTERVAL:
                self._pet_cfg_checked = now
                self._pet_resolve_config()
            if not self._pet_enabled:
                continue
            with self._pet_lock:
                self._pet_frame_idx += 1
                kitty = self._pet_renderer is not None and self._pet_renderer.mode == "kitty"
            if kitty:
                self._pet_queue_kitty_frame()
            app = getattr(self, "_app", None)
            if app is not None:
                try:
                    app.invalidate()
                except OSError as exc:
                    if getattr(exc, "errno", None) == errno.EIO:
                        self._mark_terminal_io_broken("pet_anim")
                        break
                except Exception:
                    pass

    def _pet_start_anim(self) -> None:
        if self._pet_anim_running:
            return
        self._pet_resolve_config()
        view = self._pet_view()
        if view is not None and view[1]:
            self._pet_queue_kitty_frame()
        self._pet_anim_running = True
        self._pet_anim_thread = threading.Thread(target=self._pet_anim_loop, daemon=True)
        self._pet_anim_thread.start()

    def _pet_stop_anim(self) -> None:
        self._pet_anim_running = False
        thread = self._pet_anim_thread
        if thread is not None:
            thread.join(timeout=0.3)
        self._pet_anim_thread = None

    # ── voice ─────────────────────────────────────────────────────────────────

    def _voice_record_key_label(self) -> str:
        """The push-to-talk key label every voice-facing hint advertises. Cached at startup
        (``set_voice_record_key_cache``) because the prompt_toolkit binding is registered once —
        re-reading config per render could advertise a chord that isn't bound — and this sits on
        the hot render path.

        Two reasons (Copilot round-13 on 19835): See #19835.
        """
        return getattr(self, "_voice_record_key_display_cache", None) or "Ctrl+B"

    def set_voice_record_key_cache(self, raw_key: object) -> None:
        """Populate the voice label cache from a raw ``voice.record_key``; called after the
        prompt_toolkit binding is registered so the label matches the live binding."""
        try:
            from hermes_cli.voice import format_voice_record_key_for_status
            self._voice_record_key_display_cache = format_voice_record_key_for_status(raw_key)
        except Exception:
            self._voice_record_key_display_cache = "Ctrl+B"

    def _get_voice_status_fragments(self, width: Optional[int] = None):
        """Voice status bar fragments for the interactive TUI."""
        width = width or self._get_tui_terminal_width()
        compact = self._use_minimal_tui_chrome(width=width)
        label = self._voice_record_key_label()
        if self._voice_recording:
            if compact:
                return [("class:voice-status-recording", " ● REC ")]
            return [("class:voice-status-recording", f" ● REC  {label} to stop ")]
        if self._voice_processing:
            return [("class:voice-status", " ◉ STT " if compact else " ◉ Transcribing... ")]
        if compact:
            return [("class:voice-status", f" 🎤 {label} ")]
        tts = " | TTS on" if self._voice_tts else ""
        cont = " | Continuous" if self._voice_continuous else ""
        return [("class:voice-status", f" 🎤 Voice mode{tts}{cont}  —  {label} to record ")]

    # ── status bar rendering ──────────────────────────────────────────────────

    @staticmethod
    def _status_bar_goal_segment(snapshot: Dict[str, Any]) -> str:
        """``⊙ goal 3/20`` while a goal is active, else ``""`` (paused/done goals already
        print their own glyph lines in the thread)."""
        if not snapshot.get("goal_active"):
            return ""
        used = snapshot.get("goal_turns_used") or 0
        max_turns = snapshot.get("goal_max_turns") or 0
        return f"⊙ goal {used}/{max_turns}" if max_turns else "⊙ goal"

    def _get_status_bar_field_set(self) -> Optional[frozenset]:
        """Visible status-bar fields from ``display.status_bar.fields`` (module-level
        ``CLI_CONFIG``; no per-render YAML parse). ``None`` = not customized, show everything.

        Fields: model, context_detail, context_pct, cache_hit, latency, tps, compressions,
        bg_tasks, bg_processes, bg_subagents, goal, duration, prompt_elapsed, idle_since,
        focus, yolo, stash, battery, title, total_tokens (opt-in only). Order is fixed; the
        config controls visibility only.
        """
        from cli import CLI_CONFIG
        if hasattr(self, "_status_bar_field_set_cache"):
            return self._status_bar_field_set_cache
        result = None
        try:
            display = CLI_CONFIG.get("display") if isinstance(CLI_CONFIG, dict) else None
            status_bar = (display or {}).get("status_bar") if isinstance(display, dict) else None
            fields = status_bar.get("fields") if isinstance(status_bar, dict) else None
            if isinstance(fields, list) and fields:
                result = frozenset(str(f) for f in fields)
        except Exception:
            result = None
        self._status_bar_field_set_cache = result
        return result

    def _status_bar_segments(
        self, snapshot, width: int, field_set, yolo_active: bool, *, styled: bool) -> list:
        """Ordered status-bar segments for one width tier (<52 / <76 / wide), each a list of
        ``(style, text)`` fragments. Shared by the plain-text and prompt_toolkit renderers so
        the two can never drift; ``styled`` selects the graphical context bar."""
        from cli import format_token_count_compact
        model_short = snapshot["model_short"]
        duration_label = snapshot["duration"]
        goal_segment = self._status_bar_goal_segment(snapshot)
        focus_label = snapshot.get("focus_label") or ""

        def _ok(name: str) -> bool:
            return field_set is None or name in field_set

        segs: list = []

        def add(name, style, text):
            if _ok(name):
                segs.append([(style, text)])

        def add_count(name, key, glyph, style=_STRONG):
            count = snapshot.get(key, 0)
            if count:
                add(name, style(count) if callable(style) else style, f"{glyph} {count}")

        if _ok("model"):
            if styled:
                segs.append([(_SB, " ⚕ "), (_STRONG, model_short)])
            else:
                segs.append([("", f"⚕ {model_short}")])
        narrow, wide = width < 52, width >= 76
        if narrow:
            # Narrow bars put duration ahead of the goal segment; the other tiers reverse it.
            add("duration", _DIM, duration_label)
        else:
            percent = snapshot["context_percent"]
            percent_label = f"{percent}%" if percent is not None else "--"
            if wide and _ok("context_detail"):
                if snapshot["context_length"]:
                    ctx_total = _format_context_length(snapshot["context_length"])
                    ctx_used = format_token_count_compact(snapshot["context_tokens"])
                    context_label = f"{ctx_used}/{ctx_total}"
                else:
                    context_label = "ctx --"
                segs.append([(_DIM, context_label)])
            if _ok("context_pct"):
                bar_style = self._status_bar_context_style(percent)
                if wide and styled:
                    segs.append([
                        (bar_style, self._build_context_bar(percent)), (_DIM, " "), (bar_style, percent_label),
                    ])
                else:
                    segs.append([(bar_style, percent_label)])
            cache = self._cache_hit_rate(snapshot, precision=1 if wide else 0)
            if cache:
                add("cache_hit", self._cache_hit_rate_style(cache[0]), cache[1])
            if wide:
                for name, key, glyph in (
                    ("latency", "avg_latency_label", "◷"), ("tps", "avg_velocity_label", "↑")):
                    label = snapshot.get(key) or ""
                    if label:
                        add(name, _DIM, f"{glyph} {label}")
            add_count("compressions", "compressions", "🗜️", self._compression_count_style)
            add_count("bg_tasks", "active_background_tasks", "▶")
            add_count("bg_processes", "active_background_processes", "⚙")
            add_count("bg_subagents", "active_background_subagents", "⛓")
        if goal_segment:
            add("goal", _STRONG, goal_segment)
        if not narrow:
            add("duration", _DIM, duration_label)
        if wide:
            for name in ("prompt_elapsed", "idle_since"):
                label = snapshot.get(name)
                if label:
                    add(name, _DIM, label)
        if focus_label:
            add("focus", _STRONG, focus_label)
        if yolo_active:
            add("yolo", "class:status-bar-yolo", "⚠ YOLO")
        if wide:
            # Session token total (Σ) — opt-in only via an explicit fields list.
            total_tokens = snapshot.get("session_total_tokens", 0)
            if total_tokens and field_set is not None and "total_tokens" in field_set:
                segs.append([(_DIM, f"Σ{format_token_count_compact(total_tokens)}")])
        return segs

    def _build_status_bar_text(self, width: Optional[int] = None) -> str:
        """Compact one-line session status string for the TUI footer."""
        try:
            snapshot = self._get_status_bar_snapshot()
            if width is None:
                width = self._get_tui_terminal_width()
            model_short = snapshot["model_short"]
            battery_label = snapshot.get("battery_label") or ""
            field_set = self._get_status_bar_field_set()
            show_title = field_set is None or "title" in field_set
            session_title = (snapshot.get("session_title") or "") if show_title else ""
            segs = self._status_bar_segments(
                snapshot, width, field_set, self._is_session_yolo_active(), styled=False)
            parts = ["".join(t for _, t in seg) for seg in segs] or [f"⚕ {model_short}"]
            # Narrow bars always join the battery with │; wider tiers use the tier separator.
            if battery_label:
                parts.insert(0, battery_label)
            if width < 52:
                text = f"{parts[0]} │ " + " · ".join(parts[1:]) if battery_label else " · ".join(parts)
            else:
                text = (" · " if width < 76 else " │ ").join(parts)
            return self._right_align_status_title(text, session_title, width)
        except Exception:
            return f"⚕ {self.model if getattr(self, 'model', None) else 'Hermes'}"

    def _get_status_bar_fragments(self):
        if (
            not self._status_bar_visible
            or getattr(self, "_model_picker_state", None)
            or getattr(self, "_command_palette_state", None)):
            return []
        try:
            snapshot = self._get_status_bar_snapshot()
            # prompt_toolkit's own width: shutil's can be stale (esp. over SSH) and an overflow
            # produces duplicated status-bar rows over long sessions.
            width = self._get_tui_terminal_width()
            field_set = self._get_status_bar_field_set()

            def _ok(name: str) -> bool:
                return field_set is None or name in field_set

            session_title = (snapshot.get("session_title") or "") if _ok("title") else ""
            segs = self._status_bar_segments(
                snapshot, width, field_set, self._is_session_yolo_active(), styled=True)
            sep = " · " if width < 76 else " │ "
            frags: list = []
            for seg in segs or [[(_SB, " ⚕ "), (_STRONG, snapshot["model_short"])]]:
                if frags:
                    frags.append((_DIM, sep))
                frags.extend(seg)
            # Stash indicator (📌 N) after every width tier so a parked draft is never
            # invisible; before the battery prepend, and the first thing the trim drops.
            try:
                stash_indicator = self._prompt_stash.indicator()
            except Exception:
                stash_indicator = ""
            if stash_indicator and _ok("stash"):
                frags.extend([(_DIM, " · "), (_STRONG, stash_indicator)])
            frags.append((_SB, " "))  # one-cell right margin
            # Battery is the first element when enabled: prepend ahead of the ⚕ marker.
            battery_label = snapshot.get("battery_label") or ""
            if battery_label and _ok("battery"):
                battery_style = self._battery_status_style(snapshot.get("battery_category", "dim"))
                frags[0:0] = [(_SB, " "), (battery_style, battery_label), (_DIM, " │")]

            frags = self._right_align_status_title_fragments(frags, session_title, width)
            total_width = sum(self._status_bar_display_width(text) for _, text in frags)
            if total_width > width:
                plain_text = "".join(text for _, text in frags)
                return [(_SB, self._trim_status_bar_text(plain_text, width))]
            return frags
        except Exception:
            return [(_SB, f" {self._build_status_bar_text()} ")]

    # ── prompt stash panel ────────────────────────────────────────────────────

    @staticmethod
    def _fmt_stash_age(stashed_at: float) -> str:
        secs = int(time.monotonic() - stashed_at)
        if secs < 10:
            return "just now"
        if secs < 90:
            return f"{secs}s ago"
        mins = secs // 60
        return f"{mins} min ago" if mins < 60 else f"{mins // 60}h ago"

    def _render_stash_panel(self, stash_list: list, cursor: int, width: int) -> list:
        """prompt_toolkit fragments for the stash panel box. Every horizontal measurement uses
        ``_status_bar_display_width`` (display cells), not ``len()`` — 📌 is one codepoint but
        two cells, and CJK previews would otherwise bleed past the right border."""
        cw = self._status_bar_display_width
        W = max(12, min(width - 4, 80))

        n = len(stash_list)
        hdr_prefix_str = f"╭─ 📌 Stash ({n} item{'s' if n != 1 else ''}) "
        HDR_SUFFIX = " Ctrl+S ─╮"
        FTR_PREFIX = "╰"
        FTR_SUFFIX = " ↑↓ Enter=restore  D=delete  Esc ─╯"

        # On narrow terminals the full hint text is wider than the box: drop to compact
        # affordances rather than letting the frame bleed past the right edge.
        if cw(hdr_prefix_str) + cw(HDR_SUFFIX) > W:
            hdr_prefix_str = f"╭─ 📌 {n} "
            HDR_SUFFIX = "─╮"
        if cw(FTR_PREFIX) + cw(FTR_SUFFIX) > W:
            FTR_SUFFIX = " ↑↓ ⏎ D Esc ─╯"
        if cw(FTR_PREFIX) + cw(FTR_SUFFIX) > W:
            FTR_SUFFIX = "─╯"

        hdr_dashes = max(0, W - cw(hdr_prefix_str) - cw(HDR_SUFFIX))
        ftr_dashes = max(0, W - cw(FTR_PREFIX) - cw(FTR_SUFFIX))
        INNER = W - 2  # minus the two '│' border cells
        frags: list = []

        def line(text: str, style: str = "") -> None:
            # Final guard: never emit a line wider than the box.
            frags.append((style, self._trim_status_bar_text(text, W) + "\n"))

        line(f"{hdr_prefix_str}{'─' * hdr_dashes}{HDR_SUFFIX}", "class:subagent-border")
        for i, item in enumerate(stash_list):
            age = self._fmt_stash_age(item["stashed_at"])
            marker = "►" if i == cursor else " "
            prefix = f" {marker} [{i + 1}] {age:<10} "
            if cw(prefix) > INNER - 2:
                prefix = f" {marker} [{i + 1}] "
            avail = max(0, INNER - cw(prefix) - 1)
            preview = self._trim_status_bar_text(item.get("preview") or "", avail)
            preview = preview + " " * max(0, avail - cw(preview))
            if i == cursor:
                row = self._trim_status_bar_text(f"│{prefix}{preview} │", W)
                frags.append(("class:subagent-selected", row + "\n"))
            else:
                frags.append(("class:subagent-border", "│"))
                frags.append(("class:subagent-sub", f"{prefix}{preview} "))
                frags.append(("class:subagent-border", "│\n"))
        line(f"{FTR_PREFIX}{'─' * ftr_dashes}{FTR_SUFFIX}", "class:subagent-border")
        return frags

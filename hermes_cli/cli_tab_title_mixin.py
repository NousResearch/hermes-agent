"""Opt-in activity titles for the classic CLI terminal window/tab."""

import os
import sqlite3
import sys
import time

from hermes_platform.host.facts import os_family


class CLITabTitleMixin:
    """Title templates, native Windows updates, and TTY-gated OSC output."""

    def _configure_tab_activity_indicator(self, display: dict) -> None:
        """Load opt-in terminal tab title indicator settings."""
        self._tab_activity_indicator = bool(display.get("tab_activity_indicator", False))
        self._tab_title_idle = str(display.get("tab_title_idle", "{title}") or "")
        self._tab_title_busy = str(display.get("tab_title_busy", "{spinner} Hermes") or "")
        self._tab_title_done = str(display.get("tab_title_done", "{title}") or "")
        frames = display.get("tab_title_spinner_frames") or [
            "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
        ]
        if isinstance(frames, str):
            frames = list(frames)
        elif not isinstance(frames, (list, tuple)):
            frames = ["*"]
        self._tab_title_spinner_frames = [str(frame) for frame in frames if str(frame)]
        if not self._tab_title_spinner_frames:
            self._tab_title_spinner_frames = ["*"]
        self._tab_title_frame_idx = 0
        self._tab_title_last = None
        self._tab_title_next_update = 0.0

    def _tab_title_context(
        self,
        template: str,
        spinner: str = "",
        *,
        allow_db_lookup: bool = True,
    ) -> dict[str, str]:
        label = "Hermes"
        session_title = ""
        if "{title" in template or "{session" in template:
            session_title = str(getattr(self, "_pending_title", "") or "")
            if (
                allow_db_lookup
                and not session_title
                and getattr(self, "_session_db", None)
                and getattr(self, "session_id", None)
            ):
                try:
                    session_title = self._session_db.get_session_title(self.session_id) or ""
                except (sqlite3.Error, OSError):
                    session_title = ""
        title = session_title or label
        session_id = str(getattr(self, "session_id", "") or "")
        return {
            "label": label,
            "spinner": spinner,
            "title": title,
            "session": title,
            "session_id": session_id,
            "short_session_id": session_id[:8],
        }

    @staticmethod
    def _sanitize_tab_title(title: str) -> str:
        sanitized = []
        for char in str(title):
            codepoint = ord(char)
            if char in {"\r", "\n", "\t"}:
                sanitized.append(" ")
            elif codepoint < 32 or 127 <= codepoint <= 159:
                continue
            else:
                sanitized.append(char)
        return "".join(sanitized)[:255]

    def _write_tab_title(self, title: str) -> None:
        if not getattr(self, "_tab_activity_indicator", False) or not title:
            return
        title = self._sanitize_tab_title(title)
        if title == getattr(self, "_tab_title_last", None):
            return
        native_updated = False
        if os_family() == "win32":
            try:
                import ctypes

                native_updated = bool(ctypes.windll.kernel32.SetConsoleTitleW(title))
            except (AttributeError, OSError, ValueError):
                pass

        osc_updated = False
        windows_osc_capable = bool(
            os.environ.get("MSYSTEM")
            or os.environ.get("TERM_PROGRAM")
            or os.environ.get("WT_SESSION")
            or os.environ.get("TERM", "").lower() not in {"", "dumb"}
        )
        if os_family() != "win32" or windows_osc_capable:
            stream = getattr(self, "_tab_title_stream", None)
            if stream is None:
                # prompt_toolkit.patch_stdout() replaces sys.stdout with a proxy whose
                # normal write path escapes VT/OSC control bytes. Use the original
                # terminal stream so tab-title OSC sequences stay raw.
                stream = getattr(sys, "__stdout__", None) or sys.stdout
            try:
                if hasattr(stream, "isatty") and stream.isatty():
                    stream.write(f"\033]0;{title}\a")
                    stream.flush()
                    osc_updated = True
            except (AttributeError, OSError, ValueError):
                pass

        if native_updated or osc_updated:
            self._tab_title_last = title

    def _set_tab_title_state(self, state: str, *, force: bool = False) -> None:
        if not getattr(self, "_tab_activity_indicator", False):
            return
        if state == "busy":
            frames = getattr(self, "_tab_title_spinner_frames", ["*"])
            idx = getattr(self, "_tab_title_frame_idx", 0)
            spinner = frames[idx % len(frames)]
            self._tab_title_frame_idx = idx + 1
            template = getattr(self, "_tab_title_busy", "{spinner} Hermes")
        elif state == "done":
            spinner = ""
            template = getattr(self, "_tab_title_done", "{title}")
        else:
            spinner = ""
            template = getattr(self, "_tab_title_idle", "{title}")
        try:
            title = template.format(
                **self._tab_title_context(
                    template,
                    spinner=spinner,
                    allow_db_lookup=(state != "busy"),
                )
            )
        except (KeyError, IndexError, ValueError, AttributeError, TypeError):
            title = template
        if force:
            self._tab_title_last = None
        self._write_tab_title(title)

    def _tick_tab_activity_indicator(self) -> None:
        if not getattr(self, "_tab_activity_indicator", False):
            return
        now = time.monotonic()
        if now < getattr(self, "_tab_title_next_update", 0.0):
            return
        self._tab_title_next_update = now + 0.25
        self._set_tab_title_state("busy")

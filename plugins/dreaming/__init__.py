"""
Dreaming — automatic background memory consolidation for Hermes.

Reference implementation from Willow 2.0 (github.com/rudi193-cmd/willow-2.0).
Adapted to the Hermes plugin interface.

3-phase pipeline:
  Light Sleep  — scan recent transcripts, deduplicate, score candidates
  REM          — extract themes via local LLM, write dream diary
  Deep Sleep   — promote high-signal entries to MEMORY.md; route meta-entries
                 to SKILL.md rather than polluting long-term memory

Opt-in: disabled by default. Enable in $HERMES_HOME/dreaming/config.yaml
(or the dreaming: section of config.yaml) after `hermes plugins enable dreaming`.
"""
from __future__ import annotations

import threading
import time

from . import _config, _diary, _schedule

_HELP = """\
/dream            — show status and last diary entry
/dream run        — force a consolidation cycle now
/dream status     — check conditions (hours since last, sessions queued)
/dream diary      — show the last dream diary entry
"""


class _DreamThread(threading.Thread):
    def __init__(self, poll_seconds: int) -> None:
        super().__init__(daemon=True, name="hermes-dreaming")
        self._poll_seconds = poll_seconds
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(timeout=self._poll_seconds):
            try:
                cfg = _config.load_config()
                if not _config.is_enabled(cfg):
                    continue
                sched = _config.schedule(cfg)
                if _schedule.dream_check(
                    min_hours=sched["min_hours"],
                    min_sessions=sched["min_sessions"],
                ):
                    _schedule.dream_run(cfg=cfg)
            except Exception:
                pass  # never crash the daemon thread

    def stop(self) -> None:
        self._stop.set()


_thread: _DreamThread | None = None


def _on_post_llm_call(
    *,
    user_message: str = "",
    assistant_response: str = "",
    **_: object,
) -> None:
    """Persist transcript-bearing turns without counting each as a session."""
    transcript = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_response},
    ]
    if not user_message and not assistant_response:
        return
    try:
        _schedule.enqueue_session(transcript, count_session=False)
    except Exception:
        pass


def _on_session_end(**_: object) -> None:
    """Record the session boundary after its turns have been staged."""
    try:
        _schedule.mark_session_complete()
    except Exception:
        pass


def _disabled_message() -> str:
    path = _config.user_config_path()
    return (
        "Dreaming is disabled. Enable the plugin, then set "
        f"`enabled: true` in {path} (or under `dreaming:` in config.yaml).\n\n"
        + _HELP
    )


def _handle_slash(raw_args: str) -> str:
    args = raw_args.split() if raw_args else []
    cfg = _config.load_config()

    if not args:
        state = _schedule._read_state()
        hours_ago = (time.time() - state["last_dream_at"]) / 3600
        sessions = state.get("sessions_since_dream", 0)
        last = _diary.last_entry()
        enabled = "on" if _config.is_enabled(cfg) else "off"
        lines = [
            f"**Dreaming ({enabled})** | last cycle: {hours_ago:.1f}h ago | "
            f"sessions queued: {sessions}",
        ]
        if last:
            lines += ["", last]
        return "\n".join(lines)

    if args and args[0] == "run":
        try:
            result = _schedule.dream_run(force=True, cfg=cfg)
            return (
                f"Dream cycle complete. "
                f"Scanned {result['candidates_scanned']} candidates, "
                f"promoted {result['promoted']}, "
                f"routed {result['skipped_meta']} meta-entries to SKILL.md."
            )
        except Exception as e:
            return f"Dream cycle failed: {e}"

    if args and args[0] == "status":
        state = _schedule._read_state()
        hours_ago = (time.time() - state["last_dream_at"]) / 3600
        sessions = state.get("sessions_since_dream", 0)
        sched = _config.schedule(cfg)
        ready = _schedule.dream_check(
            min_hours=sched["min_hours"],
            min_sessions=sched["min_sessions"],
        )
        return (
            f"Enabled: {'yes' if _config.is_enabled(cfg) else 'no'}\n"
            f"Hours since last dream: {hours_ago:.1f}\n"
            f"Sessions since last dream: {sessions}\n"
            f"Ready to dream: {'yes' if ready else 'no'}"
        )

    if args and args[0] == "diary":
        entry = _diary.last_entry()
        return entry if entry else "No dream diary entries yet."

    return _HELP


def register(ctx) -> None:
    global _thread

    _config.ensure_user_config()

    ctx.register_command(
        "dream",
        handler=_handle_slash,
        description="Automatic background memory consolidation — status, force run, diary.",
    )

    cfg = _config.load_config()
    if not _config.is_enabled(cfg):
        return

    ctx.register_hook("post_llm_call", _on_post_llm_call)
    ctx.register_hook("on_session_end", _on_session_end)

    sched = _config.schedule(cfg)
    _thread = _DreamThread(poll_seconds=sched["poll_seconds"])
    _thread.start()

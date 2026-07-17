"""life_ops Discord adapter — fork features layered on the bundled adapter.

Subclasses the bundled ``plugins.platforms.discord.adapter.DiscordAdapter``
and is swapped in via ``platform_registry`` (last-writer-wins, see
``plugins/life_ops/__init__.py``), so the upstream adapter file stays
byte-identical to NousResearch/hermes-agent and upstream syncs never
conflict with fork code.

Adds:
- Bedtime overnight-sprint scheduler + Start/Skip prompt (``BedtimeView``)
- Daily journal-approvals dispatcher + per-todo [Approve] embeds
  (``JournalApproveView``)
- Daily todo-closure control — select + Mark Done / Dismiss / Snooze
  (``TodoClosureView``)
- Slash commands /rpe, /done, /dismiss, /snooze, /away-on, /away-off
  (logic in ``plugins.life_ops.discord_commands``)

View classes are defined lazily via ``_define_life_ops_view_classes()``,
mirroring the bundled adapter's ``_define_discord_view_classes`` pattern,
because ``discord.py`` may be absent (or lazily installed) at import time.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional

from gateway.platforms.base import SendResult
from plugins.platforms.discord.adapter import (
    _DISCORD_SELECT_FIELD_LIMIT,
    DiscordAdapter,
    _component_check_auth,
    _read_discord_prompt_timeout,
    _truncate_discord_component_text,
)

logger = logging.getLogger(__name__)

try:
    import discord  # type: ignore

    DISCORD_AVAILABLE = True
except Exception:  # pragma: no cover - environment without discord.py
    discord = None  # type: ignore
    DISCORD_AVAILABLE = False

# Populated by _define_life_ops_view_classes() once discord.py is importable.
JournalApproveView = None  # type: ignore
BedtimeView = None  # type: ignore
TodoClosureView = None  # type: ignore


def _ensure_view_classes() -> bool:
    """Define the View classes if possible; return True when they exist.

    Handles the bundled adapter's lazy-install path: discord.py may become
    importable only after gateway startup installed it, so retry the import
    here instead of trusting the module-import-time snapshot.
    """
    global discord, DISCORD_AVAILABLE
    if JournalApproveView is not None:
        return True
    if discord is None:
        try:
            import discord as _discord  # type: ignore
        except Exception:
            return False
        discord = _discord
        DISCORD_AVAILABLE = True
    _define_life_ops_view_classes()
    return JournalApproveView is not None


def _read_bedtime_config() -> dict:
    """Return the bedtime scheduler configuration from environment variables.

    Keys:
        enabled  — True when DISCORD_BEDTIME_HOUR is set and parseable
        hour     — int, UTC hour (0-23)
        minute   — int, UTC minute (0-59, default 0)
        timeout  — int, seconds the BedtimeView waits for a click (default 300)
    """
    raw_hour = os.getenv("DISCORD_BEDTIME_HOUR", "").strip()
    if not raw_hour:
        return {"enabled": False, "hour": 0, "minute": 0, "timeout": 300}
    try:
        hour = int(raw_hour)
    except ValueError:
        return {"enabled": False, "hour": 0, "minute": 0, "timeout": 300}
    raw_minute = os.getenv("DISCORD_BEDTIME_MINUTE", "0").strip()
    try:
        minute = int(raw_minute)
    except ValueError:
        minute = 0
    raw_timeout = os.getenv("DISCORD_BEDTIME_TIMEOUT", "300").strip()
    try:
        timeout = max(60, int(raw_timeout))
    except ValueError:
        timeout = 300
    return {"enabled": True, "hour": hour, "minute": minute, "timeout": timeout}


def _read_approvals_config() -> dict:
    """Return the journal-approvals dispatcher configuration from environment variables.

    Keys:
        enabled — True when DISCORD_APPROVALS_HOUR is set and parseable
        hour    — int, UTC hour (0-23)
        minute  — int, UTC minute (0-59, default 0)
    """
    raw_hour = os.getenv("DISCORD_APPROVALS_HOUR", "").strip()
    if not raw_hour:
        return {"enabled": False, "hour": 0, "minute": 0}
    try:
        hour = int(raw_hour)
    except ValueError:
        return {"enabled": False, "hour": 0, "minute": 0}
    raw_minute = os.getenv("DISCORD_APPROVALS_MINUTE", "0").strip()
    try:
        minute = int(raw_minute)
    except ValueError:
        minute = 0
    return {"enabled": True, "hour": hour, "minute": minute}


def _read_todo_closure_config() -> dict:
    """Return the todo-closure-view scheduler configuration from environment
    variables.

    Keys:
        enabled — True when DISCORD_TODO_CLOSURE_HOUR is set and parseable
        hour    — int, UTC hour (0-23)
        minute  — int, UTC minute (0-59, default 0)

    Same shape/env-var convention as :func:`_read_approvals_config` /
    :func:`_read_bedtime_config` — fires once per day, posting the
    TodoClosureView (see send_todo_closure_view) to
    discord.morning_brief_channel_id.
    """
    raw_hour = os.getenv("DISCORD_TODO_CLOSURE_HOUR", "").strip()
    if not raw_hour:
        return {"enabled": False, "hour": 0, "minute": 0}
    try:
        hour = int(raw_hour)
    except ValueError:
        return {"enabled": False, "hour": 0, "minute": 0}
    raw_minute = os.getenv("DISCORD_TODO_CLOSURE_MINUTE", "0").strip()
    try:
        minute = int(raw_minute)
    except ValueError:
        minute = 0
    return {"enabled": True, "hour": hour, "minute": minute}


def _read_journal_brief_path() -> str:
    """Return the path to journal_brief.latest.json.

    Honors JOURNAL_BRIEF_PATH like scripts/morning_brief_composer.py; falls
    back to HERMES_HOME/contracts/journal_brief.latest.json (same default
    the composer uses).
    """
    raw = os.environ.get("JOURNAL_BRIEF_PATH", "").strip()
    if raw:
        return raw
    home = os.environ.get("HERMES_HOME", "").strip() or os.path.expanduser("~/.hermes")
    return os.path.join(home, "contracts", "journal_brief.latest.json")


def _read_morning_brief_channel_id() -> str:
    """Return ``discord.morning_brief_channel_id`` from config.yaml, or "".

    Same lookup used by the morning-brief delivery script
    (plugins/life_ops/scripts/morning_brief_discord.py).
    """
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config() or {}
        discord_cfg = cfg.get("discord", {}) or {}
        raw = discord_cfg.get("morning_brief_channel_id")
    except Exception:
        return ""
    return str(raw or "").strip()


def _read_idle_nudge_config() -> dict:
    """Return the idle-day nudge scheduler configuration from environment variables.

    Keys:
        enabled — True when BOTH DISCORD_NUDGE_IDLE_HOUR and DISCORD_NUDGE_IDLE_MINUTE are set
        hour    — int, UTC hour (0-23)
        minute  — int, UTC minute (0-59)

    Both vars must be set; if either is absent the scheduler does not start.
    """
    raw_hour = os.getenv("DISCORD_NUDGE_IDLE_HOUR", "").strip()
    raw_minute = os.getenv("DISCORD_NUDGE_IDLE_MINUTE", "").strip()
    if not raw_hour or not raw_minute:
        return {"enabled": False, "hour": 0, "minute": 0}
    try:
        hour = int(raw_hour)
    except ValueError:
        return {"enabled": False, "hour": 0, "minute": 0}
    try:
        minute = int(raw_minute)
    except ValueError:
        return {"enabled": False, "hour": 0, "minute": 0}
    return {"enabled": True, "hour": hour, "minute": minute}


def _read_stale_nudge_config() -> dict:
    """Return the stale-todo nudge scheduler configuration from environment variables.

    Keys:
        enabled — True when DISCORD_NUDGE_STALE_HOUR is set and parseable
        hour    — int, UTC hour (0-23)
        minute  — int, UTC minute (0-59, default 0)
        days    — int, stale threshold in days (default 5)
    """
    raw_hour = os.getenv("DISCORD_NUDGE_STALE_HOUR", "").strip()
    if not raw_hour:
        return {"enabled": False, "hour": 0, "minute": 0, "days": 5}
    try:
        hour = int(raw_hour)
    except ValueError:
        return {"enabled": False, "hour": 0, "minute": 0, "days": 5}
    raw_minute = os.getenv("DISCORD_NUDGE_STALE_MINUTE", "0").strip()
    try:
        minute = int(raw_minute)
    except ValueError:
        minute = 0
    raw_days = os.getenv("DISCORD_NUDGE_STALE_DAYS", "5").strip()
    try:
        days = int(raw_days)
    except ValueError:
        days = 5
    return {"enabled": True, "hour": hour, "minute": minute, "days": days}


class LifeOpsDiscordAdapter(DiscordAdapter):
    """Bundled DiscordAdapter + life_ops schedulers, views, and commands."""

    def __init__(self, config):
        super().__init__(config)
        self._bedtime_task: Optional[asyncio.Task] = None
        self._approvals_task: Optional[asyncio.Task] = None
        self._todo_closure_task: Optional[asyncio.Task] = None
        self._stale_nudge_task: Optional[asyncio.Task] = None
        self._idle_nudge_task: Optional[asyncio.Task] = None

    # ── lifecycle wiring ─────────────────────────────────────────────────

    async def _run_post_connect_initialization(self) -> None:
        if not self._client:
            return
        self._start_bedtime_scheduler()
        self._start_approvals_scheduler()
        self._start_todo_closure_scheduler()
        self._start_stale_todo_nudge_scheduler()
        self._start_idle_day_nudge_scheduler()
        await super()._run_post_connect_initialization()

    async def cancel_background_tasks(self) -> None:
        await self._cancel_bedtime_task()
        await self._cancel_approvals_task()
        await self._cancel_todo_closure_task()
        await self._cancel_stale_todo_nudge_task()
        await self._cancel_idle_day_nudge_task()
        await super().cancel_background_tasks()

    def _register_slash_commands(self) -> None:
        """Register life_ops slash commands, then the bundled set.

        Order matters: the base method's Discord 100-command-cap accounting
        introspects the live ``tree`` (``already_registered`` /
        ``slot_cap``), so the fork commands must be on the tree BEFORE
        ``super()._register_slash_commands()`` runs — otherwise the
        auto-register loop could fill every slot and the fork commands
        would push the total past Discord's hard cap (error 30032, which
        breaks ALL slash commands).
        """
        if not self._client:
            return

        tree = self._client.tree

        # ── /rpe — training feedback ingestion ────────────────────────────
        try:
            from plugins.life_ops.discord_commands import register_rpe_command
            register_rpe_command(tree)
        except Exception as _rpe_err:
            logger.debug("Failed to register /rpe command: %s", _rpe_err)

        # ── /done, /dismiss, /snooze — persistent todo store closure ──────
        try:
            from plugins.life_ops.discord_commands import (
                register_dismiss_command,
                register_done_command,
                register_snooze_command,
            )
            register_done_command(tree)
            register_dismiss_command(tree)
            register_snooze_command(tree)
        except Exception as _todo_cmd_err:
            logger.debug("Failed to register /done, /dismiss, /snooze commands: %s", _todo_cmd_err)

        # ── /away-on, /away-off — away-mode kill switch ────────────────────
        try:
            from plugins.life_ops.discord_commands import (
                register_away_off_command,
                register_away_on_command,
            )
            register_away_on_command(tree)
            register_away_off_command(tree)
        except Exception as _away_cmd_err:
            logger.debug("Failed to register /away-on, /away-off commands: %s", _away_cmd_err)

        super()._register_slash_commands()

    # ── bedtime scheduler ────────────────────────────────────────────────

    def _start_bedtime_scheduler(self) -> None:
        """Start the bedtime overnight-sprint scheduler if configured.

        Idempotent: a second call while the task is live is a no-op.
        The loop fires once per night at DISCORD_BEDTIME_HOUR:DISCORD_BEDTIME_MINUTE (UTC),
        fetches the current backlog count from Commander, and posts the
        Start/Skip prompt to DISCORD_HOME_CHANNEL.
        """
        cfg = _read_bedtime_config()
        if not cfg["enabled"]:
            return
        if self._bedtime_task and not self._bedtime_task.done():
            return
        self._bedtime_task = asyncio.create_task(self._bedtime_scheduler_loop(cfg))
        logger.info(
            "[%s] Bedtime scheduler started (fire at %02d:%02d UTC)",
            self.name, cfg["hour"], cfg["minute"],
        )

    async def _bedtime_scheduler_loop(self, cfg: dict) -> None:
        """Loop indefinitely, posting the bedtime prompt once per night."""
        import datetime

        while True:
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            target = now_utc.replace(
                hour=cfg["hour"], minute=cfg["minute"], second=0, microsecond=0
            )
            if target <= now_utc:
                target += datetime.timedelta(days=1)
            delay = (target - now_utc).total_seconds()
            logger.debug(
                "[%s] Bedtime scheduler: sleeping %.0fs until %s UTC",
                self.name, delay, target.isoformat(),
            )
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return

            # Away-mode kill switch: skip the prompt entirely (no LLM call,
            # no backlog fetch — just the store check) while away, then go
            # back to sleep for the next night. Recompute "today" fresh here
            # rather than reusing the pre-sleep ``now_utc`` — the sleep can
            # span a midnight UTC rollover.
            today_str = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
            try:
                from plugins.life_ops import away_mode
                is_away = away_mode.is_away(today_str)
            except Exception as exc:
                logger.warning(
                    "[%s] Bedtime: away_mode check failed (%s); proceeding as not-away",
                    self.name, exc,
                )
                is_away = False
            if is_away:
                try:
                    until = away_mode.away_status(today_str).get("until")
                except Exception:
                    until = None
                logger.info(
                    "[%s] Bedtime prompt skipped — away mode active until %s",
                    self.name, until or "(indefinite)",
                )
                continue

            # Fetch backlog count
            try:
                from plugins.life_ops.bedtime import fetch_backlog_count
                backlog_n = await asyncio.to_thread(fetch_backlog_count)
            except Exception as exc:
                logger.error(
                    "[%s] Bedtime: failed to fetch backlog count: %s", self.name, exc
                )
                home_channel_id = os.getenv("DISCORD_HOME_CHANNEL", "").strip()
                if home_channel_id and self._client:
                    try:
                        ch = self._client.get_channel(int(home_channel_id))
                        if ch:
                            await ch.send(
                                "❌ Could not fetch backlog count for overnight sprint prompt."
                                f" Error: {exc}"
                            )
                    except Exception:
                        pass
                continue

            # Build allowed-user set from env for button auth
            raw_users = os.getenv("DISCORD_ALLOWED_USERS", "")
            allowed_user_ids = {
                u.strip() for u in raw_users.split(",") if u.strip()
            }
            # Allow all users when DISCORD_ALLOW_ALL_USERS is set
            if os.getenv("DISCORD_ALLOW_ALL_USERS", "").strip().lower() in {"true", "1", "yes"}:
                allowed_user_ids = {"*"}

            prompt_text = (
                f"Backlog has {backlog_n} tickets. Start the overnight sprint?"
            )

            home_channel_id = os.getenv("DISCORD_HOME_CHANNEL", "").strip()
            if not home_channel_id or not self._client:
                logger.warning(
                    "[%s] Bedtime: DISCORD_HOME_CHANNEL not set; skipping prompt", self.name
                )
                continue

            if not _ensure_view_classes():
                logger.warning(
                    "[%s] Bedtime: discord.py unavailable; skipping prompt", self.name
                )
                continue

            try:
                ch = self._client.get_channel(int(home_channel_id))
                if ch is None:
                    logger.warning(
                        "[%s] Bedtime: channel %s not found in cache", self.name, home_channel_id
                    )
                    continue
                view = BedtimeView(
                    backlog_count=backlog_n,
                    allowed_user_ids=allowed_user_ids,
                )
                msg = await ch.send(prompt_text, view=view)
                view._message = msg
                logger.info(
                    "[%s] Bedtime prompt posted (backlog=%d, channel=%s)",
                    self.name, backlog_n, home_channel_id,
                )
            except Exception as exc:
                logger.error(
                    "[%s] Bedtime: failed to post prompt: %s", self.name, exc
                )

    async def _cancel_bedtime_task(self) -> None:
        """Cancel and await the bedtime scheduler task, if running."""
        if self._bedtime_task and not self._bedtime_task.done():
            self._bedtime_task.cancel()
            try:
                await self._bedtime_task
            except asyncio.CancelledError:
                pass
        self._bedtime_task = None

    # ── journal-approvals scheduler ──────────────────────────────────────

    def _start_approvals_scheduler(self) -> None:
        """Start the daily journal-approvals dispatcher if configured.

        Idempotent: a second call while the task is live is a no-op. The
        loop fires once per day at DISCORD_APPROVALS_HOUR:DISCORD_APPROVALS_MINUTE
        (UTC), reads dev-category todos from the journal brief, and posts
        one [Approve] embed per todo to config.yaml's
        discord.morning_brief_channel_id.
        """
        cfg = _read_approvals_config()
        if not cfg["enabled"]:
            logger.debug("[%s] Journal approvals scheduler disabled", self.name)
            return
        if self._approvals_task and not self._approvals_task.done():
            return
        self._approvals_task = asyncio.create_task(self._approvals_scheduler_loop(cfg))
        logger.info(
            "[%s] Journal approvals scheduler started (fire at %02d:%02d UTC)",
            self.name, cfg["hour"], cfg["minute"],
        )

    async def _approvals_scheduler_loop(self, cfg: dict) -> None:
        """Loop indefinitely, firing the journal-approvals dispatcher once per day."""
        import datetime

        while True:
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            target = now_utc.replace(
                hour=cfg["hour"], minute=cfg["minute"], second=0, microsecond=0
            )
            if target <= now_utc:
                target += datetime.timedelta(days=1)
            delay = (target - now_utc).total_seconds()
            logger.debug(
                "[%s] Journal approvals scheduler: sleeping %.0fs until %s UTC",
                self.name, delay, target.isoformat(),
            )
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return

            try:
                await self._fire_journal_approvals()
            except Exception as exc:
                logger.error(
                    "[%s] Journal approvals: unexpected error firing dispatcher: %s",
                    self.name, exc,
                )

    async def _fire_journal_approvals(self) -> None:
        """Post one [Approve] embed per dev-category journal todo.

        Skips (with a log line) when no approve projects (JOURNAL_APPROVE_PROJECTS
        or JOURNAL_APPROVE_PROJECT) or the target channel is not configured, when
        there are zero dev todos, or when today's date is already recorded as
        posted (gateway restart guard).
        """
        import datetime

        from plugins.life_ops import journal_approve
        from plugins.life_ops.config import get_approve_projects

        today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        if journal_approve.has_posted_approvals_today(today):
            logger.debug(
                "[%s] Journal approvals: already posted for %s; skipping",
                self.name, today,
            )
            return

        projects = get_approve_projects()
        if not projects:
            logger.warning(
                "[%s] Journal approvals: no JOURNAL_APPROVE_PROJECTS/JOURNAL_APPROVE_PROJECT set; skipping",
                self.name,
            )
            return

        brief_path = _read_journal_brief_path()
        try:
            raw_todos = await asyncio.to_thread(journal_approve.load_dev_todos, brief_path)
        except Exception as exc:
            logger.error(
                "[%s] Journal approvals: failed to load dev todos from %s: %s",
                self.name, brief_path, exc,
            )
            return

        todos = journal_approve.map_dev_todos_for_send(raw_todos)
        if not todos:
            logger.info("[%s] Journal approvals: no dev todos to post", self.name)
            return

        channel_id = _read_morning_brief_channel_id()
        if not channel_id or not self._client:
            logger.warning(
                "[%s] Journal approvals: discord.morning_brief_channel_id not set; skipping",
                self.name,
            )
            return

        result = await self.send_journal_dev_todos(channel_id, todos, projects=projects)
        if result.success:
            journal_approve.mark_approvals_posted(today)
            logger.info(
                "[%s] Journal approvals posted (%d todo(s), channel=%s)",
                self.name, len(todos), channel_id,
            )
        else:
            logger.error(
                "[%s] Journal approvals: failed to post: %s", self.name, result.error,
            )

    async def _cancel_approvals_task(self) -> None:
        """Cancel and await the journal-approvals scheduler task, if running."""
        if self._approvals_task and not self._approvals_task.done():
            self._approvals_task.cancel()
            try:
                await self._approvals_task
            except asyncio.CancelledError:
                pass
        self._approvals_task = None

    # ── todo-closure scheduler ───────────────────────────────────────────

    def _start_todo_closure_scheduler(self) -> None:
        """Start the daily todo-closure-view dispatcher if configured.

        Idempotent: a second call while the task is live is a no-op. Same
        in-process live-client posting pattern as bedtime/approvals — the
        loop fires once per day at DISCORD_TODO_CLOSURE_HOUR:MINUTE (UTC)
        and posts the interactive [Mark Done/Dismiss/Snooze] control (see
        send_todo_closure_view) for the current open todos.
        """
        cfg = _read_todo_closure_config()
        if not cfg["enabled"]:
            logger.debug("[%s] Todo closure view scheduler disabled", self.name)
            return
        if self._todo_closure_task and not self._todo_closure_task.done():
            return
        self._todo_closure_task = asyncio.create_task(self._todo_closure_scheduler_loop(cfg))
        logger.info(
            "[%s] Todo closure view scheduler started (fire at %02d:%02d UTC)",
            self.name, cfg["hour"], cfg["minute"],
        )

    async def _todo_closure_scheduler_loop(self, cfg: dict) -> None:
        """Loop indefinitely, firing the todo-closure-view dispatcher once per day."""
        import datetime

        while True:
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            target = now_utc.replace(
                hour=cfg["hour"], minute=cfg["minute"], second=0, microsecond=0
            )
            if target <= now_utc:
                target += datetime.timedelta(days=1)
            delay = (target - now_utc).total_seconds()
            logger.debug(
                "[%s] Todo closure view scheduler: sleeping %.0fs until %s UTC",
                self.name, delay, target.isoformat(),
            )
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return

            try:
                await self._fire_todo_closure()
            except Exception as exc:
                logger.error(
                    "[%s] Todo closure view: unexpected error firing dispatcher: %s",
                    self.name, exc,
                )

    async def _fire_todo_closure(self) -> None:
        """Post the todo-closure control for the current open todos.

        Skips (with a log line) when discord.morning_brief_channel_id is not
        configured, or when there are zero open todos.
        """
        from plugins.life_ops import todo_store

        try:
            open_todos = await asyncio.to_thread(todo_store.get_open_todos)
        except Exception as exc:
            logger.error(
                "[%s] Todo closure view: failed to read open todos: %s", self.name, exc,
            )
            return

        if not open_todos:
            logger.info("[%s] Todo closure view: no open todos to post", self.name)
            return

        channel_id = _read_morning_brief_channel_id()
        if not channel_id or not self._client:
            logger.warning(
                "[%s] Todo closure view: discord.morning_brief_channel_id not set; skipping",
                self.name,
            )
            return

        result = await self.send_todo_closure_view(channel_id, open_todos)
        if result.success:
            logger.info(
                "[%s] Todo closure view posted (%d open todo(s), channel=%s)",
                self.name, len(open_todos), channel_id,
            )
        else:
            logger.error(
                "[%s] Todo closure view: failed to post: %s", self.name, result.error,
            )

    async def _cancel_todo_closure_task(self) -> None:
        """Cancel and await the todo-closure-view scheduler task, if running."""
        if self._todo_closure_task and not self._todo_closure_task.done():
            self._todo_closure_task.cancel()
            try:
                await self._todo_closure_task
            except asyncio.CancelledError:
                pass
        self._todo_closure_task = None

    # ── stale-todo nudge scheduler ───────────────────────────────────────

    def _start_stale_todo_nudge_scheduler(self) -> None:
        """Start the daily stale-todo nudge scheduler if configured.

        Idempotent: a second call while the task is live is a no-op. The
        loop fires once per day at DISCORD_NUDGE_STALE_HOUR:DISCORD_NUDGE_STALE_MINUTE
        (UTC), fetches stale todos via get_stale_todos(threshold_days=DISCORD_NUDGE_STALE_DAYS),
        and posts a TodoClosureView to discord.morning_brief_channel_id.
        Disabled by default (DISCORD_NUDGE_STALE_HOUR not set).
        """
        cfg = _read_stale_nudge_config()
        if not cfg["enabled"]:
            logger.debug("[%s] Stale-todo nudge scheduler disabled", self.name)
            return
        if self._stale_nudge_task and not self._stale_nudge_task.done():
            return
        self._stale_nudge_task = asyncio.create_task(self._stale_todo_nudge_scheduler_loop(cfg))
        logger.info(
            "[%s] Stale-todo nudge scheduler started (fire at %02d:%02d UTC, threshold=%d days)",
            self.name, cfg["hour"], cfg["minute"], cfg["days"],
        )

    async def _stale_todo_nudge_scheduler_loop(self, cfg: dict) -> None:
        """Loop indefinitely, firing the stale-todo nudge once per day."""
        import datetime

        while True:
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            target = now_utc.replace(
                hour=cfg["hour"], minute=cfg["minute"], second=0, microsecond=0
            )
            if target <= now_utc:
                target += datetime.timedelta(days=1)
            delay = (target - now_utc).total_seconds()
            logger.debug(
                "[%s] Stale-todo nudge scheduler: sleeping %.0fs until %s UTC",
                self.name, delay, target.isoformat(),
            )
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return

            try:
                await self._fire_stale_todo_nudge(cfg["days"])
            except Exception as exc:
                logger.error(
                    "[%s] Stale-todo nudge: unexpected error firing: %s",
                    self.name, exc,
                )

    async def _fire_stale_todo_nudge(self, threshold_days: int) -> None:
        """Post a stale-todo nudge with a TodoClosureView if stale todos exist.

        Skips when:
        - away mode is active
        - get_stale_todos() returns an empty list
        - discord.morning_brief_channel_id is not configured
        """
        import datetime

        today_str = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        try:
            from plugins.life_ops import away_mode
            is_away = away_mode.is_away(today_str)
        except Exception as exc:
            logger.warning(
                "[%s] Stale-todo nudge: away_mode check failed (%s); proceeding as not-away",
                self.name, exc,
            )
            is_away = False

        if is_away:
            logger.info("[%s] Stale-todo nudge skipped — away mode active", self.name)
            return

        try:
            from plugins.life_ops import todo_store
            stale_todos = await asyncio.to_thread(
                todo_store.get_stale_todos, threshold_days=threshold_days
            )
        except Exception as exc:
            logger.error(
                "[%s] Stale-todo nudge: failed to read stale todos: %s", self.name, exc,
            )
            return

        if not stale_todos:
            logger.info(
                "[%s] Stale-todo nudge: no stale todos (threshold=%d days); skipping",
                self.name, threshold_days,
            )
            return

        channel_id = _read_morning_brief_channel_id()
        if not channel_id or not self._client:
            logger.warning(
                "[%s] Stale-todo nudge: discord.morning_brief_channel_id not set; skipping",
                self.name,
            )
            return

        if not _ensure_view_classes():
            logger.warning(
                "[%s] Stale-todo nudge: discord.py unavailable; skipping", self.name
            )
            return

        try:
            channel = self._client.get_channel(int(channel_id))
            if not channel:
                channel = await self._client.fetch_channel(int(channel_id))

            count = len(stale_todos)
            content = (
                f"Still on these? {count} todo(s) haven't moved in {threshold_days}+ days:"
            )
            view = TodoClosureView(
                open_todos=stale_todos,
                allowed_user_ids=self._allowed_user_ids,
                allowed_role_ids=self._allowed_role_ids,
            )
            msg = await channel.send(content=content, view=view)
            view._message = msg
            logger.info(
                "[%s] Stale-todo nudge posted (%d stale todo(s), threshold=%d days, channel=%s)",
                self.name, count, threshold_days, channel_id,
            )
        except Exception as exc:
            logger.error(
                "[%s] Stale-todo nudge: failed to post: %s", self.name, exc,
            )

    async def _cancel_stale_todo_nudge_task(self) -> None:
        """Cancel and await the stale-todo nudge scheduler task, if running."""
        if self._stale_nudge_task and not self._stale_nudge_task.done():
            self._stale_nudge_task.cancel()
            try:
                await self._stale_nudge_task
            except asyncio.CancelledError:
                pass
        self._stale_nudge_task = None

    # ── idle-day nudge scheduler ─────────────────────────────────────────

    def _start_idle_day_nudge_scheduler(self) -> None:
        """Start the idle-day nudge scheduler if both env vars are configured.

        Idempotent: a second call while the task is live is a no-op. The loop
        fires once per day at DISCORD_NUDGE_IDLE_HOUR:DISCORD_NUDGE_IDLE_MINUTE
        (UTC). Both vars must be set; if either is absent the scheduler does not
        start.
        """
        cfg = _read_idle_nudge_config()
        if not cfg["enabled"]:
            logger.debug("[%s] Idle-day nudge scheduler disabled", self.name)
            return
        if self._idle_nudge_task and not self._idle_nudge_task.done():
            return
        self._idle_nudge_task = asyncio.create_task(self._idle_day_nudge_scheduler_loop(cfg))
        logger.info(
            "[%s] Idle-day nudge scheduler started (fire at %02d:%02d UTC)",
            self.name, cfg["hour"], cfg["minute"],
        )

    async def _idle_day_nudge_scheduler_loop(self, cfg: dict) -> None:
        """Loop indefinitely, firing the idle-day nudge once per day."""
        import datetime

        while True:
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            target = now_utc.replace(
                hour=cfg["hour"], minute=cfg["minute"], second=0, microsecond=0
            )
            if target <= now_utc:
                target += datetime.timedelta(days=1)
            delay = (target - now_utc).total_seconds()
            logger.debug(
                "[%s] Idle-day nudge scheduler: sleeping %.0fs until %s UTC",
                self.name, delay, target.isoformat(),
            )
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return

            try:
                await self._fire_idle_day_nudge()
            except Exception as exc:
                logger.error(
                    "[%s] Idle-day nudge: unexpected error firing: %s",
                    self.name, exc,
                )

    async def _fire_idle_day_nudge(self) -> None:
        """Post an idle-day nudge when all three conditions hold simultaneously.

        Conditions: away mode inactive AND count_todos_closed_today() == 0
        AND len(get_open_todos()) > 0. Silent skip (no error) otherwise.
        """
        import datetime

        today_str = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        try:
            from plugins.life_ops import away_mode
            is_away = away_mode.is_away(today_str)
        except Exception as exc:
            logger.warning(
                "[%s] Idle-day nudge: away_mode check failed (%s); proceeding as not-away",
                self.name, exc,
            )
            is_away = False

        if is_away:
            logger.info("[%s] Idle-day nudge skipped — away mode active", self.name)
            return

        try:
            from plugins.life_ops import todo_store
            closed_today = await asyncio.to_thread(todo_store.count_todos_closed_today)
        except Exception as exc:
            logger.error(
                "[%s] Idle-day nudge: failed to count closed todos: %s", self.name, exc,
            )
            return

        if closed_today > 0:
            logger.info(
                "[%s] Idle-day nudge skipped — %d todo(s) already closed today",
                self.name, closed_today,
            )
            return

        try:
            open_todos = await asyncio.to_thread(todo_store.get_open_todos)
        except Exception as exc:
            logger.error(
                "[%s] Idle-day nudge: failed to read open todos: %s", self.name, exc,
            )
            return

        if not open_todos:
            logger.info("[%s] Idle-day nudge skipped — no open todos", self.name)
            return

        channel_id = _read_morning_brief_channel_id()
        if not channel_id or not self._client:
            logger.warning(
                "[%s] Idle-day nudge: discord.morning_brief_channel_id not set; skipping",
                self.name,
            )
            return

        if not _ensure_view_classes():
            logger.warning(
                "[%s] Idle-day nudge: discord.py unavailable; skipping", self.name
            )
            return

        try:
            channel = self._client.get_channel(int(channel_id))
            if not channel:
                channel = await self._client.fetch_channel(int(channel_id))

            view = TodoClosureView(
                open_todos=open_todos,
                allowed_user_ids=self._allowed_user_ids,
                allowed_role_ids=self._allowed_role_ids,
            )
            msg = await channel.send(
                content="Haven't touched your list today — want to review it?",
                view=view,
            )
            view._message = msg
            logger.info(
                "[%s] Idle-day nudge posted (%d open todo(s), channel=%s)",
                self.name, len(open_todos), channel_id,
            )
        except Exception as exc:
            logger.error(
                "[%s] Idle-day nudge: failed to post: %s", self.name, exc,
            )

    async def _cancel_idle_day_nudge_task(self) -> None:
        """Cancel and await the idle-day nudge scheduler task, if running."""
        if self._idle_nudge_task and not self._idle_nudge_task.done():
            self._idle_nudge_task.cancel()
            try:
                await self._idle_nudge_task
            except asyncio.CancelledError:
                pass
        self._idle_nudge_task = None

    # ── senders ──────────────────────────────────────────────────────────

    async def send_journal_dev_todos(
        self,
        chat_id: str,
        todos: list,
        project: str = "",
        projects: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send dev-category journal todos with per-todo [Approve] buttons.

        Each todo gets its own message with a JournalApproveView.
        Only users in the adapter's allowlist may click Approve (AC2).

        ``todos`` is a list of dicts with keys: id, title, body.
        ``projects`` is the list of candidate target repos (``owner/repo``
        format) the approver may route the ticket to. When omitted, falls
        back to wrapping the legacy ``project`` string into a single-item
        list for back-compat.
        """
        if not self._client or not _ensure_view_classes():
            return SendResult(success=False, error="Not connected")

        if not todos:
            return SendResult(success=True, message_id="")

        if projects is None:
            projects = [project] if project else []

        try:
            target_id = chat_id
            if metadata and metadata.get("thread_id"):
                target_id = metadata["thread_id"]

            channel = self._client.get_channel(int(target_id))
            if not channel:
                channel = await self._client.fetch_channel(int(target_id))

            last_msg_id: str = ""
            for todo in todos:
                todo_id = str(todo.get("id") or "")
                title = str(todo.get("title") or "")
                body = str(todo.get("body") or "")
                label_title = _truncate_discord_component_text(title, 200)
                embed = discord.Embed(
                    title=f"📋 Dev Todo: {label_title}",
                    description=body[:4000] if body else "(no description)",
                    color=discord.Color.blurple(),
                )
                embed.set_footer(text=f"id: {todo_id}")
                view = JournalApproveView(
                    todo_id=todo_id,
                    title=title,
                    body=body,
                    projects=projects,
                    allowed_user_ids=self._allowed_user_ids,
                    allowed_role_ids=self._allowed_role_ids,
                )
                msg = await channel.send(embed=embed, view=view)
                view._message = msg
                last_msg_id = str(msg.id)

            return SendResult(success=True, message_id=last_msg_id)
        except Exception as exc:
            logger.warning("[%s] send_journal_dev_todos failed: %s", self.name, exc)
            return SendResult(success=False, error=str(exc))

    async def send_todo_closure_view(
        self,
        chat_id: str,
        open_todos: list,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Post a standalone [Mark Done / Dismiss / Snooze] control for open todos.

        Companion to the morning brief's plain-text todo section (rendered
        by the morning_brief_composer script): the brief text itself is
        typically delivered by a separate no-agent cron process
        (plugins/life_ops/scripts/morning_brief_discord.py) via raw REST,
        which has no live client and so cannot host interactive components.
        This method is the live-gateway-process path — the caller must own
        a connected ``self._client`` (e.g. the gateway's own
        ``_todo_closure_scheduler_loop`` above), so the same running
        process that posts the view also receives its interactions. This
        mirrors exactly how JournalApproveView / BedtimeView already work:
        there is no explicit "registration" step beyond passing ``view=``
        to ``channel.send`` — the poster and the interaction-handler are
        the same long-running process for as long as it stays up and the
        view hasn't timed out.

        No-op (success, empty message_id) when there are no open todos.
        """
        if not self._client or not _ensure_view_classes():
            return SendResult(success=False, error="Not connected")
        if not open_todos:
            return SendResult(success=True, message_id="")

        try:
            target_id = chat_id
            if metadata and metadata.get("thread_id"):
                target_id = metadata["thread_id"]

            channel = self._client.get_channel(int(target_id))
            if not channel:
                channel = await self._client.fetch_channel(int(target_id))

            view = TodoClosureView(
                open_todos=open_todos,
                allowed_user_ids=self._allowed_user_ids,
                allowed_role_ids=self._allowed_role_ids,
            )
            msg = await channel.send(
                content="Manage your open todos — select then click an action:",
                view=view,
            )
            view._message = msg
            return SendResult(success=True, message_id=str(msg.id))
        except Exception as exc:
            logger.warning("[%s] send_todo_closure_view failed: %s", self.name, exc)
            return SendResult(success=False, error=str(exc))


def _define_life_ops_view_classes() -> None:
    """Define the interactive View classes once discord.py is importable.

    Mirrors the bundled adapter's ``_define_discord_view_classes`` pattern:
    the classes subclass ``discord.ui.View``, so they can only be defined
    when the library is present.
    """
    global JournalApproveView, BedtimeView, TodoClosureView

    class JournalApproveView(discord.ui.View):
        """[Approve] view for promoting a dev journal todo to the backlog.

        One view instance is created per todo. When more than one candidate
        project is configured (JOURNAL_APPROVE_PROJECTS), a project-select
        dropdown is shown above the button so the approver can route the
        ticket to any of the user's Commander-tracked projects; the first
        entry is selected by default. When exactly one project is
        configured, the dropdown is omitted and behavior matches the
        original fixed-project flow.

        Clicking [Approve]:
          1. Auth-checks the clicker against the adapter's allowed users/roles.
          2. Calls handle_journal_approve() to POST to Commander's /api/tickets/create,
             targeting the currently-selected project.
          3. Confirms in-channel with "✅ Ticket created: #<N>" or an error message.
          4. Disables the button to prevent duplicate clicks.

        Unauthorised clicks (button or dropdown) receive an ephemeral
        rejection reply (AC2).
        """

        def __init__(
            self,
            todo_id: str,
            title: str,
            body: str,
            projects: list,
            allowed_user_ids: set,
            allowed_role_ids: Optional[set] = None,
        ):
            super().__init__(timeout=_read_discord_prompt_timeout())
            self.todo_id = todo_id
            self.title = title
            self.body = body
            # Discord caps select options at 25 — truncate defensively.
            self.projects = list(projects or [])[:25]
            self.project = self.projects[0] if self.projects else ""
            self.allowed_user_ids = allowed_user_ids
            self.allowed_role_ids = allowed_role_ids or set()
            self.resolved = False

            if len(self.projects) > 1:
                self._build_project_select()

        def _check_auth(self, interaction: discord.Interaction) -> bool:
            return _component_check_auth(
                interaction, self.allowed_user_ids, self.allowed_role_ids
            )

        def _build_project_select(self):
            """Add a project-picker dropdown, defaulting to the first entry."""
            options = [
                discord.SelectOption(
                    label=_truncate_discord_component_text(
                        repo, _DISCORD_SELECT_FIELD_LIMIT
                    ),
                    value=_truncate_discord_component_text(
                        repo, _DISCORD_SELECT_FIELD_LIMIT
                    ),
                    default=(repo == self.project),
                )
                for repo in self.projects
            ]
            select = discord.ui.Select(
                placeholder="Choose a project...",
                options=options,
                custom_id="journal_approve_project_select",
                row=0,
            )
            select.callback = self._on_project_selected
            self.add_item(select)

        async def _on_project_selected(self, interaction: discord.Interaction):
            """Update the current project selection. Does not resolve the view."""
            if self.resolved:
                await interaction.response.send_message(
                    "This todo has already been approved.", ephemeral=True
                )
                return

            if not self._check_auth(interaction):
                await interaction.response.send_message(
                    "You're not authorised to approve journal todos.", ephemeral=True
                )
                return

            values = (interaction.data or {}).get("values") or []
            selected = values[0] if values else self.project
            self.project = selected

            for item in self.children:
                if isinstance(item, discord.ui.Select):
                    for opt in item.options:
                        opt.default = (opt.value == selected)

            await interaction.response.edit_message(view=self)

        @discord.ui.button(label="Approve", style=discord.ButtonStyle.green, row=1)
        async def approve(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ):
            if self.resolved:
                await interaction.response.send_message(
                    "This todo has already been approved.", ephemeral=True
                )
                return

            if not self._check_auth(interaction):
                await interaction.response.send_message(
                    "You're not authorised to approve journal todos.", ephemeral=True
                )
                return

            self.resolved = True
            for child in self.children:
                child.disabled = True

            user_id = str(getattr(getattr(interaction, "user", None), "id", "unknown"))

            try:
                from plugins.life_ops.journal_approve import handle_journal_approve
                result = handle_journal_approve(
                    todo_id=self.todo_id,
                    title=self.title,
                    body=self.body,
                    project=self.project,
                    user_id=user_id,
                )
            except Exception as exc:
                logger.error("journal_approve error for todo=%s: %s", self.todo_id, exc)
                result = {"success": False, "duplicate": False, "error": str(exc), "ticket_number": None}

            if result.get("duplicate"):
                reply = f"⚠️ Todo `{self.todo_id}` was already approved — no duplicate ticket created."
            elif result.get("success"):
                num = result.get("ticket_number")
                reply = f"✅ Ticket created: #{num}" if num else "✅ Ticket created."
            else:
                reply = f"❌ Could not create ticket: {result.get('error', 'unknown error')}"

            try:
                await interaction.response.edit_message(content=reply, view=self)
            except Exception:
                try:
                    await interaction.followup.send(content=reply, ephemeral=False)
                except Exception:
                    pass

        async def on_timeout(self):
            self.resolved = True
            for child in self.children:
                child.disabled = True
            msg = getattr(self, "_message", None)
            if msg:
                try:
                    await msg.edit(view=self)
                except Exception:
                    pass

    class BedtimeView(discord.ui.View):
        """Two-button prompt for the bedtime overnight-sprint interaction.

        Posts once per night at the configured DISCORD_BEDTIME_HOUR/MINUTE.
        Clicking [Start] calls Commander's POST /api/sprints/run (after
        confirming no sprint is already running).  Clicking [Skip] (or timeout)
        updates the message and writes an audit log entry.

        Idempotency: self.resolved is set on the first handled interaction;
        subsequent clicks receive an ephemeral "Already handled" reply.
        """

        def __init__(
            self,
            *,
            backlog_count: int,
            allowed_user_ids: set,
            allowed_role_ids: Optional[set] = None,
        ) -> None:
            super().__init__(timeout=_read_discord_prompt_timeout())
            self.backlog_count = backlog_count
            self.allowed_user_ids = allowed_user_ids
            self.allowed_role_ids = allowed_role_ids or set()
            self.resolved = False

        def _check_auth(self, interaction: discord.Interaction) -> bool:
            return _component_check_auth(
                interaction, self.allowed_user_ids, self.allowed_role_ids
            )

        @discord.ui.button(label="Start", style=discord.ButtonStyle.green, custom_id="bedtime_start")
        async def start(self, interaction: discord.Interaction, button: discord.ui.Button):
            if self.resolved:
                await interaction.response.send_message(
                    "Already handled.", ephemeral=True
                )
                return
            if not self._check_auth(interaction):
                await interaction.response.send_message(
                    "You're not authorised to use this button.", ephemeral=True
                )
                return

            self.resolved = True
            user = getattr(interaction, "user", None)
            user_id = str(getattr(user, "id", "unknown"))
            username = getattr(user, "name", "unknown")
            for child in self.children:
                child.disabled = True

            from plugins.life_ops import bedtime as _bedtime

            try:
                sprint_status = await asyncio.to_thread(_bedtime.check_running_sprint)
            except Exception as exc:
                logger.error("BedtimeView: check_running_sprint failed: %s", exc)
                await interaction.response.send_message(
                    f"❌ Error checking sprint status: {exc}", ephemeral=True
                )
                self.resolved = False
                for child in self.children:
                    child.disabled = False
                return

            if sprint_status["running"]:
                _bedtime.log_bedtime_action(
                    user_id=user_id,
                    username=username,
                    action="skip",
                    sprint_id=sprint_status.get("sprint_id"),
                )
                await interaction.response.send_message(
                    "A sprint is already running — skipped.", ephemeral=True
                )
                return

            result = await asyncio.to_thread(_bedtime.start_sprint)
            if not result["success"]:
                _bedtime.log_bedtime_action(
                    user_id=user_id,
                    username=username,
                    action="start",
                    sprint_id=None,
                )
                await interaction.response.send_message(
                    f"❌ Could not start sprint: {result.get('error', 'unknown error')}",
                    ephemeral=True,
                )
                return

            sprint_id = result["sprint_id"]
            _bedtime.log_bedtime_action(
                user_id=user_id,
                username=username,
                action="start",
                sprint_id=sprint_id,
            )
            try:
                await interaction.response.edit_message(
                    content=f"✅ Overnight sprint started! (ID: {sprint_id})",
                    view=None,
                )
            except Exception:
                try:
                    await interaction.followup.send(
                        content=f"✅ Overnight sprint started! (ID: {sprint_id})"
                    )
                except Exception:
                    pass

        @discord.ui.button(label="Skip", style=discord.ButtonStyle.grey, custom_id="bedtime_skip")
        async def skip(self, interaction: discord.Interaction, button: discord.ui.Button):
            if self.resolved:
                await interaction.response.send_message(
                    "Already handled.", ephemeral=True
                )
                return
            if not self._check_auth(interaction):
                await interaction.response.send_message(
                    "You're not authorised to use this button.", ephemeral=True
                )
                return

            self.resolved = True
            user = getattr(interaction, "user", None)
            user_id = str(getattr(user, "id", "unknown"))
            username = getattr(user, "name", "unknown")
            for child in self.children:
                child.disabled = True

            from plugins.life_ops import bedtime as _bedtime
            _bedtime.log_bedtime_action(
                user_id=user_id,
                username=username,
                action="skip",
                sprint_id=None,
            )
            try:
                await interaction.response.edit_message(
                    content="⏭ Overnight sprint skipped.",
                    view=self,
                )
            except Exception:
                try:
                    await interaction.followup.send(
                        content="⏭ Overnight sprint skipped.", ephemeral=False
                    )
                except Exception:
                    pass

        async def on_timeout(self) -> None:
            self.resolved = True
            for child in self.children:
                child.disabled = True
            from plugins.life_ops import bedtime as _bedtime
            _bedtime.log_bedtime_action(
                user_id="system",
                username="system",
                action="timeout",
                sprint_id=None,
            )
            msg = getattr(self, "_message", None)
            if msg:
                try:
                    await msg.edit(
                        content="⏰ Overnight sprint prompt timed out — no action taken.",
                        view=self,
                    )
                except Exception:
                    pass

    class TodoClosureView(discord.ui.View):
        """Select-then-act view for closing out open todos from the morning brief.

        A multi-select (0-25 of the current open todos, key -> "{glyph}
        {text}" label) plus three buttons — Mark Done / Dismiss / Snooze 1
        week — that apply the chosen ``todo_store`` action to whatever is
        currently checked in the select. Modeled directly on
        JournalApproveView's select-then-button pattern (project dropdown +
        Approve button): the select updates ``self.selected`` and the
        buttons act on it, rather than each option carrying its own
        per-item action button (Discord selects don't support that).

        Same auth gate as the other component views (_component_check_auth).
        Closed keys are removed from the select's options after a
        successful action so they can't be re-clicked; the underlying
        message is edited in place rather than requiring a full brief
        re-render.
        """

        def __init__(
            self,
            open_todos: list,
            allowed_user_ids: set,
            allowed_role_ids: Optional[set] = None,
        ):
            super().__init__(timeout=_read_discord_prompt_timeout())
            # Discord caps select options at 25 — truncate defensively.
            self.open_todos = {
                str(t.get("key")): t for t in (open_todos or [])[:25] if t.get("key")
            }
            self.allowed_user_ids = allowed_user_ids
            self.allowed_role_ids = allowed_role_ids or set()
            self.selected: set = set()
            self._select: Optional["discord.ui.Select"] = None
            self._build_select()

        def _check_auth(self, interaction: discord.Interaction) -> bool:
            return _component_check_auth(
                interaction, self.allowed_user_ids, self.allowed_role_ids
            )

        def _build_select(self) -> None:
            options = []
            for key, todo in self.open_todos.items():
                glyph = "!" if str(todo.get("priority") or "").lower() == "high" else "·"
                text = str(todo.get("text") or "")
                label = _truncate_discord_component_text(
                    f"{glyph} {text}", _DISCORD_SELECT_FIELD_LIMIT
                )
                options.append(
                    discord.SelectOption(
                        label=label or key,
                        value=_truncate_discord_component_text(key, _DISCORD_SELECT_FIELD_LIMIT),
                    )
                )
            select = discord.ui.Select(
                placeholder="Choose todo(s) to act on...",
                options=options or [discord.SelectOption(label="(no open todos)", value="__none__")],
                min_values=0,
                max_values=max(1, len(options)),
                disabled=not options,
                custom_id="todo_closure_select",
                row=0,
            )
            select.callback = self._on_select
            self._select = select
            self.add_item(select)

        async def _on_select(self, interaction: discord.Interaction) -> None:
            """Record the current selection. Does not apply any action."""
            if not self._check_auth(interaction):
                await interaction.response.send_message(
                    "You're not authorised to manage todos.", ephemeral=True
                )
                return

            values = (interaction.data or {}).get("values") or []
            self.selected = {v for v in values if v != "__none__"}
            try:
                await interaction.response.send_message(
                    f"Selected {len(self.selected)} todo(s) — click a button below to apply.",
                    ephemeral=True,
                )
            except Exception:
                pass

        async def _apply(self, interaction: discord.Interaction, action: str, label: str) -> None:
            if not self._check_auth(interaction):
                await interaction.response.send_message(
                    "You're not authorised to manage todos.", ephemeral=True
                )
                return
            if not self.selected:
                await interaction.response.send_message(
                    "Select at least one todo first.", ephemeral=True
                )
                return

            from plugins.life_ops import todo_store

            snooze_until = None
            if action == "snooze":
                import datetime as _dt
                snooze_until = (_dt.date.today() + _dt.timedelta(days=7)).isoformat()

            ok_keys: list = []
            err_parts: list = []
            for key in sorted(self.selected):
                try:
                    result = await asyncio.to_thread(
                        todo_store.close_todo, key, action, "discord:select", snooze_until,
                    )
                except Exception as exc:
                    result = {"ok": False, "error": str(exc)}
                if result.get("ok"):
                    ok_keys.append(key)
                else:
                    err_parts.append(f"`{key}`: {result.get('error', 'unknown error')}")

            # Remove closed options from the select so they can't be
            # re-clicked; disable it entirely once nothing is left.
            if self._select is not None:
                self._select.options = [
                    opt for opt in self._select.options if opt.value not in ok_keys
                ]
                if not self._select.options:
                    self._select.options = [
                        discord.SelectOption(label="(no open todos)", value="__none__")
                    ]
                    self._select.disabled = True
            self.selected = set()

            reply_parts = []
            if ok_keys:
                reply_parts.append(f"✅ {label}: " + ", ".join(f"`{k}`" for k in ok_keys))
            if err_parts:
                reply_parts.append("⚠️ " + "; ".join(err_parts))
            reply = "\n".join(reply_parts) if reply_parts else "No changes made."

            try:
                await interaction.response.edit_message(view=self)
                await interaction.followup.send(content=reply, ephemeral=True)
            except Exception:
                try:
                    await interaction.followup.send(content=reply, ephemeral=True)
                except Exception:
                    pass

        @discord.ui.button(label="Mark Done", style=discord.ButtonStyle.green, custom_id="todo_closure_done", row=1)
        async def mark_done(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._apply(interaction, "done", "Marked done")

        @discord.ui.button(label="Dismiss", style=discord.ButtonStyle.grey, custom_id="todo_closure_dismiss", row=1)
        async def dismiss(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._apply(interaction, "dismiss", "Dismissed")

        @discord.ui.button(label="Snooze 1 week", style=discord.ButtonStyle.blurple, custom_id="todo_closure_snooze", row=1)
        async def snooze(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._apply(interaction, "snooze", "Snoozed 1 week")

        async def on_timeout(self) -> None:
            for child in self.children:
                child.disabled = True
            msg = getattr(self, "_message", None)
            if msg:
                try:
                    await msg.edit(view=self)
                except Exception:
                    pass


if DISCORD_AVAILABLE:
    _define_life_ops_view_classes()

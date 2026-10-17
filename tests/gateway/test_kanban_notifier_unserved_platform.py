"""A notify subscription naming a platform this build cannot serve is a dead route.

``_Collector._claim_for_sub`` skips a row whose platform has no adapter on this
gateway with only a DEBUG line: the row stays visible in ``notify-list``, the
task's terminal events stay unseen forever, and nothing surfaces the dead end
(#140268). Permanent dead-ends are reported ONCE per row at WARNING elsewhere
(route-pin mismatch #115460, anchorless thread #110919); this row class must be
too.

``tui`` is the one deliberate exception: the gateway never holds a ``tui``
adapter because the TUI polls its own board's subscriptions
(``tui_gateway/session_notifications.py``), so that skip is the designed path.
"""

import logging
from pathlib import Path

from gateway.config import GatewayConfig, Platform
from gateway.kanban_watchers_notifier import _notifier_collect
from gateway.profile_routing import parse_profile_routes
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kbn
import gateway.kanban_watchers_notifier as notifier


class RecordingAdapter:
    supports_async_delivery = True

    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, **kwargs):
        self.sent.append((chat_id, text, kwargs))

    async def handle_message(self, event):
        event._gateway_accepted = True


def _runner(tmp_path, monkeypatch):
    """A gateway that hosts exactly one platform (discord) for profile ``yuki``."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    profile = home / "profiles" / "yuki"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("{}\n", encoding="utf-8")
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.DISCORD: RecordingAdapter()}
    runner._profile_adapters = {"yuki": {}}
    runner._primary_profile_name = "default"
    runner._kanban_notifier_profile = "default"
    runner._kanban_dispatcher_lock_handle = object()
    runner.config = GatewayConfig(multiplex_profiles=True, profile_routes=parse_profile_routes([
        dict(platform="discord", guild_id="guild", chat_id="parent", profile="yuki"),
    ]))
    return runner


def _subscribe(*, platform, chat, thread=None, profile="default", metadata=None, mode="notify+wake"):
    with kbc.connect() as conn:
        task = kb.create_task(conn, title=f"{platform} completion", assignee="worker")
        kbn.add_notify_sub(conn, task_id=task, platform=platform, chat_id=chat, thread_id=thread,
                           notifier_profile=profile, delivery_mode=mode, delivery_metadata=metadata)
        kb.complete_task(conn, task, result="finished")
    return task


def _collect(runner):
    return _notifier_collect(runner, kb, notifier_profile="default", gc_due=False, gc_retention_days=30)


def _warnings(caplog, needle):
    return [r for r in caplog.records if r.levelno >= logging.WARNING and needle in r.getMessage()]


def test_unserved_platform_subscription_is_reported_once_instead_of_silent_skip(tmp_path, monkeypatch, caplog):
    runner = _runner(tmp_path, monkeypatch)
    unserved = _subscribe(platform="webui", chat="webui-session-1")
    hosted = _subscribe(platform="discord", chat="post", thread="post", profile="yuki",
                        metadata={"guild_id": "guild", "scope_id": "guild", "parent_chat_id": "parent"})

    with caplog.at_level(logging.WARNING, logger=notifier.logger.name):
        rows = _collect(runner)

    # Delivery semantics unchanged: only the served row is claimed.
    assert [row["task"].id for row in rows] == [hosted]
    assert not _warnings(caplog, hosted), "a served subscription must not warn"

    warned = _warnings(caplog, unserved)
    assert len(warned) == 1, f"dead route must be reported exactly once: {[r.getMessage() for r in caplog.records]}"
    message = warned[0].getMessage()
    assert "no connected adapter" in message
    assert "will not be delivered" in message
    assert "webui" in message

    # The row is a permanent dead end: its events stay unseen and nothing was claimed.
    with kbc.connect() as conn:
        _cursor, events = kbn.unseen_events_for_sub(
            conn, task_id=unserved, platform="webui", chat_id="webui-session-1", kinds=["completed"])
        assert events, "the unserved row's terminal event is still unseen (silent dead end)"
    assert [row for row in rows if row["sub"]["task_id"] == unserved] == []

    # Warn-once: the next tick re-skips the same row without repeating the warning.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=notifier.logger.name):
        _collect(runner)
    assert not _warnings(caplog, unserved)


def test_tui_subscription_skip_is_not_reported_as_unserved(tmp_path, monkeypatch, caplog):
    """The gateway holds no ``tui`` adapter by design: the TUI polls its own board."""
    runner = _runner(tmp_path, monkeypatch)
    tui_task = _subscribe(platform="tui", chat="tui-session-1", mode=None)

    with caplog.at_level(logging.WARNING, logger=notifier.logger.name):
        rows = _collect(runner)

    assert rows == []
    assert not _warnings(caplog, tui_task)

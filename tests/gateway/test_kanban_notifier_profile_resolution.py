"""Real profile resolution at the notifier boundary, with disposable board state."""
import asyncio
from pathlib import Path

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn
from tests.gateway.test_kanban_notifier import RecordingAdapter, _make_runner, _run_one_notifier_tick


def _subscription(profile, chat_id):
    with kbc.connect() as conn:
        task = kb.create_task(conn, title=f"notify {profile}", assignee="worker")
        kbn.add_notify_sub(conn, task_id=task, platform="telegram", chat_id=chat_id,
                           notifier_profile=profile, delivery_mode="notify+wake", chat_type="dm")
        kb.block_task(conn, task, reason="Required capability unavailable", kind="capability")
    return task


def test_bound_profiles_override_launcher_without_cross_profile_delivery(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_PROFILE_NAME", "launcher")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    kb.init_db()
    homes = {}
    for profile in ("alpha", "beta"):
        home = tmp_path / ".hermes" / "profiles" / profile
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("{}\n")
        homes[profile] = home
    adapters = {profile: RecordingAdapter() for profile in homes}
    for index, profile in enumerate(("alpha", "beta", "alpha")):
        chat_id = f"chat-{index}"
        _subscription(profile, chat_id)
        token = set_hermes_home_override(homes[profile])
        try:
            runner = _make_runner(adapters[profile])
            runner._primary_profile_name = profile
            asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
            assert runner._kanban_notifier_profile == profile
        finally:
            reset_hermes_home_override(token)
    assert [item["chat_id"] for item in adapters["alpha"].sent] == ["chat-0", "chat-2"]
    assert [item["chat_id"] for item in adapters["beta"].sent] == ["chat-1"]
    assert len(adapters["alpha"].handled) == 2
    assert len(adapters["beta"].handled) == 1


def test_explicit_notifier_binding_survives_conflicting_launcher_pin(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    monkeypatch.setenv("HERMES_PROFILE_NAME", "other")
    kb.init_db()
    _subscription("owner", "owner-chat")
    _subscription("other", "other-chat")
    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner._primary_profile_name = "owner"
    runner._kanban_notifier_profile = "owner"
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert runner._kanban_notifier_profile == "owner"
    assert [item["chat_id"] for item in adapter.sent] == ["owner-chat"]
    assert len(adapter.handled) == 1

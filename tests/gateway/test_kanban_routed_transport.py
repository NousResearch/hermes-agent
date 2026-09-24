"""Persisted notification routes authorize exactly one transport, including route-only profiles."""
import asyncio
from pathlib import Path

from gateway.config import GatewayConfig, Platform
from gateway.kanban_watchers_notifier import _KanbanNotification, _notifier_collect
from gateway.profile_routing import parse_profile_routes
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kbn


class RecordingAdapter:
    supports_async_delivery = True

    def __init__(self):
        self.sent = []
        self.handled = []

    async def send(self, chat_id, text, **kwargs):
        self.sent.append((chat_id, text, kwargs))

    async def handle_message(self, event):
        self.handled.append(event)
        event._gateway_accepted = True


def setup_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    for name in ("yuki", "other"):
        profile = home / "profiles" / name
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


def completion(*, profile="yuki", metadata=None, chat="post", thread="post", mode="notify+wake"):
    with kbc.connect() as conn:
        task = kb.create_task(conn, title="route completion", assignee="worker")
        kbn.add_notify_sub(conn, task_id=task, platform="discord", chat_id=chat,
                           thread_id=thread, chat_type="thread", user_id="creator",
                           notifier_profile=profile, delivery_mode=mode,
                           delivery_metadata=metadata if metadata is not None else
                           {"guild_id": "guild", "scope_id": "guild", "parent_chat_id": "parent"})
        kb.complete_task(conn, task, result="finished")
    return task


def collect(runner):
    return _notifier_collect(runner, kb, notifier_profile="default", gc_due=False, gc_retention_days=30)


async def deliver(runner, rows):
    for row in rows:
        await _KanbanNotification(runner, row, platform_cls=Platform, sub_fail_counts={}).deliver()


def unseen(task):
    with kbc.connect() as conn:
        return kbn.unseen_events_for_sub(conn, task_id=task, platform="discord", chat_id="post",
                                         thread_id="post", kinds=["completed"])[1]


def test_exact_routed_profile_delivers_once_on_its_authorized_transport(tmp_path, monkeypatch):
    runner = setup_runner(tmp_path, monkeypatch)
    primary = runner.adapters[Platform.DISCORD]
    task = completion(metadata={"scope_id": "guild", "guild_id": "stale-alias", "parent_chat_id": "parent"})
    rows = collect(runner)
    assert [row["task"].id for row in rows] == [task]
    asyncio.run(deliver(runner, rows))
    assert len(primary.sent) == len(primary.handled) == 1
    source = primary.handled[0].source
    assert (source.profile, source.guild_id, source.scope_id, source.parent_chat_id) == (
        "yuki", "guild", "guild", "parent")
    assert runner._adapter_for_source(source) is primary
    assert not collect(runner)

    # A connected secondary owns its credential even where the primary route matches.
    secondary = RecordingAdapter()
    secondary.scope_id_for_chat = lambda chat: "stale-cache"
    runner._profile_adapters["yuki"] = {Platform.DISCORD: secondary}
    task = completion(metadata={"guild_id": "guild", "parent_chat_id": "parent"})
    asyncio.run(deliver(runner, collect(runner)))
    assert len(primary.sent) == 1
    assert len(secondary.sent) == len(secondary.handled) == 1
    assert secondary.handled[0].source.scope_id == "guild"
    assert runner._adapter_for_source(secondary.handled[0].source) is secondary
    assert not unseen(task)


def test_route_denials_leave_events_retryable_at_claim_and_send(tmp_path, monkeypatch):
    runner = setup_runner(tmp_path, monkeypatch)
    primary = runner.adapters[Platform.DISCORD]
    # Unknown owners, wrong/default owners, incomplete anchors, and partial credentials
    # never become primary delivery authority.
    tasks = [completion(profile=owner) for owner in ("other", "default", None)]
    tasks += [completion(metadata=meta) for meta in (
        {"parent_chat_id": "parent"}, {"guild_id": "guild"},
        {"guild_id": "wrong", "parent_chat_id": "parent"},
    )]
    assert not collect(runner)
    assert all(unseen(task) for task in tasks)

    good = completion()
    runner._profile_adapters["yuki"] = {Platform.TELEGRAM: RecordingAdapter()}
    assert not collect(runner)
    runner._profile_adapters["yuki"] = {}
    runner.config.multiplex_profile_allowlist = ["other"]
    assert not collect(runner)
    runner.config.multiplex_profile_allowlist = ["yuki"]
    rows = collect(runner)
    assert [row["task"].id for row in rows] == [good]
    # Reassignment after the claim must rewind, never send using stale authority.
    runner.config.profile_routes = parse_profile_routes([
        dict(platform="discord", guild_id="guild", chat_id="parent", profile="other")])
    asyncio.run(deliver(runner, rows))
    assert primary.sent == primary.handled == []
    assert unseen(good)

    # Equal-specificity rules retain configuration order: an unknown parent
    # cannot skip an earlier rule, but a known conflicting parent rules it out.
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "tied-routes.db"))
    runner.config.profile_routes = parse_profile_routes([
        dict(platform="discord", guild_id="guild", chat_id="parent", profile="other"),
        dict(platform="discord", guild_id="guild", chat_id="post", profile="yuki"),
    ])
    ambiguous = completion(metadata={"scope_id": "guild"})
    exact = completion(metadata={"scope_id": "guild", "parent_chat_id": "different-parent"})
    rows = collect(runner)
    assert [row["task"].id for row in rows] == [exact]
    asyncio.run(deliver(runner, rows))
    assert len(primary.sent) == len(primary.handled) == 1
    assert unseen(ambiguous)


def test_kanban_wakes_install_the_destination_runtime_scope(tmp_path, monkeypatch):
    from agent.secret_scope import get_secret
    from gateway.run import _profile_runtime_scope
    from hermes_constants import get_hermes_home

    runner = setup_runner(tmp_path, monkeypatch)
    home = tmp_path / ".hermes"
    (home / ".env").write_text("KANBAN_TEST_SECRET=primary\n", encoding="utf-8")
    observed = []

    class ScopedAdapter(RecordingAdapter):
        async def handle_message(self, event):
            # A real yield catches scopes that mutate process-global state.
            await asyncio.sleep(0)
            observed.append((event.source.profile, get_secret("KANBAN_TEST_SECRET"), get_hermes_home()))
            await super().handle_message(event)

    for name in ("yuki", "other"):
        (home / "profiles" / name / ".env").write_text(f"KANBAN_TEST_SECRET={name}\n", encoding="utf-8")
        runner._profile_adapters[name] = {Platform.DISCORD: ScopedAdapter()}
        completion(profile=name)
    rows = collect(runner)
    assert len(rows) == 2

    async def concurrent_wakes():
        with _profile_runtime_scope(home):
            await asyncio.gather(*(_KanbanNotification(runner, row, platform_cls=Platform,
                                                       sub_fail_counts={}).deliver() for row in rows))
            assert get_secret("KANBAN_TEST_SECRET") == "primary"
    asyncio.run(concurrent_wakes())
    assert sorted(observed) == [(name, name, home / "profiles" / name) for name in ("other", "yuki")]


def test_removed_profile_never_wakes_under_the_primary_runtime(tmp_path, monkeypatch):
    import shutil

    runner = setup_runner(tmp_path, monkeypatch)
    secondary = RecordingAdapter()
    runner._profile_adapters["yuki"] = {Platform.DISCORD: secondary}
    task = completion(mode="wake")
    rows = collect(runner)
    assert len(rows) == 1
    shutil.rmtree(tmp_path / ".hermes" / "profiles" / "yuki")
    asyncio.run(deliver(runner, rows))
    assert secondary.handled == []
    assert unseen(task)

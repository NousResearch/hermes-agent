"""EP1 ``pre_result_delivery`` seam tests.

Without a plugin the core delivery path is unchanged (the directive helper
returns None); with a test plugin registered the handled/hold directives and
fail-open behaviour work. Behaviour tests only.
"""

import asyncio
import subprocess
import sys

from cron.scheduler import _FireOwnership, _RunDelivery, _pre_result_delivery_directive
from cron.scheduler_bookkeeping import _classify_delivery_outcome
from hermes_cli.plugins import VALID_HOOKS, get_plugin_manager


def _job(**over):
    job = {"id": "job1", "name": "job1", "deliver": "local", "execution_id": "exec1"}
    job.update(over)
    return job


def _delivery(job, success=True):
    return _RunDelivery(job=job, success=success, error=None, should_deliver=True)


def _fence(job):
    # No fire_claim: owner is None, the fence always owns (no ownership games in these tests).
    return _FireOwnership(job)


def _isolate_hooks(monkeypatch):
    manager = get_plugin_manager()
    saved = {name: list(callbacks) for name, callbacks in manager._hooks.items()}
    monkeypatch.setattr(manager, "_hooks", {name: list(cbs) for name, cbs in saved.items()})
    return manager


def test_hook_registered():
    assert "pre_result_delivery" in VALID_HOOKS


def test_no_plugin_normal_send(monkeypatch):
    manager = _isolate_hooks(monkeypatch)
    manager._hooks.pop("pre_result_delivery", None)
    job = _job()
    assert _pre_result_delivery_directive(_delivery(job), _fence(job), "hello") is None


def test_failure_path_skips_hook(monkeypatch):
    seen = []
    manager = _isolate_hooks(monkeypatch)
    manager._hooks.setdefault("pre_result_delivery", []).append(
        lambda **kwargs: seen.append(True) or {"action": "handled", "receipt": "r"})
    job = _job()
    assert _pre_result_delivery_directive(_delivery(job, success=False), _fence(job), "x") is None
    assert seen == []


def test_handled_skips_send(monkeypatch):
    manager = _isolate_hooks(monkeypatch)
    manager._hooks.setdefault("pre_result_delivery", []).append(
        lambda **kwargs: {"action": "handled", "receipt": "card-123"})
    job = _job()
    action, plugin, receipt = _pre_result_delivery_directive(_delivery(job), _fence(job), "hello")
    assert action == "handled"
    assert receipt == "card-123"
    assert isinstance(plugin, str) and plugin


def test_hold_withholds_as_held(monkeypatch):
    manager = _isolate_hooks(monkeypatch)
    manager._hooks.setdefault("pre_result_delivery", []).append(
        lambda **kwargs: {"action": "hold", "reason": "ingest-failed"})
    job = _job()
    action, plugin, reason = _pre_result_delivery_directive(_delivery(job), _fence(job), "hello")
    assert action == "hold"
    assert reason == "ingest-failed"
    assert _classify_delivery_outcome(
        delivery_error=f"held:{plugin}:{reason}", should_deliver=True, unresolved_origin=False,
        normalized_deliver="local", incident_acked=False, success=True,
    ) == "held"


def test_plain_error_classifies_failed_while_held_stays_held():
    """The held: marker must not swallow real delivery errors: a plain error
    string still classifies as "failed" (upstream semantics), while the
    plugin-withheld "held:<plugin>:<reason>" marker classifies as "held"."""
    assert _classify_delivery_outcome(
        delivery_error="send blew up", should_deliver=True, unresolved_origin=False,
        normalized_deliver="local", incident_acked=False, success=True,
    ) == "failed"
    assert _classify_delivery_outcome(
        delivery_error="held:kanban:ingest-failed", should_deliver=True, unresolved_origin=False,
        normalized_deliver="local", incident_acked=False, success=True,
    ) == "held"


def test_first_valid_directive_wins(monkeypatch):
    manager = _isolate_hooks(monkeypatch)
    manager._hooks.setdefault("pre_result_delivery", []).extend([
        lambda **kwargs: {"action": "deliver"},
        lambda **kwargs: {"action": "handled", "receipt": "second"},
    ])
    job = _job()
    action, _plugin, receipt = _pre_result_delivery_directive(_delivery(job), _fence(job), "hello")
    assert (action, receipt) == ("handled", "second")


def test_raise_fails_open(monkeypatch):
    manager = _isolate_hooks(monkeypatch)

    def broken(**kwargs):
        raise RuntimeError("plugin crashed")

    manager._hooks.setdefault("pre_result_delivery", []).append(broken)
    job = _job()
    assert _pre_result_delivery_directive(_delivery(job), _fence(job), "hello") is None


def test_unknown_action_ignored(monkeypatch):
    manager = _isolate_hooks(monkeypatch)
    manager._hooks.setdefault("pre_result_delivery", []).append(
        lambda **kwargs: {"action": "explode"})
    job = _job()
    assert _pre_result_delivery_directive(_delivery(job), _fence(job), "hello") is None


def test_hook_payload_contract(monkeypatch):
    manager = _isolate_hooks(monkeypatch)
    seen = {}

    def probe(**kwargs):
        seen.update(kwargs)

    manager._hooks.setdefault("pre_result_delivery", []).append(probe)
    job = _job()
    _pre_result_delivery_directive(_delivery(job), _fence(job), "hello")
    assert seen["source"] == "cron"
    assert seen["job_id"] == "job1"
    assert seen["execution_id"] == "exec1"
    assert seen["content"] == "hello"
    assert seen["targets"] == []
    assert isinstance(seen["job"], dict) and seen["job"]["id"] == "job1"


def test_kanban_handled_skips_ping(monkeypatch):
    """Kanban site: a handled directive skips the ping; no plugin means a normal send."""
    from gateway.kanban_watchers_notifier import _KanbanNotification

    manager = _isolate_hooks(monkeypatch)

    class Ev:
        kind = "completed"
        id = 7
        payload = {"summary": "done"}
        run_id = None

    sub = {"task_id": "t1", "platform": "telegram", "chat_id": "c1", "thread_id": ""}
    d = {"sub": sub, "events": [Ev()], "task": None, "board": "b", "runs": {}}
    note = _KanbanNotification.__new__(_KanbanNotification)
    note.runner = object()
    note.d = d
    note.sub = sub
    note.task = None
    note.board_slug = "b"
    note.platform_str = "telegram"
    note.task_id = "t1"
    note.title = "t1"

    async def run():
        manager._hooks.pop("pre_result_delivery", None)
        assert await note._pre_result_delivery_handled(Ev()) is False
        manager._hooks.setdefault("pre_result_delivery", []).append(
            lambda **kwargs: {"action": "handled", "receipt": "card-9"})
        assert await note._pre_result_delivery_handled(Ev()) is True

    asyncio.run(run())


def test_has_hook_works_in_restart_safe_worker():
    """R2: an ordinary (frozen-config) worker keeps hooks enabled.

    Only the admission bypass disables hooks; a restart-safe cron worker binds
    its config with ``_bind_worker_config`` and ``safe_worker_enabled()`` stays
    False, so ``has_hook`` still sees plugins. Runs isolated in a subprocess
    because the worker policy binds once per process.
    """
    code = (
        "from agent.safe_worker_policy import _bind_worker_config, safe_worker_enabled;"
        "_bind_worker_config({});"
        "assert safe_worker_enabled() is False, 'ordinary worker must keep hooks enabled';"
        "from hermes_cli import plugins as p;"
        "m = p.get_plugin_manager();"
        "m._hooks.setdefault('pre_result_delivery', []).append(lambda **k: None);"
        "assert p.has_hook('pre_result_delivery') is True;"
        "from hermes_cli.lifecycle import has_hook as life_has_hook;"
        "assert life_has_hook('pre_result_delivery') is True;"
        "print('R2-OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
        cwd="/home/kom/proj/ai_pr/hermes-seams", check=False,
        env={"PYTHONNOUSERSITE": "1", "PATH": "/usr/bin:/bin"},
    )
    assert "R2-OK" in proc.stdout, proc.stderr[-2000:]


def test_empty_content_never_reaches_hook(monkeypatch):
    """R10: the fire site sits after the empty/[SILENT] guards, so an empty draft
    leaves ``should_deliver`` False and no hook fires (soft-failure/silent path)."""
    from cron import scheduler

    seen = []
    manager = _isolate_hooks(monkeypatch)
    manager._hooks.setdefault("pre_result_delivery", []).append(
        lambda **kwargs: seen.append(True) or {"action": "handled", "receipt": "r"})
    job = _job()
    d = _RunDelivery(job=job, success=True, error=None)
    fence = _fence(job)
    scheduler._save_compose_deliver(
        d, fence, "", "",
        adapters=None, loop=None, verbose=False, execution_token=None)
    assert d.should_deliver is False
    assert seen == []

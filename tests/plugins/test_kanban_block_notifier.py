from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_plugin_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "plugins" / "kanban-block-notifier" / "__init__.py"
    spec = importlib.util.spec_from_file_location("kanban_block_notifier_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_redacts_secretish_reason_text():
    mod = _load_plugin_module()

    msg = mod._build_message(
        board="proj",
        task_id="t_1234abcd",
        title="Need Cloudflare",
        assignee="developer",
        kind="needs_input",
        reason="Set CLOUDFLARE_TOKEN=cf_1234567890abcdefghijklmnopqrstuvwxyz and https://x.test?a=1&token=supersecretvalue1234567890",
        config={"secure_drop_url": "https://drop.example/one"},
    )

    assert "cf_1234567890abcdefghijklmnopqrstuvwxyz" not in msg
    assert "supersecretvalue1234567890" not in msg
    assert "CLOUDFLARE_TOKEN=[redacted]" in msg
    assert "token=[redacted]" in msg
    assert "secure-drop: https://drop.example/one" in msg


def test_redacts_long_tokens_and_russian_secret_keywords():
    mod = _load_plugin_module()

    msg = mod._build_message(
        board="proj",
        task_id="t_1234abcd",
        title="Need secret",
        assignee="developer",
        kind="needs_input",
        reason="Нужен секрет ключ ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 для доступа",
        config={},
    )

    assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890" not in msg
    assert "[redacted]" in msg
    assert "secure-drop не настроен" in msg


def test_skips_review_required_and_dependency_blocks():
    mod = _load_plugin_module()

    assert not mod._is_human_block("needs_input", "review-required: tests pass", {})
    assert not mod._is_human_block("dependency", "waiting for parent", {})
    assert mod._is_human_block("needs_input", "need DNS choice", {})
    assert mod._is_human_block("capability", "missing access", {})
    assert mod._is_human_block(None, "Cloudflare DNS choice needed", {})


def test_dedupes_same_blocker_per_target(tmp_path, monkeypatch):
    mod = _load_plugin_module()
    sent: list[tuple[str, str]] = []

    monkeypatch.setattr(mod, "_plugin_config", lambda: {"state_db": str(tmp_path / "state.sqlite3"), "targets": ["telegram:test"]})
    monkeypatch.setattr(
        mod,
        "_fetch_task",
        lambda board, task_id: SimpleNamespace(
            id=task_id,
            title="Blocked task",
            assignee="developer",
            block_kind="needs_input",
        ),
    )
    monkeypatch.setattr(mod, "_send", lambda target, message, config: sent.append((target, message)))

    kwargs = {"board": "proj", "task_id": "t_deadbeef", "reason": "Need DNS decision"}
    mod._on_kanban_task_blocked(**kwargs)
    mod._on_kanban_task_blocked(**kwargs)

    assert len(sent) == 1
    assert sent[0][0] == "telegram:test"
    assert "Need DNS decision" in sent[0][1]


def test_dedupe_is_scoped_to_block_event_and_board(tmp_path, monkeypatch):
    mod = _load_plugin_module()
    sent: list[tuple[str, str]] = []
    state_db = tmp_path / "state.sqlite3"

    monkeypatch.setattr(mod, "_plugin_config", lambda: {"state_db": str(state_db), "targets": ["telegram:test"]})
    monkeypatch.setattr(
        mod,
        "_fetch_task",
        lambda board, task_id: SimpleNamespace(
            id=task_id,
            title="Blocked task",
            assignee="developer",
            block_kind="needs_input",
        ),
    )
    monkeypatch.setattr(mod, "_send", lambda target, message, config: sent.append((target, message)))

    kwargs = {"board": "project-board", "task_id": "t_deadbeef", "reason": "Need DNS decision"}
    mod._on_kanban_task_blocked(**kwargs, event_id=101)
    mod._on_kanban_task_blocked(**kwargs, event_id=101)
    mod._on_kanban_task_blocked(**kwargs, event_id=102)

    assert len(sent) == 2
    assert all("[project-board]" in message for _, message in sent)
    with mod.sqlite3.connect(str(state_db)) as conn:
        keys = [row[0] for row in conn.execute("SELECT dedupe_key FROM sent_block_notifications ORDER BY dedupe_key")]
    assert keys == [
        "project-board:t_deadbeef:event:101",
        "project-board:t_deadbeef:event:102",
    ]


def test_real_kanban_block_event_sends_once(tmp_path, monkeypatch):
    mod = _load_plugin_module()
    sent: list[tuple[str, str]] = []
    hook_calls: list[dict] = []
    home = tmp_path / "kanban-home"
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(mod, "_plugin_config", lambda: {"state_db": str(tmp_path / "state.sqlite3"), "targets": ["telegram:test"]})
    monkeypatch.setattr(mod, "_send", lambda target, message, config: sent.append((target, message)))

    from hermes_cli import kanban_db as kb
    import hermes_cli.plugins as plugins

    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="Configure HTTPS", assignee="developer", initial_status="running")

        def capture_hook(event, **kwargs):
            if event != "kanban_task_blocked":
                return
            # A separate connection can only observe the event after commit,
            # proving lifecycle dispatch is outside the write transaction.
            observer = kb.connect()
            try:
                row = observer.execute(
                    "SELECT id, kind FROM task_events WHERE id = ?", (kwargs["event_id"],)
                ).fetchone()
            finally:
                observer.close()
            assert row is not None
            assert row["kind"] == "blocked"
            hook_calls.append(kwargs)
            mod._on_kanban_task_blocked(**kwargs)

        monkeypatch.setattr(plugins, "invoke_hook", capture_hook)
        assert kb.block_task(conn, tid, reason="Need DNS decision", kind="needs_input")
        event_id = kb.list_events(conn, tid)[-1].id
        # Same blocker fired again through the lifecycle callback should be deduped.
        mod._on_kanban_task_blocked(board="default", task_id=tid, reason="Need DNS decision", event_id=event_id)
    finally:
        conn.close()

    assert len(sent) == 1
    assert len(hook_calls) == 1
    assert hook_calls[0]["event_id"] == event_id
    assert hook_calls[0]["kind"] == "needs_input"
    assert "Configure HTTPS" in sent[0][1]
    assert "Need DNS decision" in sent[0][1]


def test_dependency_block_hook_has_committed_event_id_and_kind(tmp_path, monkeypatch):
    home = tmp_path / "kanban-home"
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    from hermes_cli import kanban_db as kb
    import hermes_cli.plugins as plugins

    calls: list[dict] = []
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="Wait for parent", assignee="developer", initial_status="running")

        def capture_hook(event, **kwargs):
            if event != "kanban_task_blocked":
                return
            observer = kb.connect()
            try:
                row = observer.execute(
                    "SELECT id, kind FROM task_events WHERE id = ?", (kwargs["event_id"],)
                ).fetchone()
            finally:
                observer.close()
            assert row is not None
            assert row["kind"] == "dependency_wait"
            calls.append(kwargs)

        monkeypatch.setattr(plugins, "invoke_hook", capture_hook)
        assert kb.block_task(conn, tid, reason="waiting for parent", kind="dependency")
        event = kb.list_events(conn, tid)[-1]
    finally:
        conn.close()

    assert len(calls) == 1
    assert calls[0]["event_id"] == event.id
    assert calls[0]["kind"] == "dependency"


def test_state_db_path_fails_closed_without_kanban_resolver(monkeypatch):
    mod = _load_plugin_module()
    import hermes_cli

    monkeypatch.setattr(hermes_cli, "kanban_db", None, raising=False)

    assert mod._state_db_path({}) is None


def test_unblock_then_reblock_same_reason_sends_again(tmp_path, monkeypatch):
    mod = _load_plugin_module()
    sent: list[tuple[str, str]] = []
    home = tmp_path / "kanban-home"
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "project-board")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(mod, "_plugin_config", lambda: {"state_db": str(tmp_path / "state.sqlite3"), "targets": ["telegram:test"]})
    monkeypatch.setattr(mod, "_send", lambda target, message, config: sent.append((target, message)))

    from hermes_cli import kanban_db as kb
    import hermes_cli.plugins as plugins

    monkeypatch.setattr(
        plugins,
        "invoke_hook",
        lambda event, **kwargs: mod._on_kanban_task_blocked(**kwargs) if event == "kanban_task_blocked" else None,
    )
    kb.init_db(board="project-board")
    conn = kb.connect(board="project-board")
    try:
        tid = kb.create_task(conn, title="Configure HTTPS", assignee="developer", initial_status="running")
        assert kb.block_task(conn, tid, reason="Need DNS decision", kind="needs_input")
        first_event_id = kb.list_events(conn, tid)[-1].id
        # Replaying the same lifecycle event should not duplicate the notification.
        mod._on_kanban_task_blocked(board="project-board", task_id=tid, reason="Need DNS decision", event_id=first_event_id)
        ok, refusal = kb.promote_task(conn, tid, actor="test", force=True)
        assert ok, refusal
        assert kb.block_task(conn, tid, reason="Need DNS decision", kind="needs_input")
    finally:
        conn.close()

    assert len(sent) == 2
    assert all("[project-board]" in message for _, message in sent)


def test_retries_after_send_failure(tmp_path, monkeypatch):
    mod = _load_plugin_module()
    attempts = {"count": 0}

    monkeypatch.setattr(mod, "_plugin_config", lambda: {"state_db": str(tmp_path / "state.sqlite3"), "targets": ["telegram:test"]})
    monkeypatch.setattr(
        mod,
        "_fetch_task",
        lambda board, task_id: SimpleNamespace(
            id=task_id,
            title="Blocked task",
            assignee="developer",
            block_kind="needs_input",
        ),
    )

    def flaky_send(target, message, config):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("boom")

    monkeypatch.setattr(mod, "_send", flaky_send)

    kwargs = {"board": "proj", "task_id": "t_deadbeef", "reason": "Need DNS decision"}
    mod._on_kanban_task_blocked(**kwargs)
    mod._on_kanban_task_blocked(**kwargs)

    assert attempts["count"] == 2

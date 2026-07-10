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


def test_real_kanban_block_event_sends_once(tmp_path, monkeypatch):
    mod = _load_plugin_module()
    sent: list[tuple[str, str]] = []
    home = tmp_path / "kanban-home"
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
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
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="Configure HTTPS", assignee="developer", initial_status="running")
        assert kb.block_task(conn, tid, reason="Need DNS decision", kind="needs_input")
        # Same blocker fired again through the lifecycle callback should be deduped.
        mod._on_kanban_task_blocked(board="default", task_id=tid, reason="Need DNS decision")
    finally:
        conn.close()

    assert len(sent) == 1
    assert "Configure HTTPS" in sent[0][1]
    assert "Need DNS decision" in sent[0][1]


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

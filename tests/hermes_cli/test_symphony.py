import pytest

from hermes_cli import symphony


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setattr(symphony, "get_hermes_home", lambda: tmp_path)


def test_result_artifact_rejects_failed_or_missing_test_evidence(tmp_path):
    with pytest.raises(ValueError, match="passing test evidence"):
        symphony.write_result_artifact(
            tmp_path, task_id="dev_test", changed_files=[],
            tests=[{"command": "pytest", "exit_code": 1}],
            summary="failed", limitations=[],
        )
    assert not (tmp_path / ".symphony" / "dev_test-result.json").exists()


def test_result_artifact_rejects_path_escape_workspace(tmp_path):
    with pytest.raises(ValueError, match="outside the workspace"):
        symphony.write_result_artifact(
            tmp_path, task_id="dev_test", changed_files=["../secret.txt"],
            tests=[{"command": "pytest", "exit_code": 0}],
            summary="ok", limitations=[],
        )
    assert not (tmp_path / ".symphony" / "dev_test-result.json").exists()


def test_enqueue_builds_worker_policy_with_safety_and_evidence_contract(monkeypatch):
    task = symphony.create_task(
        title="Policy", source_text="Implement safely", project_id="project-a"
    )
    symphony.mark_ready(task.task_id)
    captured = {}
    monkeypatch.setattr(symphony.kb, "create_board", lambda *args, **kwargs: None)
    def fake_create_task(conn, **kwargs):
        captured["body"] = kwargs["body"]
        return "k_policy"
    monkeypatch.setattr(symphony.kb, "create_task", fake_create_task)
    monkeypatch.setattr(symphony.kb, "get_task", lambda *args, **kwargs: type("K", (), {"workspace_path": None, "branch_name": None})())
    monkeypatch.setattr(symphony.kb, "unblock_task", lambda *args, **kwargs: True)
    symphony.enqueue(task.task_id, assignee="symphony")
    assert "назначенном workspace" in captured["body"]
    assert "merge, production deploy" in captured["body"]
    assert "evidence" in captured["body"]


def test_create_is_idempotent_and_has_durable_event():
    first = symphony.create_task(
        title="Исправить тест", source_text="Исправить regression test",
        idempotency_key="telegram:123:456",
    )
    second = symphony.create_task(
        title="Дубликат", source_text="Не должен создать второй", idempotency_key="telegram:123:456",
    )

    assert second.task_id == first.task_id
    assert first.status == "draft"
    with symphony.connect() as conn:
        events = conn.execute(
            "SELECT kind FROM development_events WHERE task_id = ?", (first.task_id,)
        ).fetchall()
    assert [row["kind"] for row in events] == ["task.created"]


def test_transition_requires_ready_before_enqueue(monkeypatch):
    task = symphony.create_task(title="Тест", source_text="Сделать тест")
    with pytest.raises(ValueError, match="ready_for_development"):
        symphony.enqueue(task.task_id)

    ready = symphony.mark_ready(task.task_id)
    assert ready.status == "ready_for_development"


def test_run_slash_answer_clarification_reopens_task(monkeypatch, tmp_path):
    monkeypatch.setattr(symphony, "get_hermes_home", lambda: tmp_path)
    task = symphony.create_task(title="Короткая", source_text="fix", project_id="project-a")
    symphony.prepare_task(task.task_id)

    output = symphony.run_slash(f"answer-clarification {task.task_id} Добавить regression test для timeout")

    assert "ready_for_development" in output
    assert symphony.list_tasks(status="ready_for_development")[0].task_id == task.task_id


def test_prepare_task_builds_brief_criteria_and_risks_before_ready():
    task = symphony.create_task(
        title="Добавить API", source_text="В проекте demo добавь endpoint статуса", project_id="project-a",
    )

    prepared = symphony.prepare_task(task.task_id)

    assert prepared.status == "ready_for_development"
    assert "Технический brief" in prepared.technical_brief
    assert "Критерии готовности" in prepared.technical_brief
    assert "Риски" in prepared.technical_brief


def test_prepare_task_requires_project_and_nonempty_source():
    task = symphony.create_task(title="Неполная", source_text="Сделай", project_id=None)

    with pytest.raises(ValueError, match="registered Hermes project"):
        symphony.prepare_task(task.task_id)


def test_enqueue_requires_registered_project():
    task = symphony.create_task(title="Тест", source_text="Сделать тест")
    symphony.mark_ready(task.task_id)
    with pytest.raises(ValueError, match="registered Hermes project"):
        symphony.enqueue(task.task_id)


def test_pause_rejects_invalid_state_before_touching_kanban(monkeypatch):
    task = symphony.create_task(title="Draft", source_text="Not ready", project_id="project-a")
    monkeypatch.setattr(symphony.kb, "stop_task_for_user", lambda *args, **kwargs: pytest.fail("Kanban must not be touched"))
    with pytest.raises(ValueError, match="invalid Symphony transition"):
        symphony.pause(task.task_id)



    with pytest.raises(ValueError, match="invalid Symphony transition"):
        symphony._assert_transition("completed", "queued")


def test_cli_slash_creates_and_lists_task():
    created = symphony.run_slash('create "Русская задача" --body "Проверить очередь"')
    assert "Статус: draft" in created
    listed = symphony.run_slash("list")
    assert "Русская задача" in listed


def test_task_output_never_contains_secret_fields():
    task = symphony.create_task(title="Безопасность", source_text="Не логировать секреты")
    rendered = symphony.format_task(task)
    assert "token" not in rendered.lower()
    assert "secret" not in rendered.lower()


def test_events_are_bounded_and_redacted():
    task = symphony.create_task(title="Events", source_text="No token=sk_abcdefghijklmnop")
    events = symphony.list_events(task.task_id, limit=1)

    assert len(events) == 1
    assert events[0]["kind"] == "task.created"
    assert "sk_abcdefghijklmnop" not in str(events)

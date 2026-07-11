from __future__ import annotations

import ast
import os
import socket
import struct
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_writer as writer


pytestmark = pytest.mark.real_kanban_writer


@pytest.fixture
def service(tmp_path: Path):
    db_path = tmp_path / "kanban.db"
    with writer.privileged_maintenance(db_path):
        kb.init_db(db_path)
    instance = writer.KanbanWriterService(db_path)
    instance.start()
    try:
        yield instance
    finally:
        instance.stop()


def test_writer_protocol_never_imports_pickle() -> None:
    module_path = Path(writer.__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "pickle" not in imported, "authenticated local clients are not trusted to send pickle"


def test_same_request_id_with_different_payload_is_rejected(
    service: writer.KanbanWriterService,
) -> None:
    client = writer.KanbanWriterClient(service.db_path, timeout_seconds=1)
    client.mutate("create_task", {"title": "first"}, request_id="same-id")

    with pytest.raises(writer.WriterProtocolError, match="payload|request_id"):
        client.mutate("create_task", {"title": "different"}, request_id="same-id")


def test_same_request_id_with_different_generation_is_rejected(
    service: writer.KanbanWriterService,
) -> None:
    generations = writer.kanban_safety.read_generations(service.db_path)
    client = writer.KanbanWriterClient(service.db_path, timeout_seconds=1)
    client.mutate(
        "create_task",
        {"title": "first"},
        request_id="generation-id",
        expected_service_generation=generations.service_generation,
        expected_board_generation=generations.board_generation,
    )

    with pytest.raises(writer.WriterProtocolError, match="envelope|request_id"):
        client.mutate(
            "create_task",
            {"title": "first"},
            request_id="generation-id",
            expected_service_generation=generations.service_generation + 1,
            expected_board_generation=generations.board_generation,
        )


@pytest.mark.parametrize(
    ("second_actor", "second_source", "second_operation", "second_payload"),
    [
        ("rita", "worker", "create_task", {"title": "first"}),
        ("ava", "gateway", "create_task", {"title": "first"}),
        ("ava", "worker", "archive_task", {"task_id": "t_missing"}),
    ],
)
def test_same_request_id_with_different_provenance_is_rejected(
    service: writer.KanbanWriterService,
    second_actor: str,
    second_source: str,
    second_operation: str,
    second_payload: dict[str, str],
) -> None:
    first = writer.KanbanWriterClient(
        service.db_path, actor_profile="ava", source="worker", timeout_seconds=5
    )
    first.mutate("create_task", {"title": "first"}, request_id="provenance-id")

    second = writer.KanbanWriterClient(
        service.db_path,
        actor_profile=second_actor,
        source=second_source,
        timeout_seconds=5,
    )
    with pytest.raises(writer.WriterProtocolError, match="request_id|provenance|payload"):
        second.mutate(
            second_operation,
            second_payload,
            request_id="provenance-id",
        )


def test_slow_peer_cannot_block_following_authenticated_client(
    service: writer.KanbanWriterService,
) -> None:
    slow = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    slow.connect(str(service.socket_path))
    try:
        started = time.monotonic()
        client = writer.KanbanWriterClient(service.db_path, timeout_seconds=5)
        task_id = client.mutate(
            "create_task", {"title": "not starved"}, request_id="not-starved"
        )
        elapsed = time.monotonic() - started
    finally:
        slow.close()

    assert task_id
    # The incomplete peer must not block authenticated work indefinitely.
    # Allow scheduler/SQLite headroom on constrained CI hosts.
    assert elapsed < 5.0


def test_response_loss_replays_without_duplicate_mutation(
    service: writer.KanbanWriterService,
) -> None:
    generations = writer.kanban_safety.read_generations(service.db_path)
    request = {
        "authentication_token": service._token,
        "actor_profile": "default",
        "board": service.board,
        "expected_board_generation": generations.board_generation,
        "expected_service_generation": generations.service_generation,
        "operation": "create_task",
        "payload": writer._encode_value({"title": "commit then lose response"}),
        "request_id": "lost-response",
        "source": "worker",
        "version": writer.PROTOCOL_VERSION,
    }
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
        peer.connect(str(service.socket_path))
        writer._send_message(peer, request)

    committed = None
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with kb.connect(service.db_path) as conn:
            committed = conn.execute(
                "SELECT 1 FROM kanban_writer_requests WHERE request_id=?",
                ("lost-response",),
            ).fetchone()
        if committed:
            break
        time.sleep(0.01)
    assert committed is not None

    task_id = writer.KanbanWriterClient(service.db_path).mutate(
        "create_task",
        {"title": "commit then lose response"},
        request_id="lost-response",
        expected_service_generation=generations.service_generation,
        expected_board_generation=generations.board_generation,
    )
    with kb.connect(service.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
        assert kb.get_task(conn, task_id).title == "commit then lose response"


def test_response_loss_replays_after_writer_restart_and_generation_bump(
    service: writer.KanbanWriterService,
) -> None:
    generations = writer.kanban_safety.read_generations(service.db_path)
    request = {
        "authentication_token": service._token,
        "actor_profile": "default",
        "board": service.board,
        "expected_board_generation": generations.board_generation,
        "expected_service_generation": generations.service_generation,
        "operation": "create_task",
        "payload": writer._encode_value({"title": "restart replay"}),
        "request_id": "restart-lost-response",
        "source": "worker",
        "version": writer.PROTOCOL_VERSION,
    }
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
        peer.connect(str(service.socket_path))
        writer._send_message(peer, request)

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with kb.connect(service.db_path) as conn:
            if conn.execute(
                "SELECT 1 FROM kanban_writer_requests WHERE request_id=?",
                ("restart-lost-response",),
            ).fetchone():
                break
        time.sleep(0.01)
    else:
        pytest.fail("writer never committed the request before restart")

    service.stop()
    restarted = writer.KanbanWriterService(service.db_path, board=service.board)
    restarted.start()
    try:
        current = writer.kanban_safety.read_generations(service.db_path)
        assert current.service_generation > generations.service_generation
        retry = dict(request)
        retry["authentication_token"] = restarted._token
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
            peer.settimeout(10)
            peer.connect(str(restarted.socket_path))
            writer._send_message(peer, retry)
            response = writer._recv_message(peer)
        assert response["ok"] is True
        assert response["replayed"] is True
        task_id = writer._decode_value(response["result"])
        with kb.connect(service.db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
            assert kb.get_task(conn, task_id).title == "restart replay"
    finally:
        restarted.stop()


def test_activation_marker_precedes_serving_thread(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "kanban.db"
    with writer.privileged_maintenance(db_path):
        kb.init_db(db_path)
    service = writer.KanbanWriterService(db_path)
    original_activate = writer.activate_writer_requirement
    observed = {}

    def activate(path):
        observed["thread_alive"] = bool(
            service._thread is not None and service._thread.is_alive()
        )
        original_activate(path)

    monkeypatch.setattr(writer, "activate_writer_requirement", activate)
    service.start()
    try:
        assert observed == {"thread_alive": False}
        assert writer.is_writer_required(db_path)
    finally:
        service.stop()


def test_protocol_v1_request_is_explicitly_rejected(
    service: writer.KanbanWriterService,
) -> None:
    generations = writer.kanban_safety.read_generations(service.db_path)
    request = {
        "authentication_token": service._token,
        "actor_profile": "default",
        "board": service.board,
        "expected_board_generation": generations.board_generation,
        "expected_service_generation": generations.service_generation,
        "operation": "create_task",
        "payload": writer._encode_value({"title": "old protocol"}),
        "request_id": "protocol-v1",
        "source": "worker",
        "version": 1,
    }
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
        peer.settimeout(1)
        peer.connect(str(service.socket_path))
        writer._send_message(peer, request)
        response = writer._recv_message(peer)
    assert response["ok"] is False
    assert response["error_type"] == "WriterProtocolError"
    assert "unsupported writer protocol version" in response["message"]


@pytest.mark.parametrize("bad_generation", [True, 1.0, -1])
def test_invalid_generation_types_fail_closed(
    service: writer.KanbanWriterService,
    bad_generation,
) -> None:
    generations = writer.kanban_safety.read_generations(service.db_path)
    client = writer.KanbanWriterClient(service.db_path)
    with pytest.raises(writer.WriterProtocolError, match="non-negative integers"):
        client.mutate(
            "create_task",
            {"title": "invalid generation"},
            request_id=f"bad-generation-{bad_generation!r}",
            expected_service_generation=bad_generation,
            expected_board_generation=generations.board_generation,
        )


def test_duplicate_json_keys_are_rejected(service: writer.KanbanWriterService) -> None:
    payload = b'{"version":1,"version":1}'
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
        peer.settimeout(1)
        peer.connect(str(service.socket_path))
        peer.sendall(struct.pack("!I", len(payload)) + payload)
        response = writer._recv_message(peer)
    assert response["ok"] is False
    assert response["error_type"] == "WriterProtocolError"
    assert "duplicate" in response["message"]


def test_client_rejects_non_private_token(service: writer.KanbanWriterService) -> None:
    os.chmod(service.token_path, 0o644)
    try:
        with pytest.raises(writer.WriterUnavailableError, match="private regular file"):
            writer.KanbanWriterClient(service.db_path).mutate(
                "create_task", {"title": "denied"}
            )
    finally:
        os.chmod(service.token_path, 0o600)


def test_client_rejects_token_symlink(service: writer.KanbanWriterService) -> None:
    target = service.token_path.with_suffix(".target")
    target.write_text(service._token, encoding="ascii")
    service.token_path.unlink()
    service.token_path.symlink_to(target)
    with pytest.raises(writer.WriterUnavailableError, match="private regular file"):
        writer.KanbanWriterClient(service.db_path).mutate(
            "create_task", {"title": "denied"}
        )


def test_service_refuses_to_unlink_non_socket_endpoint(tmp_path: Path) -> None:
    db_path = tmp_path / "kanban.db"
    with writer.privileged_maintenance(db_path):
        kb.init_db(db_path)
    endpoint = writer.writer_socket_path(db_path)
    endpoint.write_text("do not delete", encoding="utf-8")
    instance = writer.KanbanWriterService(db_path)
    with pytest.raises(writer.WriterUnavailableError, match="non-socket"):
        instance.start()
    assert endpoint.read_text(encoding="utf-8") == "do not delete"


def test_runtime_has_no_kanban_dml_outside_canonical_writer_boundary() -> None:
    root = Path(__file__).parents[2]
    canonical = {
        Path("hermes_cli/kanban_db.py"),
        Path("hermes_cli/kanban_writer.py"),
    }
    tables = {
        "TASKS",
        "TASK_LINKS",
        "TASK_COMMENTS",
        "TASK_EVENTS",
        "TASK_RUNS",
        "TASK_ATTACHMENTS",
        "KANBAN_NOTIFY_SUBS",
        "KANBAN_WRITER_REQUESTS",
    }
    violations: list[str] = []
    for module in root.rglob("*.py"):
        relative = module.relative_to(root)
        if (
            relative in canonical
            or "tests" in relative.parts
            or ".venv" in relative.parts
        ):
            continue
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Attribute)
                or node.func.attr not in {"execute", "executemany", "executescript"}
                or not node.args
            ):
                continue
            fragments = [
                value.value
                for value in ast.walk(node.args[0])
                if isinstance(value, ast.Constant) and isinstance(value.value, str)
            ]
            normalized = " ".join(" ".join(fragments).upper().split())
            if (
                any(verb in normalized for verb in ("INSERT ", "UPDATE ", "DELETE ", "REPLACE "))
                and any(table in normalized for table in tables)
            ):
                violations.append(f"{relative}:{node.lineno}: {normalized[:120]}")
    assert violations == [], "Kanban DML bypasses canonical writer boundary:\n" + "\n".join(
        violations
    )

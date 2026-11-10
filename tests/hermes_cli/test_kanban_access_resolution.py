"""Contract tests for the guarded Kanban access-resolution producer."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from hermes_cli.kanban_parser import build_parser


DOMAIN = b"hermes-kanban-access-resolution/v1"
ORDERED_FIELDS = (
    "version", "requestId", "mode", "machine", "cardId", "expectedAssignee",
    "expectedStatus", "expectedBlockKind", "expectedBlockEventId", "expectedBlockedAt",
    "category", "owner", "evidenceRef", "nextAction", "author", "comment",
)


def _parse_access_resolution(*args: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    build_parser(parser.add_subparsers(dest="command"))
    return parser.parse_args(["kanban", "access-resolution", *args])


def _oracle_preimage(request: dict) -> bytes:
    tokens = []
    for field in ORDERED_FIELDS:
        value = request[field]
        if value is None:
            tokens.append(b"n")
        elif type(value) is int:
            payload = str(value).encode("ascii")
            tokens.append(b"i" + struct.pack(">I", len(payload)) + payload)
        else:
            payload = value.encode("utf-8")
            tokens.append(b"s" + struct.pack(">I", len(payload)) + payload)
    return DOMAIN + b"\x00" + b"".join(tokens)


def _valid_request(**overrides) -> dict:
    request = {
        "version": 1,
        "requestId": "123e4567-e89b-12d3-a456-426614174000",
        "payloadDigest": "0" * 64,
        "mode": "route",
        "machine": "macpro",
        "cardId": "t_1234abcd",
        "expectedAssignee": "worker_1",
        "expectedStatus": "blocked",
        "expectedBlockKind": "capability",
        "expectedBlockEventId": 9,
        "expectedBlockedAt": 1_799_999_999,
        "category": "missing_tool_or_read_api",
        "owner": "Owen",
        "evidenceRef": None,
        "nextAction": "Install the approved read-only tool.",
        "author": "seth",
        "comment": "Access is still unavailable.\nOwner: Owen",
    }
    request.update(overrides)
    try:
        request["payloadDigest"] = hashlib.sha256(_oracle_preimage(request)).hexdigest()
    except (AttributeError, TypeError):
        request["payloadDigest"] = "0" * 64
    return request


def test_public_parser_exposes_stdin_and_describe_modes() -> None:
    stdin_args = _parse_access_resolution("--json-stdin")
    describe_args = _parse_access_resolution("--describe")

    assert stdin_args.kanban_action == "access-resolution"
    assert stdin_args.json_stdin is True
    assert stdin_args.describe is False
    assert describe_args.describe is True
    assert describe_args.json_stdin is False


def test_valid_request_has_frozen_preimage_and_digest() -> None:
    from hermes_cli.kanban_access_resolution import canonical_preimage, parse_request_bytes

    raw_request = _valid_request(owner="José")
    raw = json.dumps(raw_request, ensure_ascii=False, separators=(",", ":")).encode()

    parsed = parse_request_bytes(raw)

    assert canonical_preimage(parsed) == _oracle_preimage(raw_request)
    assert parsed["payloadDigest"] == hashlib.sha256(_oracle_preimage(raw_request)).hexdigest()

    default_request = _valid_request()
    assert len(_oracle_preimage(default_request)) == 313
    assert default_request["payloadDigest"] == "3bd6d69ba12a1cb481fb528f363c4938c9058ab0021f7df5b18e14dd0d133d28"


@pytest.mark.parametrize(
    ("overrides", "error_code"),
    [
        ({"version": True}, "invalid_request"),
        ({"expectedBlockEventId": True}, "invalid_request"),
        ({"expectedBlockEventId": 0}, "invalid_request"),
        ({"expectedBlockedAt": 2**63}, "invalid_request"),
        ({"machine": "MacPro"}, "invalid_request"),
        ({"cardId": "t_123"}, "invalid_request"),
        ({"expectedAssignee": "bad.profile"}, "invalid_request"),
        ({"owner": "Jose\u0301"}, "invalid_request"),
        ({"owner": "bad\x00owner"}, "invalid_request"),
        ({"owner": "é" * 129}, "invalid_request"),
        ({"evidenceRef": "é" * 1025, "mode": "grant", "nextAction": None}, "invalid_request"),
        ({"nextAction": "é" * 1025}, "invalid_request"),
        ({"author": "agent"}, "invalid_request"),
        ({"comment": "é" * 4097}, "invalid_request"),
        ({"comment": "bad\rcomment"}, "invalid_request"),
        ({"comment": "bad\tcomment"}, "invalid_request"),
        ({"comment": "bad\x7fcomment"}, "invalid_request"),
        ({"mode": "route", "evidenceRef": "receipt"}, "invalid_request"),
        ({"mode": "grant", "evidenceRef": None, "nextAction": None}, "invalid_request"),
    ],
)
def test_field_domains_fail_closed(overrides: dict, error_code: str) -> None:
    from hermes_cli.kanban_access_resolution import AccessResolutionError, parse_request_bytes

    request = _valid_request(**overrides)
    raw = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()

    with pytest.raises(AccessResolutionError) as exc:
        parse_request_bytes(raw)
    assert exc.value.code == error_code


def test_rejects_digest_mismatch_duplicate_keys_and_input_over_envelope() -> None:
    from hermes_cli.kanban_access_resolution import AccessResolutionError, parse_request_bytes

    request = _valid_request()
    mismatched = dict(request, payloadDigest="f" * 64)
    with pytest.raises(AccessResolutionError, match="digest_mismatch"):
        parse_request_bytes(json.dumps(mismatched, separators=(",", ":")).encode())

    raw = json.dumps(request, separators=(",", ":"))
    duplicate = raw.replace('"version":1', '"version":1,"version":1', 1).encode()
    with pytest.raises(AccessResolutionError, match="invalid_request"):
        parse_request_bytes(duplicate)

    padded = raw.encode() + b" " * (65_537 - len(raw.encode()))
    with pytest.raises(AccessResolutionError, match="invalid_request"):
        parse_request_bytes(padded)


def test_accepts_exact_string_integer_and_envelope_boundaries() -> None:
    from hermes_cli.kanban_access_resolution import parse_request_bytes

    accepted = (
        _valid_request(owner="é" * 128),
        _valid_request(mode="grant", evidenceRef="é" * 1024, nextAction=None),
        _valid_request(nextAction="é" * 1024),
        _valid_request(comment="é" * 4096),
        _valid_request(comment="line one\nline two"),
        _valid_request(expectedBlockEventId=2**63 - 1, expectedBlockedAt=0),
        _valid_request(expectedBlockedAt=2**63 - 1),
    )
    for request in accepted:
        raw = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
        assert parse_request_bytes(raw) == request
        assert request["payloadDigest"] == hashlib.sha256(_oracle_preimage(request)).hexdigest()

    raw = json.dumps(_valid_request(), separators=(",", ":")).encode()
    exact_envelope = raw + b" " * (65_536 - len(raw))
    assert len(exact_envelope) == 65_536
    assert parse_request_bytes(exact_envelope)["requestId"] == _valid_request()["requestId"]


@pytest.mark.parametrize(
    "raw_transform",
    [
        lambda raw: b"\xef\xbb\xbf" + raw,
        lambda raw: raw + b" {}",
        lambda raw: raw.replace(b'"expectedBlockedAt":1799999999', b'"expectedBlockedAt":-1'),
        lambda raw: raw.replace(b'"expectedBlockedAt":1799999999', b'"expectedBlockedAt":01'),
        lambda raw: raw.replace(b'"expectedBlockedAt":1799999999', b'"expectedBlockedAt":NaN'),
        lambda raw: raw[:-1] + b',"extra":1}',
        lambda raw: raw.replace(b'"owner":"Owen",', b"", 1),
        lambda raw: raw.replace(b'"owner":"Owen"', b'"owner":"\\ud800"'),
        lambda raw: raw[:-1] + b"\xff}",
    ],
)
def test_rejects_noncanonical_json_and_utf8_forms(raw_transform) -> None:
    from hermes_cli.kanban_access_resolution import AccessResolutionError, parse_request_bytes

    raw = json.dumps(_valid_request(), separators=(",", ":")).encode()
    with pytest.raises(AccessResolutionError, match="invalid_request"):
        parse_request_bytes(raw_transform(raw))


@pytest.fixture
def board_home(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _blocked_request(conn, *, mode: str = "route", parent: str | None = None) -> tuple[str, dict]:
    from hermes_cli import kanban_db as kb

    task_id = kb.create_task(conn, title="guarded", assignee="worker_1", parents=(() if parent is None else (parent,)))
    if kb.claim_task(conn, task_id, claimer="test") is None:
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (task_id,))
    assert kb.block_task(conn, task_id, reason="tool unavailable", kind="capability")
    event = conn.execute(
        "SELECT id, created_at FROM task_events WHERE task_id = ? AND kind = 'blocked' ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    request = _valid_request(
        mode=mode,
        cardId=task_id,
        expectedBlockEventId=event["id"],
        expectedBlockedAt=event["created_at"],
        evidenceRef=("receipt://approved/42" if mode == "grant" else None),
        nextAction=(None if mode == "grant" else "Install the approved read-only tool."),
    )
    return task_id, request


def test_route_commits_one_attributed_comment_receipt_and_replays(board_home) -> None:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        task_id, request = _blocked_request(conn)
        receipt = kb.resolve_access_request(conn, request, machine="macpro")
        replay = kb.resolve_access_request(conn, request, machine="macpro")
        task = kb.get_task(conn, task_id)
        comments = kb.list_comments(conn, task_id)
        events = kb.list_events(conn, task_id)
        rows = conn.execute("SELECT COUNT(*) FROM kanban_access_resolutions").fetchone()[0]

    assert receipt == {**replay, "replayed": False}
    assert replay["replayed"] is True
    assert task.status == "blocked"
    assert len(comments) == 1 and comments[0].body == request["comment"]
    assert rows == 1
    commented = next(event for event in events if event.id == receipt["commentEventId"])
    assert commented.payload["request_id"] == request["requestId"]
    assert receipt["unblockEventId"] is None
    assert receipt["blockEventIdAfter"] == request["expectedBlockEventId"]


def test_request_id_conflict_and_authority_mismatch_have_no_effect(board_home) -> None:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_access_resolution import AccessResolutionError

    with kbc.connect_closing() as conn:
        task_id, request = _blocked_request(conn)
        kb.resolve_access_request(conn, request, machine="macpro")
        changed = _valid_request(**{**request, "comment": "A different audit."})
        with pytest.raises(AccessResolutionError, match="request_id_conflict"):
            kb.resolve_access_request(conn, changed, machine="macpro")
        stale = _valid_request(**{**request, "requestId": "123e4567-e89b-12d3-a456-426614174001", "expectedBlockEventId": request["expectedBlockEventId"] + 1})
        with pytest.raises(AccessResolutionError, match="authority_mismatch"):
            kb.resolve_access_request(conn, stale, machine="macpro")
        assert len(kb.list_comments(conn, task_id)) == 1
        assert conn.execute("SELECT COUNT(*) FROM kanban_access_resolutions").fetchone()[0] == 1


def test_same_second_reblock_rejects_stale_event_generation(board_home) -> None:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_access_resolution import AccessResolutionError

    with kbc.connect_closing() as conn:
        task_id, stale = _blocked_request(conn)
        assert kb.unblock_task(conn, task_id)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'running', block_kind = NULL, block_recurrences = 0 WHERE id = ?",
                (task_id,),
            )
        assert kb.block_task(conn, task_id, reason="replacement generation", kind="capability")
        replacement = conn.execute(
            "SELECT id FROM task_events WHERE task_id = ? AND kind = 'blocked' ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert replacement is not None
        assert replacement["id"] != stale["expectedBlockEventId"]
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_events SET created_at = ? WHERE id = ?",
                (stale["expectedBlockedAt"], replacement["id"]),
            )

        with pytest.raises(AccessResolutionError, match="authority_mismatch"):
            kb.resolve_access_request(conn, stale, machine="macpro")
        assert kb.list_comments(conn, task_id) == []
        assert conn.execute("SELECT COUNT(*) FROM kanban_access_resolutions").fetchone()[0] == 0


def test_existing_database_additively_gains_receipt_table(board_home) -> None:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        with kb.write_txn(conn):
            conn.execute("DROP TABLE kanban_access_resolutions")

    kb.init_db()

    with kbc.connect_closing() as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'kanban_access_resolutions'"
        ).fetchone()
        assert row is not None


@pytest.mark.parametrize(("parent_state", "status_after"), [(None, "ready"), ("open", "todo"), ("done", "ready")])
def test_grant_uses_parent_aware_unblock_in_same_transaction(board_home, parent_state, status_after) -> None:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        parent = None
        if parent_state:
            parent = kb.create_task(conn, title="parent", assignee="parent-worker")
            if parent_state == "done":
                assert kb.complete_task(conn, parent, summary="done", force=True)
        task_id, request = _blocked_request(conn, mode="grant", parent=parent)
        receipt = kb.resolve_access_request(conn, request, machine="macpro")
        task = kb.get_task(conn, task_id)
        unblock = next(event for event in kb.list_events(conn, task_id) if event.id == receipt["unblockEventId"])

    assert task.status == status_after
    assert receipt["statusAfter"] == status_after
    assert receipt["blockEventIdAfter"] is None
    assert unblock.payload["expected_block_event_id"] == request["expectedBlockEventId"]


def test_grant_restores_review_resume_status(board_home) -> None:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="review blocker", assignee="worker_1")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
        assert kb.claim_review_task(conn, task_id, claimer="reviewer") is not None
        assert kb.block_task(conn, task_id, reason="tool unavailable", kind="capability")
        event = conn.execute(
            "SELECT id, created_at FROM task_events WHERE task_id = ? AND kind = 'blocked' ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert event is not None
        request = _valid_request(
            mode="grant",
            cardId=task_id,
            expectedBlockEventId=event["id"],
            expectedBlockedAt=event["created_at"],
            evidenceRef="receipt://approved/42",
            nextAction=None,
        )

        receipt = kb.resolve_access_request(conn, request, machine="macpro")
        task = kb.get_task(conn, task_id)

    assert task is not None and task.status == "review"
    assert receipt["statusAfter"] == "review"


def test_grant_exception_after_status_update_rolls_back_every_effect(board_home, monkeypatch) -> None:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        task_id, request = _blocked_request(conn, mode="grant")
        original = kb._unblock_task_in_txn

        def fail_after_update(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected")

        monkeypatch.setattr(kb, "_unblock_task_in_txn", fail_after_update)
        with pytest.raises(RuntimeError, match="injected"):
            kb.resolve_access_request(conn, request, machine="macpro")

        assert kb.get_task(conn, task_id).status == "blocked"
        assert kb.list_comments(conn, task_id) == []
        assert conn.execute("SELECT COUNT(*) FROM kanban_access_resolutions").fetchone()[0] == 0


def test_describe_uses_real_profile_config_before_board_initialization(tmp_path, monkeypatch, capsys) -> None:
    from hermes_cli import kanban as kc
    from hermes_cli import kanban_db as kb

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("kanban:\n  machine_identity: macpro\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MACHINE_IDENTITY", "imac")
    monkeypatch.setenv("HOSTNAME", "imac")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    rc = kc.kanban_command(_parse_access_resolution("--describe"))
    output = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert output == {
        "version": 1,
        "command": "access-resolution",
        "featureRevision": "v1",
        "machine": "macpro",
        "digestDomain": "hermes-kanban-access-resolution/v1",
        "maxInputBytes": 65_536,
    }
    assert not kb.kanban_db_path().exists()


def test_mutation_uses_configured_identity_and_public_facade(board_home, monkeypatch, capsys) -> None:
    from hermes_cli import kanban as kc
    from hermes_cli import kanban_db_connect as kbc

    (board_home / "config.yaml").write_text("kanban:\n  machine_identity: macpro\n", encoding="utf-8")
    with kbc.connect_closing() as conn:
        task_id, request = _blocked_request(conn)
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(json.dumps(request).encode()), encoding="utf-8"))

    rc = kc.kanban_command(_parse_access_resolution("--json-stdin"))
    receipt = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert receipt["cardId"] == task_id
    assert receipt["machine"] == "macpro"


def test_public_command_emits_fixed_machine_and_internal_failures(board_home, monkeypatch, capsys) -> None:
    from hermes_cli import kanban as kc
    from hermes_cli import kanban_db as kb

    (board_home / "config.yaml").write_text("kanban:\n  machine_identity: macpro\n", encoding="utf-8")
    wrong_machine = _valid_request(machine="imac")
    monkeypatch.setattr(
        "sys.stdin",
        io.TextIOWrapper(io.BytesIO(json.dumps(wrong_machine).encode()), encoding="utf-8"),
    )
    assert kc.kanban_command(_parse_access_resolution("--json-stdin")) == 1
    captured = capsys.readouterr()
    assert captured.out == '{"error":"machine_mismatch"}\n'
    assert captured.err == ""

    request = _valid_request()
    monkeypatch.setattr(
        "sys.stdin",
        io.TextIOWrapper(io.BytesIO(json.dumps(request).encode()), encoding="utf-8"),
    )
    monkeypatch.setattr(kb, "resolve_access_request", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("secret detail")))
    assert kc.kanban_command(_parse_access_resolution("--json-stdin")) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "kanban access-resolution: internal_error\n"


def test_missing_machine_identity_fails_closed_without_initializing_board(tmp_path, monkeypatch, capsys) -> None:
    from hermes_cli import kanban as kc
    from hermes_cli import kanban_db as kb

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    rc = kc.kanban_command(_parse_access_resolution("--describe"))

    assert rc == 1
    assert json.loads(capsys.readouterr().out) == {"error": "invalid_request"}
    assert not kb.kanban_db_path().exists()


def test_delegated_child_fence_denies_mutation_but_allows_describe(monkeypatch) -> None:
    from agent import delegation_context
    from hermes_cli import kanban as kc

    monkeypatch.setattr(delegation_context, "kanban_path_is_fenced", lambda _path: True)

    assert kc._is_delegated_child_cli_mutation(_parse_access_resolution("--json-stdin")) is True
    assert kc._is_delegated_child_cli_mutation(_parse_access_resolution("--describe")) is False


def _resolve_in_fresh_connection(request: dict) -> tuple[str, Any]:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_access_resolution import AccessResolutionError

    try:
        with kbc.connect_closing() as conn:
            return "ok", kb.resolve_access_request(conn, request, machine="macpro")
    except AccessResolutionError as exc:
        return "error", exc.code


def test_concurrent_same_request_commits_once_and_replays_once(board_home) -> None:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        task_id, request = _blocked_request(conn)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(_resolve_in_fresh_connection, (request, request)))

    assert [kind for kind, _ in results] == ["ok", "ok"]
    receipts = [value for _, value in results]
    assert sorted(receipt["replayed"] for receipt in receipts) == [False, True]
    assert receipts[0]["receiptId"] == receipts[1]["receiptId"]
    with kbc.connect_closing() as conn:
        assert len(kb.list_comments(conn, task_id)) == 1
        assert conn.execute("SELECT COUNT(*) FROM kanban_access_resolutions").fetchone()[0] == 1


def test_concurrent_grants_for_same_block_generation_have_one_winner(board_home) -> None:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        task_id, first = _blocked_request(conn, mode="grant")
    second = _valid_request(**{
        **first,
        "requestId": "123e4567-e89b-12d3-a456-426614174001",
        "comment": "Independent grant request for the same generation.",
    })

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(_resolve_in_fresh_connection, (first, second)))

    assert sorted(kind for kind, _ in results) == ["error", "ok"]
    assert [value for kind, value in results if kind == "error"] == ["authority_mismatch"]
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "ready"
        assert len(kb.list_comments(conn, task_id)) == 1
        assert conn.execute("SELECT COUNT(*) FROM kanban_access_resolutions").fetchone()[0] == 1

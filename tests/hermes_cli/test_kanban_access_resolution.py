"""Contract tests for the guarded Kanban access-resolution producer."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing
import os
import struct
import subprocess
import sys
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


def _resolve_process(home: str, request: dict, start, results) -> None:
    os.environ["HERMES_HOME"] = home
    start.wait()
    results.put(("started", request["requestId"]))
    results.put(_resolve_in_fresh_connection(request))


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


def test_published_versioned_fixture_corpus_is_immutable_and_independent() -> None:
    from hermes_cli.kanban_access_resolution import canonical_preimage, parse_request_bytes

    fixture_path = Path(__file__).parents[1] / "fixtures" / "kanban_access_resolution_v1.json"
    fixture_bytes = fixture_path.read_bytes()
    assert hashlib.sha256(fixture_bytes).hexdigest() == "359861befec02a3304cb82c5e5c07f85d579b1ab5560323cca5ffef60d595f93"
    fixture = json.loads(fixture_bytes)
    assert fixture["fixtureVersion"] == fixture["protocolVersion"] == 1
    assert fixture["digestDomain"] == DOMAIN.decode()
    assert fixture["orderedFields"] == list(ORDERED_FIELDS)
    for vector in fixture["vectors"]:
        request = vector["request"]
        independent = _oracle_preimage(request)
        assert independent.hex() == vector["preimageHex"]
        assert hashlib.sha256(independent).hexdigest() == vector["sha256"]
        parsed = parse_request_bytes(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode())
        assert canonical_preimage(parsed) == independent


@pytest.mark.parametrize(
    ("overrides", "error_code"),
    [
        ({"version": True}, "invalid_request"),
        ({"expectedBlockEventId": True}, "invalid_request"),
        ({"expectedBlockEventId": 0}, "invalid_request"),
        ({"expectedBlockedAt": 2**63}, "invalid_request"),
        ({"expectedBlockedAt": -1}, "invalid_request"),
        ({"requestId": "123E4567-E89B-12D3-A456-426614174000"}, "invalid_request"),
        ({"requestId": "123e4567e89b12d3a456426614174000"}, "invalid_request"),

        ({"mode": "audit"}, "invalid_request"),
        ({"mode": ["route"]}, "invalid_request"),
        ({"mode": {}}, "invalid_request"),
        ({"category": ["human_receipt"]}, "invalid_request"),
        ({"category": {}}, "invalid_request"),
        ({"machine": "MacPro"}, "invalid_request"),
        ({"machine": "1macpro"}, "invalid_request"),
        ({"machine": "m" * 33}, "invalid_request"),
        ({"cardId": "t_123"}, "invalid_request"),
        ({"cardId": "t_1234ABCD"}, "invalid_request"),
        ({"expectedAssignee": "bad.profile"}, "invalid_request"),
        ({"expectedAssignee": ""}, "invalid_request"),
        ({"expectedAssignee": "a" * 65}, "invalid_request"),
        ({"expectedStatus": "ready"}, "invalid_request"),
        ({"expectedBlockKind": "needs_input"}, "invalid_request"),
        ({"category": "other"}, "invalid_request"),
        ({"owner": ""}, "invalid_request"),
        ({"owner": "Jose\u0301"}, "invalid_request"),
        ({"owner": "bad\x00owner"}, "invalid_request"),
        ({"owner": "é" * 129}, "invalid_request"),
        ({"evidenceRef": "é" * 1025, "mode": "grant", "nextAction": None}, "invalid_request"),
        ({"evidenceRef": "", "mode": "grant", "nextAction": None}, "invalid_request"),
        ({"nextAction": "é" * 1025}, "invalid_request"),
        ({"nextAction": ""}, "invalid_request"),
        ({"author": "agent"}, "invalid_request"),
        ({"comment": "é" * 4097}, "invalid_request"),
        ({"comment": ""}, "invalid_request"),
        ({"comment": "bad\rcomment"}, "invalid_request"),
        ({"comment": "bad\tcomment"}, "invalid_request"),
        ({"comment": "bad\x7fcomment"}, "invalid_request"),
        ({"comment": "bidi \u202eoverride"}, "invalid_request"),
        ({"comment": "zero\u200bwidth"}, "invalid_request"),
        ({"comment": "line\u2028separator"}, "invalid_request"),
        ({"owner": "para\u2029separator"}, "invalid_request"),
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
    for invalid_digest in ("A" * 64, "a" * 63):
        malformed = dict(request, payloadDigest=invalid_digest)
        with pytest.raises(AccessResolutionError, match="invalid_request"):
            parse_request_bytes(json.dumps(malformed, separators=(",", ":")).encode())
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
        _valid_request(machine="a", expectedAssignee="a"),
        _valid_request(machine="m" * 32, expectedAssignee="a" * 64),
        _valid_request(category="credential_or_permission"),
        _valid_request(category="human_receipt"),
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


def test_breaker_block_after_unblock_rejects_stale_capability_generation(board_home) -> None:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_access_resolution import AccessResolutionError
    from hermes_cli.kanban_db_dispatch import _record_task_failure

    with kbc.connect_closing() as conn:
        task_id, stale = _blocked_request(conn, mode="grant")
        assert kb.unblock_task(conn, task_id)
        # The breaker parks the card with no new ``blocked`` event and keeps ``block_kind``.
        assert _record_task_failure(conn, task_id, "spawn failed", outcome="spawn_failed", force_trip=True)
        task = kb.get_task(conn, task_id)
        assert task.status == "blocked" and task.block_kind == "capability"

        with pytest.raises(AccessResolutionError, match="authority_mismatch"):
            kb.resolve_access_request(conn, stale, machine="macpro")
        assert kb.get_task(conn, task_id).status == "blocked"
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


def test_exception_immediately_after_comment_insert_rolls_back_every_effect(board_home, monkeypatch) -> None:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        task_id, request = _blocked_request(conn)
        original = kb._append_event

        def fail_on_comment_event(conn, task_id, kind, payload=None, *, run_id=None):
            if kind == "commented":
                raise RuntimeError("injected after comment insert")
            return original(conn, task_id, kind, payload, run_id=run_id)

        monkeypatch.setattr(kb, "_append_event", fail_on_comment_event)
        with pytest.raises(RuntimeError, match="after comment insert"):
            kb.resolve_access_request(conn, request, machine="macpro")

        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "blocked"
        assert kb.list_comments(conn, task_id) == []
        assert conn.execute("SELECT COUNT(*) FROM kanban_access_resolutions").fetchone()[0] == 0
        assert not any(event.kind == "commented" for event in kb.list_events(conn, task_id))


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


def test_machine_identity_is_selected_only_from_each_active_profile_home(tmp_path, monkeypatch, capsys) -> None:
    from hermes_cli import kanban as kc

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "config.yaml").write_text("kanban:\n  machine_identity: macpro\n", encoding="utf-8")
    (second / "config.yaml").write_text("kanban:\n  machine_identity: imac\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MACHINE_IDENTITY", "hostile-env")
    monkeypatch.setenv("HOSTNAME", "hostile-hostname")

    monkeypatch.setenv("HERMES_HOME", str(first))
    assert kc.kanban_command(_parse_access_resolution("--describe")) == 0
    assert json.loads(capsys.readouterr().out)["machine"] == "macpro"

    monkeypatch.setenv("HERMES_HOME", str(second))
    assert kc.kanban_command(_parse_access_resolution("--describe")) == 0
    assert json.loads(capsys.readouterr().out)["machine"] == "imac"


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


def _run_process_race(home: Path, requests: tuple[dict, dict]) -> list[tuple[str, Any]]:
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    results = ctx.Queue()
    processes = [ctx.Process(target=_resolve_process, args=(str(home), request, start, results)) for request in requests]
    for process in processes:
        process.start()
    start.set()
    messages = [results.get(timeout=30) for _ in range(4)]
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    return [message for message in messages if message[0] != "started"]


def test_real_process_races_identical_and_changed_digest_requests(board_home) -> None:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        first_task, identical = _blocked_request(conn)
    identical_results = _run_process_race(board_home, (identical, identical))
    assert sorted((kind, value["replayed"] if kind == "ok" else value) for kind, value in identical_results) == [
        ("ok", False),
        ("ok", True),
    ]
    with kbc.connect_closing() as conn:
        assert len(kb.list_comments(conn, first_task)) == 1
        second_task, winner = _blocked_request(conn)
    winner = _valid_request(**{**winner, "requestId": "123e4567-e89b-12d3-a456-426614174001"})
    conflict = _valid_request(**{**winner, "comment": "Changed digest races the same request ID."})
    changed_results = _run_process_race(board_home, (winner, conflict))
    assert sorted(kind for kind, _ in changed_results) == ["error", "ok"]
    assert [value for kind, value in changed_results if kind == "error"] == ["request_id_conflict"]
    with kbc.connect_closing() as conn:
        assert len(kb.list_comments(conn, second_task)) == 1
        assert conn.execute("SELECT COUNT(*) FROM kanban_access_resolutions").fetchone()[0] == 2


def test_waiting_process_cannot_apply_stale_g1_to_same_second_g2(board_home) -> None:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    results = ctx.Queue()
    with kbc.connect_closing() as conn:
        task_id, stale = _blocked_request(conn)
        with kb.write_txn(conn):
            g2_event = kb._append_event(
                conn,
                task_id,
                "blocked",
                {"reason": "same-second successor", "kind": "capability", "source_status": "running"},
            )
            conn.execute(
                "UPDATE task_events SET created_at = ? WHERE id = ?",
                (stale["expectedBlockedAt"], g2_event),
            )
            process = ctx.Process(target=_resolve_process, args=(str(board_home), stale, start, results))
            process.start()
            start.set()
            assert results.get(timeout=30) == ("started", stale["requestId"])

        result = results.get(timeout=30)
        process.join(timeout=30)
        assert process.exitcode == 0
        assert result == ("error", "authority_mismatch")
        latest = conn.execute(
            "SELECT id, created_at FROM task_events WHERE task_id = ? AND kind = 'blocked' ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert latest is not None and latest["id"] == g2_event
        assert latest["created_at"] == stale["expectedBlockedAt"]
        assert kb.list_comments(conn, task_id) == []
        assert conn.execute("SELECT COUNT(*) FROM kanban_access_resolutions").fetchone()[0] == 0


def test_committed_g1_receipt_does_not_mutate_later_g2(board_home) -> None:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        task_id, request = _blocked_request(conn, mode="grant")
    results = _run_process_race(board_home, (request, request))
    receipt = next(value for kind, value in results if kind == "ok" and value["replayed"] is False)
    with kbc.connect_closing() as conn:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'running', block_kind = NULL, block_recurrences = 0 WHERE id = ?",
                (task_id,),
            )
        assert kb.block_task(conn, task_id, reason="successor", kind="capability")
        g2 = conn.execute(
            "SELECT id FROM task_events WHERE task_id = ? AND kind = 'blocked' ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert g2 is not None and g2["id"] != request["expectedBlockEventId"]
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "blocked"
        assert len(kb.list_comments(conn, task_id)) == 1
        assert receipt["unblockEventId"] < g2["id"]


def _cli_argv() -> list[str]:
    return [sys.executable, "-m", "hermes_cli.main", "kanban", "access-resolution", "--json-stdin"]


def test_lost_stdout_recovers_receipt_and_fixed_argv_relay_preserves_stdin(board_home) -> None:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    (board_home / "config.yaml").write_text("kanban:\n  machine_identity: macpro\n", encoding="utf-8")
    env = {**os.environ, "HERMES_HOME": str(board_home)}
    with kbc.connect_closing() as conn:
        task_id, request = _blocked_request(conn)
    raw = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()

    dropped = subprocess.run(_cli_argv(), input=raw, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env, check=False)
    assert dropped.returncode == 0 and dropped.stderr == b""
    replay = subprocess.run(_cli_argv(), input=raw, capture_output=True, env=env, check=False)
    assert replay.returncode == 0 and replay.stderr == b""
    assert json.loads(replay.stdout)["replayed"] is True
    with kbc.connect_closing() as conn:
        assert len(kb.list_comments(conn, task_id)) == 1
        relay_task, relay_request = _blocked_request(conn)
    relay_request = _valid_request(**{**relay_request, "requestId": "123e4567-e89b-12d3-a456-426614174001"})
    relay_raw = json.dumps(relay_request, ensure_ascii=False, separators=(",", ":")).encode()
    relay_script = (
        "import subprocess,sys;"
        "p=subprocess.run([sys.executable,'-m','hermes_cli.main','kanban','access-resolution','--json-stdin'],"
        "input=sys.stdin.buffer.read());raise SystemExit(p.returncode)"
    )
    exact = subprocess.run([sys.executable, "-c", relay_script], input=relay_raw, capture_output=True, env=env, check=False)
    assert exact.returncode == 0 and json.loads(exact.stdout)["cardId"] == relay_task
    changed = relay_raw.replace(b"Access is still unavailable", b"Access was altered")
    denied = subprocess.run([sys.executable, "-c", relay_script], input=changed, capture_output=True, env=env, check=False)
    assert denied.returncode == 1
    assert json.loads(denied.stdout) == {"error": "digest_mismatch"}

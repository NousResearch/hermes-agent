from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.web_gemini import (
    ActionBinding,
    ActionLedger,
    ProtocolError,
    build_prompt,
    parse_response,
)


def _binding(*, action_index: int = 0) -> ActionBinding:
    return ActionBinding(
        task_id="t_canary",
        board_id="g57-qwen36-routing-canary",
        conversation_id="19abdcf90040a07e",
        conversation_url="https://gemini.google.com/app/19abdcf90040a07e",
        prompt_hash="a" * 64,
        nonce="WGF-20260818-CANARY-1",
        action_index=action_index,
    )


def _response(binding: ActionBinding, *, actions=None, **overrides) -> str:
    body = {
        "protocol": "web_gemini.v1",
        "board_id": binding.board_id,
        "task_id": binding.task_id,
        "conversation_id": binding.conversation_id,
        "conversation_url": binding.conversation_url,
        "prompt_hash": binding.prompt_hash,
        "nonce": binding.nonce,
        "actions": actions if actions is not None else [
            {"index": binding.action_index, "kind": "comment", "body": "CANARY_OK"}
        ],
    }
    body.update(overrides)
    return json.dumps(body, separators=(",", ":"))


def test_prompt_hash_is_deterministic_and_contains_frozen_binding():
    binding = _binding()
    prompt, prompt_hash = build_prompt(
        board_id=binding.board_id,
        task_id=binding.task_id,
        conversation_id=binding.conversation_id,
        nonce=binding.nonce,
        task_title="comment-only canary",
        task_body="Return one local comment action only.",
    )

    assert "web_gemini.v1" in prompt
    assert binding.board_id in prompt
    assert binding.task_id in prompt
    assert binding.nonce in prompt
    assert len(prompt_hash) == 64
    assert prompt_hash == build_prompt(
        board_id=binding.board_id,
        task_id=binding.task_id,
        conversation_id=binding.conversation_id,
        nonce=binding.nonce,
        task_title="comment-only canary",
        task_body="Return one local comment action only.",
    )[1]


def test_parse_response_admits_only_exactly_bound_comment_action():
    binding = _binding()

    parsed = parse_response(_response(binding), expected=binding)

    assert parsed.actions == ({"index": 0, "kind": "comment", "body": "CANARY_OK"},)

    with pytest.raises(ProtocolError, match="board_id"):
        parse_response(_response(binding, board_id="other-board"), expected=binding)

    with pytest.raises(ProtocolError, match="allowlisted"):
        parse_response(
            _response(
                binding,
                actions=[{"index": 0, "kind": "assign", "profile": "default"}],
            ),
            expected=binding,
        )

    with pytest.raises(ProtocolError, match="exact JSON"):
        parse_response(f"```json\n{_response(binding)}\n```", expected=binding)


def test_action_ledger_finalizes_once_and_replays_without_duplicate_side_effect(tmp_path: Path):
    ledger = ActionLedger(tmp_path / "ledger.sqlite3")
    binding = _binding()
    calls: list[str] = []

    first = ledger.apply_once(
        binding,
        lambda: calls.append("comment") or {"comment_id": 11},
    )
    second = ledger.apply_once(
        binding,
        lambda: calls.append("duplicate") or {"comment_id": 12},
    )

    assert first.status == "finalized"
    assert second.status == "finalized"
    assert second.replayed is True
    assert first.receipt == {"comment_id": 11}
    assert second.receipt == {"comment_id": 11}
    assert calls == ["comment"]


def test_action_ledger_never_reapplies_after_action_applied_but_receipt_not_finalized(
    tmp_path: Path,
):
    ledger = ActionLedger(tmp_path / "ledger.sqlite3")
    binding = _binding()
    ledger.record_intent(binding)
    ledger.record_applied(binding, {"comment_id": 41})

    with pytest.raises(ProtocolError, match="finalization pending"):
        ledger.apply_once(binding, lambda: pytest.fail("must not duplicate action"))


def test_comment_action_is_durable_against_real_kanban_board(tmp_path: Path):
    db_path = tmp_path / "kanban.db"
    conn = kb.connect(db_path=db_path)
    try:
        task_id = kb.create_task(
            conn,
            title="comment-only canary",
            body="No model/provider fallback.",
            initial_status="blocked",
            block_kind="capability",
            block_reason="test fixture",
        )
        binding = ActionBinding(
            task_id=task_id,
            board_id="fixture-board",
            conversation_id="fixture-conversation",
            conversation_url="https://gemini.google.com/app/fixture-conversation",
            prompt_hash="b" * 64,
            nonce="FIXTURE-NONCE",
            action_index=0,
        )
        ledger = ActionLedger(tmp_path / "ledger.sqlite3")

        result = ledger.apply_once(
            binding,
            lambda: {"comment_id": kb.add_comment(conn, task_id, "web_gemini", "CANARY_OK")},
        )
        comments = kb.list_comments(conn, task_id)

        assert result.status == "finalized"
        assert len(comments) == 1
        assert comments[0].body == "CANARY_OK"
    finally:
        conn.close()

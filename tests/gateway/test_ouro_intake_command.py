"""Tests for the admission-only /ouro-intake command surface."""

import json
import re
from types import SimpleNamespace

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli.commands import (
    ACTIVE_SESSION_BYPASS_COMMANDS,
    GATEWAY_KNOWN_COMMANDS,
    is_gateway_known_command,
    resolve_command,
    should_bypass_active_session,
)


def _seed_from_body(body: str) -> dict:
    match = re.search(r"```json seed_contract\n(.*?)\n```", body, re.S)
    assert match, body
    return json.loads(match.group(1))


@pytest.fixture()
def hermes_home(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        yield tmp_path
    finally:
        reset_hermes_home_override(token)


def test_ouro_intake_is_registered_as_gateway_known_command():
    cmd = resolve_command("ouro-intake")

    assert cmd is not None
    assert cmd.name == "ouro-intake"
    assert "ouro-intake" in GATEWAY_KNOWN_COMMANDS
    assert is_gateway_known_command("ouro-intake") is True
    assert should_bypass_active_session("ouro-intake") is True
    assert "ouro-intake" in ACTIVE_SESSION_BYPASS_COMMANDS
    assert resolve_command("ouro_intake").name == "ouro-intake"


def test_help_and_missing_goal_do_not_create_kanban_card(hermes_home):
    from gateway.ouro_intake import handle_ouro_intake_command

    help_result = handle_ouro_intake_command("")
    explicit_help = handle_ouro_intake_command("help")
    missing = handle_ouro_intake_command("project:bo")

    assert help_result.action == "help"
    assert explicit_help.action == "help"
    assert missing.action == "error"
    assert help_result.mutated is False
    assert missing.mutated is False
    assert help_result.dispatched is False
    assert "Interview -> Seed" in help_result.message
    assert not (hermes_home / "kanban.db").exists()


def test_start_runs_interview_before_kanban_admission(hermes_home):
    from gateway.ouro_intake import handle_ouro_intake_command

    result = handle_ouro_intake_command(
        'goal:"Improve billing" project:dc tenant:billing',
        actor="tester",
    )

    assert result.action == "interview_started"
    assert result.session_id
    assert result.public_id is None
    assert result.task_id is None
    assert result.dispatched is False
    assert "질문:" in result.message
    assert "그냥 평문으로 보내면 됩니다" in result.message
    assert "Ambiguity:" not in result.message
    assert "Ledger:" not in result.message
    assert "Socratic blockers" not in result.message
    assert "Kanban 카드" in result.message
    assert (hermes_home / "ouro_intake_sessions.json").exists()
    assert not (hermes_home / "kanban.db").exists()

    sessions = json.loads((hermes_home / "ouro_intake_sessions.json").read_text())
    session = sessions[result.session_id]
    assert session["status"] == "interviewing"
    assert session["seed"] is None
    review = session["values"]
    assert review["goal"] == "Improve billing"


def test_answer_can_make_seed_ready_without_admitting(hermes_home):
    from gateway.ouro_intake import handle_ouro_intake_command

    started = handle_ouro_intake_command(
        'goal:"Build Discord intake report command" project:bo tenant:kanban context:"Hermes gateway command"',
        actor="tester",
    )
    assert started.session_id

    updated = handle_ouro_intake_command(
        f'answer session:{started.session_id} answer:"pytest test_ouro_intake_command.py passes with exit code 0; no repo mutation or gateway restart is allowed"',
        actor="tester",
    )

    assert updated.action == "refine_pending"
    assert "[from-user][refined]" in updated.message
    updated = handle_ouro_intake_command(
        f"answer session:{started.session_id} answer:승인",
        actor="tester",
    )

    assert updated.action == "interview_updated"
    assert updated.session_id == started.session_id
    assert updated.public_id is None
    assert "Restate:" in updated.message
    assert "Seed는 승인 전까지 막혀 있습니다" in updated.message
    assert not (hermes_home / "kanban.db").exists()

    sessions = json.loads((hermes_home / "ouro_intake_sessions.json").read_text())
    session = sessions[started.session_id]
    assert session["status"] == "restate_pending"
    assert session["seed"] is None
    assert session["turns"][-1]["question"]["id"]

    approved = handle_ouro_intake_command(
        f"answer session:{started.session_id} answer:승인",
        actor="tester",
    )
    assert "Seed draft + QA is ready" in approved.message
    sessions = json.loads((hermes_home / "ouro_intake_sessions.json").read_text())
    seed = sessions[started.session_id]["seed"]
    assert sessions[started.session_id]["status"] == "seed_ready"
    assert seed["seed_review"]["mode"] == "seed_ready_for_admission"
    assert seed["ambiguity_score"] <= seed["seed_review"]["ambiguity_threshold"]
    assert seed["seed_qa"]["passed"] is True
    assert seed["authority"]["seed_contract_is_source_material_only"] is True
    assert seed["side_effect_boundary"]["executor_dispatch"] == "forbidden_during_admission"


def test_admit_creates_blocked_seed_contract_without_dispatch(hermes_home):
    from gateway.ouro_intake import handle_ouro_intake_command
    from hermes_cli import kanban_db as kb

    started = handle_ouro_intake_command(
        'goal:"Design the Discord intake flow" project:bo tenant:kanban context:"Hermes gateway seed only" acceptance:"Kanban readback command returns task_runs equals 0" side-effects:"no worker dispatch, PR, gateway restart, or secret/env mutation"',
        actor="tester",
    )
    assert started.session_id

    blocked = handle_ouro_intake_command(f"admit session:{started.session_id}", actor="tester")
    assert blocked.action == "admission_blocked"
    assert not (hermes_home / "kanban.db").exists()

    approved = handle_ouro_intake_command(f"answer session:{started.session_id} answer:승인", actor="tester")
    assert approved.action == "interview_updated"
    result = handle_ouro_intake_command(f"admit session:{started.session_id}", actor="tester")

    assert result.action == "created"
    assert result.mutated is True
    assert result.dispatched is False
    assert result.public_id == "BO-001"
    assert result.task_id
    assert result.session_id == started.session_id
    assert "no worker dispatched" in result.message.lower()

    with kb.connect() as conn:
        task = kb.get_task(conn, result.task_id)
        assert task is not None
        assert task.public_id == "BO-001"
        assert task.status == "blocked"
        assert task.assignee is None
        assert task.claim_lock is None
        assert task.worker_pid is None
        assert task.tenant == "kanban"
        assert task.routing_verdict["status"] == "proposed_only"
        assert task.routing_verdict["verdict"] == "blocked"
        assert task.admission_snapshot is not None
        assert task.body is not None
        assert task.admission_snapshot["executor_dispatch"] == "forbidden_during_admission"
        assert task.admission_snapshot["seed_qa_passed"] is True
        assert task.closeout_evidence["policy"] == "admission_only_no_execution"
        seed = _seed_from_body(task.body)
        assert seed["authority"]["seed_contract_is_source_material_only"] is True
        assert seed["side_effect_boundary"]["executor_dispatch"] == "forbidden_during_admission"
        assert seed["initial_routing"]["status"] == "proposed_only"
        assert seed["seed_review"]["dispatch_allowed"] is False
        assert "ontology" in seed
        assert "ambiguity_ledger" in seed
        assert "seed_qa" in seed
        runs = conn.execute(
            "SELECT COUNT(*) AS n FROM task_runs WHERE task_id = ?",
            (result.task_id,),
        ).fetchone()["n"]
        assert runs == 0
        comments = kb.list_comments(conn, result.task_id)
        assert any("Admission-only block" in c.body for c in comments)

def test_sensitive_prod_billing_env_seed_stays_decision_gated_after_admit(hermes_home):
    from gateway.ouro_intake import handle_ouro_intake_command

    started = handle_ouro_intake_command(
        'goal:"Change production billing env for Paddle checkout" project:dc tenant:billing',
        actor="tester",
    )
    assert started.action == "interview_started"
    assert "아직 Kanban 카드나 worker는 만들지 않았습니다" in started.message
    assert "Ambiguity:" not in started.message

    result = handle_ouro_intake_command(f"admit session:{started.session_id}", actor="tester")

    assert result.action == "admission_blocked"
    assert result.mutated is False
    assert result.dispatched is False
    assert result.error == "restate_not_approved"
    assert not (hermes_home / "kanban.db").exists()

    sessions = json.loads((hermes_home / "ouro_intake_sessions.json").read_text())
    session = sessions[started.session_id]
    assert session["status"] == "interviewing"
    assert session["last_question"]["track"] in {"authority", "scope", "brownfield_context"}

def test_bare_korean_autopilot_starts_contextual_single_question(hermes_home):
    from gateway.ouro_intake import handle_ouro_intake_command

    result = handle_ouro_intake_command("오토파일럿 만들고싶어", actor="tester")

    assert result.action == "interview_started"
    assert result.session_id
    assert "오토파일럿" in result.message
    assert "A) intake/카드화" in result.message
    assert "B) Kanban 실행 준비" in result.message
    assert "Socratic blockers" not in result.message
    assert result.public_id is None
    assert result.task_id is None
    assert result.dispatched is False

    sessions = json.loads((hermes_home / "ouro_intake_sessions.json").read_text())
    session = sessions[result.session_id]
    assert session["last_question"]["id"] == "autopilot_axis"
    assert session["last_question"]["track"] == "scope"
    assert session["language"] == "ko"
    assert session["turns"] == []


def test_seed_command_refuses_before_restate_approval(hermes_home):
    from gateway.ouro_intake import handle_ouro_intake_command

    started = handle_ouro_intake_command(
        'goal:"Build Discord intake report command" project:bo tenant:kanban context:"Hermes gateway command"',
        actor="tester",
    )
    handle_ouro_intake_command(
        f'answer session:{started.session_id} answer:"pytest test_ouro_intake_command.py passes with exit code 0; no repo mutation or gateway restart is allowed"',
        actor="tester",
    )

    blocked = handle_ouro_intake_command(f"seed session:{started.session_id}", actor="tester")

    assert blocked.action == "seed_blocked"
    assert blocked.mutated is False
    assert "Restate gate has not been approved" in blocked.message
    assert not (hermes_home / "kanban.db").exists()


def test_plain_reply_routes_to_bound_active_interview_session(hermes_home):
    from gateway.ouro_intake import handle_ouro_intake_command, handle_ouro_intake_plain_reply

    origin = {
        "platform": "discord",
        "chat_id": "channel-1",
        "thread_id": "thread-1",
        "user_id": "user-1",
        "user_name": "tester",
        "chat_type": "thread",
    }
    started = handle_ouro_intake_command("오토파일럿 만들고싶어", actor="tester", origin=origin)
    assert started.action == "interview_started"

    updated = handle_ouro_intake_plain_reply("A와 B에 가까워", actor="tester", origin=origin)

    assert updated is not None
    assert updated.action == "refine_pending"
    assert "[from-user][refined]" in updated.message
    updated = handle_ouro_intake_plain_reply("승인", actor="tester", origin=origin)
    assert updated is not None
    assert updated.action == "interview_updated"
    assert updated.session_id == started.session_id
    assert "Updated /ouro-intake session" in updated.message

    sessions = json.loads((hermes_home / "ouro_intake_sessions.json").read_text())
    session = sessions[started.session_id]
    assert session["turns"][-1]["answer"] == "A와 B에 가까워"
    assert session["turns"][-1]["refined_answer"]["scope_axes"] == ["intake/cardization", "kanban_execution_prep"]
    assert session["last_question"]["id"] == "first_slice"
    assert "하나만 골라" not in updated.message
    assert "복수 선택" not in updated.message
    assert session["origin_binding"]["key"] == "discord|channel-1|thread-1|user-1"


def test_plain_escape_cancels_active_capture(hermes_home):
    from gateway.ouro_intake import handle_ouro_intake_command, handle_ouro_intake_plain_reply

    origin = {"platform": "discord", "chat_id": "c", "thread_id": "t", "user_id": "u1", "user_name": "tester"}
    started = handle_ouro_intake_command("오토파일럿 만들고싶어", actor="tester", origin=origin)

    cancelled = handle_ouro_intake_plain_reply("그만", actor="tester", origin=origin)

    assert cancelled is not None
    assert cancelled.action == "cancelled"
    assert "취소" in cancelled.message
    assert handle_ouro_intake_plain_reply("탈출 확인", actor="tester", origin=origin) is None
    sessions = json.loads((hermes_home / "ouro_intake_sessions.json").read_text())
    assert sessions[started.session_id]["status"] == "cancelled"
    assert sessions[started.session_id]["origin_binding"]["expires_at"] < sessions[started.session_id]["cancelled_at"]


def test_slash_cancel_does_not_start_new_session(hermes_home):
    from gateway.ouro_intake import handle_ouro_intake_command

    origin = {"platform": "discord", "chat_id": "c", "thread_id": "t", "user_id": "u1", "user_name": "tester"}
    started = handle_ouro_intake_command("오토파일럿 만들고싶어", actor="tester", origin=origin)

    cancelled = handle_ouro_intake_command("cancel", actor="tester", origin=origin)

    assert cancelled.action == "cancelled"
    sessions = json.loads((hermes_home / "ouro_intake_sessions.json").read_text())
    assert list(sessions) == [started.session_id]
    assert sessions[started.session_id]["status"] == "cancelled"


def test_plain_reply_does_not_capture_other_user_or_slash_command(hermes_home):
    from gateway.ouro_intake import handle_ouro_intake_command, handle_ouro_intake_plain_reply

    origin = {"platform": "discord", "chat_id": "c", "thread_id": "t", "user_id": "u1", "user_name": "tester"}
    other = {"platform": "discord", "chat_id": "c", "thread_id": "t", "user_id": "u2", "user_name": "other"}
    started = handle_ouro_intake_command("오토파일럿 만들고싶어", actor="tester", origin=origin)
    assert started.session_id

    assert handle_ouro_intake_plain_reply("A와 B", actor="other", origin=other) is None
    assert handle_ouro_intake_plain_reply("/help", actor="tester", origin=origin) is None


@pytest.mark.asyncio
async def test_gateway_routes_plain_reply_to_active_ouro_intake_session(hermes_home):
    from gateway.run import GatewayRunner
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import Platform, SessionSource
    from gateway.ouro_intake import handle_ouro_intake_command

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="channel-1",
        chat_type="thread",
        user_id="user-1",
        user_name="tester",
        thread_id="thread-1",
    )
    started = handle_ouro_intake_command("오토파일럿 만들고싶어", actor="tester", origin={
        "platform": "discord",
        "chat_id": "channel-1",
        "thread_id": "thread-1",
        "user_id": "user-1",
        "user_name": "tester",
        "chat_type": "thread",
    })
    assert started.session_id

    runner = object.__new__(GatewayRunner)
    event = MessageEvent(text="A와 B에 가까워", message_type=MessageType.TEXT, source=source, message_id="m2")

    result = await runner._maybe_handle_ouro_intake_plain_reply(event)

    assert result is not None
    assert "[from-user][refined]" in result
    event = MessageEvent(text="승인", message_type=MessageType.TEXT, source=source, message_id="m3")
    result = await runner._maybe_handle_ouro_intake_plain_reply(event)
    assert result is not None
    assert "Updated /ouro-intake session" in result
    sessions = json.loads((hermes_home / "ouro_intake_sessions.json").read_text())
    assert sessions[started.session_id]["turns"][-1]["answer"] == "A와 B에 가까워"


@pytest.mark.asyncio
async def test_gateway_handler_routes_raw_args_to_controller(monkeypatch):
    from gateway.run import GatewayRunner
    from gateway.platforms.base import MessageEvent
    import gateway.ouro_intake as ouro_intake

    calls = []

    def fake_handle(raw_args, *, actor=None, origin=None):
        calls.append((raw_args, actor, origin))
        return SimpleNamespace(message="handled by ouro controller")

    monkeypatch.setattr(ouro_intake, "handle_ouro_intake_command", fake_handle)
    runner = object.__new__(GatewayRunner)
    event = MessageEvent(
        text="/ouro-intake goal:test project:bo",
        source=SimpleNamespace(user_name="tester", user_id="u1", chat_id="c1"),
        message_id="m1",
    )

    result = await runner._handle_ouro_intake_command(event)

    assert result == "handled by ouro controller"
    assert calls == [("goal:test project:bo", "tester", {"platform": "", "chat_id": "c1", "thread_id": "", "user_id": "u1", "user_name": "tester", "chat_type": ""})]


def test_cli_handler_routes_raw_args_to_controller(monkeypatch):
    import cli as cli_module
    from cli import HermesCLI
    import gateway.ouro_intake as ouro_intake

    calls = []
    printed = []

    def fake_handle(raw_args, *, actor=None, origin=None):
        calls.append((raw_args, actor, origin))
        return SimpleNamespace(message="cli handled")

    monkeypatch.setattr(ouro_intake, "handle_ouro_intake_command", fake_handle)
    monkeypatch.setattr(cli_module, "_cprint", lambda message: printed.append(message))

    should_continue = object.__new__(HermesCLI).process_command("/ouro-intake goal:test project:bo")

    assert should_continue is True
    assert calls == [("goal:test project:bo", "local-cli", None)]
    assert printed == ["cli handled"]



def test_upstream_refine_gate_requires_confirmation_for_scope_decision(hermes_home):
    from gateway.ouro_intake import handle_ouro_intake_command, handle_ouro_intake_plain_reply

    origin = {"platform": "discord", "chat_id": "c", "thread_id": "t", "user_id": "u", "user_name": "tester"}
    started = handle_ouro_intake_command("오토파일럿 만들고싶어", actor="tester", origin=origin)

    refine = handle_ouro_intake_plain_reply("A와 B에 가까워", actor="tester", origin=origin)

    assert refine is not None
    assert refine.action == "refine_pending"
    assert "Decision:" in refine.message
    assert "[from-user][refined]" in refine.message
    assert "누락" in refine.message or "missing" in refine.message.lower()

    sessions = json.loads((hermes_home / "ouro_intake_sessions.json").read_text())
    session = sessions[started.session_id]
    assert session["status"] == "refine_pending"
    pending = session["pending_refinement"]
    assert pending["raw_answer"] == "A와 B에 가까워"
    assert pending["scope_axes"] == ["intake/cardization", "kanban_execution_prep"]
    assert session["turns"] == []

    advanced = handle_ouro_intake_plain_reply("승인", actor="tester", origin=origin)
    assert advanced is not None
    assert advanced.action == "interview_updated"
    sessions = json.loads((hermes_home / "ouro_intake_sessions.json").read_text())
    session = sessions[started.session_id]
    assert session["last_question"]["id"] == "first_slice"
    assert session["turns"][-1]["refined_answer"]["source_prefix"] == "[from-user][refined]"


def test_upstream_compound_scope_answer_is_refined_not_collapsed(hermes_home):
    from gateway.ouro_intake import handle_ouro_intake_command, handle_ouro_intake_plain_reply

    origin = {"platform": "discord", "chat_id": "c", "thread_id": "t", "user_id": "u", "user_name": "tester"}
    started = handle_ouro_intake_command("오토파일럿 만들고싶어", actor="tester", origin=origin)
    handle_ouro_intake_plain_reply("A와 B에 가까워", actor="tester", origin=origin)
    handle_ouro_intake_plain_reply("승인", actor="tester", origin=origin)

    sessions = json.loads((hermes_home / "ouro_intake_sessions.json").read_text())
    turn = sessions[started.session_id]["turns"][-1]
    refined = turn["refined_answer"]
    assert refined["raw_answer"] == "A와 B에 가까워"
    assert refined["decision"] == "A) intake/cardization + B) Kanban execution prep"
    assert refined["scope_axes"] == ["intake/cardization", "kanban_execution_prep"]
    assert refined["reasoning"]
    assert refined["constraints"]
    assert refined["out_of_scope"]
    assert refined["source_prefix"] == "[from-user][refined]"


def test_upstream_same_question_not_repeated_after_responsive_answer(hermes_home):
    from gateway.ouro_intake import handle_ouro_intake_command, handle_ouro_intake_plain_reply

    origin = {"platform": "discord", "chat_id": "c", "thread_id": "t", "user_id": "u", "user_name": "tester"}
    handle_ouro_intake_command("오토파일럿 만들고싶어", actor="tester", origin=origin)
    refine = handle_ouro_intake_plain_reply("A와 B에 가까워", actor="tester", origin=origin)
    assert refine is not None and "하나만" not in refine.message
    advanced = handle_ouro_intake_plain_reply("승인", actor="tester", origin=origin)

    assert advanced is not None
    assert "오토파일럿이라고 할 때" not in advanced.message
    assert "first" in advanced.message.lower() or "첫 버전" in advanced.message


def test_upstream_restate_correction_never_bypasses_refine(hermes_home):
    from gateway.ouro_intake import handle_ouro_intake_command

    started = handle_ouro_intake_command(
        'goal:"Build Discord intake report command" project:bo tenant:kanban context:"Hermes gateway command" acceptance:"pytest passes with exit code 0" side-effects:"no repo mutation or gateway restart"',
        actor="tester",
    )
    assert started.session_id
    assert "Restate:" in started.message or "Restate" in started.message

    corrected = handle_ouro_intake_command(
        f'answer session:{started.session_id} answer:"Exclude retry scheduling from the seed."',
        actor="tester",
    )

    assert corrected.action == "refine_pending"
    sessions = json.loads((hermes_home / "ouro_intake_sessions.json").read_text())
    session = sessions[started.session_id]
    assert session["status"] == "refine_pending"
    assert session["pending_refinement"]["restate_correction"] is True
    assert session["seed"] is None


def test_upstream_seed_closer_blocks_score_only_seed_ready(hermes_home):
    from gateway.ouro_intake import handle_ouro_intake_command

    result = handle_ouro_intake_command(
        'goal:"Implement gateway command lifecycle" project:bo tenant:kanban context:"brownfield Hermes gateway" acceptance:"pytest returns exit code 0" side-effects:"no repo mutation without approval"',
        actor="tester",
    )

    sessions = json.loads((hermes_home / "ouro_intake_sessions.json").read_text())
    session = sessions[result.session_id]
    assert session["status"] == "interviewing"
    assert session["seed_closer"]["ready"] is False
    assert any("ownership" in blocker or "SSOT" in blocker for blocker in session["seed_closer"]["blockers"])
    assert session["last_question"]["id"] in {"seed_closer_material_gap", "brownfield_context"}


def test_seed_projection_preserves_upstream_seed_fields(hermes_home):
    from gateway.ouro_intake import handle_ouro_intake_command

    started = handle_ouro_intake_command(
        'goal:"Build Discord intake report command" project:bo tenant:kanban context:"Hermes gateway command" acceptance:"pytest passes with exit code 0" side-effects:"no repo mutation or gateway restart"',
        actor="tester",
    )
    approved = handle_ouro_intake_command(f"answer session:{started.session_id} answer:승인", actor="tester")
    assert approved.action == "interview_updated"

    sessions = json.loads((hermes_home / "ouro_intake_sessions.json").read_text())
    seed = sessions[started.session_id]["seed"]
    upstream_seed = seed["upstream_seed"]
    assert set(upstream_seed) >= {
        "goal",
        "task_type",
        "brownfield_context",
        "constraints",
        "acceptance_criteria",
        "ontology_schema",
        "evaluation_principles",
        "exit_conditions",
        "metadata",
    }
    assert seed["authority"]["seed_contract_is_source_material_only"] is True
    assert seed["side_effect_boundary"]["executor_dispatch"] == "forbidden_during_admission"


def test_intentional_divergences_are_documented():
    from pathlib import Path

    matrix = Path(".hermes/parity/ouro-intake-upstream-parity.md").read_text(encoding="utf-8")
    assert "DIV-001" in matrix and "No live upstream MCP call" in matrix
    assert "DIV-002" in matrix and "Kanban admission source material" in matrix
    assert "unapproved divergence" in matrix.lower()


def test_cancel_escape_expires_plain_reply_capture(hermes_home):
    from gateway.ouro_intake import handle_ouro_intake_command, handle_ouro_intake_plain_reply

    origin = {"platform": "discord", "chat_id": "c", "thread_id": "t", "user_id": "u1", "user_name": "tester"}
    started = handle_ouro_intake_command("오토파일럿 만들고싶어", actor="tester", origin=origin)
    cancelled = handle_ouro_intake_plain_reply("탈출", actor="tester", origin=origin)

    assert cancelled is not None
    assert cancelled.action == "cancelled"
    assert handle_ouro_intake_plain_reply("탈출 확인", actor="tester", origin=origin) is None
    sessions = json.loads((hermes_home / "ouro_intake_sessions.json").read_text())
    assert sessions[started.session_id]["origin_binding"]["expires_at"] < sessions[started.session_id]["cancelled_at"]


def test_admission_never_dispatches_executor(hermes_home):
    from gateway.ouro_intake import handle_ouro_intake_command
    from hermes_cli import kanban_db as kb

    started = handle_ouro_intake_command(
        'goal:"Build Discord intake report command" project:bo tenant:kanban context:"Hermes gateway command" acceptance:"pytest passes with exit code 0" side-effects:"no repo mutation or gateway restart"',
        actor="tester",
    )
    handle_ouro_intake_command(f"answer session:{started.session_id} answer:승인", actor="tester")
    result = handle_ouro_intake_command(f"admit session:{started.session_id}", actor="tester")

    assert result.action == "created"
    with kb.connect() as conn:
        runs = conn.execute("SELECT COUNT(*) AS n FROM task_runs WHERE task_id = ?", (result.task_id,)).fetchone()["n"]
        task = kb.get_task(conn, result.task_id)
        assert runs == 0
        assert task.worker_pid is None
        assert task.claim_lock is None
        assert task.routing_verdict["status"] == "proposed_only"



def test_bo062_uses_vendored_upstream_interview_state_and_seed_model(hermes_home):
    from gateway.ouro_intake import handle_ouro_intake_command
    from hermes_integrations.ouroboros_upstream.bigbang.interview import InterviewState
    from hermes_integrations.ouroboros_upstream.core.seed import Seed

    started = handle_ouro_intake_command(
        'goal:"Design gateway intake wrapper" project:bo tenant:kanban context:"Hermes gateway existing runtime" acceptance:"pytest passes"',
        actor="tester",
    )
    sessions = json.loads((hermes_home / "ouro_intake_sessions.json").read_text())
    state_payload = sessions[started.session_id]["upstream_interview_state"]
    state = InterviewState.model_validate(state_payload)
    assert state.interview_id == started.session_id
    assert state.rounds
    assert sessions[started.session_id]["upstream_interview_provider"] == "vendored_q00_ouroboros_subset"

    answer = handle_ouro_intake_command(
        f'answer session:{started.session_id} answer:"pytest test passes; no execution runner or gateway restart"',
        actor="tester",
    )
    if answer.action == "refine_pending":
        answer = handle_ouro_intake_command(f"answer session:{started.session_id} answer:승인", actor="tester")
    assert answer.action == "interview_updated"
    if "Restate:" in answer.message:
        answer = handle_ouro_intake_command(f"answer session:{started.session_id} answer:승인", actor="tester")
    assert answer.action == "interview_updated"
    sessions = json.loads((hermes_home / "ouro_intake_sessions.json").read_text())
    assert sessions[started.session_id]["status"] == "seed_ready"
    seed_payload = sessions[started.session_id]["seed"]["upstream_seed"]
    seed = Seed.from_dict(seed_payload)
    assert seed.goal == "Design gateway intake wrapper"
    assert seed.brownfield_context.project_type == "brownfield"
    assert seed.ontology_schema.fields[0].field_type == "string"
    assert seed.evaluation_principles[0].name
    assert seed.exit_conditions[0].evaluation_criteria


def test_bo062_vendored_source_records_upstream_commit():
    from pathlib import Path
    import subprocess

    repo_root = Path(__file__).resolve().parents[2]
    ledger = repo_root / "hermes_integrations/ouroboros_upstream/VENDORED_UPSTREAM.md"
    text = ledger.read_text()
    upstream_sha = subprocess.check_output(["git", "-C", "/tmp/ouroboros-upstream", "rev-parse", "HEAD"], text=True).strip()
    assert "https://github.com/Q00/ouroboros" in text
    assert upstream_sha in text


def test_bo062_gateway_wrapper_keeps_execution_runner_excluded():
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    vendored = repo_root / "hermes_integrations/ouroboros_upstream"
    assert not (vendored / "orchestrator").exists()
    assert not (vendored / "ralph_loop.py").exists()
    assert not (vendored / "orchestrator_stage.py").exists()



def test_bo062_uses_vendored_upstream_seed_reviewer_and_repairer(hermes_home):
    from gateway.ouro_intake import handle_ouro_intake_command
    from hermes_integrations.ouroboros_upstream.auto.seed_reviewer import SeedReviewer
    from hermes_integrations.ouroboros_upstream.auto.seed_repairer import SeedRepairer
    from hermes_integrations.ouroboros_upstream.core.seed import Seed

    started = handle_ouro_intake_command(
        'goal:"Ship vendored seed review" project:bo tenant:kanban context:"Hermes gateway existing runtime" acceptance:"pytest exits 0"',
        actor="tester",
    )
    answer = handle_ouro_intake_command(
        f'answer session:{started.session_id} answer:"exact pytest command exits 0 and no execution runner is called"',
        actor="tester",
    )
    if answer.action == "refine_pending":
        answer = handle_ouro_intake_command(f"answer session:{started.session_id} answer:승인", actor="tester")
    if "Restate:" in answer.message:
        answer = handle_ouro_intake_command(f"answer session:{started.session_id} answer:승인", actor="tester")
    sessions = json.loads((hermes_home / "ouro_intake_sessions.json").read_text())
    seed_contract = sessions[started.session_id]["seed"]
    assert seed_contract["upstream_auto_review"]["grade"] in {"A", "B", "C"}
    assert "scores" in seed_contract["upstream_auto_review"]
    assert isinstance(seed_contract["upstream_auto_repair_history"], list)
    seed = Seed.from_dict(seed_contract["upstream_seed"])
    review = SeedReviewer().review(seed)
    repaired_seed, repaired_review, history = SeedRepairer(max_iterations=2).converge(seed)
    assert repaired_review.grade_result.grade.value in {"A", "B", "C"}
    assert Seed.from_dict(repaired_seed.to_dict()).goal == seed.goal



def test_bo062_seed_construction_uses_vendored_seed_generator_parser():
    from hermes_integrations.ouroboros_upstream import adapter
    from hermes_integrations.ouroboros_upstream.bigbang.seed_generator import SeedGenerator
    from hermes_integrations.ouroboros_upstream.core.seed import Seed

    calls = {"count": 0}
    original = SeedGenerator.build_from_structured_response

    def spy(self, response, **kwargs):
        calls["count"] += 1
        assert "GOAL:" in response
        assert "ONTOLOGY_FIELDS:" in response
        assert "ACCEPTANCE_CRITERIA:" in response
        return original(self, response, **kwargs)

    SeedGenerator.build_from_structured_response = spy
    try:
        seed_payload = adapter.build_seed_dict(
            {
                "goal": "Ship vendored SeedGenerator path",
                "project": "bo",
                "context": "Hermes gateway existing runtime",
                "acceptance_criteria": "pytest exits 0",
            },
            {"ambiguity_score": 0.15},
            session_id="oi_test",
        )
        seed = Seed.from_dict(seed_payload)
        assert seed.goal == "Ship vendored SeedGenerator path"
        assert seed.ontology_schema.fields
    finally:
        SeedGenerator.build_from_structured_response = original

    assert calls["count"] == 1



def test_bo062_hermes_agent_extraction_override_feeds_vendored_generator():
    from hermes_integrations.ouroboros_upstream import adapter
    from hermes_integrations.ouroboros_upstream.core.seed import Seed

    extraction = """GOAL: Hermes extracted goal
CONSTRAINTS: no worker dispatch | no gateway restart
ACCEPTANCE_CRITERIA: pytest exits 0 | Seed.from_dict validates
ONTOLOGY_NAME: HermesExtractedAdmission
ONTOLOGY_DESCRIPTION: Requirements extracted by Hermes acting as the Ouroboros LLM layer.
ONTOLOGY_FIELDS: goal:string:Goal | proof:array:Proof artifacts
EVALUATION_PRINCIPLES: completeness:All criteria are present:1.0 | authority:Execution authority remains separate:1.0
EXIT_CONDITIONS: tests_pass:Tests pass:pytest exits 0 | authority_preserved:No dispatch:executor_dispatch remains forbidden
PROJECT_TYPE: brownfield
CONTEXT_REFERENCES: hermes://context:reference:existing gateway runtime
EXISTING_PATTERNS: gateway wrapper | Kanban admission
EXISTING_DEPENDENCIES: Hermes gateway
"""
    payload = adapter.build_seed_dict(
        {
            "goal": "Fallback goal should not win",
            "hermes_extraction": extraction,
        },
        {"ambiguity_score": 0.12},
        session_id="oi_hermes",
    )
    seed = Seed.from_dict(payload)
    assert seed.goal == "Hermes extracted goal"
    assert seed.brownfield_context.project_type == "brownfield"
    assert seed.ontology_schema.name == "HermesExtractedAdmission"
    assert "no worker dispatch" in seed.constraints


def test_bo062_gateway_records_hermes_extraction_source():
    import gateway.ouro_intake as ouro_intake

    extraction = """GOAL: Gateway Hermes extraction
CONSTRAINTS: no worker dispatch
ACCEPTANCE_CRITERIA: pytest exits 0
ONTOLOGY_NAME: GatewayHermesAdmission
ONTOLOGY_DESCRIPTION: Hermes extraction contract test
ONTOLOGY_FIELDS: goal:string:Goal
EVALUATION_PRINCIPLES: completeness:Complete:1.0
EXIT_CONDITIONS: tests_pass:Tests pass:pytest exits 0
PROJECT_TYPE: brownfield
CONTEXT_REFERENCES: hermes://context:reference:gateway
EXISTING_PATTERNS: gateway wrapper
EXISTING_DEPENDENCIES: Hermes gateway"""
    seed_contract = ouro_intake._build_seed_contract(
        {
            "goal": "Gateway fallback",
            "project": "bo",
            "tenant": "kanban",
            "hermes_extraction": extraction,
        },
        public_id="BO-TEST",
        actor="tester",
        session_id="oi_test",
    )
    assert seed_contract["seed_extraction"]["source"] == "hermes_agent_structured_extraction"
    assert seed_contract["seed_extraction"]["provider_call"] is False
    assert seed_contract["upstream_seed"]["goal"] == "Gateway Hermes extraction"

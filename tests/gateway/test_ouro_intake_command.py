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
    assert "Question (" in result.message
    assert "Socratic blockers" not in result.message
    assert "No Kanban card" in result.message or "no Kanban card" in result.message
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

    assert updated.action == "interview_updated"
    assert updated.session_id == started.session_id
    assert updated.public_id is None
    assert "Restate gate is pending" in updated.message
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
    assert "Seed is decision-gated" in started.message

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
    assert updated.action == "interview_updated"
    assert updated.session_id == started.session_id
    assert "Updated /ouro-intake session" in updated.message

    sessions = json.loads((hermes_home / "ouro_intake_sessions.json").read_text())
    session = sessions[started.session_id]
    assert session["turns"][-1]["answer"] == "A와 B에 가까워"
    assert session["origin_binding"]["key"] == "discord|channel-1|thread-1|user-1"


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

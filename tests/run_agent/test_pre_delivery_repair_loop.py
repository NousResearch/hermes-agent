from types import SimpleNamespace

from agent import turn_stop_gates
from agent.turn_finalizer import PreDeliveryDecision


class _Agent(SimpleNamespace):
    platform = "telegram"

    def _interim_content_was_streamed(self, _text):
        return False

    def _release_pre_delivery_text(self, text):
        self.released = text


def _block(reason="missing evidence", replacement="safe terminal degradation"):
    return PreDeliveryDecision(
        action="block",
        response_text=replacement,
        transformed=True,
        pre_transform_response="candidate",
        reason=reason,
    )


def _apply(agent, messages):
    return turn_stop_gates.apply_stop_gates(
        agent,
        {"role": "assistant", "content": "candidate", "finish_reason": "stop"},
        final_response="candidate",
        messages=messages,
        conversation_history=None,
        pending_verification_response=None,
        pending_verification_response_previewed=None,
        effective_task_id="task",
        turn_id="turn",
        original_user_message="clinical question",
    )


def _disable_other_gates(monkeypatch):
    monkeypatch.setattr(turn_stop_gates, "_verify_on_stop_nudge", lambda _agent: None)
    monkeypatch.setattr(turn_stop_gates, "_pre_verify_nudge", lambda *_args: None)
    monkeypatch.setattr(turn_stop_gates, "_kanban_stop_nudge", lambda *_args: None)
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda hook: hook == "pre_delivery")


def test_pre_delivery_block_continues_same_turn_with_structured_repair(monkeypatch):
    _disable_other_gates(monkeypatch)
    monkeypatch.setattr("agent.turn_finalizer._prepare_output_for_delivery", lambda *_a, **_k: _block())
    messages = []
    agent = _Agent()

    verdict = _apply(agent, messages)

    assert verdict.continue_turn is True
    assert verdict.final_response is None
    assert verdict.pending_verification_response is None
    assert agent._pre_delivery_repair_attempts == 1
    assert messages[0]["finish_reason"] == "pre_delivery_repair_required"
    assert messages[0]["_pre_delivery_repair_synthetic"] is True
    assert messages[1]["_pre_delivery_repair_synthetic"] is True
    assert messages[1]["content"].startswith("[PRE_DELIVERY_REPAIR]\n")
    assert '"reason": "policy_block"' in messages[1]["content"]


def test_pre_delivery_threshold_escalates_without_releasing_gate_message(monkeypatch):
    _disable_other_gates(monkeypatch)
    monkeypatch.setattr("agent.turn_finalizer._prepare_output_for_delivery", lambda *_a, **_k: _block())
    messages = []
    agent = _Agent(_pre_delivery_repair_attempts=turn_stop_gates._MAX_PRE_DELIVERY_REPAIR_ATTEMPTS)

    verdict = _apply(agent, messages)

    assert verdict.continue_turn is True
    assert verdict.final_response is None
    assert verdict.pending_verification_response is None
    assert agent._pre_delivery_repair_attempts == turn_stop_gates._MAX_PRE_DELIVERY_REPAIR_ATTEMPTS + 1
    assert not hasattr(agent, "released")
    assert messages[-1]["content"].startswith("[PRE_DELIVERY_REPAIR]\n")
    assert '"mode": "escalated"' in messages[-1]["content"]


def test_pre_delivery_repair_rows_are_never_durable():
    from agent.session_persistence import _is_ephemeral_scaffolding

    assert _is_ephemeral_scaffolding({
        "role": "assistant",
        "content": "blocked clinical candidate",
        "_pre_delivery_repair_synthetic": True,
    })
    assert _is_ephemeral_scaffolding({
        "role": "user",
        "content": "[PRE_DELIVERY_REPAIR] private repair",
        "_pre_delivery_repair_synthetic": True,
    })


def test_pre_delivery_repair_rows_are_filtered_from_trajectories(monkeypatch):
    from agent import session_persistence

    captured = {}
    class Agent(session_persistence.SessionPersistenceMixin):
        save_trajectories = True
        model = "test/model"

        @staticmethod
        def _convert_to_trajectory_format(messages, *_):
            return list(messages)

    agent = Agent()
    monkeypatch.setattr(
        session_persistence,
        "_save_trajectory_to_file",
        lambda trajectory, *_: captured.setdefault("trajectory", trajectory),
    )
    messages = [
        {"role": "assistant", "content": "PHI_SENTINEL", "_pre_delivery_repair_synthetic": True},
        {"role": "user", "content": "repair", "_pre_delivery_repair_synthetic": True},
        {"role": "assistant", "content": "safe"},
    ]

    session_persistence.SessionPersistenceMixin._save_trajectory(
        agent, messages, "query", True,
    )

    assert captured["trajectory"] == [{"role": "assistant", "content": "safe"}]


def test_pre_delivery_hook_lookup_error_still_fails_closed(monkeypatch):
    _disable_other_gates(monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.plugins.has_hook",
        lambda _hook: (_ for _ in ()).throw(RuntimeError()),
    )
    monkeypatch.setattr(
        "agent.turn_finalizer._prepare_output_for_delivery",
        lambda *_a, **_k: _block(reason="PHI_SENTINEL"),
    )
    agent = _Agent(_pre_delivery_repair_attempts=turn_stop_gates._MAX_PRE_DELIVERY_REPAIR_ATTEMPTS)

    verdict = _apply(agent, [])

    assert verdict.continue_turn is True
    assert verdict.final_response is None
    assert not hasattr(agent, "released")


def test_repair_directive_contains_only_bounded_reason_code(monkeypatch):
    _disable_other_gates(monkeypatch)
    monkeypatch.setattr(
        "agent.turn_finalizer._prepare_output_for_delivery",
        lambda *_a, **_k: _block(reason="unsupported claims: PHI_SENTINEL"),
    )
    messages = []

    verdict = _apply(_Agent(), messages)

    assert verdict.continue_turn is True
    assert "PHI_SENTINEL" not in messages[-1]["content"]


def test_repair_directive_expands_research_route_by_attempt():
    directives = [
        turn_stop_gates._build_pre_delivery_repair_directive(
            attempt=attempt,
            reason="policy_block",
        )
        for attempt in range(1, 6)
    ]

    assert "ruta clínica seleccionada" in directives[0]
    assert "proveedores alternativos" in directives[1]
    assert "texto completo" in directives[2]
    assert "descubrimiento ampliado" in directives[3]
    assert '"mode": "escalated"' in directives[4]
    assert "No te detengas en una abstención" in directives[4]

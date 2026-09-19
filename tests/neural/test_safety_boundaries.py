from neural.events.types import NeuralEvent
from neural.guardian import Guardian
from neural.reflexes import Reflex


def test_guardian_only_observes_risk_and_cannot_approve_actions():
    guardian = Guardian()
    event = NeuralEvent(source="terminal", event_type="command.requested", payload={"command": "rm -rf /"})

    assessment = guardian.assess(event)

    assert assessment.risk >= 0.0
    assert assessment.allowed is False
    assert assessment.can_override_policy is False


def test_reflex_is_deterministic_and_only_emits_observation():
    reflex = Reflex(name="hard_stop")
    event = NeuralEvent(source="approval", event_type="policy.denied")

    signal = reflex.trigger(event)

    assert signal.target == "hard_stop"
    assert signal.value == 1.0

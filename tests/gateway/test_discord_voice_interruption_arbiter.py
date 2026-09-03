"""Pure lifecycle contracts for Discord voice interruption epochs."""

from plugins.platforms.discord.voice_interruption import VoiceInterruptionArbiter


def test_claim_is_one_shot_and_requires_exact_active_token():
    arbiter = VoiceInterruptionArbiter()
    assert arbiter.open_epoch(111, 7) == 7

    assert arbiter.claim_wake(111, 6, "batch") is None
    grant = arbiter.claim_wake(111, 7, "streaming")

    assert grant is not None
    assert arbiter.validate_grant(grant) is True
    assert arbiter.claim_wake(111, 7, "batch") is None


def test_unclaimed_playback_finish_makes_epoch_terminal():
    arbiter = VoiceInterruptionArbiter()
    arbiter.open_epoch(111, 7)

    arbiter.playback_finished(111, 7)

    assert arbiter.claim_wake(111, 7, "batch") is None


def test_ack_pending_grant_survives_transport_finish_until_ack_completes():
    arbiter = VoiceInterruptionArbiter()
    arbiter.open_epoch(111, 7)
    grant = arbiter.claim_wake(111, 7, "batch")
    assert grant is not None

    arbiter.playback_finished(111, 7)

    assert arbiter.validate_grant(grant) is True
    arbiter.complete_ack(grant)
    assert arbiter.validate_grant(grant) is False


def test_hard_teardown_revokes_grant_and_returns_bound_task_for_cancellation():
    arbiter = VoiceInterruptionArbiter()
    arbiter.open_epoch(111, 7)
    grant = arbiter.claim_wake(111, 7, "streaming")
    task = object()
    assert grant is not None
    assert arbiter.bind_ack_task(grant, task) is True

    tasks = arbiter.terminate_scope(111, "leave")

    assert tasks == (task,)
    assert arbiter.validate_grant(grant) is False
    assert arbiter.bind_ack_task(grant, object()) is False


def test_new_epoch_revokes_old_grant_and_cancels_bound_task():
    class Task:
        def __init__(self):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

    arbiter = VoiceInterruptionArbiter()
    arbiter.open_epoch(111, 7)
    grant = arbiter.claim_wake(111, 7, "batch")
    task = Task()
    assert grant is not None
    assert arbiter.bind_ack_task(grant, task) is True

    assert arbiter.open_epoch(111, 8) == 8

    assert task.cancelled is True
    assert arbiter.validate_grant(grant) is False
    assert arbiter.claim_wake(111, 7, "streaming") is None
    assert arbiter.claim_wake(111, 8, "streaming") is not None

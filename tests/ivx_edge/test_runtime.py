from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from ivx_edge import (
    Command,
    DeviceRuntime,
    EnrollmentAuthority,
    EnrollmentError,
    RuntimePolicy,
    SecurityError,
    sign_command,
)


def enrolled(now=lambda: 1_000):
    authority = EnrollmentAuthority(now=now)
    code = authority.issue_code("workspace-a", "app-a")
    return authority, code, authority.enroll(code, "device-a")


def command_for(identity, *, command_id="cmd-1", nonce="nonce-1", capability="sensor.temperature.read", payload=None):
    return sign_command(
        Command(
            command_id=command_id,
            device_id=identity.device_id,
            workspace_id=identity.workspace_id,
            app_id=identity.app_id,
            capability=capability,
            payload={} if payload is None else payload,
            issued_at=1_000,
            expires_at=1_060,
            nonce=nonce,
        ),
        identity.key,
    )


def test_enrollment_code_is_one_time_and_short_lived():
    authority = EnrollmentAuthority(now=lambda: 1_000)
    code = authority.issue_code("workspace-a", "app-a", ttl_seconds=60)
    identity = authority.enroll(code, "device-a")

    assert identity.workspace_id == "workspace-a"
    with pytest.raises(EnrollmentError, match="already used"):
        authority.enroll(code, "device-b")

    expired_authority = EnrollmentAuthority(now=lambda: 2_000)
    expired = expired_authority.issue_code("workspace-a", "app-a", ttl_seconds=1)
    expired_authority._now = lambda: 2_002
    with pytest.raises(EnrollmentError, match="expired"):
        expired_authority.enroll(expired, "device-c")


def test_scope_signature_capability_and_revocation_fail_closed():
    _, _, identity = enrolled()
    runtime = DeviceRuntime(
        identity,
        RuntimePolicy(frozenset({"sensor.temperature.read"})),
        now=lambda: 1_000,
    )
    signed = command_for(identity)

    with pytest.raises(SecurityError, match="scope"):
        runtime.submit(replace(signed, workspace_id="workspace-b"))
    with pytest.raises(SecurityError, match="signature"):
        runtime.submit(replace(signed, signature="00" * 32))
    with pytest.raises(SecurityError, match="not granted"):
        runtime.submit(
            command_for(identity, capability="screen.content.set", payload={"content_id": "welcome"})
        )

    runtime.revoke()
    with pytest.raises(SecurityError, match="revoked"):
        runtime.submit(signed)


def test_mutating_adapter_requires_confirmation_and_rejects_injection():
    _, _, identity = enrolled()
    policy = RuntimePolicy(frozenset({"screen.content.set"}))
    command = command_for(
        identity,
        capability="screen.content.set",
        payload={"content_id": "welcome"},
    )
    denied = DeviceRuntime(identity, policy, now=lambda: 1_000).submit(command)
    assert denied.status == "rejected"

    runtime = DeviceRuntime(identity, policy, confirm=lambda _command: True, now=lambda: 1_000)
    completed = runtime.submit(command)
    assert completed.status == "completed"
    assert completed.memory_event["scope"] == "workspace/app/device"

    injected = command_for(
        identity,
        command_id="cmd-2",
        nonce="nonce-2",
        capability="screen.content.set",
        payload={"content_id": "welcome; shutdown -h now"},
    )
    with pytest.raises(SecurityError, match="bounded content_id"):
        runtime.submit(injected)


def test_signed_payload_is_recursively_immutable():
    _, _, identity = enrolled()
    source = {"content_id": "welcome", "nested": {"items": ["one"]}}
    command = command_for(
        identity,
        capability="screen.content.set",
        payload=source,
    )
    source["content_id"] = "substituted"
    source["nested"]["items"].append("two")

    assert command.payload["content_id"] == "welcome"
    assert command.payload["nested"]["items"] == ("one",)
    with pytest.raises(TypeError):
        command.payload["content_id"] = "mutated"


def test_offline_queue_is_idempotent_and_replay_protected():
    _, _, identity = enrolled()
    runtime = DeviceRuntime(
        identity,
        RuntimePolicy(frozenset({"sensor.temperature.read"})),
        now=lambda: 1_000,
    )
    command = command_for(identity)
    runtime.set_online(False)

    assert runtime.submit(command).status == "queued"
    assert runtime.submit(command).status == "queued"
    results = runtime.set_online(True)
    assert [result.status for result in results] == ["completed"]

    # A retry of the same signed command returns the prior result without re-execution.
    assert runtime.submit(command) == results[0]
    assert len([event for event in runtime.audit if event.status == "completed"]) == 1

    replay = command_for(identity, command_id="cmd-2", nonce="nonce-1")
    with pytest.raises(SecurityError, match="replay"):
        runtime.submit(replay)


def test_command_id_reuse_requires_identical_signed_content():
    _, _, identity = enrolled()
    runtime = DeviceRuntime(
        identity,
        RuntimePolicy(frozenset({"sensor.temperature.read"})),
        now=lambda: 1_000,
    )
    runtime.submit(command_for(identity))
    conflicting = command_for(identity, command_id="cmd-1", nonce="nonce-2")

    with pytest.raises(SecurityError, match="reused"):
        runtime.submit(conflicting)


def test_invalid_offline_entry_does_not_drop_later_work():
    clock = [1_000]
    _, _, identity = enrolled(now=lambda: clock[0])
    runtime = DeviceRuntime(
        identity,
        RuntimePolicy(frozenset({"sensor.temperature.read"})),
        now=lambda: clock[0],
    )
    runtime.set_online(False)
    runtime.submit(command_for(identity, command_id="cmd-expired", nonce="nonce-expired"))
    valid = replace(
        command_for(identity, command_id="cmd-valid", nonce="nonce-valid"),
        expires_at=1_100,
        signature="",
    )
    valid = sign_command(valid, identity.key)
    runtime.submit(valid)
    clock[0] = 1_070

    results = runtime.set_online(True)

    assert [result.status for result in results] == ["failed", "completed"]
    assert runtime._queue == []


def test_concurrent_duplicate_submission_executes_once():
    _, _, identity = enrolled()
    runtime = DeviceRuntime(
        identity,
        RuntimePolicy(frozenset({"sensor.temperature.read"})),
        now=lambda: 1_000,
    )
    command = command_for(identity)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _index: runtime.submit(command), range(16)))

    assert all(result == results[0] for result in results)
    assert len([event for event in runtime.audit if event.status == "completed"]) == 1


def test_revocation_is_serialized_with_inflight_execution():
    _, _, identity = enrolled()
    confirmation_entered = Event()
    release_confirmation = Event()

    def confirm(_command):
        confirmation_entered.set()
        release_confirmation.wait(timeout=2)
        return True

    runtime = DeviceRuntime(
        identity,
        RuntimePolicy(frozenset({"screen.content.set"})),
        confirm=confirm,
        now=lambda: 1_000,
    )
    command = command_for(
        identity,
        capability="screen.content.set",
        payload={"content_id": "welcome"},
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        execution = pool.submit(runtime.submit, command)
        assert confirmation_entered.wait(timeout=2)
        revocation = pool.submit(runtime.revoke)
        assert not revocation.done()
        release_confirmation.set()
        assert execution.result(timeout=2).status == "completed"
        revocation.result(timeout=2)

    with pytest.raises(SecurityError, match="revoked"):
        runtime.submit(command_for(identity, command_id="cmd-after", nonce="nonce-after"))


def test_rejected_commands_are_audited_without_raw_payload():
    _, _, identity = enrolled()
    runtime = DeviceRuntime(
        identity,
        RuntimePolicy(frozenset({"screen.content.set"})),
        confirm=lambda _command: True,
        now=lambda: 1_000,
    )
    injected = command_for(
        identity,
        capability="screen.content.set",
        payload={"content_id": "secret; rm -rf /"},
    )

    with pytest.raises(SecurityError):
        runtime.submit(injected)

    assert runtime.audit[-1].status == "failed"
    assert "secret" not in repr(runtime.audit[-1])


def test_oversized_rejection_still_emits_redacted_audit():
    _, _, identity = enrolled()
    runtime = DeviceRuntime(
        identity,
        RuntimePolicy(frozenset({"screen.content.set"})),
        now=lambda: 1_000,
    )
    oversized = replace(
        command_for(
            identity,
            capability="screen.content.set",
            payload={"content_id": "welcome"},
        ),
        payload={"content_id": "x" * 20_000},
        signature="invalid",
    )

    with pytest.raises(SecurityError, match="maximum canonical size"):
        runtime.submit(oversized)

    assert runtime.audit[-1].status == "failed"
    assert "x" * 100 not in repr(runtime.audit[-1])


def test_oversized_audit_digest_distinguishes_payloads():
    _, _, identity = enrolled()
    runtime = DeviceRuntime(
        identity,
        RuntimePolicy(frozenset({"screen.content.set"})),
        now=lambda: 1_000,
    )
    base = command_for(
        identity,
        capability="screen.content.set",
        payload={"content_id": "welcome"},
    )
    oversized_a = replace(base, payload={"content_id": "a" * 20_000}, signature="invalid")
    oversized_b = replace(base, payload={"content_id": "b" * 20_000}, signature="invalid")

    for command in (oversized_a, oversized_b):
        with pytest.raises(SecurityError):
            runtime.submit(command)

    assert runtime.audit[-2].payload_digest != runtime.audit[-1].payload_digest


def test_malformed_command_types_fail_closed_and_are_audited():
    _, _, identity = enrolled()
    runtime = DeviceRuntime(
        identity,
        RuntimePolicy(frozenset({"sensor.temperature.read"})),
        now=lambda: 1_000,
    )
    malformed = replace(command_for(identity), issued_at="1000")

    with pytest.raises(SecurityError, match="integer seconds"):
        runtime.submit(malformed)

    assert runtime.audit[-1].status == "failed"


@pytest.mark.parametrize("field", ["device_id", "workspace_id", "app_id"])
def test_byte_identity_fields_fail_closed_and_are_audited(field):
    _, _, identity = enrolled()
    runtime = DeviceRuntime(
        identity,
        RuntimePolicy(frozenset({"sensor.temperature.read"})),
        now=lambda: 1_000,
    )
    malformed = replace(command_for(identity), **{field: b"invalid"})

    with pytest.raises(SecurityError, match="must be a string"):
        runtime.submit(malformed)

    assert runtime.audit[-1].status == "failed"


def test_confirmation_failure_does_not_consume_command():
    _, _, identity = enrolled()
    attempts = [0]

    def confirm(_command):
        attempts[0] += 1
        if attempts[0] == 1:
            raise RuntimeError("prompt unavailable")
        return True

    runtime = DeviceRuntime(
        identity,
        RuntimePolicy(frozenset({"screen.content.set"})),
        confirm=confirm,
        now=lambda: 1_000,
    )
    command = command_for(
        identity,
        capability="screen.content.set",
        payload={"content_id": "welcome"},
    )

    with pytest.raises(SecurityError, match="failed closed"):
        runtime.submit(command)
    assert runtime.audit[-1].status == "failed"
    assert runtime.submit(command).status == "completed"


def test_audit_uses_payload_digest_not_payload_or_secret():
    _, _, identity = enrolled()
    runtime = DeviceRuntime(
        identity,
        RuntimePolicy(frozenset({"screen.content.set"})),
        confirm=lambda _command: True,
        now=lambda: 1_000,
    )
    command = command_for(
        identity,
        capability="screen.content.set",
        payload={"content_id": "private-campaign"},
    )
    runtime.submit(command)
    serialized = repr(runtime.audit)

    assert "private-campaign" not in serialized
    assert identity.key.hex() not in serialized
    assert len(runtime.audit[0].payload_digest) == 64

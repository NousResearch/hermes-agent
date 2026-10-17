"""The Windows SSH spawn payload's transport (#127320).

Desktop streams its PowerShell spawn script on the SSH session's stdin, because
cmd.exe (Windows OpenSSH's default shell) rejects the ``-EncodedCommand`` form
of it for exceeding the 8191-character command-line limit. PowerShell drains that
stream before it invokes the helper, so the spawn payload must reach
``windows_ssh_runtime`` on argv as base64 — a stdin-reading spawn sees an empty
stream on that path. Plain local invocations keep using stdin.
"""

import base64
import io
import json
import sys

import pytest

from hermes_cli.windows_ssh_runtime import _MAX_JSON, _read_payload

_PAYLOAD = {
    "ownershipId": "0" * 32,
    "spawnNonce": "0" * 16,
    "profile": "default",
    "hermesPath": r"C:\Hermes\hermes.exe",
}


class _Stdin:
    def __init__(self, data: bytes):
        self.buffer = io.BytesIO(data)


def _b64(payload) -> str:
    return base64.b64encode(json.dumps(payload).encode()).decode()


def test_payload_arrives_on_argv_as_base64():
    assert _read_payload(["--payload-b64", _b64(_PAYLOAD)]) == _PAYLOAD


def test_argv_payload_survives_an_exhausted_stdin(monkeypatch):
    """The point of the argv channel: PowerShell has already drained stdin."""
    monkeypatch.setattr(sys, "stdin", _Stdin(b""))

    assert _read_payload(["--payload-b64", _b64(_PAYLOAD)]) == _PAYLOAD


def test_plain_invocation_still_reads_stdin(monkeypatch):
    monkeypatch.setattr(sys, "stdin", _Stdin(json.dumps(_PAYLOAD).encode()))

    assert _read_payload([]) == _PAYLOAD


def test_argv_payload_rejects_invalid_base64():
    with pytest.raises(ValueError, match="not valid base64"):
        _read_payload(["--payload-b64", "not base64 !!"])


def test_argv_payload_requires_exactly_one_value():
    with pytest.raises(ValueError, match="exactly one value"):
        _read_payload(["--payload-b64"])
    with pytest.raises(ValueError, match="exactly one value"):
        _read_payload(["--payload-b64", _b64(_PAYLOAD), "extra"])


def test_argv_payload_rejects_oversize_and_non_object():
    with pytest.raises(ValueError, match="too large"):
        _read_payload(["--payload-b64", _b64({"blob": "x" * (_MAX_JSON + 1)})])
    with pytest.raises(ValueError, match="must be an object"):
        _read_payload(["--payload-b64", _b64([1, 2, 3])])


def test_stdin_payload_keeps_the_same_size_and_shape_checks(monkeypatch):
    monkeypatch.setattr(sys, "stdin", _Stdin(b'"' + b"x" * _MAX_JSON + b'"'))
    with pytest.raises(ValueError, match="too large"):
        _read_payload([])

    monkeypatch.setattr(sys, "stdin", _Stdin(b"[1, 2, 3]"))
    with pytest.raises(ValueError, match="must be an object"):
        _read_payload([])


def test_spawn_operation_accepts_the_argv_payload():
    """`spawn` must forward its argv to the payload reader, not drop it.

    A regression to the old stdin-only reader shows up here as a read of the
    test runner's stdin, never as the payload the argv carried.
    """
    from hermes_cli import windows_ssh_runtime as mod

    seen = {}

    def fake_spawn_backend(payload):
        seen["payload"] = payload
        return {"ok": True}

    original = mod.spawn_backend
    mod.spawn_backend = fake_spawn_backend
    try:
        assert mod.dispatch(["spawn", "--payload-b64", _b64(_PAYLOAD)]) == {"ok": True}
    finally:
        mod.spawn_backend = original

    assert seen["payload"] == _PAYLOAD

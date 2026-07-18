from __future__ import annotations

import os
from pathlib import Path

import pytest


ISOLATED_RUNTIME_ENV = "MUNCHO_OWNER_GATE_ISOLATED_TEST_RUNTIME"
if os.environ.get(ISOLATED_RUNTIME_ENV) != "1":
    pytest.skip(
        "runs through test_passkey_v2_isolated_runtime.py under the exact "
        "owner-gate WebAuthn dependency boundary",
        allow_module_level=True,
    )

from scripts.canary import owner_gate_host_observation as producer


def test_host_observation_security_selftest_exercises_pinned_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(producer, "SELFTEST_BASE", tmp_path)
    result = producer._runtime_security_selftest()
    assert result == {
        "webauthn": {
            "rp_id": "lomliev.com",
            "origin": "https://auth.lomliev.com",
            "user_verification_required": True,
            "forged_assertion_blocked": True,
            "wrong_challenge_blocked": True,
            "wrong_origin_blocked": True,
            "wrong_rp_blocked": True,
            "no_uv_blocked": True,
            "replay_blocked": True,
            "concurrent_exactly_one_authorized": True,
            "web_raw_grant_api_absent": True,
        },
        "public_web_can_author_envelope": False,
        "authorization_receipt_signature_self_verified": True,
        "receipt_action_binding_self_verified": True,
    }

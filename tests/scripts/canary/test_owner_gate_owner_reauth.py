from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_owner_reauth as reauth
from scripts.canary import owner_gate_trust as trust


NOW = 2_000_000_000
PRIVATE_KEY = Ed25519PrivateKey.generate()
KEY_ID = hashlib.sha256(
    PRIVATE_KEY.public_key().public_bytes_raw()
).hexdigest()
REVISION = "a" * 40


@pytest.fixture(autouse=True)
def _pin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        KEY_ID,
    )


class _Executable:
    def __init__(self, prefix: tuple[str, ...]) -> None:
        self.prefix = prefix
        self.changed = False

    def trusted_command_prefix(self) -> tuple[str, ...]:
        if not self.changed:
            return self.prefix
        return (*self.prefix[:-1], self.prefix[-1] + ".changed")


class _Configuration:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._account = reauth.OWNER_ACCOUNT
        self.changed = False

    @property
    def account(self) -> str:
        return self._account

    def assert_stable(self) -> None:
        if self.changed:
            raise RuntimeError("changed")

    def environment_values(self) -> dict[str, str]:
        return {
            "HOME": str(self._root),
            "CLOUDSDK_CONFIG": str(self._root / ".config" / "gcloud"),
        }


class _Runner:
    def __init__(self, executable: _Executable | None = None) -> None:
        self.executable = executable
        self.interactive_calls: list[tuple] = []
        self.capture_calls: list[tuple] = []
        self.tty = True
        self.returncode = 0

    def interactive_tty_verified(self) -> bool:
        return self.tty

    def run_interactive(
        self,
        argv,
        *,
        env,
        timeout_seconds,
    ) -> int:
        self.interactive_calls.append((tuple(argv), dict(env), timeout_seconds))
        if self.executable is not None:
            self.executable.changed = True
        return self.returncode

    def run_capture(self, argv, *, env, timeout_seconds):
        self.capture_calls.append((tuple(argv), dict(env), timeout_seconds))
        payload = json.dumps({
            "projectId": foundation.PROJECT,
            "projectNumber": "123456789012",
        }).encode("ascii")
        return reauth.CapturedCommand(0, payload, b"")


def _runtime(tmp_path: Path) -> tuple[_Executable, _Configuration]:
    python = tmp_path / "python" / "bin" / "python3"
    module = tmp_path / "google-cloud-sdk" / "lib" / "gcloud.py"
    python.parent.mkdir(parents=True)
    module.parent.mkdir(parents=True)
    python.write_bytes(b"#!/bin/sh\nexit 0\n")
    module.write_bytes(b"# sealed gcloud module\n")
    os.chmod(python, 0o700)
    os.chmod(module, 0o600)
    prefix = (
        str(python),
        *reauth.launcher._GCLOUD_PYTHON_ISOLATION_ARGS,
        str(module),
    )
    return _Executable(prefix), _Configuration(tmp_path)


def _clock(*values: int):
    pending = iter(values)
    return lambda: next(pending)


def _sealed_identity(prefix: tuple[str, ...]) -> dict:
    unsigned = {
        "schema": "muncho-owner-sealed-gcloud-runtime-identity.v1",
        "release_sha": REVISION,
        "command_prefix_sha256": foundation.sha256_json(list(prefix)),
        "sdk_tree_entries": 10,
        "sdk_tree_bytes": 100,
        "sdk_tree_sha256": "1" * 64,
        "sdk_publication_tree_entries": 10,
        "sdk_publication_tree_bytes": 100,
        "sdk_publication_tree_sha256": "2" * 64,
        "sdk_publication_intent_sha256": "3" * 64,
        "python_version": "3.11.2",
        "python_tree_entries": 4,
        "python_tree_bytes": 40,
        "python_tree_sha256": "4" * 64,
        "owner_support_tree_entries": 5,
        "owner_support_tree_bytes": 50,
        "owner_support_tree_sha256": "5" * 64,
        "owner_support_manifest_sha256": "6" * 64,
        "owner_support_source_tree_oid": "b" * 40,
        "bootstrap_receipt_file_sha256": "7" * 64,
    }
    return {**unsigned, "identity_sha256": foundation.sha256_json(unsigned)}


@pytest.mark.parametrize("source_tree_oid", ("a" * 39, "A" * 40, "g" * 40))
def test_sealed_identity_rejects_invalid_source_tree_oid(
    tmp_path: Path,
    source_tree_oid: str,
) -> None:
    executable, _configuration = _runtime(tmp_path)
    identity = _sealed_identity(executable.prefix)
    identity["owner_support_source_tree_oid"] = source_tree_oid
    unsigned = dict(identity)
    unsigned.pop("identity_sha256")
    identity["identity_sha256"] = foundation.sha256_json(unsigned)

    with pytest.raises(
        reauth.OwnerGateOwnerReauthError,
        match="owner_gate_owner_reauth_runtime_invalid",
    ):
        reauth._validate_sealed_runtime_identity(
            identity,
            expected_release_revision=REVISION,
            prefix=executable.prefix,
        )


def _receipt(tmp_path: Path) -> tuple[dict, _Runner]:
    executable, configuration = _runtime(tmp_path)
    runner = _Runner()
    value = reauth._produce_owner_reauth_receipt_with_runtime(
        runner=runner,
        private_key=PRIVATE_KEY,
        now_unix=_clock(NOW, NOW + 3, NOW + 4),
        gcloud_executable=executable,
        gcloud_configuration=configuration,
        expected_release_revision=REVISION,
        sealed_runtime_snapshot=lambda: _sealed_identity(executable.prefix),
    )
    return dict(value), runner


def test_producer_uses_sealed_prefix_closed_environment_and_no_token(
    tmp_path: Path,
) -> None:
    receipt, runner = _receipt(tmp_path)
    checked = reauth.validate_owner_reauth_receipt(
        receipt,
        public_key=PRIVATE_KEY.public_key(),
        now_unix=NOW + 5,
    )

    argv, environment, timeout = runner.interactive_calls[0]
    assert argv[-6:] == (
        "auth",
        "login",
        reauth.OWNER_ACCOUNT,
        "--force",
        "--brief",
        f"--configuration={reauth.GCLOUD_CONFIGURATION}",
    )
    assert timeout == reauth.INTERACTIVE_TIMEOUT_SECONDS
    assert environment["CLOUDSDK_CORE_DISABLE_PROMPTS"] == "0"
    assert environment["CLOUDSDK_CORE_LOG_HTTP"] == "0"
    assert environment["CLOUDSDK_CORE_LOG_HTTP_REDACT_TOKEN"] == "1"
    assert not {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "PYTHONPATH",
        "PYTHONHOME",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
    } & set(environment)
    assert checked["interactive_reauthentication"]["access_token_requested"] is False
    assert checked["interactive_reauthentication"][
        "credential_material_captured"
    ] is False
    assert "access_token_value" not in repr(checked).casefold()
    assert checked["expires_at_unix"] - checked["issued_at_unix"] == 900
    assert len(runner.capture_calls) == 1


def test_producer_fails_closed_on_runtime_change(tmp_path: Path) -> None:
    executable, configuration = _runtime(tmp_path)
    runner = _Runner(executable)
    with pytest.raises(
        reauth.OwnerGateOwnerReauthError,
        match="owner_gate_owner_reauth_runtime",
    ):
        reauth._produce_owner_reauth_receipt_with_runtime(
            runner=runner,
            private_key=PRIVATE_KEY,
            now_unix=_clock(NOW, NOW + 1),
            gcloud_executable=executable,
            gcloud_configuration=configuration,
            expected_release_revision=REVISION,
            sealed_runtime_snapshot=lambda: _sealed_identity(executable.prefix),
        )


def test_producer_requires_tty_and_successful_interactive_flow(tmp_path: Path) -> None:
    executable, configuration = _runtime(tmp_path)
    runner = _Runner()
    runner.tty = False
    with pytest.raises(
        reauth.OwnerGateOwnerReauthError,
        match="owner_gate_owner_reauth_interactive_tty_required",
    ):
        reauth._produce_owner_reauth_receipt_with_runtime(
            runner=runner,
            private_key=PRIVATE_KEY,
            now_unix=_clock(NOW),
            gcloud_executable=executable,
            gcloud_configuration=configuration,
            expected_release_revision=REVISION,
            sealed_runtime_snapshot=lambda: _sealed_identity(executable.prefix),
        )


def test_public_producer_rejects_structural_protocol_fakes(tmp_path: Path) -> None:
    executable, configuration = _runtime(tmp_path)
    with pytest.raises(
        reauth.OwnerGateOwnerReauthError,
        match="owner_gate_owner_reauth_runtime_invalid",
    ):
        reauth.produce_owner_reauth_receipt(
            runner=_Runner(),
            private_key=PRIVATE_KEY,
            now_unix=_clock(NOW),
            gcloud_executable=executable,  # type: ignore[arg-type]
            gcloud_configuration=configuration,  # type: ignore[arg-type]
            expected_release_revision=REVISION,
        )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("interactive_reauthentication", "access_token_requested"), True),
        (("interactive_reauthentication", "credential_material_captured"), True),
        (("trusted_runtime_identity", "account"), "attacker@example.com"),
        (("trusted_runtime_identity", "project"), "attacker-project"),
    ],
)
def test_validator_rejects_tampered_security_identity(
    tmp_path: Path,
    path: tuple[str, str],
    replacement,
) -> None:
    receipt, _ = _receipt(tmp_path)
    receipt[path[0]][path[1]] = replacement
    with pytest.raises(reauth.OwnerGateOwnerReauthError):
        reauth.validate_owner_reauth_receipt(
            receipt,
            public_key=PRIVATE_KEY.public_key(),
            now_unix=NOW + 5,
        )


def test_validator_rejects_expired_noncanonical_and_cross_domain(
    tmp_path: Path,
) -> None:
    receipt, _ = _receipt(tmp_path)
    with pytest.raises(
        reauth.OwnerGateOwnerReauthError,
        match="owner_gate_owner_reauth_receipt_expired",
    ):
        reauth.validate_owner_reauth_receipt(
            receipt,
            public_key=PRIVATE_KEY.public_key(),
            now_unix=NOW + 905,
        )

    pretty = json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8")
    with pytest.raises(reauth.OwnerGateOwnerReauthError):
        reauth.decode_canonical_owner_reauth_receipt(
            pretty,
            public_key=PRIVATE_KEY.public_key(),
            now_unix=NOW + 5,
        )

    signed = {
        key: value
        for key, value in receipt.items()
        if key != "signature_ed25519_b64url"
    }
    receipt["signature_ed25519_b64url"] = base64.urlsafe_b64encode(
        PRIVATE_KEY.sign(
            b"some-other-protocol/v1\x00"
            + foundation.canonical_json_bytes(signed)
        )
    ).rstrip(b"=").decode("ascii")
    with pytest.raises(
        reauth.OwnerGateOwnerReauthError,
        match="owner_gate_owner_reauth_signature_invalid",
    ):
        reauth.validate_owner_reauth_receipt(
            receipt,
            public_key=PRIVATE_KEY.public_key(),
            now_unix=NOW + 5,
        )


def test_validator_rejects_unpinned_key(tmp_path: Path) -> None:
    receipt, _ = _receipt(tmp_path)
    attacker = Ed25519PrivateKey.generate()
    with pytest.raises(
        reauth.OwnerGateOwnerReauthError,
        match="owner_gate_owner_reauth_signer_not_pinned",
    ):
        reauth.validate_owner_reauth_receipt(
            receipt,
            public_key=attacker.public_key(),
            now_unix=NOW + 5,
        )

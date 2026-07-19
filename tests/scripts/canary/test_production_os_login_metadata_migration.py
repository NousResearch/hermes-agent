from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Mapping

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import production_os_login_metadata_migration as migration
from scripts.canary import production_cutover_owner_launcher as owner_launcher
from tests.gateway.test_canonical_writer_production_cutover import (
    _runtime_attestation,
)
from tests.scripts.canary.test_production_cutover_owner_launcher import (
    _ProductionKnownHosts,
    _production_transport,
)


NOW = 1_800_000_000
OWNER_SUBJECT = "a" * 64
OWNER_ACCOUNT = "owner@example.com"


@pytest.fixture(autouse=True)
def _clear_process_ca_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep launcher identity tests independent of gateway import side effects."""
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)


def test_production_target_does_not_depend_on_canary_project_identity() -> None:
    source = Path(migration.__file__).read_text(encoding="utf-8")
    assert "DEDICATED_CANARY_PROJECT_NUMBER" not in source
    assert migration._TARGET["project_number"] == (
        owner_launcher.PRODUCTION_PROJECT_NUMBER
    )


class FakeBoundary:
    def __init__(self, *, fail_access: bool = False, now: int = NOW) -> None:
        self.instance = {
            "enable-oslogin": "FALSE",
            "ssh-keys": "owner:ssh-ed25519 AAAATEST owner@example.com\n",
            "unrelated-key": "opaque-value-never-emitted",
        }
        self.project = {"project-key": "unchanged"}
        self.calls: list[str] = []
        self.fail_access = fail_access
        self.now = now

    @staticmethod
    def _fingerprint(value: Mapping[str, str]) -> str:
        return hashlib.sha256(migration._canonical(dict(value))).hexdigest()[:24]

    def observe(self) -> migration.ObservedState:
        state = migration._state_name(self.instance)
        ssh_keys = self.instance.get("ssh-keys")
        identity = {
            "target": copy.deepcopy(migration._TARGET),
            "owner_subject_sha256": OWNER_SUBJECT,
            "owner_account_sha256": hashlib.sha256(OWNER_ACCOUNT.encode()).hexdigest(),
            "state": state,
            "instance_metadata_fingerprint": self._fingerprint(self.instance),
            "instance_metadata_keys": sorted(self.instance),
            "enable_oslogin": self.instance.get("enable-oslogin"),
            "ssh_keys_present": ssh_keys is not None,
            "public_ssh_keys_sha256": (
                None
                if ssh_keys is None
                else hashlib.sha256(ssh_keys.encode()).hexdigest()
            ),
            "project_metadata_fingerprint": self._fingerprint(self.project),
            "project_metadata_keys": sorted(self.project),
            "os_login_profile_sha256": "b" * 64,
            "os_login_public_key_fingerprint": "c" * 64,
            "iam_permissions": {
                permission: "GRANTED" for permission in migration._IAM_PERMISSIONS
            },
        }
        unsigned = {
            "schema": migration.PREFLIGHT_SCHEMA,
            **identity,
            "state_identity_sha256": hashlib.sha256(
                migration._canonical(identity)
            ).hexdigest(),
            "observed_at_unix": self.now,
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        }
        receipt = migration._hashed(unsigned, "receipt_sha256")
        return migration.ObservedState(
            receipt=receipt,
            instance_metadata=tuple(sorted(self.instance.items())),
            project_metadata=tuple(sorted(self.project.items())),
        )

    def set_enable_oslogin_true(self) -> None:
        self.calls.append("set-enable-oslogin-true")
        self.instance["enable-oslogin"] = "TRUE"

    def remove_instance_ssh_keys(self) -> None:
        self.calls.append("remove-instance-ssh-keys")
        self.instance.pop("ssh-keys")

    def restore_instance_ssh_keys(self, value: str) -> None:
        self.calls.append("restore-instance-ssh-keys")
        self.instance["ssh-keys"] = value

    def restore_enable_oslogin(self, value: str | None) -> None:
        self.calls.append("restore-enable-oslogin")
        if value is None:
            self.instance.pop("enable-oslogin", None)
        else:
            self.instance["enable-oslogin"] = value

    def probe_iap_os_login(self) -> Mapping[str, object]:
        self.calls.append("probe-iap-os-login")
        if self.fail_access:
            raise RuntimeError("injected IAP failure")
        unsigned = {
            "schema": migration.ACCESS_SCHEMA,
            "target": copy.deepcopy(migration._TARGET),
            "owner_subject_sha256": OWNER_SUBJECT,
            "owner_account_sha256": hashlib.sha256(OWNER_ACCOUNT.encode()).hexdigest(),
            "fixed_remote_command": "/usr/bin/true",
            "authorization_snapshot_sha256": "d" * 64,
            "iap_os_login_succeeded": True,
            "observed_at_unix": self.now,
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        }
        return migration._hashed(unsigned, "receipt_sha256")


def _authority(boundary: FakeBoundary) -> tuple[dict, dict]:
    preflight = migration.collect_migration_preflight(boundary, now_unix=NOW)
    plan, approval = migration.build_migration_plan(
        preflight_receipt=preflight,
        owner_subject_sha256=OWNER_SUBJECT,
        private_key=Ed25519PrivateKey.generate(),
        now_unix=NOW,
    )
    return dict(plan), dict(approval)


def _intent(
    boundary: FakeBoundary,
    plan: Mapping[str, object],
    approval: Mapping[str, object],
) -> Mapping[str, object]:
    return migration.build_migration_intent(
        observed=boundary.observe(),
        plan=plan,
        approval=approval,
        now_unix=NOW,
    )


def test_signed_gate_applies_only_two_keys_then_proves_iap_os_login() -> None:
    boundary = FakeBoundary()
    plan, approval = _authority(boundary)
    intent = _intent(boundary, plan, approval)

    receipt = migration.execute_migration(
        boundary=boundary,
        plan=plan,
        approval=approval,
        intent=intent,
        now_unix=NOW,
    )

    assert boundary.instance == {
        "enable-oslogin": "TRUE",
        "unrelated-key": "opaque-value-never-emitted",
    }
    assert boundary.calls == [
        "set-enable-oslogin-true",
        "remove-instance-ssh-keys",
        "probe-iap-os-login",
    ]
    assert receipt["schema"] == migration.RECEIPT_SCHEMA
    assert receipt["iap_os_login_succeeded"] is True
    assert receipt["rollback_used"] is False
    assert receipt["private_key_staged"] is False


@pytest.mark.parametrize("interrupted_state", ("intermediate", "ready"))
def test_durable_intent_recovers_after_approval_expiry_and_reapplies(
    interrupted_state: str,
) -> None:
    boundary = FakeBoundary()
    plan, approval = _authority(boundary)
    intent = _intent(boundary, plan, approval)
    boundary.instance["enable-oslogin"] = "TRUE"
    if interrupted_state == "ready":
        boundary.instance.pop("ssh-keys")
    boundary.now = NOW + 1_000

    receipt = migration.execute_migration(
        boundary=boundary,
        plan=plan,
        approval=approval,
        intent=intent,
        now_unix=boundary.now,
    )

    assert receipt["intent_sha256"] == intent["intent_sha256"]
    assert boundary.calls == [
        "restore-instance-ssh-keys",
        "restore-enable-oslogin",
        "set-enable-oslogin-true",
        "remove-instance-ssh-keys",
        "probe-iap-os-login",
    ]
    assert boundary.instance == {
        "enable-oslogin": "TRUE",
        "unrelated-key": "opaque-value-never-emitted",
    }


def test_access_failure_restores_exact_prior_metadata() -> None:
    boundary = FakeBoundary(fail_access=True)
    prior = copy.deepcopy(boundary.instance)
    plan, approval = _authority(boundary)
    intent = _intent(boundary, plan, approval)

    with pytest.raises(RuntimeError, match="injected IAP failure"):
        migration.execute_migration(
            boundary=boundary,
            plan=plan,
            approval=approval,
            intent=intent,
            now_unix=NOW,
        )

    assert boundary.instance == prior
    assert boundary.calls == [
        "set-enable-oslogin-true",
        "remove-instance-ssh-keys",
        "probe-iap-os-login",
        "restore-instance-ssh-keys",
        "restore-enable-oslogin",
    ]


def test_base_exception_during_access_still_restores_exact_prior_metadata() -> None:
    class Interrupted(BaseException):
        pass

    class InterruptedBoundary(FakeBoundary):
        def probe_iap_os_login(self) -> Mapping[str, object]:
            self.calls.append("probe-iap-os-login")
            raise Interrupted()

    boundary = InterruptedBoundary()
    prior = copy.deepcopy(boundary.instance)
    plan, approval = _authority(boundary)
    intent = _intent(boundary, plan, approval)

    with pytest.raises(Interrupted):
        migration.execute_migration(
            boundary=boundary,
            plan=plan,
            approval=approval,
            intent=intent,
            now_unix=NOW,
        )

    assert boundary.instance == prior
    assert boundary.calls[-2:] == [
        "restore-instance-ssh-keys",
        "restore-enable-oslogin",
    ]


def test_pre_mutation_drift_fails_before_any_metadata_write() -> None:
    boundary = FakeBoundary()
    plan, approval = _authority(boundary)
    intent = _intent(boundary, plan, approval)
    boundary.instance["unrelated-key"] = "concurrent-change"

    with pytest.raises(
        migration.OsLoginMetadataMigrationError,
        match="os_login_metadata_pre_mutation_drifted",
    ):
        migration.execute_migration(
            boundary=boundary,
            plan=plan,
            approval=approval,
            intent=intent,
            now_unix=NOW,
        )

    assert boundary.calls == []


def test_tampered_owner_approval_fails_before_observation_or_mutation() -> None:
    boundary = FakeBoundary()
    plan, approval = _authority(boundary)
    intent = _intent(boundary, plan, approval)
    approval["signature_ed25519_hex"] = "0" * 128
    approval["approval_sha256"] = hashlib.sha256(
        migration._canonical({
            name: item for name, item in approval.items() if name != "approval_sha256"
        })
    ).hexdigest()

    with pytest.raises(
        migration.OsLoginMetadataMigrationError,
        match="os_login_metadata_approval_invalid",
    ):
        migration.execute_migration(
            boundary=boundary,
            plan=plan,
            approval=approval,
            intent=intent,
            now_unix=NOW,
        )

    assert boundary.calls == []


def test_concrete_transport_exposes_only_fixed_metadata_operations() -> None:
    transport = object.__new__(migration.ProductionOsLoginMetadataTransport)
    calls: list[tuple[str, ...]] = []
    transport._mutate = lambda arguments: calls.append(arguments)

    transport.set_enable_oslogin_true()
    transport.remove_instance_ssh_keys()
    transport.restore_enable_oslogin(None)
    transport.restore_enable_oslogin("FALSE")

    prefix = (
        "compute",
        "instances",
        "add-metadata",
        "ai-platform-runtime-01",
        "--project=adventico-ai-platform",
        "--zone=europe-west3-a",
    )
    assert calls == [
        (*prefix, "--metadata=enable-oslogin=TRUE"),
        (
            "compute",
            "instances",
            "remove-metadata",
            "ai-platform-runtime-01",
            "--project=adventico-ai-platform",
            "--zone=europe-west3-a",
            "--keys=ssh-keys",
        ),
        (
            "compute",
            "instances",
            "remove-metadata",
            "ai-platform-runtime-01",
            "--project=adventico-ai-platform",
            "--zone=europe-west3-a",
            "--keys=enable-oslogin",
        ),
        (*prefix, "--metadata=enable-oslogin=FALSE"),
    ]


def test_concrete_preflight_binds_instance_profile_key_and_effective_iam() -> None:
    calls: list[tuple[str, ...]] = []
    instance = {
        "id": migration._TARGET["instance_id"],
        "name": migration._TARGET["vm"],
        "zone": (
            "https://www.googleapis.com/compute/v1/projects/"
            f"{migration._TARGET['project']}/zones/{migration._TARGET['zone']}"
        ),
        "metadata": {
            "fingerprint": "instance-fingerprint",
            "items": [
                {"key": "enable-oslogin", "value": "FALSE"},
                {"key": "ssh-keys", "value": "legacy-public-key"},
            ],
        },
    }
    project = {
        "name": migration._TARGET["project"],
        "commonInstanceMetadata": {
            "fingerprint": "project-fingerprint",
            "items": [],
        },
    }
    fingerprint = "e" * 64
    profile = {
        "name": migration._TARGET["os_login_profile_id"],
        "posixAccounts": [
            {
                "username": migration._TARGET["os_login_username"],
                "homeDirectory": f"/home/{migration._TARGET['os_login_username']}",
                "operatingSystemType": "LINUX",
                "primary": True,
            }
        ],
        "sshPublicKeys": {
            fingerprint: {
                "fingerprint": fingerprint,
                "key": _ProductionKnownHosts.public_key,
            }
        },
    }

    def runner(argv, **_kwargs):
        exact = tuple(argv)
        calls.append(exact)
        if "instances" in exact and "describe" in exact:
            value = instance
        elif "project-info" in exact:
            value = project
        elif "os-login" in exact:
            value = profile
        elif "troubleshoot-policy" in exact:
            value = {"overallAccessState": "CAN_ACCESS"}
        else:
            raise AssertionError(exact)
        return subprocess.CompletedProcess(
            exact,
            0,
            stdout=json.dumps(value).encode(),
            stderr=b"",
        )

    production = _production_transport(runner=runner)
    production._owner_identity.owner_subject_sha256 = OWNER_SUBJECT
    boundary = migration.ProductionOsLoginMetadataTransport(
        production, clock=lambda: NOW
    )

    observed = boundary.observe()

    assert observed.receipt["state"] == "legacy_instance_ssh_keys"
    assert observed.receipt["iam_permissions"] == {
        permission: "GRANTED" for permission in migration._IAM_PERMISSIONS
    }
    assert len(calls) == 10
    joined = "\n".join(" ".join(call) for call in calls)
    assert "ai-platform-runtime-01" in joined
    assert "policy-intelligence troubleshoot-policy iam" in joined
    assert "--format=json(overallAccessState)" in joined
    assert "compute.instances.setMetadata" in joined
    assert "compute.instances.osAdminLogin" in joined
    assert "iap.tunnelInstances.accessViaIAP" in joined


class _CliIdentity:
    owner_subject_sha256 = OWNER_SUBJECT


def _write_openssh_key(path, key: Ed25519PrivateKey) -> None:
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.OpenSSH,
            serialization.NoEncryption(),
        )
    )
    path.chmod(0o600)


def test_owner_preflight_action_builds_signed_authority_without_key_staging(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    boundary = FakeBoundary()
    key = Ed25519PrivateKey.generate()
    key_path = (tmp_path / "skyvision_mac_ops_ed25519").resolve()
    output = (tmp_path / "os-login-authority.json").resolve()
    _write_openssh_key(key_path, key)
    monkeypatch.setattr(migration.time, "time", lambda: NOW)
    monkeypatch.setattr(
        owner_launcher.canary_transport,
        "harden_owner_secret_process",
        lambda: None,
    )
    monkeypatch.setattr(
        owner_launcher,
        "build_production_os_login_metadata_boundary",
        lambda _revision: (_CliIdentity(), boundary),
    )
    monkeypatch.setattr(
        owner_launcher,
        "_active_owner_runtime_attestation",
        lambda revision: _runtime_attestation(revision),
    )

    assert (
        owner_launcher.main((
            "os-login-preflight",
            "--revision",
            "a" * 40,
            "--owner-private-key",
            str(key_path),
            "--output",
            str(output),
        ))
        == 0
    )

    authority = migration.validate_authority_bundle(
        json.loads(output.read_text()),
        release_revision="a" * 40,
        now_unix=NOW,
    )
    printed = capsys.readouterr().out
    assert authority["private_key_staged"] is False
    assert key_path.read_text() not in printed
    assert "PRIVATE KEY" not in printed
    assert boundary.calls == []


def test_owner_migrate_action_consumes_exact_authority_and_runs_gate(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = FakeBoundary()
    plan, approval = _authority(boundary)
    preflight = migration.collect_migration_preflight(boundary, now_unix=NOW)
    authority = migration.build_authority_bundle(
        release_revision="a" * 40,
        preflight=preflight,
        plan=plan,
        approval=approval,
        now_unix=NOW,
    )
    authority_path = (tmp_path / "os-login-authority.json").resolve()
    authority_path.write_bytes(migration._canonical(authority))
    authority_path.chmod(0o600)
    output = (tmp_path / "os-login-receipt.json").resolve()
    monkeypatch.setattr(migration.time, "time", lambda: NOW)
    monkeypatch.setattr(
        owner_launcher,
        "build_production_os_login_metadata_boundary",
        lambda _revision: (_CliIdentity(), boundary),
    )
    monkeypatch.setattr(
        owner_launcher,
        "_active_owner_runtime_attestation",
        lambda revision: _runtime_attestation(revision),
    )

    assert (
        owner_launcher.main((
            "os-login-migrate",
            "--revision",
            "a" * 40,
            "--authority",
            str(authority_path),
            "--output",
            str(output),
        ))
        == 0
    )

    receipt = migration.validate_migration_receipt(
        json.loads(output.read_text()),
        plan=plan,
        approval=approval,
        now_unix=NOW,
    )
    assert receipt["iap_os_login_succeeded"] is True
    assert boundary.calls == [
        "set-enable-oslogin-true",
        "remove-instance-ssh-keys",
        "probe-iap-os-login",
    ]


def test_owner_gate_recovers_when_terminal_receipt_persistence_failed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = FakeBoundary()
    plan, approval = _authority(boundary)
    authority = {"plan": plan, "approval": approval}
    output = (tmp_path / "os-login-receipt.json").resolve()
    real_write = owner_launcher._write_public_output
    fence_events: list[str] = []

    class Fence:
        def install(self) -> None:
            fence_events.append("install")

        def begin_cleanup(self) -> None:
            fence_events.append("begin-cleanup")

        def restore(self) -> None:
            fence_events.append("restore")

    def fail_terminal_write(path, value):
        if path == output:
            raise OSError("injected terminal persistence failure")
        return real_write(path, value)

    monkeypatch.setattr(
        owner_launcher.canary_transport,
        "_OwnerSignalFence",
        Fence,
    )
    monkeypatch.setattr(owner_launcher, "_write_public_output", fail_terminal_write)
    monkeypatch.setattr(migration.time, "time", lambda: NOW)

    with pytest.raises(OSError, match="terminal persistence failure"):
        owner_launcher._execute_os_login_migration_signal_safe(
            boundary=boundary,
            authority=authority,
            output_path=output,
        )

    intent_path = output.with_name(f".{output.name}.migration-intent.json")
    assert intent_path.is_file()
    assert not output.exists()
    assert boundary.instance == {
        "enable-oslogin": "TRUE",
        "unrelated-key": "opaque-value-never-emitted",
    }
    assert fence_events == ["install", "begin-cleanup", "restore"]

    boundary.now = NOW + 1_000
    monkeypatch.setattr(migration.time, "time", lambda: boundary.now)
    monkeypatch.setattr(owner_launcher, "_write_public_output", real_write)
    fence_events.clear()

    receipt, created = owner_launcher._execute_os_login_migration_signal_safe(
        boundary=boundary,
        authority=authority,
        output_path=output,
    )

    assert created is True
    assert output.is_file()
    assert receipt["iap_os_login_succeeded"] is True
    assert boundary.calls[-5:] == [
        "restore-instance-ssh-keys",
        "restore-enable-oslogin",
        "set-enable-oslogin-true",
        "remove-instance-ssh-keys",
        "probe-iap-os-login",
    ]
    assert fence_events == ["install", "begin-cleanup", "restore"]


def test_public_boundary_builder_uses_release_pinned_owner_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    trusted = object()
    configuration = object()

    class Identity:
        owner_subject_sha256 = OWNER_SUBJECT

        def __init__(self, **kwargs) -> None:
            events.append(("identity", kwargs))

        def account_for_read_only_preflight(self) -> str:
            events.append("identity-preflight")
            return OWNER_ACCOUNT

    class Transport:
        def __init__(self, identity, **kwargs) -> None:
            events.append(("transport", identity, kwargs))

    monkeypatch.setattr(
        owner_launcher,
        "_active_owner_runtime_attestation",
        lambda revision: events.append(("runtime", revision))
        or _runtime_attestation(revision),
    )
    monkeypatch.setattr(
        owner_launcher.canary_transport,
        "TrustedGcloudExecutable",
        lambda *, release_sha: events.append(("trusted", release_sha))
        or trusted,
    )
    monkeypatch.setattr(
        owner_launcher.canary_transport,
        "PinnedGcloudConfiguration",
        lambda: configuration,
    )
    monkeypatch.setattr(
        owner_launcher.canary_transport,
        "GcloudOwnerAccessToken",
        Identity,
    )
    monkeypatch.setattr(owner_launcher, "ProductionCutoverTransport", Transport)

    identity, boundary = owner_launcher.build_production_os_login_metadata_boundary(
        "a" * 40
    )

    assert isinstance(identity, Identity)
    assert isinstance(boundary, migration.ProductionOsLoginMetadataTransport)
    assert ("runtime", "a" * 40) in events
    assert ("trusted", "a" * 40) in events

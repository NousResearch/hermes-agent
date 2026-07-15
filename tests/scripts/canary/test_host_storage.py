from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys

import pytest

from scripts.canary import host as canary_host
from scripts.canary import (
    host_storage,
    host_storage_boot,
    host_storage_boot_journal,
    host_storage_preflight,
)
from scripts.canary.network_boundary import NetworkBoundarySpec
from scripts.canary.network_boundary import build_plan as build_network_plan


NOW = 1_800_000_000
SQL_IP = "10.91.0.3"
NETWORK_PLAN = build_network_plan(NetworkBoundarySpec(sql_private_ip=SQL_IP))
HOST_PLAN = canary_host.build_plan(
    canary_host.HostSpec(
        sql_private_ip=SQL_IP,
        network_plan_sha256=NETWORK_PLAN.sha256,
    )
)


@pytest.fixture
def owner_tmp_path(tmp_path):
    """Return a temp directory with the exact journal owner contract."""
    descriptor = os.open(
        tmp_path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fchown(descriptor, os.geteuid(), os.getegid())
        os.fchmod(descriptor, 0o700)
        item = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    assert stat.S_ISDIR(item.st_mode)
    assert item.st_uid == os.geteuid()
    assert item.st_gid == os.getegid()
    assert stat.S_IMODE(item.st_mode) == 0o700
    return tmp_path


def _host_report(*, target: bool, collected_at: int = NOW - 2):
    vm_passed = target
    checks = []
    for name in sorted(host_storage_preflight._HOST_CHECK_NAMES):
        checks.append({
            "name": name,
            "passed": vm_passed
            if name == "resource.vm_absent_or_exact_running"
            else True,
            "detail": "exact",
        })
    return {
        "schema": "muncho-isolated-canary-host-preflight.v1",
        "ok": target,
        "collected_at_unix": collected_at,
        "network_plan_sha256": NETWORK_PLAN.sha256,
        "sql_private_ip": SQL_IP,
        "plan_sha256": HOST_PLAN.sha256,
        "satisfied_steps": ["create_isolated_canary_vm"] if target else [],
        "checks": checks,
    }


def _instance():
    return {
        "id": host_storage.VM_INSTANCE_ID,
        "name": "muncho-canary-v2-01",
        "status": "RUNNING",
        "zone": (
            "https://www.googleapis.com/compute/v1/projects/"
            "adventico-ai-platform/zones/europe-west3-a"
        ),
        "machineType": (
            "https://www.googleapis.com/compute/v1/projects/"
            "adventico-ai-platform/zones/europe-west3-a/machineTypes/e2-medium"
        ),
        "canIpForward": False,
        "deletionProtection": False,
        "labels": {},
        "resourcePolicies": [],
        "tags": {"items": ["iap-ssh"]},
        "metadata": {
            "items": [
                {"key": "disable-legacy-endpoints", "value": "TRUE"},
                {"key": "enable-oslogin", "value": "TRUE"},
            ]
        },
        "serviceAccounts": [
            {
                "email": (
                    "muncho-canary-v2-runtime@"
                    "adventico-ai-platform.iam.gserviceaccount.com"
                ),
                "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
            }
        ],
        "networkInterfaces": [
            {
                "network": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    "adventico-ai-platform/global/networks/muncho-canary-vpc"
                ),
                "subnetwork": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    "adventico-ai-platform/regions/europe-west3/subnetworks/"
                    "muncho-canary-europe-west3"
                ),
                "stackType": "IPV4_ONLY",
                "networkIP": "10.90.0.2",
                "accessConfigs": [
                    {
                        "type": "ONE_TO_ONE_NAT",
                        "networkTier": "PREMIUM",
                        "natIP": "203.0.113.10",
                    }
                ],
            }
        ],
        "disks": [
            {
                "boot": True,
                "autoDelete": True,
                "deviceName": "persistent-disk-0",
                "mode": "READ_WRITE",
                "type": "PERSISTENT",
                "source": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    "adventico-ai-platform/zones/europe-west3-a/disks/"
                    "muncho-canary-v2-01"
                ),
            }
        ],
        "scheduling": {
            "automaticRestart": True,
            "onHostMaintenance": "MIGRATE",
            "preemptible": False,
            "provisioningModel": "STANDARD",
        },
        "shieldedInstanceConfig": {
            "enableSecureBoot": True,
            "enableVtpm": True,
            "enableIntegrityMonitoring": True,
        },
    }


def _disk(size_gb: int):
    return {
        "id": host_storage.DISK_ID,
        "name": "muncho-canary-v2-01",
        "sizeGb": str(size_gb),
        "status": "READY",
        "zone": (
            "https://www.googleapis.com/compute/v1/projects/"
            "adventico-ai-platform/zones/europe-west3-a"
        ),
        "type": (
            "https://www.googleapis.com/compute/v1/projects/"
            "adventico-ai-platform/zones/europe-west3-a/diskTypes/pd-balanced"
        ),
        "sourceImage": (
            "https://www.googleapis.com/compute/v1/projects/debian-cloud/"
            "global/images/debian-12-bookworm-v20260609"
        ),
        "architecture": "X86_64",
        "physicalBlockSizeBytes": "4096",
        "users": [
            "https://www.googleapis.com/compute/v1/projects/"
            "adventico-ai-platform/zones/europe-west3-a/instances/"
            "muncho-canary-v2-01"
        ],
    }


def _guest(*, size_bytes: int, available_bytes: int):
    return {
        "filesystems": [
            {
                "source": "/dev/sda1",
                "fstype": "ext4",
                "size": size_bytes,
                "avail": available_bytes,
                "target": "/",
            }
        ]
    }


def _evidence(
    *,
    target: bool = False,
    root_size: int | None = None,
    root_available: int | None = None,
    collected_at: int = NOW,
):
    if root_size is None:
        root_size = 42_000_000_000 if target else 20_500_000_000
    if root_available is None:
        root_available = 24_000_000_000 if target else 6_000_000_000
    return {
        "collected_at_unix": collected_at,
        "host_report": _host_report(target=target, collected_at=collected_at - 1),
        "instance": _instance(),
        "disk": _disk(40 if target else 20),
        "guest_root": _guest(
            size_bytes=root_size,
            available_bytes=root_available,
        ),
    }


def _digest_without(value, key):
    unsigned = dict(value)
    del unsigned[key]
    encoded = json.dumps(
        unsigned,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _source_preflight():
    return host_storage_preflight.evaluate(_evidence())


def _target_postflight(*, collected_at: int = NOW + 2):
    return host_storage_preflight.evaluate(
        _evidence(target=True, collected_at=collected_at)
    )


def _transition_postflight(*, collected_at: int = NOW + 2):
    return host_storage_preflight.evaluate(
        _evidence(
            target=True,
            root_size=20_500_000_000,
            root_available=6_000_000_000,
            collected_at=collected_at,
        )
    )


def _safe_service_states():
    states = []
    for unit in host_storage_boot.CANARY_RUNTIME_UNITS:
        states.append({
            "unit": unit,
            "state": "absent",
            "properties": {
                "LoadState": "not-found",
                "ActiveState": "inactive",
                "SubState": "dead",
                "UnitFileState": "",
                "MainPID": "0",
                "FragmentPath": "",
                "DropInPaths": "",
            },
        })
    return states


def _observation(*, status: str, boot: str | None, collected_at: int = NOW + 3):
    instance = _instance()
    instance["status"] = status
    return {
        "collected_at_unix": collected_at,
        "instance": instance,
        "disk": _disk(40),
        "boot_id_sha256": boot,
        "service_states": _safe_service_states() if status == "RUNNING" else None,
    }


def _apply_receipt(*, completed_at: int = NOW + 1):
    plan = host_storage.build_plan()
    return host_storage.execute_plan(
        plan,
        approved_plan_sha256=plan.sha256,
        preflight=_source_preflight(),
        runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, stdout="ok", stderr=""
        ),
        now_unix=completed_at,
    )


def _live_apply_receipt():
    return {
        "approved_plan_sha256": host_storage.build_plan().sha256,
        "completed_at_unix": 1_784_070_197,
        "filesystem_ready": False,
        "mutation_performed": True,
        "ok": True,
        "plan_sha256": host_storage.build_plan().sha256,
        "preflight_report_sha256": (
            "dce04d48ffd5807cf9560868fb9857dae84a12d95daaf7e50bddaefb2a52bd2f"
        ),
        "ready_for_packaging": False,
        "receipt_sha256": (
            "b2ada08f473cf67dee9c738852373d19d9dba473fdc8ee3cdf834f14d951dbd5"
        ),
        "receipts": [
            {
                "name": "resize_canary_boot_disk_to_40gb",
                "result": "resized",
                "returncode": 0,
                "stderr_sha256": (
                    "81a5d969c935e63a8ac3382b6675bb0e113946f06eeb90fe65641d2f07061639"
                ),
                "stdout_sha256": (
                    "98cfd9558ddf036fa13d8c38346f56611f9bb647fd4e8966c142386a3ddb7c6a"
                ),
            }
        ],
        "requires_post_apply_attestation": True,
        "schema": host_storage.STORAGE_APPLY_RECEIPT_SCHEMA,
    }


def _legacy_live_boot_receipt():
    return {
        "completed_at_unix": 1_784_072_450,
        "disk_id": host_storage.DISK_ID,
        "filesystem_ready": False,
        "ok": True,
        "opens_runtime_gate": False,
        "operation_plan_sha256": host_storage.LEGACY_BOOT_OPERATION_PLAN_SHA256,
        "parent_apply_receipt_sha256": _live_apply_receipt()["receipt_sha256"],
        "parent_storage_plan_sha256": host_storage.build_plan().sha256,
        "pre_instance_sha256": (
            "8df6bb89b3d87ba0238a534aa7c594aadefbb0dd42ef8408bd3818f54030cfbe"
        ),
        "receipt_sha256": host_storage.LEGACY_BOOT_RECEIPT_SHA256,
        "requires_storage_postflight": True,
        "schema": host_storage.LEGACY_BOOT_RECEIPT_SCHEMA,
        "secret_material_recorded": False,
        "start_receipt": {
            "returncode": 0,
            "stderr_sha256": (
                "d6c3de201ac1776891323a593a476b026fe93fac291cc6fb1917ebd740854494"
            ),
            "stdout_sha256": (
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            ),
        },
        "started_instance_sha256": (
            "8df6bb89b3d87ba0238a534aa7c594aadefbb0dd42ef8408bd3818f54030cfbe"
        ),
        "stop_receipt": {
            "returncode": 0,
            "stderr_sha256": (
                "f1bbc7511b7dfb0de33b00ae804679903c19ac3fd554123a069c954e6905d732"
            ),
            "stdout_sha256": (
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            ),
        },
        "stopped_instance_sha256": (
            "889e715a6719980ae22758c061bf99ed753419c4a4dd7a2fb68bce8c1f24c608"
        ),
        "vm_instance_id": host_storage.VM_INSTANCE_ID,
    }


def _legacy_live_readiness_receipt():
    return {
        "apply_receipt_sha256": _live_apply_receipt()["receipt_sha256"],
        "attested_at_unix": 1_784_072_692,
        "disk_id": host_storage.DISK_ID,
        "disk_size_gb": 40,
        "filesystem_ready": True,
        "ok": True,
        "opens_runtime_gate": False,
        "plan_sha256": host_storage.build_plan().sha256,
        "postflight_report_sha256": (
            "df16b507ccd988f29bfe852d4731e3a7827564fc421a0f767196567f8b88391c"
        ),
        "ready_for_packaging": True,
        "receipt_sha256": host_storage.LEGACY_READINESS_RECEIPT_SHA256,
        "root_available_bytes": 26_518_962_176,
        "root_filesystem": "ext4",
        "root_mountpoint": "/",
        "root_size_bytes": 42_025_213_952,
        "root_source": "/dev/sda1",
        "schema": host_storage.LEGACY_READINESS_RECEIPT_SCHEMA,
        "vm_instance_id": host_storage.VM_INSTANCE_ID,
    }


def _boot_artifacts(apply_receipt, journal, *, completed_at: int = NOW + 3):
    prior = "1" * 64
    current = "2" * 64
    boot_plan = host_storage_boot.build_plan()
    preflight = host_storage_boot.build_preflight(
        storage_apply_receipt=apply_receipt,
        storage_report=_transition_postflight(collected_at=completed_at - 1),
        prior_boot_id_sha256=prior,
        service_states=_safe_service_states(),
        now_unix=completed_at - 1,
    )
    states = iter([("RUNNING", prior), ("TERMINATED", None), ("RUNNING", current)])
    timeline = {"now": completed_at}

    def observer():
        timeline["now"] += 1
        status, boot = next(states)
        return _observation(
            status=status,
            boot=boot,
            collected_at=timeline["now"],
        )

    def runner(argv):
        timeline["now"] += 1
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    receipt = host_storage_boot.execute_plan(
        boot_plan,
        approved_plan_sha256=boot_plan.sha256,
        preflight=preflight,
        observer=observer,
        runner=runner,
        journal=journal,
        clock=lambda: timeline["now"],
    )
    return preflight, receipt


class _BootRuntime:
    def __init__(self, *, now: int, prior_boot: str = "1" * 64):
        self.now = now
        self.status = "RUNNING"
        self.boot = prior_boot
        self.prior_boot = prior_boot
        self.called = []

    def clock(self):
        return self.now

    def observe(self):
        self.now += 1
        return _observation(
            status=self.status,
            boot=self.boot if self.status == "RUNNING" else None,
            collected_at=self.now,
        )

    def run(self, argv):
        self.now += 2
        self.called.append(tuple(argv))
        action = argv[3]
        if action == "stop":
            self.status = "TERMINATED"
            self.boot = None
        elif action == "start":
            self.status = "RUNNING"
            self.boot = "2" * 64
        else:
            raise AssertionError(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")


def _journaled_boot_preflight():
    apply_receipt = _apply_receipt()
    return host_storage_boot.build_preflight(
        storage_apply_receipt=apply_receipt,
        storage_report=_transition_postflight(),
        prior_boot_id_sha256="1" * 64,
        service_states=_safe_service_states(),
        now_unix=NOW + 2,
    )


def test_canonical_host_and_storage_plan_target_exact_40gb_only():
    plan = host_storage.build_plan()

    assert plan.spec.source_size_gb == 20
    assert plan.spec.target_size_gb == 40
    assert len(plan.steps) == 1
    assert plan.steps[0].name == "resize_canary_boot_disk_to_40gb"
    assert plan.steps[0].argv == (
        "gcloud",
        "compute",
        "disks",
        "resize",
        "muncho-canary-v2-01",
        "--project=adventico-ai-platform",
        "--zone=europe-west3-a",
        "--account=lomliev@adventico.com",
        "--size=40GB",
        "--quiet",
    )
    rendered = json.dumps(plan.report(), sort_keys=True)
    assert "instances create" not in rendered
    assert "ssh" not in rendered
    assert "snapshot" not in " ".join(plan.steps[0].argv)
    assert "iam" not in " ".join(plan.steps[0].argv)
    assert plan.architecture["mutates_guest"] is False
    assert plan.architecture["opens_runtime_gate"] is False


def test_transitional_20gb_source_is_exact_and_digest_sealed():
    report = _source_preflight()

    assert report["ok"] is True
    assert report["state"] == "source_ready"
    assert report["disk_size_gb"] == 20
    assert report["normal_host_preflight_ok"] is False
    assert report["satisfied_steps"] == []
    assert report["ready_for_packaging"] is False
    assert report["report_sha256"] == _digest_without(report, "report_sha256")
    assert {check["name"] for check in report["checks"]} == (
        host_storage_preflight._STORAGE_CHECK_NAMES
    )
    assert all(check["passed"] for check in report["checks"])


def test_findmnt_decimal_strings_are_normalized_before_receipt_sealing():
    evidence = _evidence()
    evidence["guest_root"]["filesystems"][0]["size"] = "20500000000"
    evidence["guest_root"]["filesystems"][0]["avail"] = "6000000000"

    report = host_storage_preflight.evaluate(evidence)

    assert report["ok"] is True
    assert report["root_size_bytes"] == 20_500_000_000
    assert report["root_available_bytes"] == 6_000_000_000


def test_source_requires_normal_host_failure_only_for_40gb_target_shape():
    evidence = _evidence()
    evidence["host_report"]["checks"][0]["passed"] = False

    report = host_storage_preflight.evaluate(evidence)

    assert report["ok"] is False
    assert report["state"] == "invalid"
    assert report["checks"][0]["name"] == "host.lifecycle_state_exact"
    assert report["checks"][0]["passed"] is False


def test_source_transition_requires_every_non_disk_host_field_exact():
    evidence = _evidence()
    evidence["instance"]["networkInterfaces"][0]["network"] = (
        "https://www.googleapis.com/compute/v1/projects/"
        "adventico-ai-platform/global/networks/ai-platform-vpc"
    )

    report = host_storage_preflight.evaluate(evidence)

    assert report["ok"] is False
    assert report["state"] == "invalid"
    failed = {item["name"] for item in report["checks"] if not item["passed"]}
    assert failed == {"host.lifecycle_state_exact"}


def test_target_is_not_ready_until_normal_host_and_guest_capacity_pass():
    evidence = _evidence(
        target=True,
        root_size=20_500_000_000,
        root_available=9_000_000_000,
    )

    report = host_storage_preflight.evaluate(evidence)

    assert report["ok"] is False
    assert report["state"] == "transition_pending"
    failed = {item["name"] for item in report["checks"] if not item["passed"]}
    assert failed == {"filesystem.capacity_matches_disk_state"}


def test_target_requires_eight_gibibytes_of_packaging_headroom():
    report = host_storage_preflight.evaluate(
        _evidence(
            target=True,
            root_available=host_storage.MINIMUM_PACKAGING_FREE_BYTES - 1,
        )
    )

    assert report["ok"] is False
    assert report["state"] == "transition_pending"
    failed = {item["name"] for item in report["checks"] if not item["passed"]}
    assert failed == {"filesystem.packaging_headroom_if_target"}


def test_replacement_instance_or_disk_fails_closed():
    for field, value in (("instance", "999"), ("disk", "888")):
        evidence = _evidence()
        evidence[field]["id"] = value

        report = host_storage_preflight.evaluate(evidence)

        assert report["ok"] is False
        assert report["state"] == "invalid"


def test_apply_requires_exact_approved_digest_and_never_calls_runner_on_drift():
    plan = host_storage.build_plan()
    called = []

    with pytest.raises(RuntimeError, match="approved storage plan"):
        host_storage.execute_plan(
            plan,
            approved_plan_sha256="0" * 64,
            preflight=_source_preflight(),
            runner=lambda argv: called.append(argv),
            now_unix=NOW,
        )

    assert called == []


def test_apply_runs_only_exact_resize_and_returns_hash_only_receipt():
    plan = host_storage.build_plan()
    called = []

    def runner(argv):
        called.append(tuple(argv))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="provider-success-with-sensitive-context",
            stderr="provider-warning-with-sensitive-context",
        )

    receipt = host_storage.execute_plan(
        plan,
        approved_plan_sha256=plan.sha256,
        preflight=_source_preflight(),
        runner=runner,
        now_unix=NOW + 1,
    )

    assert called == [plan.steps[0].argv]
    assert receipt["ok"] is True
    assert receipt["mutation_performed"] is True
    assert receipt["filesystem_ready"] is False
    assert receipt["ready_for_packaging"] is False
    assert receipt["requires_post_apply_attestation"] is True
    assert receipt["receipt_sha256"] == _digest_without(receipt, "receipt_sha256")
    rendered = json.dumps(receipt)
    assert "provider-success" not in rendered
    assert "provider-warning" not in rendered


def test_failed_resize_returns_terminal_failure_without_readiness_claim():
    plan = host_storage.build_plan()

    receipt = host_storage.execute_plan(
        plan,
        approved_plan_sha256=plan.sha256,
        preflight=_source_preflight(),
        runner=lambda argv: subprocess.CompletedProcess(
            argv, 1, stdout="failure", stderr="denied"
        ),
        now_unix=NOW + 1,
    )

    assert receipt["ok"] is False
    assert receipt["requires_post_apply_attestation"] is False
    assert receipt["filesystem_ready"] is False
    assert receipt["ready_for_packaging"] is False


def test_default_resize_runner_uses_only_pinned_gcloud_and_closed_environment(
    monkeypatch,
):
    from scripts.canary import full_canary_owner_launcher as owner_launcher

    calls = []

    class Executable:
        def __init__(self, *, candidates, python_candidates):
            assert candidates == (host_storage.TRUSTED_GCLOUD,)
            assert python_candidates == (host_storage.TRUSTED_PYTHON,)

        @staticmethod
        def trusted_command_prefix():
            calls.append("sdk_attested")
            return ("/trusted/python", "-B", "-I", "/trusted/gcloud.py")

    class Configuration:
        @staticmethod
        def assert_stable():
            calls.append("config_attested")

    class Identity:
        def __init__(self, **kwargs):
            assert isinstance(kwargs["gcloud_executable"], Executable)
            assert isinstance(kwargs["gcloud_configuration"], Configuration)

        @staticmethod
        def account_for_read_only_preflight():
            return host_storage.OWNER_ACCOUNT

        @staticmethod
        def require_stable():
            calls.append("owner_attested")

    monkeypatch.setattr(owner_launcher, "TrustedGcloudExecutable", Executable)
    monkeypatch.setattr(owner_launcher, "PinnedGcloudConfiguration", Configuration)
    monkeypatch.setattr(owner_launcher, "GcloudOwnerAccessToken", Identity)
    monkeypatch.setattr(
        owner_launcher,
        "_owner_gcloud_environment",
        lambda configuration, python: {
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )

    def run(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=b"updated\n", stderr=b"")

    monkeypatch.setattr(host_storage.subprocess, "run", run)
    plan = host_storage.build_plan()
    completed = host_storage._runner(plan.steps[0].argv)

    executed = next(item for item in calls if isinstance(item, tuple))
    assert executed[0] == (
        "/trusted/python",
        "-B",
        "-I",
        "/trusted/gcloud.py",
        *plan.steps[0].argv[1:],
    )
    assert executed[1]["env"] == {
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    assert executed[1]["shell"] is False
    assert completed.stdout == "updated\n"
    assert completed.stderr == ""
    assert calls.count("sdk_attested") == 2
    assert "config_attested" in calls
    assert "owner_attested" in calls


def test_default_resize_runner_rejects_any_non_plan_argv_before_execution():
    with pytest.raises(RuntimeError, match="argv is not exact"):
        host_storage._runner(("gcloud", "compute", "disks", "delete", "wrong"))


def test_tampered_preflight_blocks_before_cloud_mutation():
    plan = host_storage.build_plan()
    preflight = _source_preflight()
    preflight["disk_id"] = "8" * 19
    called = []

    with pytest.raises(RuntimeError, match="preflight digest mismatch"):
        host_storage.execute_plan(
            plan,
            approved_plan_sha256=plan.sha256,
            preflight=preflight,
            runner=lambda argv: called.append(argv),
            now_unix=NOW,
        )

    assert called == []


def test_digest_sealed_but_extended_preflight_is_rejected():
    plan = host_storage.build_plan()
    preflight = _source_preflight()
    preflight["unexpected_authority"] = True
    preflight["report_sha256"] = _digest_without(preflight, "report_sha256")

    with pytest.raises(RuntimeError, match="fields are not exact"):
        host_storage.execute_plan(
            plan,
            approved_plan_sha256=plan.sha256,
            preflight=preflight,
            now_unix=NOW,
        )


def test_readiness_binds_fresh_exact_40gb_host_and_guest_postflight(owner_tmp_path):
    plan = host_storage.build_plan()
    apply_receipt = _apply_receipt(completed_at=NOW + 1)
    journal = host_storage_boot_journal.StorageBootJournal._for_test(
        owner_tmp_path / "j"
    )
    boot_preflight, boot_receipt = _boot_artifacts(
        apply_receipt, journal, completed_at=NOW + 3
    )
    postflight = _target_postflight(collected_at=NOW + 9)

    readiness = host_storage.build_readiness_receipt(
        plan,
        apply_receipt=apply_receipt,
        postflight=postflight,
        boot_preflight=boot_preflight,
        boot_receipt=boot_receipt,
        now_unix=NOW + 10,
    )

    assert readiness["ok"] is True
    assert readiness["disk_size_gb"] == 40
    assert readiness["filesystem_ready"] is True
    assert readiness["ready_for_packaging"] is True
    assert readiness["boot_expansion_required"] is True
    assert readiness["boot_receipt_sha256"] == boot_receipt["receipt_sha256"]
    assert readiness["opens_runtime_gate"] is False
    assert readiness["apply_receipt_sha256"] == apply_receipt["receipt_sha256"]
    assert readiness["postflight_report_sha256"] == postflight["report_sha256"]
    assert readiness["receipt_sha256"] == _digest_without(readiness, "receipt_sha256")


def test_readiness_rejects_postflight_collected_before_apply(owner_tmp_path):
    plan = host_storage.build_plan()
    apply_receipt = _apply_receipt(completed_at=NOW + 2)
    journal = host_storage_boot_journal.StorageBootJournal._for_test(
        owner_tmp_path / "j"
    )
    boot_preflight, boot_receipt = _boot_artifacts(
        apply_receipt, journal, completed_at=NOW + 4
    )

    with pytest.raises(RuntimeError, match="postflight"):
        host_storage.build_readiness_receipt(
            plan,
            apply_receipt=apply_receipt,
            postflight=_target_postflight(collected_at=NOW + 1),
            boot_preflight=boot_preflight,
            boot_receipt=boot_receipt,
            now_unix=NOW + 10,
        )


def test_readiness_rejects_digest_sealed_extended_apply_receipt():
    plan = host_storage.build_plan()
    apply_receipt = host_storage.execute_plan(
        plan,
        approved_plan_sha256=plan.sha256,
        preflight=_source_preflight(),
        runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, stdout="ok", stderr=""
        ),
        now_unix=NOW + 1,
    )
    apply_receipt["unexpected_authority"] = True
    apply_receipt["receipt_sha256"] = _digest_without(apply_receipt, "receipt_sha256")

    with pytest.raises(RuntimeError, match="fields are not exact"):
        host_storage.build_readiness_receipt(
            plan,
            apply_receipt=apply_receipt,
            postflight=_target_postflight(collected_at=NOW + 2),
            now_unix=NOW + 3,
        )


def test_exact_existing_target_is_idempotent_but_still_requires_new_postflight():
    plan = host_storage.build_plan()
    called = []
    target = _target_postflight(collected_at=NOW)

    receipt = host_storage.execute_plan(
        plan,
        approved_plan_sha256=plan.sha256,
        preflight=target,
        runner=lambda argv: called.append(argv),
        now_unix=NOW,
    )

    assert called == []
    assert receipt["ok"] is True
    assert receipt["mutation_performed"] is False
    assert receipt["receipts"] == [
        {
            "name": "resize_canary_boot_disk_to_40gb",
            "result": "verified_existing",
        }
    ]
    assert receipt["requires_post_apply_attestation"] is True


def test_guest_collector_command_is_fixed_read_only_and_hardened():
    command = host_storage_preflight._guest_command()

    assert command == (
        "/usr/bin/findmnt",
        "--json",
        "--bytes",
        "--output=SOURCE,FSTYPE,SIZE,AVAIL,TARGET",
        "/",
    )


def test_guest_read_reuses_sealed_iap_transport_and_existing_owner_profile():
    calls = []

    class Identity:
        @staticmethod
        def account_for_read_only_preflight():
            return "lomliev@adventico.com"

    class Transport:
        @staticmethod
        def _run_remote(argv, **kwargs):
            calls.append((tuple(argv), kwargs))
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=json.dumps(
                    _guest(
                        size_bytes=20_500_000_000,
                        available_bytes=6_000_000_000,
                    )
                ).encode(),
                stderr=b"",
            )

    value = host_storage_preflight._run_guest_json(
        host_storage_preflight._guest_command(),
        owner_identity=Identity(),
        transport=Transport(),
    )

    assert value == _guest(
        size_bytes=20_500_000_000,
        available_bytes=6_000_000_000,
    )
    assert calls == [
        (
            host_storage_preflight._guest_command(),
            {
                "account": "lomliev@adventico.com",
                "timeout_seconds": 90.0,
                "maximum_output_bytes": 64 * 1024,
            },
        )
    ]


def test_collector_reuses_host_read_only_inventory_and_fixed_guest_runner(monkeypatch):
    host_evidence = {
        "planned_vm": _instance(),
        "planned_vm_disk": _disk(20),
    }
    seen = []
    monkeypatch.setattr(
        host_storage_preflight,
        "collect_host",
        lambda *, run_json: seen.append(("host", run_json)) or host_evidence,
    )
    monkeypatch.setattr(
        host_storage_preflight,
        "evaluate_host",
        lambda value: _host_report(target=False),
    )

    def cloud_runner(argv):
        raise AssertionError(f"unexpected direct cloud command: {argv}")

    def guest_runner(argv):
        seen.append(("guest", tuple(argv)))
        return _guest(size_bytes=20_500_000_000, available_bytes=6_000_000_000)

    evidence = host_storage_preflight.collect(
        run_json=cloud_runner,
        run_guest_json=guest_runner,
    )

    assert seen == [
        ("host", cloud_runner),
        ("guest", host_storage_preflight._guest_command()),
    ]
    assert evidence["instance"] == _instance()
    assert evidence["disk"] == _disk(20)


def test_storage_boot_plan_is_exact_and_has_no_guest_or_shell_authority():
    plan = host_storage_boot.build_plan()

    assert plan.storage_plan_sha256 == host_storage.build_plan().sha256
    assert [step.name for step in plan.steps] == [
        host_storage_boot.STOP_STEP,
        host_storage_boot.START_STEP,
    ]
    assert [step.argv[3] for step in plan.steps] == ["stop", "start"]
    assert all(step.argv[4] == "muncho-canary-v2-01" for step in plan.steps)
    rendered = json.dumps(plan.report(), sort_keys=True)
    assert "ssh" not in rendered
    assert "systemctl start" not in rendered
    assert "growpart" not in rendered
    assert "resize2fs" not in rendered
    assert plan.architecture["guest_command_authority"] is False
    assert plan.architecture["shell_authority"] is False
    assert plan.architecture["opens_runtime_gate"] is False

    assert {
        "hermes-cloud-gateway.service",
        "muncho-canonical-writer-phase-b-readiness.service",
        "muncho-canonical-writer.service",
        "muncho-isolated-worker.socket",
        "muncho-isolated-worker.service",
        "muncho-capability-browser.service",
        "muncho-discord-connector.service",
        "muncho-discord-egress.service",
        "muncho-mac-ops-edge.service",
    } <= set(host_storage_boot.CANARY_RUNTIME_UNITS)


def test_guest_state_normalizes_only_exact_processless_pid_omissions():
    def service_stdout(unit, *, omit_pid):
        values = {
            "LoadState": "not-found",
            "ActiveState": "inactive",
            "SubState": "dead",
            "UnitFileState": "",
            "MainPID": "0",
            "FragmentPath": "",
            "DropInPaths": "",
        }
        if omit_pid:
            values.pop("MainPID")
        return "".join(
            f"{name}={values[name]}\n"
            for name in host_storage_boot._SERVICE_PROPERTIES
            if name in values
        ).encode()

    class Transport:
        def __init__(self, *, omit_service_pid=False):
            self.omit_service_pid = omit_service_pid

        def _run_remote(self, argv, **_kwargs):
            if tuple(argv) == host_storage_boot.BOOT_ID_COMMAND:
                stdout = b"c535e272-8c19-4b4b-8287-b5c4ad354ab1\n"
            else:
                unit = argv[-1]
                processless = unit in {
                    "muncho-canonical-writer-export.timer",
                    "muncho-isolated-worker.socket",
                }
                stdout = service_stdout(
                    unit,
                    omit_pid=processless or self.omit_service_pid,
                )
            return subprocess.CompletedProcess(argv, 0, stdout, b"")

    _boot, states = host_storage_boot._collect_guest_state(
        Transport(),
        account="owner@example.com",
    )
    by_unit = {item["unit"]: item for item in states}
    assert by_unit["muncho-canonical-writer-export.timer"]["properties"][
        "MainPID"
    ] == "0"
    assert by_unit["muncho-isolated-worker.socket"]["properties"]["MainPID"] == "0"

    with pytest.raises(RuntimeError, match="incomplete"):
        host_storage_boot._collect_guest_state(
            Transport(omit_service_pid=True),
            account="owner@example.com",
        )


def test_storage_boot_runner_rejects_every_non_plan_mutation():
    with pytest.raises(RuntimeError, match="argv is not exact"):
        host_storage_boot._runner((
            "gcloud",
            "compute",
            "instances",
            "reset",
            "muncho-canary-v2-01",
        ))


def test_storage_boot_runner_uses_only_sealed_owner_gcloud(monkeypatch):
    from scripts.canary import full_canary_owner_launcher as owner_launcher

    attestations = []

    class Executable:
        def __init__(self, *, candidates, python_candidates):
            assert candidates == (host_storage_boot.TRUSTED_GCLOUD,)
            assert python_candidates == (host_storage_boot.TRUSTED_PYTHON,)

        @staticmethod
        def trusted_command_prefix():
            attestations.append("sdk")
            return ("/trusted/python", "-B", "-I", "/trusted/gcloud.py")

    class Configuration:
        @staticmethod
        def assert_stable():
            attestations.append("config")

    class Identity:
        def __init__(self, **kwargs):
            assert isinstance(kwargs["gcloud_executable"], Executable)
            assert isinstance(kwargs["gcloud_configuration"], Configuration)

        @staticmethod
        def account_for_read_only_preflight():
            return host_storage_boot.OWNER_ACCOUNT

        @staticmethod
        def require_stable():
            attestations.append("owner")

    monkeypatch.setattr(owner_launcher, "TrustedGcloudExecutable", Executable)
    monkeypatch.setattr(owner_launcher, "PinnedGcloudConfiguration", Configuration)
    monkeypatch.setattr(owner_launcher, "GcloudOwnerAccessToken", Identity)
    monkeypatch.setattr(
        owner_launcher,
        "_owner_gcloud_environment",
        lambda configuration, python: {"PATH": "/usr/bin:/bin"},
    )
    calls = []

    def run(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=b"done\n", stderr=b"")

    monkeypatch.setattr(host_storage_boot.subprocess, "run", run)
    step = host_storage_boot.build_plan().steps[0]
    completed = host_storage_boot._runner(step.argv)

    assert calls[0][0] == (
        "/trusted/python",
        "-B",
        "-I",
        "/trusted/gcloud.py",
        *step.argv[1:],
    )
    assert calls[0][1]["env"] == {"PATH": "/usr/bin:/bin"}
    assert calls[0][1]["shell"] is False
    assert completed.stdout == "done\n"
    assert attestations.count("sdk") == 2
    assert {"config", "owner"} <= set(attestations)


def test_storage_boot_preflight_binds_resize_and_stopped_runtime_inventory():
    apply_receipt = _apply_receipt()
    report = host_storage_boot.build_preflight(
        storage_apply_receipt=apply_receipt,
        storage_report=_transition_postflight(),
        prior_boot_id_sha256="1" * 64,
        service_states=_safe_service_states(),
        now_unix=NOW + 2,
    )

    assert report["ok"] is True
    assert report["state"] == "boot_required"
    assert report["storage_apply_receipt_sha256"] == apply_receipt["receipt_sha256"]
    assert report["disk_size_gb"] == 40
    assert report["runtime_units"] == list(host_storage_boot.CANARY_RUNTIME_UNITS)
    assert all(item["passed"] for item in report["checks"])
    assert report["report_sha256"] == _digest_without(report, "report_sha256")


def test_storage_boot_preflight_rejects_active_or_enabled_runtime_unit():
    apply_receipt = _apply_receipt()
    states = _safe_service_states()
    states[0] = {
        "unit": states[0]["unit"],
        "state": "disabled_inactive",
        "properties": {
            "LoadState": "loaded",
            "ActiveState": "active",
            "SubState": "running",
            "UnitFileState": "enabled",
            "MainPID": "42",
            "FragmentPath": f"/etc/systemd/system/{states[0]['unit']}",
            "DropInPaths": "",
        },
    }

    report = host_storage_boot.build_preflight(
        storage_apply_receipt=apply_receipt,
        storage_report=_transition_postflight(),
        prior_boot_id_sha256="1" * 64,
        service_states=states,
        now_unix=NOW + 2,
    )

    assert report["ok"] is False
    assert report["state"] == "invalid"
    failed = {item["name"] for item in report["checks"] if not item["passed"]}
    assert failed == {
        "runtime.units_inventory_exact",
        "runtime.units_stopped_exact",
    }


def test_storage_boot_executes_only_stop_start_and_validates_receipt(owner_tmp_path):
    apply_receipt = _apply_receipt()
    journal = host_storage_boot_journal.StorageBootJournal._for_test(
        owner_tmp_path / "j"
    )
    preflight, receipt = _boot_artifacts(apply_receipt, journal)
    plan = host_storage_boot.build_plan()

    validated = host_storage_boot.validate_receipt(
        plan=plan,
        preflight=preflight,
        receipt=receipt,
        now_unix=NOW + 20,
        journal=journal,
    )

    assert validated == receipt
    assert receipt["execution_state"] == "stop_start_completed"
    assert receipt["mutation_performed"] is True
    assert [item["name"] for item in receipt["receipts"]] == [
        host_storage_boot.STOP_STEP,
        host_storage_boot.START_STEP,
    ]
    assert receipt["prior_boot_id_sha256"] != receipt["current_boot_id_sha256"]
    assert receipt["requires_storage_postflight"] is True
    assert receipt["opens_runtime_gate"] is False


def test_storage_boot_unrelated_terminated_state_has_no_recovery_authority(
    owner_tmp_path,
):
    apply_receipt = _apply_receipt()
    prior = "1" * 64
    plan = host_storage_boot.build_plan()
    preflight = host_storage_boot.build_preflight(
        storage_apply_receipt=apply_receipt,
        storage_report=_transition_postflight(),
        prior_boot_id_sha256=prior,
        service_states=_safe_service_states(),
        now_unix=NOW + 2,
    )
    called = []
    journal = host_storage_boot_journal.StorageBootJournal._for_test(
        owner_tmp_path / "j"
    )

    with pytest.raises(RuntimeError, match="durable pre-stop intent"):
        host_storage_boot.execute_plan(
            plan,
            approved_plan_sha256=plan.sha256,
            preflight=preflight,
            observer=lambda: _observation(
                status="TERMINATED", boot=None, collected_at=NOW + 3
            ),
            runner=lambda argv: called.append(tuple(argv)),
            journal=journal,
            now_unix=NOW + 3,
        )

    assert called == []


def test_storage_boot_unrelated_new_boot_has_no_attestation_authority(
    owner_tmp_path,
):
    apply_receipt = _apply_receipt()
    plan = host_storage_boot.build_plan()
    preflight = host_storage_boot.build_preflight(
        storage_apply_receipt=apply_receipt,
        storage_report=_transition_postflight(),
        prior_boot_id_sha256="1" * 64,
        service_states=_safe_service_states(),
        now_unix=NOW + 2,
    )
    called = []
    journal = host_storage_boot_journal.StorageBootJournal._for_test(
        owner_tmp_path / "j"
    )

    with pytest.raises(RuntimeError, match="durable pre-stop intent"):
        host_storage_boot.execute_plan(
            plan,
            approved_plan_sha256=plan.sha256,
            preflight=preflight,
            observer=lambda: _observation(
                status="RUNNING", boot="2" * 64, collected_at=NOW + 3
            ),
            runner=lambda argv: called.append(tuple(argv)),
            journal=journal,
            now_unix=NOW + 3,
        )

    assert called == []


def test_storage_boot_same_boot_requires_fresh_preflight_before_stop(owner_tmp_path):
    apply_receipt = _apply_receipt()
    plan = host_storage_boot.build_plan()
    preflight = host_storage_boot.build_preflight(
        storage_apply_receipt=apply_receipt,
        storage_report=_transition_postflight(),
        prior_boot_id_sha256="1" * 64,
        service_states=_safe_service_states(),
        now_unix=NOW + 2,
    )
    called = []
    journal = host_storage_boot_journal.StorageBootJournal._for_test(
        owner_tmp_path / "j"
    )

    with pytest.raises(RuntimeError, match="stale|preflight"):
        host_storage_boot.execute_plan(
            plan,
            approved_plan_sha256=plan.sha256,
            preflight=preflight,
            observer=lambda: _observation(status="RUNNING", boot="1" * 64),
            runner=lambda argv: called.append(tuple(argv)),
            journal=journal,
            now_unix=NOW + 10_000,
        )

    assert called == []


def test_storage_boot_rechecks_exact_vm_and_disk_before_any_mutation(
    owner_tmp_path,
):
    apply_receipt = _apply_receipt()
    plan = host_storage_boot.build_plan()
    preflight = host_storage_boot.build_preflight(
        storage_apply_receipt=apply_receipt,
        storage_report=_transition_postflight(),
        prior_boot_id_sha256="1" * 64,
        service_states=_safe_service_states(),
        now_unix=NOW + 2,
    )
    wrong = _observation(status="RUNNING", boot="1" * 64)
    wrong["disk"]["id"] = "9" * 19
    called = []
    journal = host_storage_boot_journal.StorageBootJournal._for_test(
        owner_tmp_path / "j"
    )

    with pytest.raises(RuntimeError, match="identity is not exact"):
        host_storage_boot.execute_plan(
            plan,
            approved_plan_sha256=plan.sha256,
            preflight=preflight,
            observer=lambda: wrong,
            runner=lambda argv: called.append(tuple(argv)),
            journal=journal,
            now_unix=NOW + 3,
        )

    assert called == []


def test_storage_boot_rechecks_service_state_digest_before_publishing_intent(
    owner_tmp_path,
):
    plan = host_storage_boot.build_plan()
    preflight = _journaled_boot_preflight()
    journal = host_storage_boot_journal.StorageBootJournal._for_test(
        owner_tmp_path / "j"
    )
    live = _observation(
        status="RUNNING",
        boot="1" * 64,
        collected_at=NOW + 4,
    )
    unit = host_storage_boot.CANARY_RUNTIME_UNITS[0]
    live["service_states"][0] = {
        "unit": unit,
        "state": "disabled_inactive",
        "properties": {
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "SubState": "dead",
            "UnitFileState": "disabled",
            "MainPID": "0",
            "FragmentPath": f"/etc/systemd/system/{unit}",
            "DropInPaths": "",
        },
    }
    called = []

    with pytest.raises(RuntimeError, match="service state diverged"):
        host_storage_boot.execute_plan(
            plan,
            approved_plan_sha256=plan.sha256,
            preflight=preflight,
            observer=lambda: live,
            runner=lambda argv: called.append(tuple(argv)),
            journal=journal,
            now_unix=NOW + 4,
        )

    transaction_id = host_storage_boot._transaction_id(plan, preflight)
    assert journal.read(transaction_id, "intent") is None
    assert called == []


def test_storage_readiness_fails_closed_without_required_boot_receipt():
    plan = host_storage.build_plan()
    apply_receipt = _apply_receipt()

    with pytest.raises(RuntimeError, match="requires the exact boot receipt"):
        host_storage.build_readiness_receipt(
            plan,
            apply_receipt=apply_receipt,
            postflight=_target_postflight(collected_at=NOW + 4),
            now_unix=NOW + 5,
        )


def test_existing_target_path_needs_no_boot_and_remains_idempotent():
    plan = host_storage.build_plan()
    target = _target_postflight(collected_at=NOW)
    apply_receipt = host_storage.execute_plan(
        plan,
        approved_plan_sha256=plan.sha256,
        preflight=target,
        runner=lambda argv: (_ for _ in ()).throw(AssertionError(argv)),
        now_unix=NOW,
    )
    postflight = _target_postflight(collected_at=NOW + 1)

    readiness = host_storage.build_readiness_receipt(
        plan,
        apply_receipt=apply_receipt,
        postflight=postflight,
        now_unix=NOW + 2,
    )

    assert readiness["boot_expansion_required"] is False
    assert readiness["boot_plan_sha256"] is None
    assert readiness["boot_preflight_report_sha256"] is None
    assert readiness["boot_receipt_sha256"] is None


@pytest.mark.parametrize(
    "crash_checkpoint",
    [
        "intent_published",
        "stop_command_completed",
        "stop_recorded",
        "start_command_completed",
        "start_recorded",
    ],
)
def test_storage_boot_signal_recovery_requires_and_reuses_durable_intent(
    owner_tmp_path, crash_checkpoint
):
    plan = host_storage_boot.build_plan()
    preflight = _journaled_boot_preflight()
    journal = host_storage_boot_journal.StorageBootJournal._for_test(
        owner_tmp_path / "j"
    )
    runtime = _BootRuntime(now=NOW + 3)
    crashed = {"done": False}

    def checkpoint(name):
        if name == crash_checkpoint and not crashed["done"]:
            crashed["done"] = True
            raise KeyboardInterrupt(name)

    with pytest.raises(KeyboardInterrupt, match=crash_checkpoint):
        host_storage_boot.execute_plan(
            plan,
            approved_plan_sha256=plan.sha256,
            preflight=preflight,
            observer=runtime.observe,
            runner=runtime.run,
            journal=journal,
            clock=runtime.clock,
            checkpoint=checkpoint,
        )

    transaction_id = host_storage_boot._transaction_id(plan, preflight)
    intent = journal.read(transaction_id, "intent")
    assert intent is not None
    assert intent["state"] == "intent_published_before_stop"
    assert journal.read(transaction_id, "completion") is None

    receipt = host_storage_boot.execute_plan(
        plan,
        approved_plan_sha256=plan.sha256,
        preflight=preflight,
        observer=runtime.observe,
        runner=runtime.run,
        journal=journal,
        clock=runtime.clock,
    )

    assert receipt["ok"] is True
    assert receipt["transaction_id"] == transaction_id
    assert receipt["journal_intent_sha256"] == intent["intent_sha256"]
    assert (
        receipt["started_observation_collected_at_unix"]
        > receipt["initial_observation_collected_at_unix"]
    )
    if receipt["stopped_observation_collected_at_unix"] is not None:
        assert (
            receipt["initial_observation_collected_at_unix"]
            < receipt["stopped_observation_collected_at_unix"]
            < receipt["started_observation_collected_at_unix"]
        )

    called = list(runtime.called)
    replay = host_storage_boot.execute_plan(
        plan,
        approved_plan_sha256=plan.sha256,
        preflight=preflight,
        observer=lambda: (_ for _ in ()).throw(AssertionError("no observation")),
        runner=lambda argv: (_ for _ in ()).throw(AssertionError(argv)),
        journal=journal,
        clock=runtime.clock,
    )
    assert replay == receipt
    assert runtime.called == called


def test_storage_boot_replays_completion_published_before_signal(owner_tmp_path):
    plan = host_storage_boot.build_plan()
    preflight = _journaled_boot_preflight()
    journal = host_storage_boot_journal.StorageBootJournal._for_test(
        owner_tmp_path / "j"
    )
    runtime = _BootRuntime(now=NOW + 3)

    with pytest.raises(KeyboardInterrupt, match="completion_recorded"):
        host_storage_boot.execute_plan(
            plan,
            approved_plan_sha256=plan.sha256,
            preflight=preflight,
            observer=runtime.observe,
            runner=runtime.run,
            journal=journal,
            clock=runtime.clock,
            checkpoint=lambda name: (
                (_ for _ in ()).throw(KeyboardInterrupt(name))
                if name == "completion_recorded"
                else None
            ),
        )

    transaction_id = host_storage_boot._transaction_id(plan, preflight)
    durable = journal.read(transaction_id, "completion")
    assert durable is not None
    replay = host_storage_boot.execute_plan(
        plan,
        approved_plan_sha256=plan.sha256,
        preflight=preflight,
        observer=lambda: (_ for _ in ()).throw(AssertionError("no observation")),
        runner=lambda argv: (_ for _ in ()).throw(AssertionError(argv)),
        journal=journal,
        clock=runtime.clock,
    )
    assert replay == durable


def test_storage_boot_rejects_live_boot_drift_after_durable_start(owner_tmp_path):
    plan = host_storage_boot.build_plan()
    preflight = _journaled_boot_preflight()
    journal = host_storage_boot_journal.StorageBootJournal._for_test(
        owner_tmp_path / "j"
    )
    runtime = _BootRuntime(now=NOW + 3)

    with pytest.raises(KeyboardInterrupt, match="start_recorded"):
        host_storage_boot.execute_plan(
            plan,
            approved_plan_sha256=plan.sha256,
            preflight=preflight,
            observer=runtime.observe,
            runner=runtime.run,
            journal=journal,
            clock=runtime.clock,
            checkpoint=lambda name: (
                (_ for _ in ()).throw(KeyboardInterrupt(name))
                if name == "start_recorded"
                else None
            ),
        )
    runtime.boot = "3" * 64
    called = list(runtime.called)

    with pytest.raises(RuntimeError, match="contradicts live observation"):
        host_storage_boot.execute_plan(
            plan,
            approved_plan_sha256=plan.sha256,
            preflight=preflight,
            observer=runtime.observe,
            runner=runtime.run,
            journal=journal,
            clock=runtime.clock,
        )

    assert runtime.called == called
    transaction_id = host_storage_boot._transaction_id(plan, preflight)
    assert journal.read(transaction_id, "completion") is None


def test_storage_boot_rejects_expired_intent_for_terminated_recovery(
    owner_tmp_path,
):
    plan = host_storage_boot.build_plan()
    preflight = _journaled_boot_preflight()
    journal = host_storage_boot_journal.StorageBootJournal._for_test(
        owner_tmp_path / "j"
    )
    runtime = _BootRuntime(now=NOW + 3)

    with pytest.raises(KeyboardInterrupt):
        host_storage_boot.execute_plan(
            plan,
            approved_plan_sha256=plan.sha256,
            preflight=preflight,
            observer=runtime.observe,
            runner=runtime.run,
            journal=journal,
            clock=runtime.clock,
            checkpoint=lambda name: (
                (_ for _ in ()).throw(KeyboardInterrupt())
                if name == "stop_command_completed"
                else None
            ),
        )
    assert runtime.status == "TERMINATED"
    runtime.now += 60 * 60 + 1
    called = list(runtime.called)

    with pytest.raises(RuntimeError, match="stale or unrelated"):
        host_storage_boot.execute_plan(
            plan,
            approved_plan_sha256=plan.sha256,
            preflight=preflight,
            observer=runtime.observe,
            runner=runtime.run,
            journal=journal,
            clock=runtime.clock,
        )

    assert runtime.called == called
    assert runtime.status == "TERMINATED"


def test_storage_boot_rejects_cached_post_stop_observation(owner_tmp_path):
    plan = host_storage_boot.build_plan()
    preflight = _journaled_boot_preflight()
    journal = host_storage_boot_journal.StorageBootJournal._for_test(
        owner_tmp_path / "j"
    )
    runtime = _BootRuntime(now=NOW + 3)
    first = runtime.observe()
    observations = iter([
        first,
        {
            **_observation(
                status="TERMINATED",
                boot=None,
                collected_at=first["collected_at_unix"],
            )
        },
    ])

    with pytest.raises(RuntimeError, match="stale or out of order"):
        host_storage_boot.execute_plan(
            plan,
            approved_plan_sha256=plan.sha256,
            preflight=preflight,
            observer=lambda: next(observations),
            runner=runtime.run,
            journal=journal,
            clock=runtime.clock,
        )

    assert runtime.called == [plan.steps[0].argv]


def test_storage_boot_rechecks_intent_expiry_before_start_mutation(owner_tmp_path):
    plan = host_storage_boot.build_plan()
    preflight = _journaled_boot_preflight()
    journal = host_storage_boot_journal.StorageBootJournal._for_test(
        owner_tmp_path / "j"
    )
    runtime = _BootRuntime(now=NOW + 3)

    def runner(argv):
        completed = runtime.run(argv)
        if argv == plan.steps[0].argv:
            runtime.now += 60 * 60 + 1
        return completed

    with pytest.raises(RuntimeError, match="stale or unrelated"):
        host_storage_boot.execute_plan(
            plan,
            approved_plan_sha256=plan.sha256,
            preflight=preflight,
            observer=runtime.observe,
            runner=runner,
            journal=journal,
            clock=runtime.clock,
        )

    assert runtime.called == [plan.steps[0].argv]
    assert runtime.status == "TERMINATED"


def test_owner_journal_recovers_atomic_pending_publication(owner_tmp_path, monkeypatch):
    journal = host_storage_boot_journal.StorageBootJournal._for_test(
        owner_tmp_path / "j"
    )
    transaction_id = "a" * 64
    artifact = {"schema": "test", "value": "exact"}
    real_link = host_storage_boot_journal.os.link
    crashed = {"done": False}

    def crash_once(*args, **kwargs):
        if not crashed["done"]:
            crashed["done"] = True
            raise KeyboardInterrupt("link")
        return real_link(*args, **kwargs)

    monkeypatch.setattr(host_storage_boot_journal.os, "link", crash_once)
    with pytest.raises(KeyboardInterrupt, match="link"):
        journal.publish(transaction_id, "intent", artifact)
    monkeypatch.setattr(host_storage_boot_journal.os, "link", real_link)

    assert journal.publish(transaction_id, "intent", artifact) == artifact
    root = owner_tmp_path / "j" / transaction_id
    assert not (root / ".intent.pending").exists()
    assert stat.S_IMODE((owner_tmp_path / "j").stat().st_mode) == 0o700
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "intent.json").stat().st_mode) == 0o600
    assert (root / "intent.json").stat().st_uid == os.geteuid()
    assert (root / "intent.json").stat().st_gid == os.getegid()

    (owner_tmp_path / "j").chmod(0o750)
    with pytest.raises(PermissionError, match="directory is not owner-only"):
        journal.read(transaction_id, "intent")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("initial_observation_sha256", None, "chronology"),
        ("started_observation_sha256", None, "chronology"),
        ("stop_command_completed_at_unix", None, "chronology"),
        ("journal_start_sha256", "not-a-digest", "not exact"),
    ],
)
def test_storage_boot_receipt_rejects_missing_causal_evidence_without_journal(
    owner_tmp_path, field, value, message
):
    apply_receipt = _apply_receipt()
    journal = host_storage_boot_journal.StorageBootJournal._for_test(
        owner_tmp_path / "j"
    )
    preflight, receipt = _boot_artifacts(apply_receipt, journal)
    tampered = dict(receipt)
    tampered[field] = value
    tampered["receipt_sha256"] = _digest_without(tampered, "receipt_sha256")

    with pytest.raises(RuntimeError, match=message):
        host_storage_boot.validate_receipt(
            plan=host_storage_boot.build_plan(),
            preflight=preflight,
            receipt=tampered,
            now_unix=NOW + 20,
        )


def test_storage_boot_journal_transition_requires_exact_result_shape(
    owner_tmp_path,
):
    apply_receipt = _apply_receipt()
    journal = host_storage_boot_journal.StorageBootJournal._for_test(
        owner_tmp_path / "j"
    )
    preflight, receipt = _boot_artifacts(apply_receipt, journal)
    transaction_id = receipt["transaction_id"]
    intent = journal.read(transaction_id, "intent")
    start = journal.read(transaction_id, "start")
    assert intent is not None and start is not None
    start["command_completed_at_unix"] = None
    start["start_sha256"] = _digest_without(start, "start_sha256")

    with pytest.raises(RuntimeError, match="transition is invalid"):
        host_storage_boot._require_transition_record(
            start,
            schema=host_storage_boot.BOOT_EXPANSION_START_SCHEMA,
            digest_key="start_sha256",
            transaction_id=transaction_id,
            intent=intent,
            allowed_results={"started", "verified_new_boot_after_intent"},
        )


def test_exact_legacy_live_reconciliation_mints_v2_without_reboot():
    plan = host_storage.build_plan()
    apply_receipt = _live_apply_receipt()
    fresh_postflight = _target_postflight(collected_at=NOW)

    receipt = host_storage.build_readiness_receipt(
        plan,
        apply_receipt=apply_receipt,
        postflight=fresh_postflight,
        legacy_boot_receipt=_legacy_live_boot_receipt(),
        legacy_readiness_receipt=_legacy_live_readiness_receipt(),
        now_unix=NOW + 1,
    )

    assert receipt["schema"] == host_storage.STORAGE_READINESS_RECEIPT_SCHEMA
    assert receipt["boot_evidence_kind"] == "legacy_live_receipt_reconciliation"
    assert receipt["legacy_boot_receipt_sha256"] == (
        host_storage.LEGACY_BOOT_RECEIPT_SHA256
    )
    assert receipt["legacy_readiness_receipt_sha256"] == (
        host_storage.LEGACY_READINESS_RECEIPT_SHA256
    )
    assert receipt["preboot_boot_id_evidence"] is False
    assert receipt["preboot_service_state_evidence"] is False
    assert receipt["follow_on_stopped_runtime_gate_required"] is True
    assert receipt["opens_runtime_gate"] is False
    assert receipt["postflight_report_sha256"] == fresh_postflight["report_sha256"]


def test_legacy_owner_archive_loader_requires_exact_owner_only_file(
    owner_tmp_path, monkeypatch
):
    archive = owner_tmp_path / "archive"
    archive.mkdir(mode=0o700)
    path = archive / "apply.json"
    value = _live_apply_receipt()
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)
    monkeypatch.setattr(host_storage, "_OWNER_UID", os.geteuid())
    monkeypatch.setattr(host_storage, "_OWNER_GID", os.getegid())

    assert (
        host_storage._load_exact_legacy_owner_receipt(path, expected_path=path) == value
    )
    path.chmod(0o644)
    with pytest.raises(SystemExit, match="owner-only exact"):
        host_storage._load_exact_legacy_owner_receipt(path, expected_path=path)


def test_storage_attest_cli_rejects_mixed_boot_modes_before_loading(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "host_storage",
            "attest",
            "--apply-receipt",
            "/missing/apply.json",
            "--postflight",
            "/missing/postflight.json",
            "--boot-preflight",
            "/missing/boot-preflight.json",
            "--legacy-boot-receipt",
            str(host_storage.LEGACY_BOOT_RECEIPT_PATH),
        ],
    )

    with pytest.raises(SystemExit, match="mutually exclusive"):
        host_storage.main()


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("boot", "operation_plan_sha256", "0" * 64, "not exact"),
        ("boot", "vm_instance_id", "999", "not exact"),
        ("readiness", "root_size_bytes", 20_000_000_000, "not exact"),
        ("readiness", "attested_at_unix", 1_784_000_000, "not exact"),
    ],
)
def test_legacy_reconciliation_rejects_digest_sealed_semantic_drift(
    target, field, value, message
):
    boot = _legacy_live_boot_receipt()
    readiness = _legacy_live_readiness_receipt()
    selected = boot if target == "boot" else readiness
    selected[field] = value
    selected["receipt_sha256"] = _digest_without(selected, "receipt_sha256")

    with pytest.raises(RuntimeError, match=message):
        host_storage.build_readiness_receipt(
            host_storage.build_plan(),
            apply_receipt=_live_apply_receipt(),
            postflight=_target_postflight(collected_at=NOW),
            legacy_boot_receipt=boot,
            legacy_readiness_receipt=readiness,
            now_unix=NOW + 1,
        )


def test_readiness_rejects_mixed_journaled_and_legacy_boot_evidence(
    owner_tmp_path,
):
    apply_receipt = _apply_receipt()
    journal = host_storage_boot_journal.StorageBootJournal._for_test(
        owner_tmp_path / "j"
    )
    preflight, receipt = _boot_artifacts(apply_receipt, journal)

    with pytest.raises(RuntimeError, match="mutually exclusive"):
        host_storage.build_readiness_receipt(
            host_storage.build_plan(),
            apply_receipt=apply_receipt,
            postflight=_target_postflight(collected_at=NOW + 20),
            boot_preflight=preflight,
            boot_receipt=receipt,
            legacy_boot_receipt=_legacy_live_boot_receipt(),
            legacy_readiness_receipt=_legacy_live_readiness_receipt(),
            now_unix=NOW + 21,
        )

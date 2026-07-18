from __future__ import annotations

import copy
import json
import shlex
from typing import Any, Mapping, Sequence

import pytest

from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_interpreter_provenance as provenance
from scripts.canary import owner_gate_owner_reauth as owner_reauth


RELEASE = "a" * 40
NOW = 2_000_000_000
IMAGE_ID = "1234567890123456789"
PYTHON_SHA = "9" * 64


def _raw(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class _Executable:
    prefix = ("/trusted/python", "-I", "-S", "-B", "/trusted/gcloud.py")

    def trusted_command_prefix(self) -> tuple[str, ...]:
        return self.prefix

    def sealed_runtime_identity(self, *, expected_release_sha: str) -> Mapping[str, Any]:
        assert expected_release_sha == RELEASE
        return {"identity_sha256": "8" * 64}


class _Configuration:
    account = owner_reauth.OWNER_ACCOUNT

    def assert_stable(self) -> None:
        return None

    def environment_values(self) -> Mapping[str, str]:
        return {"HOME": "/owner", "CLOUDSDK_CONFIG": "/owner/.config/gcloud"}


class _KnownHosts:
    def __init__(self, *, expected_instance_id: str) -> None:
        self.instance_id = expected_instance_id

    def absolute_path(self) -> str:
        return "/owner/.ssh/google_compute_known_hosts"

    def private_key_path(self) -> str:
        return "/owner/.ssh/google_compute_engine"

    def public_key_line(self) -> str:
        return "ssh-ed25519 AAAATEST owner"

    def server_host_key_line(self, instance_id: str) -> str:
        assert instance_id == self.instance_id
        return f"compute.{instance_id} ssh-ed25519 AAAAHOST"


class _Runner:
    def __init__(self, *, second_sha: str = PYTHON_SHA, bad_image: bool = False) -> None:
        self.second_sha = second_sha
        self.bad_image = bad_image
        self.ssh_calls = 0

    @staticmethod
    def _dry_run(argv: Sequence[str]) -> bytes:
        host_arg = next(item for item in argv if item.startswith("lomliev_"))
        name = host_arg.split("@", 1)[1]
        host = next(item for item in provenance.FIXED_HOSTS if item.name == name)
        prefix = _Executable.prefix
        remote = next(
            item.split("=", 1)[1] for item in argv if item.startswith("--command=")
        )
        flags = tuple(
            item.removeprefix("--ssh-flag=")
            for item in argv
            if item.startswith("--ssh-flag=")
        )
        proxy = "ProxyCommand " + " ".join((
            *prefix,
            "compute",
            "start-iap-tunnel",
            host.name,
            "%p",
            "--listen-on-stdin",
            f"--project={foundation.PROJECT}",
            f"--zone={foundation.ZONE}",
            "--verbosity=error",
        ))
        expected = (
            "/usr/bin/ssh",
            "-T",
            "-o",
            proxy,
            "-o",
            "ProxyUseFdpass=no",
            *flags,
            f"{launcher.OS_LOGIN_USERNAME}@compute.{host.instance_id}",
            "--",
            *remote.split(" "),
        )
        return (shlex.join(expected) + "\n").encode()

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        timeout_seconds: float,
    ) -> provenance.CapturedCommand:
        assert env["CLOUDSDK_CORE_DISABLE_PROMPTS"] == "1"
        assert timeout_seconds > 0
        args = tuple(argv)
        if "images" in args:
            value = {
                "id": IMAGE_ID,
                "name": provenance.DEBIAN_IMAGE_NAME,
                "selfLink": provenance.DEBIAN_IMAGE_SELF_LINK,
                "status": "READY",
                "architecture": "X86_64",
            }
            return provenance.CapturedCommand(0, _raw(value))
        if "instances" in args:
            name = args[args.index("describe") + 1]
            host = next(item for item in provenance.FIXED_HOSTS if item.name == name)
            disk = f"{name}-boot"
            value = {
                "id": host.instance_id,
                "name": name,
                "zone": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    f"{foundation.PROJECT}/zones/{foundation.ZONE}"
                ),
                "status": "RUNNING",
                "disks": [{
                    "boot": True,
                    "deviceName": disk,
                    "source": (
                        "https://www.googleapis.com/compute/v1/projects/"
                        f"{foundation.PROJECT}/zones/{foundation.ZONE}/disks/{disk}"
                    ),
                }],
            }
            return provenance.CapturedCommand(0, _raw(value))
        if "disks" in args:
            name = args[args.index("describe") + 1]
            zone = (
                "https://www.googleapis.com/compute/v1/projects/"
                f"{foundation.PROJECT}/zones/{foundation.ZONE}"
            )
            value = {
                "id": "4567890123456789012",
                "name": name,
                "selfLink": f"{zone}/disks/{name}",
                "sourceImage": (
                    "projects/other/global/images/tamper"
                    if self.bad_image
                    else provenance.DEBIAN_IMAGE_SELF_LINK
                ),
                "sourceImageId": IMAGE_ID,
                "status": "READY",
                "zone": zone,
            }
            return provenance.CapturedCommand(0, _raw(value))
        if "ssh" in args and "--dry-run" in args:
            return provenance.CapturedCommand(0, self._dry_run(args[:-1]))
        if "ssh" in args:
            self.ssh_calls += 1
            digest = PYTHON_SHA if self.ssh_calls == 1 else self.second_sha
            output = (
                "link=python3.11\n"
                "linkstat=0|0|777|1|symbolic link\n"
                "stat=0|0|755|1|regular file|6831736\n"
                "owner=python3.11-minimal: /usr/bin/python3.11\n"
                "package=ii |python3.11-minimal|3.11.2-6+deb12u7|amd64\n"
                "verify=clean\n"
                "version=Python 3.11.2\n"
                f"sha256={digest}\n"
            ).encode()
            return provenance.CapturedCommand(0, output)
        raise AssertionError(args)


def _collect(runner: _Runner) -> Mapping[str, Any]:
    return provenance._collect_with_runner(
        release_revision=RELEASE,
        collected_at_unix=NOW,
        gcloud_executable=_Executable(),  # type: ignore[arg-type]
        gcloud_configuration=_Configuration(),  # type: ignore[arg-type]
        runner=runner,
        known_hosts_factory=_KnownHosts,  # type: ignore[arg-type]
    )


def test_two_fixed_hosts_bind_exact_image_package_and_digest() -> None:
    evidence = _collect(_Runner())

    assert evidence["interpreter_sha256"] == PYTHON_SHA
    assert evidence["image"]["id"] == IMAGE_ID
    assert [item["host"]["instance_id"] for item in evidence["hosts"]] == [
        foundation.PRODUCTION_SOURCE_VM_ID,
        launcher.VM_INSTANCE_ID,
    ]
    assert evidence["package_version"] == "3.11.2-6+deb12u7"
    assert evidence["evidence_sha256"] == provenance._sha256(
        provenance._canonical({
            key: value for key, value in evidence.items() if key != "evidence_sha256"
        })
    )


def test_two_host_interpreter_mismatch_fails_closed() -> None:
    with pytest.raises(
        provenance.OwnerGateInterpreterProvenanceError,
        match="owner_gate_interpreter_hosts_mismatch",
    ):
        _collect(_Runner(second_sha="7" * 64))


def test_boot_disk_image_tamper_fails_before_ssh() -> None:
    runner = _Runner(bad_image=True)
    with pytest.raises(
        provenance.OwnerGateInterpreterProvenanceError,
        match="owner_gate_interpreter_disk_invalid",
    ):
        _collect(runner)
    assert runner.ssh_calls == 0


def test_probe_rejects_non_root_or_hardlinked_interpreter() -> None:
    raw = (
        "link=python3.11\n"
        "linkstat=0|0|777|1|symbolic link\n"
        "stat=0|0|755|2|regular file|6831736\n"
        "owner=python3.11-minimal: /usr/bin/python3.11\n"
        "package=ii |python3.11-minimal|3.11.2-6+deb12u7|amd64\n"
        "verify=clean\n"
        "version=Python 3.11.2\n"
        f"sha256={PYTHON_SHA}\n"
    ).encode()
    with pytest.raises(
        provenance.OwnerGateInterpreterProvenanceError,
        match="owner_gate_interpreter_probe_invalid",
    ):
        provenance._validate_probe(raw)


def test_validator_rejects_wrong_self_hash() -> None:
    evidence = dict(_collect(_Runner()))
    evidence["evidence_sha256"] = "0" * 64

    with pytest.raises(
        provenance.OwnerGateInterpreterProvenanceError,
        match="owner_gate_interpreter_evidence_invalid",
    ):
        provenance.validate_interpreter_provenance(
            evidence,
            expected_release_revision=RELEASE,
            now_unix=NOW,
        )


def test_validator_rejects_expired_evidence() -> None:
    evidence = _collect(_Runner())

    with pytest.raises(
        provenance.OwnerGateInterpreterProvenanceError,
        match="owner_gate_interpreter_evidence_invalid",
    ):
        provenance.validate_interpreter_provenance(
            evidence,
            expected_release_revision=RELEASE,
            now_unix=NOW + provenance.MAX_EVIDENCE_AGE_SECONDS + 1,
        )


@pytest.mark.parametrize(
    "tamper",
    ["host_instance_id", "boot_image_id", "probe_digest"],
)
def test_validator_rejects_rehashed_fixed_host_tamper(tamper: str) -> None:
    evidence = copy.deepcopy(_collect(_Runner()))
    if tamper == "host_instance_id":
        evidence["hosts"][0]["host"]["instance_id"] = "999"
    elif tamper == "boot_image_id":
        evidence["hosts"][0]["boot_disk"]["sourceImageId"] = "999"
    else:
        evidence["hosts"][0]["probe"]["interpreter_sha256"] = "7" * 64
    unsigned = {
        key: value for key, value in evidence.items() if key != "evidence_sha256"
    }
    evidence["evidence_sha256"] = provenance._sha256(
        provenance._canonical(unsigned)
    )

    with pytest.raises(
        provenance.OwnerGateInterpreterProvenanceError,
        match=(
            "owner_gate_interpreter_hosts_mismatch"
            if tamper == "probe_digest"
            else "owner_gate_interpreter_evidence_invalid"
        ),
    ):
        provenance.validate_interpreter_provenance(
            evidence,
            expected_release_revision=RELEASE,
            now_unix=NOW,
        )

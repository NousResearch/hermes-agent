#!/usr/bin/env python3
"""Fixed read-only Debian interpreter provenance for foundation authoring.

The collector deliberately has no caller-selectable project, image, host,
path, package, provider, or command.  It binds the exact reviewed Debian image
to two independently identified fixed VMs, then requires byte-identical
package-integrity observations from both hosts before returning a digest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_owner_reauth as owner_reauth


EVIDENCE_SCHEMA = "muncho-owner-gate-interpreter-provenance.v1"
DEBIAN_IMAGE_NAME = "debian-12-bookworm-v20260609"
DEBIAN_IMAGE_PROJECT = "debian-cloud"
DEBIAN_IMAGE_SHORT_LINK = (
    f"projects/{DEBIAN_IMAGE_PROJECT}/global/images/{DEBIAN_IMAGE_NAME}"
)
DEBIAN_IMAGE_SELF_LINK = (
    "https://www.googleapis.com/compute/v1/" + DEBIAN_IMAGE_SHORT_LINK
)
PYTHON_PATH = "/usr/bin/python3.11"
PYTHON_LINK = "/usr/bin/python3"
PYTHON_LINK_TARGET = "python3.11"
PYTHON_PACKAGE = "python3.11-minimal"
PYTHON_VERSION = "3.11.2"
MAX_EVIDENCE_AGE_SECONDS = 600
MAX_OUTPUT_BYTES = 1024 * 1024

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NUMERIC_ID = re.compile(r"^[1-9][0-9]{0,31}$")
_PACKAGE_VERSION = re.compile(r"^3\.11\.2-[0-9]+\+deb12u[0-9]+$")
_DISK_NAME = re.compile(r"^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")


@dataclass(frozen=True)
class FixedHost:
    name: str
    instance_id: str


FIXED_HOSTS = (
    FixedHost(foundation.PRODUCTION_SOURCE_VM, foundation.PRODUCTION_SOURCE_VM_ID),
    FixedHost(launcher.VM_NAME, launcher.VM_INSTANCE_ID),
)


class OwnerGateInterpreterProvenanceError(RuntimeError):
    """Stable, secret-free provenance failure."""


def _error(code: str, _cause: BaseException | None = None) -> None:
    raise OwnerGateInterpreterProvenanceError(code) from None


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        _error("owner_gate_interpreter_evidence_json_invalid", exc)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class CapturedCommand:
    returncode: int
    stdout: bytes


class ReadOnlyRunner(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        timeout_seconds: float,
    ) -> CapturedCommand: ...


class _SubprocessReadOnlyRunner:
    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        timeout_seconds: float,
    ) -> CapturedCommand:
        try:
            completed = subprocess.run(
                tuple(argv),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=dict(env),
                shell=False,
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            _error("owner_gate_interpreter_command_failed", exc)
        return CapturedCommand(completed.returncode, completed.stdout)


def _decode_json(raw: bytes, *, code: str) -> Mapping[str, Any]:
    if not raw or len(raw) > MAX_OUTPUT_BYTES:
        _error(code)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate")
            result[key] = item
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _item: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        _error(code, exc)
    if not isinstance(value, Mapping):
        _error(code)
    return dict(value)


def _run_json(
    *,
    prefix: Sequence[str],
    args: Sequence[str],
    environment: Mapping[str, str],
    runner: ReadOnlyRunner,
) -> Mapping[str, Any]:
    completed = runner.run(
        (*prefix, *args),
        env=environment,
        timeout_seconds=60.0,
    )
    if completed.returncode != 0:
        _error("owner_gate_interpreter_inventory_failed")
    return _decode_json(
        completed.stdout,
        code="owner_gate_interpreter_inventory_invalid",
    )


def _image_command(account: str) -> tuple[str, ...]:
    return (
        "compute",
        "images",
        "describe",
        DEBIAN_IMAGE_NAME,
        f"--project={DEBIAN_IMAGE_PROJECT}",
        f"--account={account}",
        "--format=json(id,name,selfLink,status,architecture)",
        "--quiet",
    )


def _instance_command(host: FixedHost, account: str) -> tuple[str, ...]:
    return (
        "compute",
        "instances",
        "describe",
        host.name,
        f"--project={foundation.PROJECT}",
        f"--zone={foundation.ZONE}",
        f"--account={account}",
        "--format=json(id,name,zone,status,disks[].boot,disks[].deviceName,disks[].source)",
        "--quiet",
    )


def _disk_command(name: str, account: str) -> tuple[str, ...]:
    if _DISK_NAME.fullmatch(name or "") is None:
        _error("owner_gate_interpreter_disk_invalid")
    return (
        "compute",
        "disks",
        "describe",
        name,
        f"--project={foundation.PROJECT}",
        f"--zone={foundation.ZONE}",
        f"--account={account}",
        "--format=json(id,name,selfLink,sourceImage,sourceImageId,status,zone)",
        "--quiet",
    )


def _disk_name_from_source(
    source: object,
    *,
    zone: str,
    error_code: str,
) -> str:
    prefix = f"{zone}/disks/"
    if not isinstance(source, str) or not source.startswith(prefix):
        _error(error_code)
    name = source[len(prefix) :]
    if _DISK_NAME.fullmatch(name) is None or source != f"{prefix}{name}":
        _error(error_code)
    return name


def _validate_image(value: Mapping[str, Any]) -> Mapping[str, str]:
    if (
        set(value) != {"id", "name", "selfLink", "status", "architecture"}
        or _NUMERIC_ID.fullmatch(str(value.get("id", ""))) is None
        or value.get("name") != DEBIAN_IMAGE_NAME
        or value.get("selfLink") != DEBIAN_IMAGE_SELF_LINK
        or value.get("status") != "READY"
        or value.get("architecture") != "X86_64"
    ):
        _error("owner_gate_interpreter_image_invalid")
    return {
        "id": str(value["id"]),
        "name": DEBIAN_IMAGE_NAME,
        "selfLink": DEBIAN_IMAGE_SELF_LINK,
        "shortLink": DEBIAN_IMAGE_SHORT_LINK,
        "status": "READY",
        "architecture": "X86_64",
    }


def _validate_instance(
    value: Mapping[str, Any],
    *,
    host: FixedHost,
) -> tuple[Mapping[str, Any], str]:
    disks = value.get("disks")
    zone = (
        "https://www.googleapis.com/compute/v1/projects/"
        f"{foundation.PROJECT}/zones/{foundation.ZONE}"
    )
    if (
        set(value) != {"id", "name", "zone", "status", "disks"}
        or value.get("id") != host.instance_id
        or value.get("name") != host.name
        or value.get("zone") != zone
        or value.get("status") != "RUNNING"
        or not isinstance(disks, list)
    ):
        _error("owner_gate_interpreter_instance_invalid")
    boot = [item for item in disks if isinstance(item, Mapping) and item.get("boot") is True]
    if len(boot) != 1 or any(not isinstance(item, Mapping) for item in disks):
        _error("owner_gate_interpreter_instance_invalid")
    selected = dict(boot[0])
    if set(selected) != {"boot", "deviceName", "source"}:
        _error("owner_gate_interpreter_instance_invalid")
    device = selected.get("deviceName")
    source = selected.get("source")
    if (
        not isinstance(device, str)
        or _DISK_NAME.fullmatch(device) is None
    ):
        _error("owner_gate_interpreter_instance_invalid")
    disk_name = _disk_name_from_source(
        source,
        zone=zone,
        error_code="owner_gate_interpreter_instance_invalid",
    )
    normalized = {
        "id": host.instance_id,
        "name": host.name,
        "zone": zone,
        "status": "RUNNING",
        "boot_disk": {"deviceName": device, "source": source},
    }
    return normalized, disk_name


def _validate_disk(
    value: Mapping[str, Any],
    *,
    name: str,
    image_numeric_id: str,
) -> Mapping[str, str]:
    zone = (
        "https://www.googleapis.com/compute/v1/projects/"
        f"{foundation.PROJECT}/zones/{foundation.ZONE}"
    )
    self_link = f"{zone}/disks/{name}"
    if (
        set(value)
        != {"id", "name", "selfLink", "sourceImage", "sourceImageId", "status", "zone"}
        or _NUMERIC_ID.fullmatch(str(value.get("id", ""))) is None
        or value.get("name") != name
        or value.get("selfLink") != self_link
        or value.get("sourceImage") != DEBIAN_IMAGE_SELF_LINK
        or value.get("sourceImageId") != image_numeric_id
        or value.get("status") != "READY"
        or value.get("zone") != zone
    ):
        _error("owner_gate_interpreter_disk_invalid")
    return {
        "id": str(value["id"]),
        "name": name,
        "selfLink": self_link,
        "sourceImage": DEBIAN_IMAGE_SELF_LINK,
        "sourceImageId": image_numeric_id,
        "status": "READY",
        "zone": zone,
    }


def _remote_probe_command() -> str:
    script = " ".join((
        "set -eu;",
        "export LC_ALL=C;",
        "/usr/bin/printf 'link='; /usr/bin/readlink -- /usr/bin/python3;",
        "/usr/bin/printf 'linkstat='; /usr/bin/stat -c '%u|%g|%a|%h|%F' -- /usr/bin/python3;",
        "/usr/bin/printf 'stat='; /usr/bin/stat -c '%u|%g|%a|%h|%F|%s' -- /usr/bin/python3.11;",
        "/usr/bin/printf 'owner='; /usr/bin/dpkg-query -S /usr/bin/python3.11;",
        "/usr/bin/printf 'package='; /usr/bin/dpkg-query -W -f='${db:Status-Abbrev}|${binary:Package}|${Version}|${Architecture}\\n' python3.11-minimal;",
        "/usr/bin/dpkg --verify python3.11-minimal; /usr/bin/printf 'verify=clean\\n';",
        "/usr/bin/printf 'version='; /usr/bin/python3.11 -I -S -B --version 2>&1;",
        "/usr/bin/printf 'sha256='; /usr/bin/sha256sum -- /usr/bin/python3.11 | /usr/bin/cut -d' ' -f1;",
    ))
    return shlex.join(("/bin/sh", "-c", script))


def _ssh_flags(known_hosts: str, private_key: str) -> tuple[str, ...]:
    return launcher.IapCoordinatorTransport._ssh_flags(known_hosts, private_key)


def _ssh_argv(
    *,
    prefix: Sequence[str],
    host: FixedHost,
    account: str,
    known_hosts: launcher.PinnedGoogleComputeKnownHosts,
) -> tuple[str, ...]:
    known_hosts_path = known_hosts.absolute_path()
    private_key = known_hosts.private_key_path()
    known_hosts.public_key_line()
    known_hosts.server_host_key_line(host.instance_id)
    return (
        *prefix,
        "compute",
        "ssh",
        f"{launcher.OS_LOGIN_USERNAME}@{host.name}",
        f"--project={foundation.PROJECT}",
        f"--zone={foundation.ZONE}",
        f"--account={account}",
        "--plain",
        "--tunnel-through-iap",
        "--quiet",
        f"--command={_remote_probe_command()}",
        *_ssh_flags(known_hosts_path, private_key),
    )


def _validate_ssh_dry_run(
    raw: bytes,
    *,
    argv: Sequence[str],
    prefix: Sequence[str],
    host: FixedHost,
    known_hosts: launcher.PinnedGoogleComputeKnownHosts,
) -> None:
    if (
        not raw
        or len(raw) > 256 * 1024
        or not raw.endswith(b"\n")
        or b"\n" in raw[:-1]
    ):
        _error("owner_gate_interpreter_ssh_dry_run_invalid")
    try:
        observed = tuple(
            shlex.split(raw[:-1].decode("utf-8", errors="strict"), posix=True)
        )
    except (UnicodeError, ValueError) as exc:
        _error("owner_gate_interpreter_ssh_dry_run_invalid", exc)
    known_hosts_path = known_hosts.absolute_path()
    private_key = known_hosts.private_key_path()
    remote = next(
        (item.split("=", 1)[1] for item in argv if item.startswith("--command=")),
        None,
    )
    try:
        remote_argv = () if remote is None else tuple(shlex.split(remote, posix=True))
    except ValueError as exc:
        _error("owner_gate_interpreter_ssh_dry_run_invalid", exc)
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
    ssh_options = tuple(
        item.removeprefix("--ssh-flag=")
        for item in _ssh_flags(known_hosts_path, private_key)
    )
    expected = (
        "/usr/bin/ssh",
        "-T",
        "-o",
        proxy,
        "-o",
        "ProxyUseFdpass=no",
        *ssh_options,
        f"{launcher.OS_LOGIN_USERNAME}@compute.{host.instance_id}",
        "--",
        *remote_argv,
    )
    if remote is None or observed != expected:
        _error("owner_gate_interpreter_ssh_dry_run_invalid")


def _validate_probe(raw: bytes) -> Mapping[str, Any]:
    if not raw or len(raw) > 16 * 1024:
        _error("owner_gate_interpreter_probe_invalid")
    try:
        lines = raw.decode("ascii", errors="strict").splitlines()
    except UnicodeError as exc:
        _error("owner_gate_interpreter_probe_invalid", exc)
    prefixes = (
        "link=",
        "linkstat=",
        "stat=",
        "owner=",
        "package=",
        "verify=",
        "version=",
        "sha256=",
    )
    if len(lines) != len(prefixes) or any(
        not line.startswith(prefix) for line, prefix in zip(lines, prefixes)
    ):
        _error("owner_gate_interpreter_probe_invalid")
    values = {prefix[:-1]: line[len(prefix):] for prefix, line in zip(prefixes, lines)}
    package = values["package"].split("|")
    stat_value = values["stat"].split("|")
    if (
        values["link"] != PYTHON_LINK_TARGET
        or values["linkstat"] != "0|0|777|1|symbolic link"
        or len(stat_value) != 6
        or stat_value[:5] != ["0", "0", "755", "1", "regular file"]
        or not stat_value[5].isdigit()
        or int(stat_value[5]) <= 0
        or len(package) != 4
        or package[0] != "ii "
        or package[1] not in {PYTHON_PACKAGE, f"{PYTHON_PACKAGE}:amd64"}
        or _PACKAGE_VERSION.fullmatch(package[2] or "") is None
        or package[3] != "amd64"
        or values["owner"]
        not in {
            f"{PYTHON_PACKAGE}: {PYTHON_PATH}",
            f"{PYTHON_PACKAGE}:amd64: {PYTHON_PATH}",
        }
        or values["verify"] != "clean"
        or values["version"] != f"Python {PYTHON_VERSION}"
        or _SHA256.fullmatch(values["sha256"] or "") is None
    ):
        _error("owner_gate_interpreter_probe_invalid")
    return {
        "link_path": PYTHON_LINK,
        "link_target": PYTHON_LINK_TARGET,
        "link_stat": values["linkstat"],
        "interpreter_path": PYTHON_PATH,
        "interpreter_stat": values["stat"],
        "package_owner": values["owner"],
        "package_name": package[1],
        "package_version": package[2],
        "package_architecture": package[3],
        "package_integrity": "dpkg_verify_clean",
        "python_version": PYTHON_VERSION,
        "interpreter_sha256": values["sha256"],
    }


def _collect_inventory(
    *,
    prefix: Sequence[str],
    account: str,
    environment: Mapping[str, str],
    runner: ReadOnlyRunner,
) -> tuple[Mapping[str, str], tuple[tuple[FixedHost, Mapping[str, Any], Mapping[str, str]], ...]]:
    image = _validate_image(_run_json(
        prefix=prefix,
        args=_image_command(account),
        environment=environment,
        runner=runner,
    ))
    hosts: list[tuple[FixedHost, Mapping[str, Any], Mapping[str, str]]] = []
    for host in FIXED_HOSTS:
        instance, disk_name = _validate_instance(
            _run_json(
                prefix=prefix,
                args=_instance_command(host, account),
                environment=environment,
                runner=runner,
            ),
            host=host,
        )
        disk = _validate_disk(
            _run_json(
                prefix=prefix,
                args=_disk_command(disk_name, account),
                environment=environment,
                runner=runner,
            ),
            name=disk_name,
            image_numeric_id=image["id"],
        )
        hosts.append((host, instance, disk))
    return image, tuple(hosts)


def _collect_with_runner(
    *,
    release_revision: str,
    collected_at_unix: int,
    gcloud_executable: launcher.TrustedGcloudExecutable,
    gcloud_configuration: launcher.PinnedGcloudConfiguration,
    runner: ReadOnlyRunner,
    known_hosts_factory: Callable[..., launcher.PinnedGoogleComputeKnownHosts],
) -> Mapping[str, Any]:
    if (
        re.fullmatch(r"[0-9a-f]{40}", release_revision or "") is None
        or type(collected_at_unix) is not int
        or collected_at_unix <= 0
    ):
        _error("owner_gate_interpreter_request_invalid")
    try:
        before_runtime = gcloud_executable.sealed_runtime_identity(
            expected_release_sha=release_revision
        )
        gcloud_configuration.assert_stable()
        account = gcloud_configuration.account
        prefix = gcloud_executable.trusted_command_prefix()
        environment = launcher._owner_gcloud_environment(
            gcloud_configuration,
            prefix[0],
        )
    except launcher.OwnerLauncherError as exc:
        _error("owner_gate_interpreter_runtime_invalid", exc)
    if account != owner_reauth.OWNER_ACCOUNT:
        _error("owner_gate_interpreter_account_invalid")
    image_before, inventory_before = _collect_inventory(
        prefix=prefix,
        account=account,
        environment=environment,
        runner=runner,
    )
    observations: list[Mapping[str, Any]] = []
    for host, instance, disk in inventory_before:
        try:
            known_hosts = known_hosts_factory(expected_instance_id=host.instance_id)
            argv = _ssh_argv(
                prefix=prefix,
                host=host,
                account=account,
                known_hosts=known_hosts,
            )
            dry_run = runner.run(
                (*argv, "--dry-run"),
                env=environment,
                timeout_seconds=60.0,
            )
            if dry_run.returncode != 0 or not dry_run.stdout:
                _error("owner_gate_interpreter_ssh_dry_run_invalid")
            _validate_ssh_dry_run(
                dry_run.stdout,
                argv=argv,
                prefix=prefix,
                host=host,
                known_hosts=known_hosts,
            )
            probe = runner.run(argv, env=environment, timeout_seconds=120.0)
        except launcher.OwnerLauncherError as exc:
            _error("owner_gate_interpreter_ssh_boundary_invalid", exc)
        if probe.returncode != 0:
            _error("owner_gate_interpreter_probe_failed")
        observations.append({
            "host": {"name": host.name, "instance_id": host.instance_id},
            "instance": instance,
            "boot_disk": disk,
            "probe": _validate_probe(probe.stdout),
            "ssh_command_sha256": _sha256(_canonical(list(argv))),
            "ssh_dry_run_sha256": _sha256(dry_run.stdout),
        })
    image_after, inventory_after = _collect_inventory(
        prefix=prefix,
        account=account,
        environment=environment,
        runner=runner,
    )
    try:
        after_runtime = gcloud_executable.sealed_runtime_identity(
            expected_release_sha=release_revision
        )
        gcloud_configuration.assert_stable()
    except launcher.OwnerLauncherError as exc:
        _error("owner_gate_interpreter_runtime_invalid", exc)
    if (
        image_after != image_before
        or inventory_after != inventory_before
        or after_runtime != before_runtime
    ):
        _error("owner_gate_interpreter_provenance_changed")
    first = observations[0]["probe"]
    second = observations[1]["probe"]
    exact_fields = (
        "link_target",
        "link_stat",
        "interpreter_stat",
        "package_owner",
        "package_name",
        "package_version",
        "package_architecture",
        "package_integrity",
        "python_version",
        "interpreter_sha256",
    )
    if any(first[field] != second[field] for field in exact_fields):
        _error("owner_gate_interpreter_hosts_mismatch")
    unsigned = {
        "schema": EVIDENCE_SCHEMA,
        "release_revision": release_revision,
        "project": foundation.PROJECT,
        "zone": foundation.ZONE,
        "image": image_before,
        "hosts": observations,
        "interpreter_path": PYTHON_PATH,
        "interpreter_sha256": first["interpreter_sha256"],
        "python_version": PYTHON_VERSION,
        "package_name": first["package_name"],
        "package_version": first["package_version"],
        "package_architecture": first["package_architecture"],
        "collected_at_unix": collected_at_unix,
        "expires_at_unix": collected_at_unix + MAX_EVIDENCE_AGE_SECONDS,
        "trusted_runtime_identity_sha256": before_runtime["identity_sha256"],
    }
    result = {**unsigned, "evidence_sha256": _sha256(_canonical(unsigned))}
    return validate_interpreter_provenance(
        result,
        expected_release_revision=release_revision,
        now_unix=collected_at_unix,
    )


def validate_interpreter_provenance(
    value: Any,
    *,
    expected_release_revision: str,
    now_unix: int,
) -> Mapping[str, Any]:
    """Validate the complete canonical two-host provenance contract."""

    fields = {
        "schema",
        "release_revision",
        "project",
        "zone",
        "image",
        "hosts",
        "interpreter_path",
        "interpreter_sha256",
        "python_version",
        "package_name",
        "package_version",
        "package_architecture",
        "collected_at_unix",
        "expires_at_unix",
        "trusted_runtime_identity_sha256",
        "evidence_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        _error("owner_gate_interpreter_evidence_invalid")
    evidence = dict(value)
    unsigned = {key: item for key, item in evidence.items() if key != "evidence_sha256"}
    image = evidence.get("image")
    hosts = evidence.get("hosts")
    collected = evidence.get("collected_at_unix")
    expires = evidence.get("expires_at_unix")
    if (
        evidence.get("schema") != EVIDENCE_SCHEMA
        or evidence.get("release_revision") != expected_release_revision
        or re.fullmatch(r"[0-9a-f]{40}", expected_release_revision or "") is None
        or evidence.get("project") != foundation.PROJECT
        or evidence.get("zone") != foundation.ZONE
        or evidence.get("interpreter_path") != PYTHON_PATH
        or _SHA256.fullmatch(str(evidence.get("interpreter_sha256", ""))) is None
        or evidence.get("python_version") != PYTHON_VERSION
        or evidence.get("package_name") not in {PYTHON_PACKAGE, f"{PYTHON_PACKAGE}:amd64"}
        or _PACKAGE_VERSION.fullmatch(str(evidence.get("package_version", ""))) is None
        or evidence.get("package_architecture") != "amd64"
        or type(collected) is not int
        or type(expires) is not int
        or type(now_unix) is not int
        or collected <= 0
        or expires != collected + MAX_EVIDENCE_AGE_SECONDS
        or now_unix < collected
        or now_unix > expires
        or _SHA256.fullmatch(
            str(evidence.get("trusted_runtime_identity_sha256", ""))
        )
        is None
        or evidence.get("evidence_sha256") != _sha256(_canonical(unsigned))
        or not isinstance(image, Mapping)
        or not isinstance(hosts, list)
        or len(hosts) != 2
    ):
        _error("owner_gate_interpreter_evidence_invalid")
    if (
        set(image)
        != {"id", "name", "selfLink", "shortLink", "status", "architecture"}
        or _NUMERIC_ID.fullmatch(str(image.get("id", ""))) is None
        or image.get("name") != DEBIAN_IMAGE_NAME
        or image.get("selfLink") != DEBIAN_IMAGE_SELF_LINK
        or image.get("shortLink") != DEBIAN_IMAGE_SHORT_LINK
        or image.get("status") != "READY"
        or image.get("architecture") != "X86_64"
    ):
        _error("owner_gate_interpreter_evidence_invalid")
    checked_image = dict(image)
    checked_probes: list[Mapping[str, Any]] = []
    for expected_host, observation in zip(FIXED_HOSTS, hosts):
        if not isinstance(observation, Mapping) or set(observation) != {
            "host",
            "instance",
            "boot_disk",
            "probe",
            "ssh_command_sha256",
            "ssh_dry_run_sha256",
        }:
            _error("owner_gate_interpreter_evidence_invalid")
        host = observation.get("host")
        instance = observation.get("instance")
        disk = observation.get("boot_disk")
        probe = observation.get("probe")
        if (
            host
            != {"name": expected_host.name, "instance_id": expected_host.instance_id}
            or not isinstance(instance, Mapping)
            or not isinstance(disk, Mapping)
            or not isinstance(probe, Mapping)
            or _SHA256.fullmatch(str(observation.get("ssh_command_sha256", "")))
            is None
            or _SHA256.fullmatch(str(observation.get("ssh_dry_run_sha256", "")))
            is None
        ):
            _error("owner_gate_interpreter_evidence_invalid")
        expected_zone = (
            "https://www.googleapis.com/compute/v1/projects/"
            f"{foundation.PROJECT}/zones/{foundation.ZONE}"
        )
        boot = instance.get("boot_disk")
        if (
            set(instance) != {"id", "name", "zone", "status", "boot_disk"}
            or instance.get("id") != expected_host.instance_id
            or instance.get("name") != expected_host.name
            or instance.get("zone") != expected_zone
            or instance.get("status") != "RUNNING"
            or not isinstance(boot, Mapping)
            or set(boot) != {"deviceName", "source"}
            or not isinstance(boot.get("deviceName"), str)
            or _DISK_NAME.fullmatch(boot.get("deviceName")) is None
        ):
            _error("owner_gate_interpreter_evidence_invalid")
        boot_disk_resource_name = _disk_name_from_source(
            boot.get("source"),
            zone=expected_zone,
            error_code="owner_gate_interpreter_evidence_invalid",
        )
        if (
            disk.get("name") != boot_disk_resource_name
            or disk.get("selfLink") != boot.get("source")
            or disk.get("sourceImage") != DEBIAN_IMAGE_SELF_LINK
            or disk.get("sourceImageId") != checked_image["id"]
            or disk.get("status") != "READY"
            or disk.get("zone") != expected_zone
            or set(disk)
            != {"id", "name", "selfLink", "sourceImage", "sourceImageId", "status", "zone"}
            or _NUMERIC_ID.fullmatch(str(disk.get("id", ""))) is None
        ):
            _error("owner_gate_interpreter_evidence_invalid")
        probe_fields = {
            "link_path",
            "link_target",
            "link_stat",
            "interpreter_path",
            "interpreter_stat",
            "package_owner",
            "package_name",
            "package_version",
            "package_architecture",
            "package_integrity",
            "python_version",
            "interpreter_sha256",
        }
        stat_parts = str(probe.get("interpreter_stat", "")).split("|")
        if (
            set(probe) != probe_fields
            or probe.get("link_path") != PYTHON_LINK
            or probe.get("link_target") != PYTHON_LINK_TARGET
            or probe.get("link_stat") != "0|0|777|1|symbolic link"
            or probe.get("interpreter_path") != PYTHON_PATH
            or len(stat_parts) != 6
            or stat_parts[:5] != ["0", "0", "755", "1", "regular file"]
            or not stat_parts[5].isdigit()
            or int(stat_parts[5]) <= 0
            or probe.get("package_name") not in {PYTHON_PACKAGE, f"{PYTHON_PACKAGE}:amd64"}
            or _PACKAGE_VERSION.fullmatch(str(probe.get("package_version", "")))
            is None
            or probe.get("package_architecture") != "amd64"
            or probe.get("package_integrity") != "dpkg_verify_clean"
            or probe.get("python_version") != PYTHON_VERSION
            or _SHA256.fullmatch(str(probe.get("interpreter_sha256", ""))) is None
        ):
            _error("owner_gate_interpreter_evidence_invalid")
        checked_probes.append(dict(probe))
    first, second = checked_probes
    if (
        first != second
        or evidence["interpreter_sha256"] != first["interpreter_sha256"]
        or evidence["package_name"] != first["package_name"]
        or evidence["package_version"] != first["package_version"]
        or evidence["package_architecture"] != first["package_architecture"]
    ):
        _error("owner_gate_interpreter_hosts_mismatch")
    return evidence


def collect_interpreter_provenance(
    *,
    release_revision: str,
    collected_at_unix: int,
    gcloud_executable: launcher.TrustedGcloudExecutable,
    gcloud_configuration: launcher.PinnedGcloudConfiguration,
) -> Mapping[str, Any]:
    """Collect only through the exact sealed production owner capability."""

    if (
        type(gcloud_executable) is not launcher.TrustedGcloudExecutable
        or type(gcloud_configuration) is not launcher.PinnedGcloudConfiguration
    ):
        _error("owner_gate_interpreter_runtime_invalid")
    return _collect_with_runner(
        release_revision=release_revision,
        collected_at_unix=collected_at_unix,
        gcloud_executable=gcloud_executable,
        gcloud_configuration=gcloud_configuration,
        runner=_SubprocessReadOnlyRunner(),
        known_hosts_factory=launcher.PinnedGoogleComputeKnownHosts,
    )


__all__ = [
    "DEBIAN_IMAGE_SHORT_LINK",
    "EVIDENCE_SCHEMA",
    "OwnerGateInterpreterProvenanceError",
    "PYTHON_VERSION",
    "collect_interpreter_provenance",
    "validate_interpreter_provenance",
]

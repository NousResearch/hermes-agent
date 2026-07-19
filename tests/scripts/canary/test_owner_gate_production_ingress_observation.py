"""Fixed production-ingress observer, transport, and signature tests."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import owner_gate_production_ingress_contract as contract
from scripts.canary import owner_gate_production_ingress_observation as ingress
from scripts.canary import owner_gate_trust as trust


REVISION = "a" * 40
PLAN_SHA256 = "b" * 64
NOW = 1_800_000_000


def _systemctl_output(unit: str, **changes: str) -> bytes:
    if unit == ingress.OLD_V1_UNIT:
        values = {
            "Id": ingress.OLD_V1_UNIT,
            "LoadState": "loaded",
            "ActiveState": "active",
            "SubState": "running",
            "UnitFileState": "enabled",
            "FragmentPath": str(ingress.OLD_V1_FRAGMENT_PATH),
            "DropInPaths": "",
            "MainPID": "4343",
            "ExecMainPID": "4343",
            "NeedDaemonReload": "no",
            "User": ingress.OLD_V1_USER,
            "Group": ingress.OLD_V1_GROUP,
            "ExecStart": (
                "{ path="
                f"{ingress.OLD_V1_EXEC_START_ARGV[0]} ; argv[]="
                f"{' '.join(ingress.OLD_V1_EXEC_START_ARGV)} ; "
                "ignore_errors=no ; start_time=[n/a] ; "
                "stop_time=[n/a] ; pid=4343 ; code=(null) ; "
                "status=0/0 }"
            ),
        }
    elif unit == ingress.CADDY_UNIT:
        values = {
            "LoadState": "loaded",
            "ActiveState": "active",
            "SubState": "running",
            "UnitFileState": "enabled",
            "FragmentPath": ingress.CADDY_UNIT_FRAGMENT,
            "DropInPaths": "",
            "MainPID": "4242",
            "ExecStart": (
                "{ path="
                f"{ingress.CADDY_EXECUTABLE} ; argv[]="
                f"{' '.join(ingress.CADDY_EXPECTED_ARGV)} ; "
                "ignore_errors=no ; start_time=[n/a] ; "
                "stop_time=[n/a] ; pid=4242 ; code=(null) ; "
                "status=0/0 }"
            ),
        }
    else:  # pragma: no cover - a test fixture guard
        raise AssertionError(f"unexpected unit: {unit}")
    values.update(changes)
    properties = (
        ingress.CADDY_SYSTEMD_PROPERTIES
        if unit == ingress.CADDY_UNIT
        else ingress.SYSTEMD_PROPERTIES
    )
    return "".join(f"{name}={values[name]}\n" for name in properties).encode()


def _adapted(*, dial: str = "127.0.0.1:7341", duplicate_route: bool = False) -> bytes:
    route: dict[str, Any] = {
        "match": [{"host": [ingress.PUBLIC_HOST]}],
        "handle": [
            {
                "handler": "subroute",
                "routes": [
                    {
                        "handle": [
                            {
                                "handler": "reverse_proxy",
                                "upstreams": [{"dial": dial}],
                            }
                        ]
                    }
                ],
            }
        ],
    }
    if duplicate_route:
        first = copy.deepcopy(route)
        first["match"][0]["path"] = ["/first/*"]
        routes = [first, copy.deepcopy(route)]
    else:
        routes = [route]
    value = {
        "apps": {
            "http": {
                "servers": {
                    "srv0": {
                        "listen": [":443"],
                        "routes": routes,
                    }
                }
            }
        }
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _proxy_route(
    dial: str,
    *,
    host: str | None = None,
    nested: bool = False,
) -> dict[str, Any]:
    proxy: dict[str, Any] = {
        "handler": "reverse_proxy",
        "upstreams": [{"dial": dial}],
    }
    handle: dict[str, Any]
    if nested:
        handle = {
            "handler": "subroute",
            "routes": [{"handle": [proxy]}],
        }
    else:
        handle = proxy
    route: dict[str, Any] = {"handle": [handle]}
    if host is not None:
        route["match"] = [{"host": [host]}]
    return route


def _adapted_value(raw: bytes | None = None) -> dict[str, Any]:
    return json.loads(raw if raw is not None else _adapted())


def _adapted_raw(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class _CommandFixture:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.old_changes: dict[str, str] = {}
        self.caddy_changes: dict[str, str] = {}
        self.adapted_outputs: list[bytes] = [_adapted(), _adapted()]
        self.live_outputs: list[bytes] | None = None
        self.last_adapted: bytes | None = None
        self.old_process_snapshots: list[ingress._OldV1ProcessSnapshot] = [
            ingress._OldV1ProcessSnapshot(
                pid=4343,
                uid=ingress.OLD_V1_UID,
                gid=ingress.OLD_V1_GID,
                start_time_ticks=90,
            ),
            ingress._OldV1ProcessSnapshot(
                pid=4343,
                uid=ingress.OLD_V1_UID,
                gid=ingress.OLD_V1_GID,
                start_time_ticks=90,
            ),
        ]
        self.process_snapshots: list[ingress._CaddyProcessSnapshot] = [
            ingress._CaddyProcessSnapshot(
                pid=4242,
                identity=(4242, 100, ingress.CADDY_EXECUTABLE, "socket"),
                admin_host="127.0.0.1",
                admin_socket_inode="12345",
            ),
            ingress._CaddyProcessSnapshot(
                pid=4242,
                identity=(4242, 100, ingress.CADDY_EXECUTABLE, "socket"),
                admin_host="127.0.0.1",
                admin_socket_inode="12345",
            ),
        ]

    def __call__(self, argv: tuple[str, ...], *, maximum_output_bytes: int) -> bytes:
        del maximum_output_bytes
        self.calls.append(argv)
        if argv == ingress._systemctl_command(ingress.OLD_V1_UNIT):
            return _systemctl_output(ingress.OLD_V1_UNIT, **self.old_changes)
        if argv == ingress._systemctl_command(ingress.CADDY_UNIT):
            return _systemctl_output(ingress.CADDY_UNIT, **self.caddy_changes)
        if argv == ingress._caddy_adapt_command():
            if not self.adapted_outputs:
                raise AssertionError("too many adapt calls")
            self.last_adapted = self.adapted_outputs.pop(0)
            return self.last_adapted
        raise AssertionError(f"unexpected command: {argv}")


@pytest.fixture
def production_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _CommandFixture:
    caddyfile = tmp_path / "etc/caddy/Caddyfile"
    caddyfile.parent.mkdir(parents=True)
    caddyfile.write_text("# secret-looking input is never emitted\nTOKEN super-secret-token\n")
    caddyfile.chmod(0o644)
    fragment = tmp_path / "etc/systemd/system/muncho-passkey-stepup.service"
    fragment.parent.mkdir(parents=True)
    fragment.write_text("[Unit]\nDescription=exact legacy v1 fixture\n")
    fragment.chmod(0o644)
    fixture_owner = caddyfile.stat()
    os.chown(fragment, fixture_owner.st_uid, fixture_owner.st_gid)
    assert (fragment.stat().st_uid, fragment.stat().st_gid) == (
        fixture_owner.st_uid,
        fixture_owner.st_gid,
    )
    fragment_sha256 = hashlib.sha256(fragment.read_bytes()).hexdigest()

    monkeypatch.setattr(ingress, "CADDYFILE_PATH", caddyfile)
    monkeypatch.setattr(
        ingress,
        "CADDY_EXPECTED_ARGV",
        (
            ingress.CADDY_EXECUTABLE,
            "run",
            "--environ",
            "--config",
            str(caddyfile),
        ),
    )
    monkeypatch.setattr(ingress, "OLD_V1_FRAGMENT_PATH", fragment)
    monkeypatch.setattr(ingress, "OLD_V1_FRAGMENT_SHA256", fragment_sha256)
    monkeypatch.setattr(ingress, "EXPECTED_ROOT_UID", fixture_owner.st_uid)
    monkeypatch.setattr(ingress, "EXPECTED_ROOT_GID", fixture_owner.st_gid)
    monkeypatch.setattr(contract, "CADDYFILE_PATH", caddyfile)
    monkeypatch.setattr(contract, "OLD_V1_FRAGMENT_PATH", fragment)
    monkeypatch.setattr(contract, "OLD_V1_FRAGMENT_SHA256", fragment_sha256)
    monkeypatch.setattr(contract, "EXPECTED_ROOT_UID", fixture_owner.st_uid)
    monkeypatch.setattr(contract, "EXPECTED_ROOT_GID", fixture_owner.st_gid)
    monkeypatch.setattr(ingress.sys, "platform", "linux")
    monkeypatch.setattr(ingress.os, "geteuid", lambda: 0)
    commands = _CommandFixture()
    monkeypatch.setattr(ingress, "_run_command", commands)
    monkeypatch.setattr(
        ingress,
        "_caddy_process_snapshot",
        lambda _service: commands.process_snapshots.pop(0),
    )
    monkeypatch.setattr(
        ingress,
        "_old_v1_process_snapshot",
        lambda _service: commands.old_process_snapshots.pop(0),
    )

    def read_live(_process: ingress._CaddyProcessSnapshot) -> bytes:
        if commands.live_outputs is not None:
            if not commands.live_outputs:
                raise AssertionError("too many live config reads")
            return commands.live_outputs.pop(0)
        if commands.last_adapted is None:
            raise AssertionError("live config read before disk adaptation")
        return commands.last_adapted

    monkeypatch.setattr(ingress, "_read_live_caddy_config", read_live)
    return commands


def _collect() -> dict[str, Any]:
    return dict(
        ingress.collect_production_ingress_observation(
            phase="inert",
            release_revision=REVISION,
            plan_sha256=PLAN_SHA256,
            now_unix=NOW,
        )
    )


def _authority() -> dict[str, Any]:
    return {
        "kind": "pinned_owner_gcloud_iap_ssh_read_only",
        "project": ingress.PROJECT,
        "zone": ingress.ZONE,
        "vm": ingress.VM_NAME,
        "instance_id": ingress.INSTANCE_ID,
        "known_hosts_file_sha256": "1" * 64,
        "observer_source_sha256": "2" * 64,
        "instance_authorization_sha256": "3" * 64,
        "project_authorization_sha256": "4" * 64,
        "oslogin_authorization_sha256": "5" * 64,
    }


def test_standalone_reexports_the_pure_fork_pinned_contract() -> None:
    assert ingress.validate_production_ingress_observation is (
        contract.validate_production_ingress_observation
    )
    assert ingress.validate_signed_production_ingress_observation is (
        contract.validate_signed_production_ingress_observation
    )
    assert ingress.ProductionIngressObservationError is (
        contract.ProductionIngressObservationError
    )
    assert contract.PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256 == (
        trust.PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256
    )


def test_provenance_checked_bundle_executes_under_isolated_python_without_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_paths: list[Path] = []

    def local_digest(self, release_revision: str) -> str:
        assert release_revision == REVISION
        observed_paths.append(self._module_path)
        return hashlib.sha256(self._module_path.read_bytes()).hexdigest()

    checkout_root = Path(launcher.__file__).absolute().parents[2]
    sealed_source = tmp_path / f"owner-support-{REVISION}" / "source"
    sealed_observer = (
        sealed_source
        / "scripts/canary/owner_gate_production_ingress_observation.py"
    )
    sealed_contract = (
        sealed_source
        / "scripts/canary/owner_gate_production_ingress_contract.py"
    )
    sealed_observer.parent.mkdir(parents=True)
    sealed_observer.write_bytes(
        (
            checkout_root
            / "scripts/canary/owner_gate_production_ingress_observation.py"
        ).read_bytes()
    )
    sealed_contract.write_bytes(
        (
            checkout_root
            / "scripts/canary/owner_gate_production_ingress_contract.py"
        ).read_bytes()
    )
    monkeypatch.setattr(
        ingress,
        "__file__",
        str(sealed_observer),
    )
    monkeypatch.setattr(
        contract,
        "__file__",
        str(sealed_contract),
    )
    monkeypatch.setattr(
        launcher.LocalLauncherProvenance,
        "__call__",
        local_digest,
    )
    source, source_sha256 = ingress._observer_source(REVISION)
    assert observed_paths == [
        checkout_root
        / "scripts/canary/owner_gate_production_ingress_observation.py",
        checkout_root / "scripts/canary/owner_gate_production_ingress_contract.py",
    ]
    assert source_sha256 == hashlib.sha256(source).hexdigest()
    assert len(source) <= ingress.MAX_REMOTE_SOURCE_BYTES

    completed = subprocess.run(
        (
            sys.executable,
            "-B",
            "-I",
            "-",
            "inert",
            "--release-revision",
            REVISION,
            "--plan-sha256",
            PLAN_SHA256,
        ),
        input=source,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        timeout=30.0,
        check=False,
    )
    assert completed.returncode in {0, 1}
    assert completed.stderr == b""
    result = json.loads(completed.stdout)
    assert result["schema"] in {
        ingress.OBSERVATION_SCHEMA,
        ingress.FAILURE_SCHEMA,
    }
    if completed.returncode:
        assert result["error_code"] != (
            "owner_gate_production_ingress_unexpected_failure"
        )


def test_provenance_checked_bundle_rejects_changed_sealed_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout_root = Path(launcher.__file__).absolute().parents[2]
    sealed_source = tmp_path / f"owner-support-{REVISION}" / "source"
    sealed_observer = (
        sealed_source
        / "scripts/canary/owner_gate_production_ingress_observation.py"
    )
    sealed_contract = (
        sealed_source
        / "scripts/canary/owner_gate_production_ingress_contract.py"
    )
    sealed_observer.parent.mkdir(parents=True)
    sealed_observer.write_bytes(
        (
            checkout_root
            / "scripts/canary/owner_gate_production_ingress_observation.py"
        ).read_bytes()
    )
    sealed_contract.write_bytes(
        (
            checkout_root
            / "scripts/canary/owner_gate_production_ingress_contract.py"
        ).read_bytes()
        + b"\n"
    )

    def local_digest(self, release_revision: str) -> str:
        assert release_revision == REVISION
        return hashlib.sha256(self._module_path.read_bytes()).hexdigest()

    monkeypatch.setattr(ingress, "__file__", str(sealed_observer))
    monkeypatch.setattr(contract, "__file__", str(sealed_contract))
    monkeypatch.setattr(
        launcher.LocalLauncherProvenance,
        "__call__",
        local_digest,
    )
    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_source_invalid",
    ):
        ingress._observer_source(REVISION)


def _resign_envelope(
    value: dict[str, Any],
    private_key: Ed25519PrivateKey,
    *,
    domain: bytes = ingress.SIGNATURE_DOMAIN,
) -> dict[str, Any]:
    unsigned = {
        key: item
        for key, item in value.items()
        if key not in {"signature_ed25519_b64url", "envelope_sha256"}
    }
    signature = private_key.sign(domain + ingress._canonical(unsigned))
    signed = {
        **unsigned,
        "signature_ed25519_b64url": base64.urlsafe_b64encode(signature)
        .rstrip(b"=")
        .decode(),
    }
    return {
        **signed,
        "envelope_sha256": hashlib.sha256(ingress._canonical(signed)).hexdigest(),
    }


def test_collects_exact_safe_projection_without_secret_or_secret_digest(
    production_files: _CommandFixture,
) -> None:
    report = _collect()

    assert report["target"] == {
        "project": ingress.PROJECT,
        "zone": ingress.ZONE,
        "vm": ingress.VM_NAME,
        "instance_id": ingress.INSTANCE_ID,
    }
    assert report["old_v1"]["unit_file_state"] == "enabled"
    assert report["old_v1"]["active_state"] == "active"
    assert report["old_v1"]["fragment_sha256"] == (
        ingress.OLD_V1_FRAGMENT_SHA256
    )
    assert report["old_v1"]["process_cmdline"] == list(
        ingress.OLD_V1_PROCESS_CMDLINE
    )
    assert report["old_v1"]["trusted_for_v2"] is False
    assert report["caddy"]["reverse_proxy_upstreams"] == [
        ingress.LEGACY_V1_UPSTREAM
    ]
    assert report["caddy"]["legacy_v1_upstream_active"] is True
    assert report["caddy"]["private_v2_upstream_active"] is False
    assert report["caddy"]["config_validated"] is True
    assert report["fresh_through_unix"] == NOW + ingress.FRESHNESS_SECONDS
    serialized = ingress._canonical(report)
    assert b"super-secret-token" not in serialized
    assert b"caddyfile_sha256" not in serialized
    assert report["secret_material_recorded"] is False
    assert report["secret_digest_recorded"] is False

    assert production_files.calls.count(ingress._caddy_adapt_command()) == 2
    assert ingress._caddy_adapt_command() == (
        ingress.CADDY_EXECUTABLE,
        "adapt",
        "--config",
        str(ingress.CADDYFILE_PATH),
        "--adapter",
        "caddyfile",
    )
    assert production_files.calls.count(
        ingress._systemctl_command(ingress.OLD_V1_UNIT)
    ) == 2
    assert production_files.calls.count(
        ingress._systemctl_command(ingress.CADDY_UNIT)
    ) == 2
    assert all(
        not any(token in command for token in ("start", "stop", "restart", "reload"))
        for command in production_files.calls
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("ActiveState", "inactive"),
        ("SubState", "dead"),
        ("UnitFileState", "disabled"),
        ("LoadState", "masked"),
        ("FragmentPath", "/dev/null"),
        ("User", "root"),
        ("Group", "root"),
        ("NeedDaemonReload", "yes"),
        ("DropInPaths", "/run/systemd/system/attacker.conf"),
    ),
)
def test_rejects_each_old_v1_unit_drift(
    production_files: _CommandFixture,
    field: str,
    value: str,
) -> None:
    production_files.old_changes[field] = value
    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_old_v1_unsafe",
    ):
        _collect()


def test_rejects_symlinked_v1_fragment(
    production_files: _CommandFixture,
    tmp_path: Path,
) -> None:
    del production_files
    fragment = ingress.OLD_V1_FRAGMENT_PATH
    raw = fragment.read_bytes()
    fragment.unlink()
    target = tmp_path / "attacker-unit"
    target.write_bytes(raw)
    target.chmod(0o644)
    fragment.symlink_to(target)
    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_v1_fragment_invalid",
    ):
        _collect()


def test_rejects_v1_process_change_during_observation(
    production_files: _CommandFixture,
) -> None:
    production_files.old_process_snapshots[1] = ingress._OldV1ProcessSnapshot(
        pid=4343,
        uid=ingress.OLD_V1_UID,
        gid=ingress.OLD_V1_GID,
        start_time_ticks=91,
    )
    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_old_v1_changed",
    ):
        _collect()


def _legacy_process_service() -> Mapping[str, Any]:
    return {
        "main_pid": 4343,
        "exec_main_pid": 4343,
        "exec_start_argv": list(ingress.OLD_V1_EXEC_START_ARGV),
    }


def _legacy_process_stat(start_time: int = 90) -> bytes:
    fields = ["S", *("0" for _ in range(18)), str(start_time)]
    return f"4343 (legacy v1) {' '.join(fields)}\n".encode()


def test_reads_exact_pinned_v1_process_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "status": (
            f"Name:\tpython\nUid:\t{ingress.OLD_V1_UID}\t"
            f"{ingress.OLD_V1_UID}\t{ingress.OLD_V1_UID}\t"
            f"{ingress.OLD_V1_UID}\nGid:\t{ingress.OLD_V1_GID}\t"
            f"{ingress.OLD_V1_GID}\t{ingress.OLD_V1_GID}\t"
            f"{ingress.OLD_V1_GID}\n"
        ).encode(),
        "cmdline": b"\x00".join(
            item.encode() for item in ingress.OLD_V1_PROCESS_CMDLINE
        ) + b"\x00",
        "cgroup": (
            f"0::/system.slice/{ingress.OLD_V1_UNIT}\n"
        ).encode(),
        "stat": _legacy_process_stat(),
    }
    monkeypatch.setattr(
        ingress,
        "_bounded_old_v1_proc_read",
        lambda path: values[path.name],
    )

    snapshot = ingress._old_v1_process_snapshot(_legacy_process_service())

    assert snapshot == ingress._OldV1ProcessSnapshot(
        pid=4343,
        uid=ingress.OLD_V1_UID,
        gid=ingress.OLD_V1_GID,
        start_time_ticks=90,
    )


@pytest.mark.parametrize("drift", ("uid", "cmdline", "cgroup", "restart"))
def test_rejects_each_pinned_v1_process_drift(
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    stat_reads = 0

    def read(path: Path) -> bytes:
        nonlocal stat_reads
        if path.name == "status":
            uid = 0 if drift == "uid" else ingress.OLD_V1_UID
            return (
                f"Uid:\t{uid}\t{uid}\t{uid}\t{uid}\n"
                f"Gid:\t{ingress.OLD_V1_GID}\t{ingress.OLD_V1_GID}\t"
                f"{ingress.OLD_V1_GID}\t{ingress.OLD_V1_GID}\n"
            ).encode()
        if path.name == "cmdline":
            if drift == "cmdline":
                return b"/tmp/attacker\x00"
            return b"\x00".join(
                item.encode() for item in ingress.OLD_V1_PROCESS_CMDLINE
            ) + b"\x00"
        if path.name == "cgroup":
            unit = "attacker.service" if drift == "cgroup" else ingress.OLD_V1_UNIT
            return f"0::/system.slice/{unit}\n".encode()
        if path.name == "stat":
            stat_reads += 1
            return _legacy_process_stat(
                91 if drift == "restart" and stat_reads == 2 else 90
            )
        raise AssertionError(path)

    monkeypatch.setattr(ingress, "_bounded_old_v1_proc_read", read)
    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_old_v1_process_unsafe",
    ):
        ingress._old_v1_process_snapshot(_legacy_process_service())


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("ActiveState", "inactive"),
        ("SubState", "dead"),
        ("UnitFileState", "disabled"),
        ("FragmentPath", "/etc/systemd/system/caddy.service"),
        ("DropInPaths", "/etc/systemd/system/caddy.service.d/attacker.conf"),
    ),
)
def test_rejects_each_caddy_service_drift(
    production_files: _CommandFixture,
    field: str,
    value: str,
) -> None:
    production_files.caddy_changes[field] = value
    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_caddy_service_unsafe",
    ):
        _collect()


def test_rejects_modified_base_unit_with_alternate_config(
    production_files: _CommandFixture,
) -> None:
    production_files.caddy_changes["ExecStart"] = (
        "{ path=/usr/bin/caddy ; "
        "argv[]=/usr/bin/caddy run --environ --config /tmp/attacker.json ; "
        "ignore_errors=no ; start_time=[n/a] ; stop_time=[n/a] ; "
        "pid=4242 ; code=(null) ; status=0/0 }"
    )
    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_caddy_service_unsafe",
    ):
        _collect()


def test_rejects_safe_disk_config_when_live_admin_route_is_private_v2(
    production_files: _CommandFixture,
) -> None:
    malicious = _adapted(dial=ingress.PRIVATE_V2_UPSTREAM)
    production_files.live_outputs = [malicious, malicious]
    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_private_v2_already_active",
    ):
        _collect()


def test_rejects_changed_live_process_identity(
    production_files: _CommandFixture,
) -> None:
    production_files.process_snapshots[1] = ingress._CaddyProcessSnapshot(
        pid=4243,
        identity=(4243, 101, ingress.CADDY_EXECUTABLE, "socket"),
        admin_host="127.0.0.1",
        admin_socket_inode="54321",
    )
    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_caddy_changed",
    ):
        _collect()


def test_process_snapshot_binds_running_executable_and_exact_cmdline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "caddy"
    executable.write_bytes(b"trusted-caddy-binary")
    executable.chmod(0o755)
    state = executable.stat()
    expected_argv = (
        str(executable),
        "run",
        "--environ",
        "--config",
        "/etc/caddy/Caddyfile",
    )
    monkeypatch.setattr(ingress, "CADDY_EXECUTABLE", str(executable))
    monkeypatch.setattr(ingress, "CADDY_EXPECTED_ARGV", expected_argv)
    monkeypatch.setattr(ingress, "EXPECTED_ROOT_UID", state.st_uid)
    monkeypatch.setattr(ingress, "EXPECTED_ROOT_GID", state.st_gid)
    monkeypatch.setattr(ingress.os, "readlink", lambda _path: str(executable))
    original_stat = ingress.os.stat

    def observed_stat(
        path: int | str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        if str(path) == "/proc/4242/exe":
            return state
        return original_stat(
            path,
            dir_fd=dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(ingress.os, "stat", observed_stat)
    expected_cmdline = b"\x00".join(
        item.encode() for item in expected_argv
    ) + b"\x00"
    observed_cmdline = expected_cmdline
    monkeypatch.setattr(
        ingress,
        "_bounded_proc_read",
        lambda path: observed_cmdline
        if str(path) == "/proc/4242/cmdline"
        else b"unused",
    )
    monkeypatch.setattr(ingress, "_process_start_time", lambda _pid: 100)
    monkeypatch.setattr(
        ingress,
        "_admin_listener_for_pid",
        lambda _pid: ("127.0.0.1", "12345"),
    )
    service = {"main_pid": 4242, "exec_start_argv": list(expected_argv)}

    snapshot = ingress._caddy_process_snapshot(service)
    assert snapshot.pid == 4242
    assert snapshot.admin_socket_inode == "12345"

    observed_cmdline = expected_cmdline.replace(
        b"/etc/caddy/Caddyfile",
        b"/tmp/attacker.json",
    )
    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_caddy_process_unsafe",
    ):
        ingress._caddy_process_snapshot(service)


@pytest.mark.parametrize(
    "adapted",
    (
        _adapted(dial=ingress.PRIVATE_V2_UPSTREAM),
        _adapted(duplicate_route=True),
        b'{"apps":{"http":{"servers":{"srv0":{"routes":[]}}}}}',
    ),
)
def test_rejects_private_v2_missing_or_ambiguous_auth_route(
    production_files: _CommandFixture,
    adapted: bytes,
) -> None:
    production_files.adapted_outputs = [adapted, adapted]
    with pytest.raises(ingress.ProductionIngressObservationError):
        _collect()


@pytest.mark.parametrize(
    "malicious_route",
    (
        _proxy_route(ingress.PRIVATE_V2_UPSTREAM),
        _proxy_route("[::ffff:10.80.3.2]:8080"),
        _proxy_route(
            ingress.PRIVATE_V2_UPSTREAM,
            host="*.lomliev.com",
        ),
        _proxy_route(
            ingress.PRIVATE_V2_UPSTREAM,
            nested=True,
        ),
    ),
)
def test_rejects_private_v2_in_any_public_host_capable_route(
    production_files: _CommandFixture,
    malicious_route: Mapping[str, Any],
) -> None:
    value = _adapted_value()
    routes = value["apps"]["http"]["servers"]["srv0"]["routes"]
    routes.insert(0, copy.deepcopy(malicious_route))
    adapted = _adapted_raw(value)
    production_files.adapted_outputs = [adapted, adapted]
    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_private_v2_already_active",
    ):
        _collect()


def test_rejects_private_v2_on_overlapping_listener_server(
    production_files: _CommandFixture,
) -> None:
    value = _adapted_value()
    value["apps"]["http"]["servers"]["shadow"] = {
        "listen": ["0.0.0.0:443"],
        "routes": [_proxy_route(ingress.PRIVATE_V2_UPSTREAM)],
    }
    adapted = _adapted_raw(value)
    production_files.adapted_outputs = [adapted, adapted]
    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_private_v2_already_active",
    ):
        _collect()


def test_rejects_unknown_handler_in_effective_public_route(
    production_files: _CommandFixture,
) -> None:
    value = _adapted_value()
    route = value["apps"]["http"]["servers"]["srv0"]["routes"][0]
    route["handle"] = [
        {"handler": "third_party_network_tunnel"},
        *route["handle"],
    ]
    adapted = _adapted_raw(value)
    production_files.adapted_outputs = [adapted, adapted]
    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_caddy_route_unsafe",
    ):
        _collect()


def test_rejects_unreachable_local_proxy_after_terminal_handler(
    production_files: _CommandFixture,
) -> None:
    value = _adapted_value()
    route = value["apps"]["http"]["servers"]["srv0"]["routes"][0]
    route["handle"] = [
        {"handler": "static_response", "status_code": 200},
        {
            "handler": "reverse_proxy",
            "upstreams": [{"dial": "127.0.0.1:7341"}],
        },
    ]
    adapted = _adapted_raw(value)
    production_files.adapted_outputs = [adapted, adapted]
    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_caddy_route_unsafe",
    ):
        _collect()


def test_private_v2_on_disjoint_listener_is_not_public_route(
    production_files: _CommandFixture,
) -> None:
    value = _adapted_value()
    value["apps"]["http"]["servers"]["internal"] = {
        "listen": ["127.0.0.1:8443"],
        "routes": [_proxy_route(ingress.PRIVATE_V2_UPSTREAM)],
    }
    adapted = _adapted_raw(value)
    production_files.adapted_outputs = [adapted, adapted]
    report = _collect()
    assert report["caddy"]["still_on_current_host"] is True
    assert report["caddy"]["private_v2_upstream_active"] is False


def test_unrelated_host_route_does_not_enter_public_route_graph(
    production_files: _CommandFixture,
) -> None:
    value = _adapted_value()
    routes = value["apps"]["http"]["servers"]["srv0"]["routes"]
    routes.insert(
        0,
        _proxy_route(
            ingress.PRIVATE_V2_UPSTREAM,
            host="unrelated.example.com",
            nested=True,
        ),
    )
    adapted = _adapted_raw(value)
    production_files.adapted_outputs = [adapted, adapted]
    report = _collect()
    assert report["caddy"]["still_on_current_host"] is True


@pytest.mark.parametrize(
    "dial",
    (
        "127.0.0.1:7342",
        "192.0.2.10:7341",
        "localhost:7341",
        "{env.LEGACY_UPSTREAM}:7341",
    ),
)
def test_still_on_current_host_requires_positive_local_upstream_evidence(
    production_files: _CommandFixture,
    dial: str,
) -> None:
    adapted = _adapted(dial=dial)
    production_files.adapted_outputs = [adapted, adapted]
    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_caddy_route_unsafe",
    ):
        _collect()


def test_rejects_two_nonidentical_adapted_documents_even_when_projection_matches(
    production_files: _CommandFixture,
) -> None:
    second = _adapted_value()
    second["apps"]["tls"] = {
        "certificates": {"automate": [ingress.PUBLIC_HOST]}
    }
    production_files.adapted_outputs = [
        _adapted(),
        _adapted_raw(second),
    ]
    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_caddy_changed",
    ):
        _collect()


def test_rejects_caddyfile_symlink(
    production_files: _CommandFixture,
    tmp_path: Path,
) -> None:
    del production_files
    path = ingress.CADDYFILE_PATH
    raw = path.read_bytes()
    path.unlink()
    target = tmp_path / "attacker-caddyfile"
    target.write_bytes(raw)
    target.chmod(0o644)
    path.symlink_to(target)
    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_caddyfile_invalid",
    ):
        _collect()


def test_rejects_caddyfile_change_between_adaptations(
    production_files: _CommandFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = production_files.__call__
    adaptations = 0

    def changing(argv: tuple[str, ...], *, maximum_output_bytes: int) -> bytes:
        nonlocal adaptations
        result = original(argv, maximum_output_bytes=maximum_output_bytes)
        if argv == ingress._caddy_adapt_command():
            adaptations += 1
            if adaptations == 1:
                ingress.CADDYFILE_PATH.write_text("changed while observing\n")
                ingress.CADDYFILE_PATH.chmod(0o644)
        return result

    monkeypatch.setattr(ingress, "_run_command", changing)
    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_caddy_changed",
    ):
        _collect()


def test_remote_report_validation_rejects_semantic_tamper_and_staleness(
    production_files: _CommandFixture,
) -> None:
    del production_files
    report = _collect()
    tampered = copy.deepcopy(report)
    tampered["old_v1"]["trusted_for_v2"] = True
    unsigned = {key: item for key, item in tampered.items() if key != "report_sha256"}
    tampered["report_sha256"] = hashlib.sha256(ingress._canonical(unsigned)).hexdigest()
    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_old_v1_invalid",
    ):
        ingress.validate_production_ingress_observation(
            tampered,
            phase="inert",
            release_revision=REVISION,
            plan_sha256=PLAN_SHA256,
            now_unix=NOW,
        )
    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_observation_invalid",
    ):
        ingress.validate_production_ingress_observation(
            report,
            phase="inert",
            release_revision=REVISION,
            plan_sha256=PLAN_SHA256,
            now_unix=NOW + ingress.FRESHNESS_SECONDS + 1,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("fragment_sha256", "f" * 64),
        ("process_cmdline", ["/tmp/attacker"]),
        ("active_process_stable", False),
    ),
)
def test_remote_report_rejects_resigned_v1_identity_drift(
    production_files: _CommandFixture,
    field: str,
    value: Any,
) -> None:
    del production_files
    tampered = copy.deepcopy(_collect())
    tampered["old_v1"][field] = value
    unsigned = {
        key: item for key, item in tampered.items() if key != "report_sha256"
    }
    tampered["report_sha256"] = hashlib.sha256(
        ingress._canonical(unsigned)
    ).hexdigest()

    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_old_v1_invalid",
    ):
        ingress.validate_production_ingress_observation(
            tampered,
            phase="inert",
            release_revision=REVISION,
            plan_sha256=PLAN_SHA256,
            now_unix=NOW,
        )


def test_remote_report_rejects_resigned_alternate_local_caddy_route(
    production_files: _CommandFixture,
) -> None:
    del production_files
    tampered = copy.deepcopy(_collect())
    tampered["caddy"]["reverse_proxy_upstreams"] = ["127.0.0.1:7342"]
    route_projection = {
        name: tampered["caddy"][name]
        for name in (
            "auth_host_route_count",
            "reverse_proxy_handler_count",
            "reverse_proxy_upstream_count",
            "reverse_proxy_upstreams",
            "legacy_v1_upstream_active",
            "still_on_current_host",
            "private_v2_upstream_active",
        )
    }
    tampered["caddy"]["live_route_projection_sha256"] = hashlib.sha256(
        ingress._canonical(route_projection)
    ).hexdigest()
    unsigned = {
        key: item for key, item in tampered.items() if key != "report_sha256"
    }
    tampered["report_sha256"] = hashlib.sha256(
        ingress._canonical(unsigned)
    ).hexdigest()

    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_caddy_invalid",
    ):
        ingress.validate_production_ingress_observation(
            tampered,
            phase="inert",
            release_revision=REVISION,
            plan_sha256=PLAN_SHA256,
            now_unix=NOW,
        )


def test_release_signed_envelope_round_trip_and_exact_digest_semantics(
    production_files: _CommandFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del production_files
    observation = _collect()
    private_key = Ed25519PrivateKey.generate()
    key_id = hashlib.sha256(private_key.public_key().public_bytes_raw()).hexdigest()
    monkeypatch.setattr(trust, "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256", key_id)
    monkeypatch.setattr(
        contract,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        key_id,
    )
    transport = object.__new__(ingress.OwnerGateProductionIngressTransport)
    monkeypatch.setattr(
        ingress.OwnerGateProductionIngressTransport,
        "observe",
        lambda self, **kwargs: (observation, _authority()),
    )

    envelope = ingress.collect_and_sign_production_ingress_observation(
        transport,
        phase="inert",
        release_revision=REVISION,
        plan_sha256=PLAN_SHA256,
        release_private_key=private_key,
        now_unix=NOW,
    )

    assert envelope["observer_report_sha256"] == observation["report_sha256"]
    assert envelope["fresh_through_unix"] == observation["fresh_through_unix"]
    assert envelope["signer_key_id"] == key_id
    signed = {key: item for key, item in envelope.items() if key != "envelope_sha256"}
    assert envelope["envelope_sha256"] == hashlib.sha256(
        ingress._canonical(signed)
    ).hexdigest()
    assert ingress.validate_signed_production_ingress_observation(
        envelope,
        phase="inert",
        release_revision=REVISION,
        plan_sha256=PLAN_SHA256,
        release_public_key=private_key.public_key(),
        now_unix=NOW,
    ) == envelope


def test_signed_envelope_rejects_wrong_domain_tamper_wrong_key_and_staleness(
    production_files: _CommandFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del production_files
    observation = _collect()
    private_key = Ed25519PrivateKey.generate()
    key_id = hashlib.sha256(private_key.public_key().public_bytes_raw()).hexdigest()
    monkeypatch.setattr(trust, "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256", key_id)
    monkeypatch.setattr(
        contract,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        key_id,
    )
    transport = object.__new__(ingress.OwnerGateProductionIngressTransport)
    monkeypatch.setattr(
        ingress.OwnerGateProductionIngressTransport,
        "observe",
        lambda self, **kwargs: (observation, _authority()),
    )
    envelope = dict(
        ingress.collect_and_sign_production_ingress_observation(
            transport,
            phase="inert",
            release_revision=REVISION,
            plan_sha256=PLAN_SHA256,
            release_private_key=private_key,
            now_unix=NOW,
        )
    )

    wrong_domain = _resign_envelope(
        envelope,
        private_key,
        domain=b"attacker-domain\x00",
    )
    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_signature_invalid",
    ):
        ingress.validate_signed_production_ingress_observation(
            wrong_domain,
            phase="inert",
            release_revision=REVISION,
            plan_sha256=PLAN_SHA256,
            release_public_key=private_key.public_key(),
            now_unix=NOW,
        )

    tampered = copy.deepcopy(envelope)
    tampered["transport_authority"]["instance_id"] = "9999999999999999999"
    tampered = _resign_envelope(tampered, private_key)
    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_transport_authority_invalid",
    ):
        ingress.validate_signed_production_ingress_observation(
            tampered,
            phase="inert",
            release_revision=REVISION,
            plan_sha256=PLAN_SHA256,
            release_public_key=private_key.public_key(),
            now_unix=NOW,
        )

    wrong_key = Ed25519PrivateKey.generate()
    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_signer_not_pinned",
    ):
        ingress.validate_signed_production_ingress_observation(
            envelope,
            phase="inert",
            release_revision=REVISION,
            plan_sha256=PLAN_SHA256,
            release_public_key=wrong_key.public_key(),
            now_unix=NOW,
        )

    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_envelope_invalid",
    ):
        ingress.validate_signed_production_ingress_observation(
            envelope,
            phase="inert",
            release_revision=REVISION,
            plan_sha256=PLAN_SHA256,
            release_public_key=private_key.public_key(),
            now_unix=NOW + ingress.FRESHNESS_SECONDS + 1,
        )


def test_transport_runs_exact_committed_source_over_bounded_pinned_stdin(
    production_files: _CommandFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del production_files
    observation = _collect()
    captured: dict[str, Any] = {}

    class Identity:
        def account_for_read_only_preflight(self) -> str:
            return "owner@example.com"

        def require_stable(self) -> None:
            captured["stable"] = int(captured.get("stable", 0)) + 1

    class KnownHosts:
        @staticmethod
        def absolute_path() -> str:
            return "/trusted/google_compute_known_hosts"

    class Transport:
        _owner_identity = Identity()
        _known_hosts = KnownHosts()

        @staticmethod
        def _fixed_remote_environment(*, chdir: str) -> tuple[str, ...]:
            assert chdir == "/"
            return ("/usr/bin/env", "-i", "--chdir=/")

        @staticmethod
        def _authorization_snapshot(account: str) -> tuple[str, str, str]:
            assert account == "owner@example.com"
            return ("1" * 64, "2" * 64, "3" * 64)

        @staticmethod
        def _run_remote_input(command: tuple[str, ...], **kwargs: Any):
            captured["command"] = command
            captured.update(kwargs)
            stdout = ingress._canonical(observation) + b"\n"
            return subprocess.CompletedProcess(command, 0, stdout, b"")

    source = b"print('reviewed observer source')\n"
    monkeypatch.setattr(ingress, "_observer_source", lambda revision: (source, "9" * 64))
    monkeypatch.setattr(ingress.time, "time", lambda: NOW)
    monkeypatch.setattr(
        ingress,
        "_stable_owner_file",
        lambda path, *, maximum: b"trusted-known-hosts\n",
    )

    observed, authority = ingress.OwnerGateProductionIngressTransport(
        Transport()
    ).observe(
        phase="inert",
        release_revision=REVISION,
        plan_sha256=PLAN_SHA256,
    )

    assert observed == observation
    assert captured["command"] == (
        "/usr/bin/env",
        "-i",
        "--chdir=/",
        ingress.REMOTE_PYTHON,
        "-B",
        "-I",
        "-",
        "inert",
        "--release-revision",
        REVISION,
        "--plan-sha256",
        PLAN_SHA256,
    )
    assert "-c" not in captured["command"]
    assert captured["input_bytes"] == source
    assert captured["maximum_input_bytes"] == ingress.MAX_REMOTE_SOURCE_BYTES
    assert captured["maximum_output_bytes"] == ingress.MAX_REMOTE_OUTPUT_BYTES
    assert captured["timeout_seconds"] == 120
    assert captured["stable"] == 2
    assert authority["observer_source_sha256"] == "9" * 64
    assert authority["known_hosts_file_sha256"] == hashlib.sha256(
        b"trusted-known-hosts\n"
    ).hexdigest()


def test_transport_rejects_authorization_change_after_remote_observation(
    production_files: _CommandFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del production_files
    observation = _collect()
    snapshots = iter(
        [
            ("1" * 64, "2" * 64, "3" * 64),
            ("4" * 64, "5" * 64, "6" * 64),
        ]
    )

    class Identity:
        @staticmethod
        def account_for_read_only_preflight() -> str:
            return "owner@example.com"

        @staticmethod
        def require_stable() -> None:
            return None

    class KnownHosts:
        @staticmethod
        def absolute_path() -> str:
            return "/trusted/google_compute_known_hosts"

    class Transport:
        _owner_identity = Identity()
        _known_hosts = KnownHosts()

        @staticmethod
        def _fixed_remote_environment(*, chdir: str) -> tuple[str, ...]:
            return ("/usr/bin/env", "-i", f"--chdir={chdir}")

        @staticmethod
        def _authorization_snapshot(account: str) -> tuple[str, str, str]:
            del account
            return next(snapshots)

        @staticmethod
        def _run_remote_input(command: tuple[str, ...], **kwargs: Any):
            del kwargs
            return subprocess.CompletedProcess(
                command,
                0,
                ingress._canonical(observation) + b"\n",
                b"",
            )

    monkeypatch.setattr(
        ingress,
        "_observer_source",
        lambda revision: (b"print('source')\n", "9" * 64),
    )
    monkeypatch.setattr(ingress.time, "time", lambda: NOW)
    with pytest.raises(
        ingress.ProductionIngressObservationError,
        match="owner_gate_production_ingress_transport_changed",
    ):
        ingress.OwnerGateProductionIngressTransport(Transport()).observe(
            phase="inert",
            release_revision=REVISION,
            plan_sha256=PLAN_SHA256,
        )

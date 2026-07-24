from __future__ import annotations

import base64
import copy
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.operational_edge_catalog import (
    operation_catalog,
    required_cron_operations,
)
from gateway.operational_edge_readiness import (
    PROBE_PACKET_SCHEMA,
    build_operational_edge_readiness,
)
from ops.muncho.runtime import trusted_cron_collector_rail as rail


EDGE_BOOT_ID_SHA256 = "f" * 64
EDGE_OBSERVED_AT_UNIX = 1_800_000_000
EDGE_COLLECTOR_NONCE = "12345678-1234-4123-8123-123456789abc"


def _dependency_facts() -> dict[str, str]:
    return {
        path: f"{index + 1:064x}"
        for index, path in enumerate(
            sorted(
                {str(rail.SETPRIV)}
                | {
                    path
                    for spec in rail.COLLECTOR_SPECS
                    for path in spec.dependency_paths
                }
            )
        )
    }


def _package() -> dict:
    return rail.build_package_manifest(
        revision="a" * 40,
        rail_sha256="b" * 64,
        dependency_facts=_dependency_facts(),
    )


def _operational_edge_receipt() -> dict:
    required = dict(required_cron_operations())
    catalog = operation_catalog()
    jobs = []
    for index, (job_id, operation_id) in enumerate(required.items(), start=1):
        operation = catalog[operation_id]
        jobs.append(
            {
                "source_job_id": job_id,
                "operation_id": operation_id,
                "domain": operation.domain,
                "service_unit": (
                    f"muncho-operational-edge-{operation.domain}.service"
                ),
                "service_uid": 3000 + index,
                "service_gid": 4000 + index,
                "socket_path": (
                    f"/run/muncho-operational-edge/{operation.domain}/edge.sock"
                ),
                "socket_uid": 3000 + index,
                "socket_gid": 4000 + index,
                "socket_mode": "0660",
                "main_pid": 5000 + index,
                "peer_round_trip": True,
                "probe_operation_id": (
                    operation.probe_operation_id or operation.operation_id
                ),
                "probe_return_code": 0,
                "probe_packet_schema": PROBE_PACKET_SCHEMA,
                "probe_packet_sha256": f"{index:064x}",
                "meaningful_packet": True,
                "error_only_packet": False,
            }
        )
    return build_operational_edge_readiness(
        revision="a" * 40,
        required_jobs=required,
        jobs=jobs,
        boot_id_sha256=EDGE_BOOT_ID_SHA256,
        observed_at_unix=EDGE_OBSERVED_AT_UNIX,
        collector_nonce=EDGE_COLLECTOR_NONCE,
    )


def _unit_namespace_receipt(package: dict | None = None) -> dict:
    package = _package() if package is None else package
    rows = [
        {
            "source_job_id": item.source_job_id,
            "execution_boundary": item.execution_boundary,
            "unit_profile_sha256": rail.service_profile_sha256(
                item,
                revision=package["release_revision"],
            ),
            "safe_probe_operation_id": rail._safe_probe_operation_id(item),
            "return_code": 0,
            "namespace_probe_succeeded": True,
            "job_mutation_executed": False,
            "secret_material_recorded": False,
        }
        for item in rail.COLLECTOR_SPECS
    ]
    return rail._namespace_readiness_receipt(
        package,
        rows,
        collected_at="2026-07-24T00:00:00Z",
        boot_id_sha256=EDGE_BOOT_ID_SHA256,
        observed_at_unix=EDGE_OBSERVED_AT_UNIX,
        collector_nonce=EDGE_COLLECTOR_NONCE,
    )


def test_package_is_exact_credential_isolated_and_never_activates() -> None:
    package = _package()
    units = rail.render_package_unit_files(package)
    contract = rail.catalog_public_contract()

    assert rail.validate_package_manifest(package) == package
    assert len(contract) == 21
    assert len(units) == 42
    assert package["isolated_service_user"] == "ai-platform-brain"
    assert package["isolated_service_group"] == "ai-platform-brain"
    assert package["scoped_service_user"] == "muncho-projector"
    assert package["scoped_service_group"] == "muncho-projector"
    assert package["reader_group"] == "ai-platform-brain"
    assert package["provider_or_model_dependency"] is False
    assert package["gateway_credential_dependency"] is False
    assert package["discord_credential_dependency"] is False
    assert package["timer_enabled_by_package"] is False
    assert package["timer_started_by_package"] is False
    assert all(row["semantic_judgment_allowed"] is False for row in contract)
    assert all(row["provider_or_model_allowed"] is False for row in contract)
    assert all(row["direct_discord_allowed"] is False for row in contract)
    assert all(
        row["historical_source_delivery_eligible"] is False
        for row in contract
    )
    by_job = {row["source_job_id"]: row for row in contract}
    assert by_job["06ef64d72891"]["historical_source_delivery"] == (
        "discord:1504852355588423801"
    )
    assert by_job["e62f55ca93ca"]["historical_source_delivery"] == (
        "discord:1504852355588423801:1524321461714681976"
    )
    assert any(
        row["execution_boundary"] == rail.EXECUTION_BOUNDARY_SCOPED
        for row in contract
    )

    services = b"\n".join(
        value for name, value in units.items() if name.endswith(".service")
    )
    timers = b"\n".join(
        value for name, value in units.items() if name.endswith(".timer")
    )
    for forbidden in (
        b"EnvironmentFile=",
        b"LoadCredential=",
        b"OPENAI_API_KEY",
        b"DISCORD_BOT_TOKEN",
        b"PassEnvironment=",
    ):
        assert forbidden not in services
    assert b"User=root" in services
    assert b"Group=root" in services
    assert b"/usr/bin/setpriv --reuid ai-platform-brain" in services
    assert b"/usr/bin/setpriv --reuid muncho-projector" in services
    assert b"SupplementaryGroups=" in services
    assert b"PrivateNetwork=yes" in services
    assert b"Persistent=true" not in timers
    assert b"Persistent=false" in timers


def test_units_use_exact_split_identity_and_whitelist_namespaces() -> None:
    units = rail.render_package_unit_files(_package())
    contract = {row["source_job_id"]: row for row in rail.catalog_public_contract()}

    for job_id, row in contract.items():
        service = units[f"systemd/muncho-cron-{job_id}.service"]
        text = service.decode()
        item = next(
            spec for spec in rail.COLLECTOR_SPECS
            if spec.source_job_id == job_id
        )
        release = (
            rail.RELEASES_ROOT
            / f"hermes-agent-{'a' * 12}"
        )
        assert "User=root\n" in text
        assert "Group=root\n" in text
        assert "SupplementaryGroups=\n" in text
        assert "SetLoginEnvironment=no\n" in text
        assert b"PrivateNetwork=yes\n" in service
        assert b"PrivateIPC=yes\n" in service
        assert b"KeyringMode=private\n" in service
        assert b"RestrictAddressFamilies=AF_UNIX\n" in service
        assert b"ExecStartPre=" not in service
        assert b"prepare-isolated-access" not in service
        assert b"setfacl" not in service
        assert b"CAP_DAC" not in service
        assert (
            b"CapabilityBoundingSet=CAP_SETUID CAP_SETGID\n"
            in service
        )
        assert b"AmbientCapabilities=\n" in service
        assert "ReadWritePaths=" not in text
        assert f"BindPaths={rail.STATE_ROOT}\n" not in text
        assert "InaccessiblePaths=-" not in text
        for root in rail.NAMESPACE_MASKS:
            assert f"TemporaryFileSystem={root}:ro\n" in text
        assert (
            f"BindReadOnlyPaths={release}:{release}:norbind\n"
            in text
        )
        assert (
            f"BindReadOnlyPaths={rail.MANIFEST_PATH}:"
            f"{rail.MANIFEST_PATH}:norbind\n"
        ) in text
        packet_root = rail.PACKET_ROOT / job_id
        assert (
            f"BindPaths={packet_root}:{packet_root}:norbind\n"
            in text
        )
        assert (
            f"ExecStart={rail.SETPRIV} " in text
            and " --inh-caps=-all --ambient-caps=-all "
            "--bounding-set=-all -- " in text
        )
        if row["execution_boundary"] == rail.EXECUTION_BOUNDARY_SCOPED:
            _operation_id, domain = rail._scoped_operation(item)
            assert (
                f"--reuid muncho-projector --regid muncho-projector "
                f"--groups muncho-edge-{domain}-c "
                in text
            )
            assert (
                f"BindReadOnlyPaths={rail.OPERATIONAL_EDGE_CLIENT_CONFIG}:"
                f"{rail.OPERATIONAL_EDGE_CLIENT_CONFIG}:norbind\n"
                in text
            )
            trust_key = (
                rail.OPERATIONAL_EDGE_TRUST_ROOT
                / f"{domain}-receipt-public.pem"
            )
            assert (
                f"BindReadOnlyPaths={trust_key}:{trust_key}:norbind\n"
                in text
            )
            assert (
                f"BindReadOnlyPaths={rail.OPERATIONAL_EDGE_TRUST_ROOT}:"
                not in text
            )
            assert (
                f"BindReadOnlyPaths={rail.OPERATIONAL_EDGE_SOCKET_ROOT / domain / 'edge.sock'}:"
                f"{rail.OPERATIONAL_EDGE_SOCKET_ROOT / domain / 'edge.sock'}:"
                "norbind\n"
                in text
            )
            for path in (
                Path("/run/systemd/private"),
                Path("/run/systemd/system"),
                Path("/run/dbus/system_bus_socket"),
            ):
                assert (
                    f"BindReadOnlyPaths={path}:{path}:norbind\n"
                    in text
                )
            assert "InaccessiblePaths=/proc\n" not in text
        else:
            assert (
                "--reuid ai-platform-brain --regid ai-platform-brain "
                "--clear-groups " in text
            )
            assert "InaccessiblePaths=/proc\n" in text
            assert "SystemCallFilter=~kill tkill tgkill" in text
            for root in item.json_roots:
                assert (
                    f"BindReadOnlyPaths={root}:{root}:norbind\n"
                    in text
                )
            if item.mode == "voice_stage":
                assert (
                    f"BindPaths={rail.VOICE_ROOT}:{rail.VOICE_ROOT}:norbind\n"
                    in text
                )
            else:
                assert str(rail.VOICE_ROOT) not in text
            assert str(rail.OPERATIONAL_EDGE_CLIENT_CONFIG) not in text


def test_transient_probe_reuses_the_final_service_profile() -> None:
    package = _package()
    item = next(
        spec for spec in rail.COLLECTOR_SPECS
        if spec.source_job_id == "a7b15e3dea75"
    )
    manifest_path = Path("/run/readiness/manifest.json")
    argv = rail.transient_probe_argv(
        item,
        manifest_path,
        package,
        unit_name=(
            "muncho-cron-readiness-a7b15e3dea75-"
            "1234567890abcdef.service"
        ),
    )
    properties = {
        argument.removeprefix("--property=")
        for argument in argv
        if argument.startswith("--property=")
    }

    assert set(
        rail.service_profile_properties(
            item,
            revision=package["release_revision"],
            manifest_path=manifest_path,
        )
    ).issubset(properties)
    separator = argv.index("--")
    command = argv[separator + 1 :]
    assert command[:2] == (str(rail.SETPRIV), "--reuid")
    assert "probe" in command
    assert "run" not in command
    assert "prepare-isolated-access" not in command


def test_collect_unit_namespace_readiness_runs_all_jobs_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _package()
    directory = tmp_path / "namespace"
    directory.mkdir()
    manifest_path = directory / "manifest.json"
    manifest_path.write_text("{}", encoding="ascii")
    calls: list[list[str]] = []

    monkeypatch.setattr(
        rail,
        "validate_setpriv_toolchain",
        lambda _manifest: None,
    )
    monkeypatch.setattr(
        rail,
        "_write_transient_manifest",
        lambda _manifest: (directory, manifest_path),
    )
    monkeypatch.setattr(
        rail,
        "_namespace_boot_id_sha256",
        lambda: EDGE_BOOT_ID_SHA256,
    )
    monkeypatch.setattr(
        rail.time,
        "time",
        lambda: EDGE_OBSERVED_AT_UNIX,
    )

    def runner(argv, **_kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    receipt = rail.collect_unit_namespace_readiness(
        package,
        runner=runner,
    )

    assert rail.validate_unit_namespace_readiness(
        receipt,
        manifest=package,
        expected_boot_id_sha256=EDGE_BOOT_ID_SHA256,
        now_unix=EDGE_OBSERVED_AT_UNIX,
    ) == receipt
    assert receipt["job_count"] == 21
    assert receipt["all_jobs_ready"] is True
    assert receipt["safe_read_probes_only"] is True
    assert receipt["job_mutation_executed"] is False
    assert len(calls) == 21
    assert all("probe" in call for call in calls)
    assert all(
        "run" not in call[call.index("--") + 1 :]
        for call in calls
    )
    assert not directory.exists()


def test_unit_namespace_readiness_rejects_profile_or_job_drift() -> None:
    package = _package()
    receipt = _unit_namespace_receipt(package)
    receipt["jobs"][0]["unit_profile_sha256"] = "f" * 64
    unsigned = {
        name: item
        for name, item in receipt.items()
        if name != "receipt_sha256"
    }
    receipt["receipt_sha256"] = rail._sha256(rail._canonical(unsigned))

    with pytest.raises(
        rail.TrustedCronCollectorError,
        match="unit_namespace_readiness_invalid",
    ):
        rail.validate_unit_namespace_readiness(
            receipt,
            manifest=package,
            expected_boot_id_sha256=EDGE_BOOT_ID_SHA256,
            now_unix=EDGE_OBSERVED_AT_UNIX,
        )


def test_unit_namespace_readiness_is_boot_bound_and_short_lived() -> None:
    package = _package()
    receipt = _unit_namespace_receipt(package)

    with pytest.raises(
        rail.TrustedCronCollectorError,
        match="unit_namespace_readiness_invalid",
    ):
        rail.validate_unit_namespace_readiness(
            receipt,
            manifest=package,
            expected_boot_id_sha256="e" * 64,
            now_unix=EDGE_OBSERVED_AT_UNIX,
        )

    with pytest.raises(
        rail.TrustedCronCollectorError,
        match="unit_namespace_readiness_invalid",
    ):
        rail.validate_unit_namespace_readiness(
            receipt,
            manifest=package,
            expected_boot_id_sha256=EDGE_BOOT_ID_SHA256,
            now_unix=(
                EDGE_OBSERVED_AT_UNIX
                + rail.UNIT_NAMESPACE_READINESS_MAXIMUM_AGE_SECONDS
                + 1
            ),
        )


def test_isolated_tree_probe_is_complete_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    child = root / "nested"
    child.mkdir(parents=True)
    payload = child / "evidence.json"
    payload.write_text("{}", encoding="utf-8")

    assert rail._probe_isolated_tree(root) == 3

    original_access = os.access
    monkeypatch.setattr(
        rail.os,
        "access",
        lambda path, mode, **kwargs: (
            False
            if Path(path) == payload
            else original_access(path, mode, **kwargs)
        ),
    )
    with pytest.raises(
        rail.TrustedCronCollectorError,
        match="isolated_probe_denied",
    ):
        rail._probe_isolated_tree(root)


def test_json_tree_never_silently_skips_a_read_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    payload = root / "evidence.json"
    payload.write_text("{}", encoding="utf-8")
    item = rail.CollectorSpec(
        source_job_id="123456789abc",
        rail_id="test-read",
        schedule={"kind": "interval", "minutes": 1},
        mode="json_tree",
        model_review_required=True,
        json_roots=(str(root),),
        json_include_content=True,
    )
    original = rail._stable_read
    monkeypatch.setattr(
        rail,
        "_stable_read",
        lambda path, **kwargs: (
            (_ for _ in ()).throw(
                rail.TrustedCronCollectorError("read_failed")
            )
            if path == payload
            else original(path, **kwargs)
        ),
    )

    with pytest.raises(
        rail.TrustedCronCollectorError,
        match="read_failed",
    ):
        rail._json_tree(item)


def test_json_tree_rejects_a_descendant_added_during_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    payload = root / "evidence.json"
    payload.write_text("{}", encoding="utf-8")
    added = root / "added.json"
    item = rail.CollectorSpec(
        source_job_id="123456789abc",
        rail_id="test-read",
        schedule={"kind": "interval", "minutes": 1},
        mode="json_tree",
        model_review_required=True,
        json_roots=(str(root),),
        json_include_content=True,
    )
    original = rail._isolated_tree_state
    calls = 0

    def state(selected: Path):
        nonlocal calls
        calls += 1
        if calls == 2:
            added.write_text("{}", encoding="utf-8")
        return original(selected)

    monkeypatch.setattr(rail, "_isolated_tree_state", state)

    with pytest.raises(
        rail.TrustedCronCollectorError,
        match="json_tree_changed",
    ):
        rail._json_tree(item)


def test_process_identity_attestation_rejects_every_extra_group_or_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = next(
        spec
        for spec in rail.COLLECTOR_SPECS
        if spec.execution_boundary == rail.EXECUTION_BOUNDARY_ISOLATED
    )

    monkeypatch.setattr(
        rail.pwd,
        "getpwnam",
        lambda _name: SimpleNamespace(pw_uid=1001, pw_gid=1002),
    )
    monkeypatch.setattr(
        rail.grp,
        "getgrnam",
        lambda _name: SimpleNamespace(gr_gid=1002),
    )
    monkeypatch.setattr(rail.os, "geteuid", lambda: 1001)
    monkeypatch.setattr(rail.os, "getegid", lambda: 1002)
    monkeypatch.setattr(rail.os, "getgroups", lambda: [])
    monkeypatch.setattr(rail, "_effective_capabilities", lambda: 0)

    rail._attest_process_identity(item)

    monkeypatch.setattr(rail.os, "getgroups", lambda: [9999])
    with pytest.raises(
        rail.TrustedCronCollectorError,
        match="process_identity_invalid",
    ):
        rail._attest_process_identity(item)

    monkeypatch.setattr(rail.os, "getgroups", lambda: [])
    monkeypatch.setattr(rail, "_effective_capabilities", lambda: 1)
    with pytest.raises(
        rail.TrustedCronCollectorError,
        match="process_identity_invalid",
    ):
        rail._attest_process_identity(item)


def test_setpriv_toolchain_requires_exact_root_owned_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b"reviewed setpriv binary"
    facts = _dependency_facts()
    facts[str(rail.SETPRIV)] = rail._sha256(raw)
    package = rail.build_package_manifest(
        revision="a" * 40,
        rail_sha256="b" * 64,
        dependency_facts=facts,
    )
    monkeypatch.setattr(
        rail,
        "_root_owned_executable_sha256",
        lambda _path, **_kwargs: rail._sha256(raw),
    )

    rail.validate_setpriv_toolchain(package)

    monkeypatch.setattr(
        rail,
        "_root_owned_executable_sha256",
        lambda _path, **_kwargs: "f" * 64,
    )
    with pytest.raises(
        rail.TrustedCronCollectorError,
        match="setpriv_toolchain_invalid",
    ):
        rail.validate_setpriv_toolchain(package)


def test_setpriv_metadata_and_digest_are_attested_on_one_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b"reviewed setpriv binary"
    metadata = SimpleNamespace(
        st_mode=stat.S_IFREG | 0o755,
        st_uid=0,
        st_gid=0,
        st_nlink=1,
        st_size=len(raw),
        st_dev=1,
        st_ino=2,
        st_mtime_ns=3,
        st_ctime_ns=4,
    )
    chunks = iter((raw, b""))
    observed: dict[str, object] = {}
    monkeypatch.setattr(rail.os, "open", lambda *_args, **_kwargs: 17)
    monkeypatch.setattr(rail.os, "fstat", lambda descriptor: metadata)
    monkeypatch.setattr(
        rail.os,
        "read",
        lambda descriptor, _size: (
            observed.setdefault("read_descriptor", descriptor),
            next(chunks),
        )[1],
    )
    monkeypatch.setattr(
        rail.os,
        "close",
        lambda descriptor: observed.setdefault("closed_descriptor", descriptor),
    )

    assert rail._root_owned_executable_sha256(
        rail.SETPRIV,
        maximum=1024,
    ) == rail._sha256(raw)
    assert observed == {
        "read_descriptor": 17,
        "closed_descriptor": 17,
    }

    metadata.st_uid = 1000
    with pytest.raises(
        rail.TrustedCronCollectorError,
        match="setpriv_toolchain_invalid",
    ):
        rail._root_owned_executable_sha256(
            rail.SETPRIV,
            maximum=1024,
        )


def test_readiness_checks_real_service_identity_paths_and_scoped_edge() -> None:
    package = _package()

    readiness = rail.collect_execution_readiness(
        package,
        account_lookup=lambda _name: SimpleNamespace(pw_uid=2003),
        group_lookup=lambda _name: SimpleNamespace(gr_gid=2004),
    )

    assert rail.validate_execution_readiness(
        readiness,
        manifest=package,
    ) == readiness
    assert readiness["direct_dependencies_ready"] is False
    assert readiness["unit_namespace_readiness_packaged"] is False
    assert readiness["unit_namespace_readiness_receipt_sha256"] is None
    assert readiness["unit_namespace_readiness_job_count"] == 0
    assert readiness["unit_namespace_readiness_boot_id_sha256"] is None
    assert readiness["unit_namespace_readiness_observed_at_unix"] is None
    assert readiness["unit_namespace_readiness_maximum_age_seconds"] == 0
    assert readiness["scoped_execution_edge_required_count"] > 0
    assert readiness["scoped_execution_edge_packaged"] is False
    assert readiness["activation_ready"] is False
    assert readiness["permissions_widened"] is False
    assert readiness["credential_content_read"] is False


def test_readiness_receipt_rejects_a_claimed_ready_edge() -> None:
    package = _package()
    readiness = rail.collect_execution_readiness(
        package,
        account_lookup=lambda _name: SimpleNamespace(pw_uid=2003),
        group_lookup=lambda _name: SimpleNamespace(gr_gid=2004),
    )
    readiness["scoped_execution_edge_packaged"] = True
    readiness["activation_ready"] = True
    readiness["readiness_sha256"] = rail._sha256(
        rail._canonical({
            name: item
            for name, item in readiness.items()
            if name != "readiness_sha256"
        })
    )

    with pytest.raises(
        rail.TrustedCronCollectorError,
        match="execution_readiness_invalid",
    ):
        rail.validate_execution_readiness(readiness, manifest=package)


def test_readiness_requires_exact_meaningful_operational_edge_receipt() -> None:
    package = _package()
    edge = _operational_edge_receipt()
    namespace = _unit_namespace_receipt(package)
    readiness = rail.collect_execution_readiness(
        package,
        unit_namespace_receipt=namespace,
        operational_edge_receipt=edge,
        expected_boot_id_sha256=EDGE_BOOT_ID_SHA256,
        now_unix=EDGE_OBSERVED_AT_UNIX,
        account_lookup=lambda _name: SimpleNamespace(pw_uid=2003),
        group_lookup=lambda _name: SimpleNamespace(gr_gid=2004),
    )

    assert rail.validate_execution_readiness(
        readiness,
        manifest=package,
        unit_namespace_receipt=namespace,
        operational_edge_receipt=edge,
        expected_boot_id_sha256=EDGE_BOOT_ID_SHA256,
        now_unix=EDGE_OBSERVED_AT_UNIX,
    ) == readiness
    assert readiness["direct_dependencies_ready"] is True
    assert readiness["unit_namespace_readiness_packaged"] is True
    assert readiness["unit_namespace_readiness_receipt_sha256"] == namespace[
        "receipt_sha256"
    ]
    assert readiness["unit_namespace_readiness_job_count"] == 21
    assert (
        readiness["unit_namespace_readiness_boot_id_sha256"]
        == EDGE_BOOT_ID_SHA256
    )
    assert (
        readiness["unit_namespace_readiness_observed_at_unix"]
        == EDGE_OBSERVED_AT_UNIX
    )
    assert (
        readiness["unit_namespace_readiness_maximum_age_seconds"]
        == rail.UNIT_NAMESPACE_READINESS_MAXIMUM_AGE_SECONDS
    )
    assert readiness["scoped_execution_edge_packaged"] is True
    assert readiness["scoped_execution_edge_receipt_sha256"] == edge[
        "receipt_sha256"
    ]
    assert readiness["scoped_execution_edge_meaningful_packet_count"] == len(
        required_cron_operations()
    )
    assert readiness["activation_ready"] is True

    with pytest.raises(
        rail.TrustedCronCollectorError,
        match="execution_readiness_invalid",
    ):
        rail.validate_execution_readiness(readiness, manifest=package)


def test_scoped_packet_uses_verified_operational_edge_not_local_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import operational_edge_client

    item = next(
        spec for spec in rail.COLLECTOR_SPECS
        if spec.source_job_id == "8d09136f7da5"
    )
    operation_id, domain = rail._scoped_operation(item)
    config = SimpleNamespace(domain=domain)
    observed: dict[str, object] = {}

    class Client:
        def __init__(self, selected, *, main_pid_provider):
            observed["config"] = selected
            observed["provider"] = main_pid_provider

        def invoke_verified_evidence(
            self,
            selected_operation,
            arguments,
            **kwargs,
        ):
            observed["operation"] = selected_operation
            observed["arguments"] = arguments
            observed["kwargs"] = kwargs
            payload = {
                "outcome": "succeeded",
                "return_code": 0,
                "service_unit": (
                    f"muncho-operational-edge-{domain}.service"
                ),
                "release_revision": "a" * 40,
                "executable_sha256": "b" * 64,
                "stdout_b64": base64.b64encode(b"healthy\n").decode(),
                "stderr_b64": "",
                "mutation_performed": False,
                "secret_material_recorded": False,
            }
            return {
                "payload": payload,
                "signed_envelope_sha256": "c" * 64,
                "request_sha256": "d" * 64,
                "peer": {
                    "pid": 123,
                    "uid": 456,
                    "gid": 789,
                    "service_unit": payload["service_unit"],
                },
            }

    monkeypatch.setattr(
        operational_edge_client,
        "load_operational_edge_client_configs",
        lambda: {domain: config},
    )
    monkeypatch.setattr(
        operational_edge_client,
        "OperationalEdgeClient",
        Client,
    )
    monkeypatch.setenv("INVOCATION_ID", "e" * 32)

    packet = rail.collect_packet(item, revision="a" * 40)

    assert packet is not None
    assert observed["config"] is config
    assert isinstance(
        observed["provider"],
        operational_edge_client.SystemctlMainPidProvider,
    )
    assert observed["operation"] == operation_id
    assert observed["arguments"] == {}
    assert packet["evidence"]["operation_id"] == operation_id
    assert packet["evidence"]["domain"] == domain
    assert packet["evidence"]["stdout_utf8"] == "healthy\n"
    assert packet["evidence"]["credential_content_read_by_collector"] is False
    assert packet["evidence"]["mutation_performed"] is False


def test_scoped_namespace_probe_invokes_only_catalog_bound_read_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = next(
        spec
        for spec in rail.COLLECTOR_SPECS
        if spec.source_job_id == "8d09136f7da5"
    )
    primary_operation, _domain = rail._scoped_operation(item)
    probe_operation = rail._safe_probe_operation_id(item)
    observed: dict[str, object] = {}

    def evidence(
        selected,
        *,
        revision,
        selected_operation_id=None,
        idempotency_prefix="cron",
    ):
        observed.update(
            {
                "item": selected,
                "revision": revision,
                "operation": selected_operation_id,
                "prefix": idempotency_prefix,
            }
        )
        return {
            "operation_id": selected_operation_id,
            "outcome": "succeeded",
            "return_code": 0,
            "mutation_performed": False,
            "secret_material_recorded": False,
        }

    monkeypatch.setattr(rail, "_operational_edge_evidence", evidence)

    rail._probe_scoped_access(item, revision="a" * 40)

    assert observed["item"] is item
    assert observed["revision"] == "a" * 40
    assert observed["operation"] == probe_operation
    assert observed["operation"] != primary_operation
    assert observed["prefix"] == "cron-probe"


def test_package_rejects_dependency_or_unit_drift() -> None:
    package = _package()
    drifted = copy.deepcopy(package)
    dependency = next(iter(drifted["dependency_sha256"]))
    drifted["dependency_sha256"][dependency] = "f" * 64

    with pytest.raises(
        rail.TrustedCronCollectorError,
        match="package_manifest_invalid",
    ):
        rail.validate_package_manifest(drifted)

    drifted = copy.deepcopy(package)
    first = rail.COLLECTOR_SPECS[0].source_job_id
    drifted["units"][first]["service_sha256"] = "f" * 64
    unsigned = {
        key: value
        for key, value in drifted.items()
        if key != "manifest_sha256"
    }
    drifted["manifest_sha256"] = rail._sha256(rail._canonical(unsigned))
    with pytest.raises(
        rail.TrustedCronCollectorError,
        match="package_manifest_invalid",
    ):
        rail.validate_package_manifest(drifted)


def _voice_stage(tmp_path: Path) -> tuple[dict, Path]:
    voice_root = tmp_path / "voice"
    voice_root.mkdir()
    source = voice_root / "discord_g12345678901234567_vc12345678901234568_1.jsonl"
    source.write_text(
        json.dumps(
            {
                "type": "discord_voice_context.transcript",
                "timestamp": "2026-07-15T00:00:00Z",
                "guild_id": "12345678901234567",
                "voice_channel_id": "12345678901234568",
                "voice_channel_name": "Public voice",
                "text_channel_id": "12345678901234569",
                "user_id": "12345678901234570",
                "transcript": "Follow up on the deployment.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cursor = tmp_path / "state/cursor.json"
    stage = rail.stage_voice_packet(
        voice_root=voice_root,
        cursor_path=cursor,
        created_at="2026-07-15T00:01:00Z",
    )
    assert stage is not None
    return stage, cursor


def test_voice_cursor_moves_only_after_sent_readback_and_canonical_sent(
    tmp_path: Path,
) -> None:
    stage, cursor = _voice_stage(tmp_path)
    assert not cursor.exists()

    blocked = rail.build_voice_delivery_proof(
        stage=stage,
        connector_send_receipt_sha256="1" * 64,
        connector_readback_receipt_sha256=None,
        canonical_event_type="route_back.blocked",
        canonical_event_sha256="2" * 64,
    )
    result = rail.commit_voice_cursor(
        stage=stage,
        proof=blocked,
        cursor_path=cursor,
    )
    assert result["cursor_committed"] is False
    assert result["canonical_terminal_event"] == "route_back.blocked"
    assert not cursor.exists()

    with pytest.raises(
        rail.TrustedCronCollectorError,
        match="voice_delivery_proof_invalid",
    ):
        rail.build_voice_delivery_proof(
            stage=stage,
            connector_send_receipt_sha256="1" * 64,
            connector_readback_receipt_sha256=None,
            canonical_event_type="route_back.sent",
            canonical_event_sha256="2" * 64,
        )

    sent = rail.build_voice_delivery_proof(
        stage=stage,
        connector_send_receipt_sha256="1" * 64,
        connector_readback_receipt_sha256="3" * 64,
        canonical_event_type="route_back.sent",
        canonical_event_sha256="2" * 64,
    )
    result = rail.commit_voice_cursor(
        stage=stage,
        proof=sent,
        cursor_path=cursor,
    )
    persisted = json.loads(cursor.read_text(encoding="utf-8"))
    assert result["cursor_committed"] is True
    assert result["canonical_terminal_event"] == "route_back.sent"
    assert persisted["files"] == stage["proposed_offsets"]
    assert persisted["delivery_proof_sha256"] == sent["proof_sha256"]


def test_voice_cursor_drift_fails_closed(tmp_path: Path) -> None:
    stage, cursor = _voice_stage(tmp_path)
    cursor.parent.mkdir(parents=True)
    cursor.write_text(
        json.dumps(
            {
                "schema": rail.VOICE_CURSOR_SCHEMA,
                "files": {"unrelated.jsonl": 10},
            }
        ),
        encoding="utf-8",
    )
    proof = rail.build_voice_delivery_proof(
        stage=stage,
        connector_send_receipt_sha256="1" * 64,
        connector_readback_receipt_sha256="3" * 64,
        canonical_event_type="route_back.sent",
        canonical_event_sha256="2" * 64,
    )

    with pytest.raises(
        rail.TrustedCronCollectorError,
        match="voice_cursor_drifted",
    ):
        rail.commit_voice_cursor(
            stage=stage,
            proof=proof,
            cursor_path=cursor,
        )

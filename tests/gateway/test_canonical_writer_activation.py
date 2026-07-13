from __future__ import annotations

import json
import os
import stat
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import canonical_writer_activation as activation


@pytest.fixture(autouse=True)
def _tmp_path_uses_process_primary_group(tmp_path):
    """Keep BSD tmp-path group inheritance aligned with the test process."""

    os.chown(tmp_path, -1, os.getgid())


def _collector_bound_native_plan():
    ca = b"trusted-ca"
    return activation.NativeObservationPlan(
        value={
            "boot_id_sha256": "1" * 64,
            "host_identity_sha256": "2" * 64,
            "revision": "a" * 40,
            "artifact_sha256": "3" * 64,
            "release_manifest_file_sha256": "4" * 64,
            "config_collector_receipt_sha256": "5" * 64,
            "writer_config": {"sha256": "6" * 64},
            "gateway_config": {"sha256": "7" * 64},
            "database": {
                "ip_network": "10.91.0.3/32",
                "tls_server_name": (
                    "14-11111111-1111-4111-8111-111111111111."
                    "europe-west3.sql.goog"
                ),
                "ca_path": "/etc/muncho/trust/cloudsql-server-ca.pem",
                "ca_sha256": activation.hashlib.sha256(ca).hexdigest(),
            },
            "discord": {
                "config_path": "/etc/muncho/discord-edge.json",
                "token_path": "/etc/muncho/discord-edge-credentials/bot-token",
                "socket_path": "/run/muncho-discord-egress/edge.sock",
            },
            "legacy_helper_path": "/run/muncho/legacy-helper.sock",
        }
    )


def _owner_approval(scope: str, plan_sha256: str):
    now = int(time.time())
    receipt = activation.OwnerApprovalReceipt.from_mapping({
        "schema": activation.OWNER_APPROVAL_SCHEMA,
        "scope": scope,
        "plan_sha256": plan_sha256,
        "authority_kind": "trusted_root_bootstrap_out_of_band_owner",
        "cryptographic_owner_proof": False,
        "owner_subject_sha256": "1" * 64,
        "approval_source_sha256": "2" * 64,
        "nonce_sha256": "3" * 64,
        "approved_at_unix": now - 1,
        "expires_at_unix": now + 300,
    })
    receipt.require(scope=scope, plan_sha256=plan_sha256, now_unix=now)
    return receipt


def test_strict_json_rejects_duplicates_nan_and_noncanonical_bytes():
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        activation._decode_strict_json(b'{"a":1,"a":2}', label="test")
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        activation._decode_strict_json(b'{"a":NaN}', label="test")
    with pytest.raises(ValueError, match="canonical JSON"):
        activation._decode_strict_json(b'{ "a": 1 }\n', label="test")
    assert activation._decode_strict_json(b'{"a":1}\n', label="test") == {"a": 1}


def test_command_has_fixed_clean_environment_and_rejects_shell():
    command = activation.Command((activation.SYSTEMCTL, "daemon-reload"))

    assert command.environment == {
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
        "TZ": "UTC",
    }
    with pytest.raises(ValueError, match="argv"):
        activation.Command(("/bin/sh", "-c", "true"))


def test_activation_lock_is_under_root_controlled_run_not_world_writable_run_lock():
    assert activation.ACTIVATION_LOCK_PATH == Path("/run/muncho-writer-activation.lock")


def test_renewed_owner_approvals_are_append_only_receipt_addressed():
    first = _owner_approval("activation", "a" * 64)
    renewed = activation.OwnerApprovalReceipt.from_mapping({
        **first.to_mapping(),
        "nonce_sha256": "4" * 64,
    })

    first_path = activation.owner_approval_receipt_path(first)
    renewed_path = activation.owner_approval_receipt_path(renewed)

    assert first_path.parent == renewed_path.parent
    assert first_path.name == f"{first.sha256}.json"
    assert renewed_path.name == f"{renewed.sha256}.json"
    assert first_path != renewed_path


def test_external_iam_loader_enforces_minimum_remaining_lifetime(monkeypatch):
    calls = []

    class Receipt:
        value = {"source_approval_sha256": "b" * 64}

        def require_fresh(self, now, *, minimum_remaining_seconds=0):
            calls.append((now, minimum_remaining_seconds))

    plan = activation.NativeObservationPlan(
        value={"external_iam_policy_sha256": "a" * 64}
    )
    monkeypatch.setattr(activation, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        activation,
        "load_trusted_external_iam_receipt",
        lambda *_args, **_kwargs: Receipt(),
    )

    activation.load_external_iam_receipt(
        plan,
        path_value=activation.DEFAULT_EXTERNAL_IAM_LIVE_PATH,
        now_unix=123,
        minimum_remaining_seconds=720,
        expected_source_approval_sha256="b" * 64,
    )

    assert calls == [(123, 720)]

    with pytest.raises(ValueError, match="another approval"):
        activation.load_external_iam_receipt(
            plan,
            path_value=activation.DEFAULT_EXTERNAL_IAM_LIVE_PATH,
            now_unix=123,
            expected_source_approval_sha256="c" * 64,
        )


def test_external_iam_install_requires_exact_owner_approval_chain(monkeypatch):
    policy_sha256 = "b" * 64
    plan = activation.NativeObservationPlan(
        value={"external_iam_policy_sha256": policy_sha256}
    )
    plan_sha256 = plan.sha256
    owner = _owner_approval("native_observation", plan_sha256)
    archived = []
    replaced = []

    class Receipt:
        sha256 = "c" * 64

        def __init__(self, source_approval_sha256):
            self.policy_sha256 = policy_sha256
            self.value = {
                "source_approval_sha256": source_approval_sha256,
            }

        def require_fresh(self, _now):
            return None

        def to_mapping(self):
            return dict(self.value)

    current = Receipt(owner.sha256)
    monkeypatch.setattr(activation, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        activation.NativeObservationPlan,
        "from_mapping",
        lambda _value: plan,
    )
    monkeypatch.setattr(
        activation,
        "_read_trusted_file",
        lambda *_args, **_kwargs: activation._canonical_bytes(
            current.to_mapping()
        ),
    )
    monkeypatch.setattr(
        activation.ExternalIAMReceipt,
        "from_mapping",
        lambda _value: current,
    )
    monkeypatch.setattr(
        activation,
        "_archive_external_iam_at",
        lambda receipt, path: archived.append((receipt, path)) or {"path": str(path)},
    )
    monkeypatch.setattr(
        activation,
        "_external_iam_archive_path",
        lambda _receipt: Path("/archive/iam.json"),
    )
    monkeypatch.setattr(
        activation,
        "_replace_live_external_iam",
        lambda receipt: replaced.append(receipt) or True,
    )

    observed, _archive, changed = activation.install_staged_external_iam_receipt(
        activation.DEFAULT_STAGED_EXTERNAL_IAM_PATH,
        authorized_plan=plan,
        owner_approval_receipt=owner,
    )

    assert observed is current
    assert changed is True
    assert archived and replaced == [current]

    current.value["source_approval_sha256"] = "d" * 64
    with pytest.raises(PermissionError, match="not bound to owner approval"):
        activation.install_staged_external_iam_receipt(
            activation.DEFAULT_STAGED_EXTERNAL_IAM_PATH,
            authorized_plan=plan,
            owner_approval_receipt=owner,
        )


def test_final_cli_installs_iam_only_through_exact_plan_approval_chain(
    monkeypatch,
    capsys,
):
    plan = SimpleNamespace(
        sha256="a" * 64,
        digests=SimpleNamespace(external_iam_policy_sha256="b" * 64),
    )
    owner = _owner_approval("activation", plan.sha256)
    iam = SimpleNamespace(
        value={"schema": "external-iam.v1"},
        sha256="c" * 64,
        policy_sha256=plan.digests.external_iam_policy_sha256,
    )
    calls = []
    monkeypatch.setattr(
        activation,
        "load_activation_plan",
        lambda path: calls.append(("plan", path)) or plan,
    )
    monkeypatch.setattr(
        activation,
        "load_owner_approval_receipt",
        lambda path, **kwargs: calls.append(("approval", path, kwargs)) or owner,
    )
    monkeypatch.setattr(
        activation,
        "install_staged_external_iam_receipt",
        lambda path, **kwargs: calls.append(("install", path, kwargs))
        or (iam, {"path": "/archive/iam.json"}, True),
    )

    result = activation.main([
        "install-external-iam",
        "--plan",
        str(activation.DEFAULT_PLAN_PATH),
        "--staged-receipt",
        str(activation.DEFAULT_STAGED_EXTERNAL_IAM_PATH),
        "--approved-plan-sha256",
        plan.sha256,
        "--owner-approval-receipt",
        "/approval.json",
        "--external-iam-policy-sha256",
        plan.digests.external_iam_policy_sha256,
    ])

    assert result == 0
    assert calls == [
        ("plan", activation.DEFAULT_PLAN_PATH),
        (
            "approval",
            "/approval.json",
            {"scope": "activation", "plan_sha256": plan.sha256},
        ),
        (
            "install",
            str(activation.DEFAULT_STAGED_EXTERNAL_IAM_PATH),
            {"authorized_plan": plan, "owner_approval_receipt": owner},
        ),
    ]
    output = json.loads(capsys.readouterr().out)
    assert output["scope"] == "activation"
    assert output["plan_sha256"] == plan.sha256
    assert output["owner_approval_receipt_sha256"] == owner.sha256


def test_final_cli_preflight_requires_and_passes_exact_owner_approval(
    monkeypatch,
    capsys,
):
    plan = SimpleNamespace(sha256="a" * 64)
    owner = _owner_approval("activation", plan.sha256)
    calls = []
    monkeypatch.setattr(
        activation,
        "load_activation_plan",
        lambda path: calls.append(("plan", path)) or plan,
    )
    monkeypatch.setattr(
        activation,
        "load_owner_approval_receipt",
        lambda path, **kwargs: calls.append(("approval", path, kwargs)) or owner,
    )
    monkeypatch.setattr(
        activation,
        "activation_read_only_preflight",
        lambda observed_plan, **kwargs: calls.append(
            ("preflight", observed_plan, kwargs)
        )
        or ({"ok": True}, None, object()),
    )

    result = activation.main([
        "validate-plan",
        "--plan",
        str(activation.DEFAULT_PLAN_PATH),
        "--approved-plan-sha256",
        plan.sha256,
        "--owner-approval-receipt",
        "/approval.json",
    ])

    assert result == 0
    assert calls == [
        ("plan", str(activation.DEFAULT_PLAN_PATH)),
        (
            "approval",
            "/approval.json",
            {"scope": "activation", "plan_sha256": plan.sha256},
        ),
        (
            "preflight",
            plan,
            {"owner_approval_receipt": owner},
        ),
    ]
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_durable_native_chain_rereads_exact_approval_and_archived_iam(monkeypatch):
    policy_sha256 = "b" * 64
    plan = activation.NativeObservationPlan(
        value={
            "revision": "a" * 40,
            "external_iam_policy_sha256": policy_sha256,
        }
    )
    owner = _owner_approval("native_observation", plan.sha256)
    iam_sha256 = "c" * 64
    receipt = activation.NativeObservationReceipt(
        value={
            "owner_approval_receipt_sha256": owner.sha256,
            "external_iam_receipt_sha256": iam_sha256,
        }
    )
    iam = SimpleNamespace(
        sha256=iam_sha256,
        policy_sha256=policy_sha256,
        value={"source_approval_sha256": owner.sha256},
    )
    owner_path = activation.owner_approval_receipt_path(owner)
    iam_path = (
        activation.DEFAULT_NATIVE_OBSERVATION_EVIDENCE_ROOT
        / plan.value["revision"]
        / plan.sha256
        / "external-iam"
        / f"{iam_sha256}.json"
    )
    reads = []

    def read(path, **_kwargs):
        reads.append(Path(path))
        if Path(path) == owner_path:
            return activation._canonical_bytes(owner.to_mapping())
        if Path(path) == iam_path:
            return b'{"archived":true}'
        raise AssertionError(path)

    monkeypatch.setattr(activation, "_read_trusted_file", read)
    monkeypatch.setattr(
        activation.ExternalIAMReceipt,
        "from_mapping",
        lambda _value: iam,
    )

    loaded_owner, loaded_iam, evidence = (
        activation._load_durable_native_authority_chain(plan, receipt)
    )

    assert loaded_owner.sha256 == owner.sha256
    assert loaded_iam is iam
    assert reads == [owner_path, iam_path]
    assert evidence["path"] == str(iam_path)
    assert evidence["sha256"] == iam_sha256

    iam.value["source_approval_sha256"] = "d" * 64
    with pytest.raises(RuntimeError, match="IAM approval chain drifted"):
        activation._load_durable_native_authority_chain(plan, receipt)


def test_preflight_archive_distinguishes_report_and_file_digests(monkeypatch):
    installed = []
    plan = SimpleNamespace(
        revision="a" * 40,
        sha256="b" * 64,
        paths=SimpleNamespace(evidence_root=Path("/evidence")),
    )
    report = {
        "schema": "muncho-writer-activation-read-only-preflight.v1",
        "ok": True,
    }
    report["report_sha256"] = activation._sha256_json(report)
    monkeypatch.setattr(activation, "_ensure_root_directory", lambda _path: None)
    monkeypatch.setattr(
        activation,
        "_install_exact_bytes",
        lambda path, payload, **kwargs: installed.append((path, payload, kwargs)),
    )

    evidence = activation._seal_activation_preflight_report(plan, report)

    assert evidence["report_sha256"] == report["report_sha256"]
    assert evidence["file_sha256"] == activation._sha256_bytes(installed[0][1])
    assert evidence["file_sha256"] != evidence["report_sha256"]


def test_preflight_does_not_swallow_process_cancellation():
    plan = SimpleNamespace(revision="a" * 40, sha256="b" * 64)

    with pytest.raises(KeyboardInterrupt):
        activation._run_checked_preflight(
            plan,
            (("cancel", lambda: (_ for _ in ()).throw(KeyboardInterrupt())),),
        )


def test_root_receipt_archive_cross_binds_exact_evidence_bundle(monkeypatch):
    revision = "a" * 40
    plan_sha = "b" * 64
    bundle = {"schema": "canonical-writer-root-evidence-bundle.v1", "ok": True}
    bundle_raw = activation._canonical_bytes(bundle)
    bundle_sha = activation._sha256_bytes(bundle_raw)
    bundle_path = (
        activation.DEFAULT_ROOT_EVIDENCE_ROOT
        / revision
        / plan_sha
        / f"{bundle_sha}.json"
    )
    root_path = Path("/run/muncho-canonical-preflight/root-preflight.json")
    root = {
        "evidence_bundle_path": str(bundle_path),
        "evidence_bundle_sha256": bundle_sha,
        "external_iam_receipt_sha256": "c" * 64,
        "host_preparation_receipt_sha256": "d" * 64,
    }
    root_raw = activation._canonical_bytes(root)
    plan = SimpleNamespace(
        revision=revision,
        sha256=plan_sha,
        paths=SimpleNamespace(
            root_receipt_path=root_path,
            evidence_root=Path("/var/lib/muncho-writer-activation"),
        ),
    )
    installed = []

    def read(path, **_kwargs):
        return bundle_raw if Path(path) == bundle_path else root_raw

    monkeypatch.setattr(activation, "_read_trusted_file", read)
    monkeypatch.setattr(activation, "_ensure_root_directory", lambda _path: None)
    monkeypatch.setattr(
        activation,
        "_install_exact_bytes",
        lambda path, payload, **_kwargs: installed.append((path, payload)),
    )

    archive = activation._archive_root_receipt(
        plan,
        expected_sha256=activation._sha256_bytes(root_raw),
    )

    assert archive["evidence_bundle"] == {
        "path": str(bundle_path),
        "sha256": bundle_sha,
        "mode": "0400",
        "owner_uid": 0,
        "group_gid": 0,
    }
    assert archive["host_preparation_receipt_sha256"] == "d" * 64
    assert installed[0][1] == root_raw


def test_durable_native_replay_rehashes_external_library_before_mutation(
    monkeypatch,
):
    path = Path("/usr/lib/muncho-reviewed.so")
    payload = b"current-reviewed-native-library"
    digest = activation._sha256_bytes(payload)
    receipt = activation.NativeObservationReceipt(
        value={
            "plan": {},
            "observation": {
                "gateway_service": {
                    "external_native_mappings": [{"path": str(path), "sha256": digest}]
                },
                "writer_service": {
                    "external_native_mappings": [{"path": str(path), "sha256": digest}]
                },
            },
        }
    )
    native_plan = SimpleNamespace(
        value={
            "artifact_root": "/opt/muncho-canary-releases/" + "a" * 40,
            "native_discovery_policy": {
                "allowed_roots": ["/usr/lib"],
                "maximum_mappings": 256,
                "required_owner_uid": 0,
                "required_owner_gid": 0,
            },
        }
    )
    observed = SimpleNamespace(
        st_mode=stat.S_IFREG | 0o444,
        st_nlink=1,
        st_uid=0,
        st_gid=0,
        st_size=len(payload),
    )
    monkeypatch.setattr(
        activation.NativeObservationPlan,
        "from_mapping",
        lambda _value: native_plan,
    )
    monkeypatch.setattr(activation.os, "lstat", lambda _path: observed)
    monkeypatch.setattr(activation, "_list_xattrs", lambda _path: ())
    monkeypatch.setattr(
        activation,
        "_read_trusted_file",
        lambda *_args, **_kwargs: payload,
    )

    activation._rehash_native_receipt_external_mappings(receipt)
    receipt.value["observation"]["writer_service"]["external_native_mappings"][0][
        "sha256"
    ] = "f" * 64

    with pytest.raises(RuntimeError, match="mapping digest drifted"):
        activation._rehash_native_receipt_external_mappings(receipt)


def test_atomic_install_link_race_never_unlinks_existing_target(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "target"
    existing = b"existing-owner-content"
    target.write_bytes(existing)
    target.chmod(0o400)

    monkeypatch.setattr(activation, "_validate_root_parent_chain", lambda _path: None)
    monkeypatch.setattr(activation, "_fsync_directory", lambda _path: None)
    monkeypatch.setattr(activation.os, "fchown", lambda *_args: None)
    real_link = os.link
    real_lstat = os.lstat

    def pretend_absent(path, *args, **kwargs):
        if Path(path) == target:
            raise FileNotFoundError(path)
        return real_lstat(path, *args, **kwargs)

    def collision(_source, destination, **_kwargs):
        assert Path(destination) == target
        raise FileExistsError(destination)

    monkeypatch.setattr(activation.os, "lstat", pretend_absent)
    monkeypatch.setattr(activation.os, "link", collision)

    with pytest.raises(FileExistsError):
        activation._install_exact_bytes(
            target,
            b"new-content",
            uid=os.getuid(),
            gid=os.getgid(),
            mode=0o400,
        )

    monkeypatch.setattr(activation.os, "link", real_link)
    assert target.read_bytes() == existing


def test_systemd_absent_state_requires_debian_252_empty_unit_file_state(
    monkeypatch,
):
    monkeypatch.setattr(
        activation,
        "_systemd_show",
        lambda _unit, runner: {
            "LoadState": "not-found",
            "ActiveState": "inactive",
            "SubState": "dead",
            "MainPID": "0",
            "UnitFileState": "",
            "FragmentPath": "",
            "DropInPaths": "",
            "NeedDaemonReload": "no",
        },
    )
    activation._require_off_disabled(
        activation.EXPORTER_UNIT,
        runner=lambda _command: None,
        absent=True,
    )

    monkeypatch.setattr(
        activation,
        "_systemd_show",
        lambda _unit, runner: {
            "LoadState": "not-found",
            "ActiveState": "inactive",
            "SubState": "dead",
            "MainPID": "0",
            "UnitFileState": "not-found",
            "FragmentPath": "",
            "DropInPaths": "",
            "NeedDaemonReload": "no",
        },
    )
    with pytest.raises(RuntimeError, match="stopped/disabled"):
        activation._require_off_disabled(
            activation.EXPORTER_UNIT,
            runner=lambda _command: None,
            absent=True,
        )


def test_loaded_unit_rejects_run_override_or_dropin(monkeypatch):
    base = {
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "MainPID": "0",
        "UnitFileState": "disabled",
        "FragmentPath": str(activation.DEFAULT_WRITER_UNIT_PATH),
        "DropInPaths": "",
        "NeedDaemonReload": "no",
    }
    monkeypatch.setattr(
        activation,
        "_systemd_show",
        lambda _unit, runner: dict(base),
    )
    activation._require_off_disabled(
        activation.WRITER_UNIT,
        runner=lambda _command: None,
    )

    base["FragmentPath"] = "/run/systemd/system/muncho-canonical-writer.service"
    with pytest.raises(RuntimeError, match="stopped/disabled"):
        activation._require_off_disabled(
            activation.WRITER_UNIT,
            runner=lambda _command: None,
        )

    base["FragmentPath"] = str(activation.DEFAULT_WRITER_UNIT_PATH)
    base["DropInPaths"] = "/run/systemd/system/muncho-canonical-writer.service.d/x.conf"
    with pytest.raises(RuntimeError, match="stopped/disabled"):
        activation._require_off_disabled(
            activation.WRITER_UNIT,
            runner=lambda _command: None,
        )


def test_exporter_stdout_receipt_is_exact_and_count_bounded():
    invocation = "a" * 32
    commands = []

    def runner(command):
        commands.append(command)
        stdout = (
            b""
            if command.argv == (activation.JOURNALCTL, "--sync")
            else b'{"event_count":2,"success":true}\n'
        )
        return subprocess.CompletedProcess(command.argv, 0, stdout, b"")

    assert activation._exporter_stdout_receipt(invocation, runner=runner) == {
        "event_count": 2,
        "success": True,
    }
    assert commands[1].argv == (
        activation.JOURNALCTL,
        "--no-pager",
        "--output=cat",
        f"_SYSTEMD_INVOCATION_ID={invocation}",
        "PRIORITY=6",
    )
    assert all("sh" not in command.argv[:1] for command in commands)


def test_projection_validation_binds_canonical_count_hash_and_identity(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "canonical-events.json"
    raw = b'{"events":[{"event_id":"a"}]}\n'
    target.write_bytes(raw)
    target.chmod(0o640)
    identities = activation.NumericIdentities(
        gateway_uid=activation.CANARY_GATEWAY_UID,
        gateway_gid=activation.CANARY_GATEWAY_GID,
        gateway_home="/var/lib/hermes-gateway",
        writer_uid=os.getuid(),
        writer_gid=activation.CANARY_WRITER_GID,
        writer_home="/nonexistent",
        socket_client_gid=activation.CANARY_SOCKET_CLIENT_GID,
        projector_uid=activation.CANARY_PROJECTOR_UID,
        projector_gid=os.getgid(),
        projector_home="/nonexistent",
    )
    monkeypatch.setattr(activation, "_list_xattrs", lambda _path: ())

    value = activation._validate_projection(target, identities)

    assert value["event_count"] == 1
    assert value["size"] == len(raw)
    assert value["owner_uid"] == os.getuid()
    assert value["group_gid"] == os.getgid()
    assert len(value["sha256"]) == 64


def test_direct_apply_requires_matching_owner_approval_before_any_mutation(
    monkeypatch,
):
    calls = []
    fake_plan = SimpleNamespace(sha256="a" * 64)
    executor = activation.ActivationExecutor(
        fake_plan,
        runner=lambda command: calls.append(command),
    )
    monkeypatch.setattr(
        activation,
        "_require_root_linux",
        lambda: calls.append("root"),
    )

    with pytest.raises(PermissionError, match="approval digest"):
        executor.apply(
            approved_plan_sha256="b" * 64,
            owner_approval_receipt=None,
        )

    assert calls == []


@pytest.mark.parametrize(
    ("expire_on_require", "expected_started", "expected_sealed"),
    (
        (3, [], False),  # before the first mutation is safely retryable
        (4, [], True),  # after permanent artifact installation
        (5, [], True),  # after the temporary projection exporter
        (6, [activation.WRITER_UNIT], True),  # gateway withheld, writer stopped
    ),
)
def test_final_activation_rechecks_owner_freshness_before_each_dangerous_step(
    tmp_path,
    monkeypatch,
    expire_on_require,
    expected_started,
    expected_sealed,
):
    commands = []
    lifecycle = []
    invalidations = []
    seals = []
    paths = SimpleNamespace(
        quarantine_path=tmp_path / "quarantine.json",
        root_receipt_path=tmp_path / "root.json",
        evidence_root=tmp_path / "evidence",
        external_iam_receipt_path=tmp_path / "iam.json",
    )
    plan = SimpleNamespace(
        sha256="a" * 64,
        revision="b" * 40,
        paths=paths,
        native_observation_receipt={"plan": {}},
        digests=SimpleNamespace(
            native_observation_plan_sha256="c" * 64,
            native_observation_receipt_sha256="d" * 64,
        ),
    )
    owner = _owner_approval("activation", plan.sha256)
    real_require = activation.OwnerApprovalReceipt.require
    require_calls = 0

    def expiring_require(self, *, scope, plan_sha256, now_unix):
        nonlocal require_calls
        require_calls += 1
        if require_calls == expire_on_require:
            raise PermissionError("owner approval does not authorize this exact action")
        return real_require(
            self,
            scope=scope,
            plan_sha256=plan_sha256,
            now_unix=now_unix,
        )

    def runner(command):
        commands.append(command.argv)
        return subprocess.CompletedProcess(command.argv, 0, b"", b"")

    executor = activation.ActivationExecutor(plan, runner=runner)
    monkeypatch.setattr(activation.OwnerApprovalReceipt, "require", expiring_require)
    monkeypatch.setattr(activation, "_host_activation_lock", lambda: nullcontext())
    monkeypatch.setattr(
        activation,
        "activation_read_only_preflight",
        lambda *_args, **_kwargs: (
            {"ok": True},
            None,
            SimpleNamespace(value={"source_approval_sha256": owner.sha256}),
        ),
    )
    monkeypatch.setattr(
        activation.NativeObservationPlan,
        "from_mapping",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        activation,
        "_archive_plan_external_iam",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        activation,
        "_load_lifecycle_external_iam",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(activation, "_require_root_linux", lambda: None)
    monkeypatch.setattr(activation, "_verify_release_tree", lambda _plan: None)
    monkeypatch.setattr(
        activation,
        "_install_plan_artifacts",
        lambda _plan: lifecycle.append("install") or (),
    )
    monkeypatch.setattr(executor, "_verify_installed", lambda: None)
    monkeypatch.setattr(
        executor,
        "_run_projection_export",
        lambda: lifecycle.append("export") or {},
    )
    monkeypatch.setattr(
        executor,
        "_invalidate_root_receipt",
        lambda: invalidations.append(True),
    )
    monkeypatch.setattr(
        activation,
        "_require_active",
        lambda unit, **_kwargs: 111 if unit == activation.WRITER_UNIT else 222,
    )
    monkeypatch.setattr(
        activation, "_require_off_disabled", lambda *_args, **_kwargs: None
    )

    def sealed(*_args, **_kwargs):
        seals.append(True)
        raise RuntimeError("sealed after expired approval")

    monkeypatch.setattr(activation, "_seal_activation_failure", sealed)

    expected_error = (
        "sealed after expired approval"
        if expected_sealed
        else "safely stopped before mutation"
    )
    with pytest.raises(RuntimeError, match=expected_error):
        executor.apply(
            approved_plan_sha256=plan.sha256,
            owner_approval_receipt=owner,
        )

    started = [
        argv[2]
        for argv in commands
        if argv[:2] == (activation.SYSTEMCTL, "start")
    ]
    assert started == expected_started
    assert activation.GATEWAY_UNIT not in started
    assert bool(seals) is expected_sealed
    if expire_on_require <= 3:
        assert lifecycle == []
        assert invalidations == []
    elif expire_on_require == 4:
        assert lifecycle == ["install"]
    else:
        assert lifecycle == ["install", "export"]


def test_native_expired_approval_before_host_mutation_is_retryable_not_forensic(
    tmp_path,
    monkeypatch,
):
    plan = activation.NativeObservationPlan(value={
        "revision": "b" * 40,
        "external_iam_policy_sha256": "c" * 64,
    })
    owner = _owner_approval("native_observation", plan.sha256)
    real_require = activation.OwnerApprovalReceipt.require
    require_calls = 0
    calls = []

    def expiring_require(self, *, scope, plan_sha256, now_unix):
        nonlocal require_calls
        require_calls += 1
        if require_calls == 3:
            raise PermissionError("owner approval does not authorize this exact action")
        return real_require(
            self,
            scope=scope,
            plan_sha256=plan_sha256,
            now_unix=now_unix,
        )

    executor = activation.NativeObservationExecutor(
        plan,
        runner=lambda command: calls.append(command.argv)
        or subprocess.CompletedProcess(command.argv, 0, b"", b""),
    )
    monkeypatch.setattr(activation.OwnerApprovalReceipt, "require", expiring_require)
    monkeypatch.setattr(activation, "_host_activation_lock", lambda: nullcontext())
    monkeypatch.setattr(activation, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        activation,
        "DEFAULT_QUARANTINE_PATH",
        tmp_path / "quarantine.json",
    )
    monkeypatch.setattr(
        activation,
        "_native_receipt_path",
        lambda _plan: tmp_path / "native-receipt.json",
    )
    monkeypatch.setattr(
        activation,
        "_native_stage_path",
        lambda _plan: tmp_path / "native-stage.json",
    )
    monkeypatch.setattr(
        activation,
        "_load_lifecycle_external_iam",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        activation,
        "_verify_native_preflight_inputs",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        activation,
        "_archive_plan_external_iam",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        activation,
        "_host_identity_snapshot",
        lambda: pytest.fail("host mutation boundary must not be entered"),
    )
    monkeypatch.setattr(
        activation, "_require_off_or_absent", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        activation, "_require_off_disabled", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        activation,
        "_native_failure_path",
        lambda _plan: pytest.fail("retryable expiry must not create failure evidence"),
    )

    with pytest.raises(RuntimeError, match="safely stopped before mutation"):
        executor.observe(
            approved_plan_sha256=plan.sha256,
            owner_approval_receipt=owner,
            external_iam_receipt_path=activation.DEFAULT_EXTERNAL_IAM_LIVE_PATH,
        )

    assert not [
        argv for argv in calls if argv[:2] == (activation.SYSTEMCTL, "start")
    ]


def test_native_expired_approval_after_host_mutation_is_forensic(
    tmp_path,
    monkeypatch,
):
    plan = activation.NativeObservationPlan(value={
        "revision": "b" * 40,
        "external_iam_policy_sha256": "c" * 64,
    })
    owner = _owner_approval("native_observation", plan.sha256)
    real_require = activation.OwnerApprovalReceipt.require
    require_calls = 0
    commands = []
    writes = []

    def expiring_require(self, *, scope, plan_sha256, now_unix):
        nonlocal require_calls
        require_calls += 1
        if require_calls == 4:
            raise PermissionError("owner approval does not authorize this exact action")
        return real_require(
            self,
            scope=scope,
            plan_sha256=plan_sha256,
            now_unix=now_unix,
        )

    executor = activation.NativeObservationExecutor(
        plan,
        runner=lambda command: commands.append(command.argv)
        or subprocess.CompletedProcess(command.argv, 0, b"", b""),
    )
    quarantine = tmp_path / "quarantine.json"
    failure = tmp_path / "failure.json"
    monkeypatch.setattr(activation.OwnerApprovalReceipt, "require", expiring_require)
    monkeypatch.setattr(activation, "_host_activation_lock", lambda: nullcontext())
    monkeypatch.setattr(activation, "_require_root_linux", lambda: None)
    monkeypatch.setattr(activation, "DEFAULT_QUARANTINE_PATH", quarantine)
    monkeypatch.setattr(
        activation,
        "_native_receipt_path",
        lambda _plan: tmp_path / "native-receipt.json",
    )
    monkeypatch.setattr(
        activation,
        "_native_stage_path",
        lambda _plan: tmp_path / "native-stage.json",
    )
    monkeypatch.setattr(
        activation,
        "_load_lifecycle_external_iam",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        activation,
        "_verify_native_preflight_inputs",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        activation,
        "_archive_plan_external_iam",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        activation,
        "_host_identity_snapshot",
        lambda: {"state": "before"},
    )
    monkeypatch.setattr(
        activation,
        "prepare_canary_host_identities",
        lambda *_args, **_kwargs: {
            "changed": True,
            "before": {"state": "before"},
            "after": {"state": "after"},
        },
    )
    monkeypatch.setattr(
        activation,
        "_record_host_preparation",
        lambda *_args, **_kwargs: {
            "receipt_path": str(tmp_path / "host.json"),
            "receipt_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(
        activation,
        "_install_native_observation_artifacts",
        lambda _plan: pytest.fail("expired approval must block artifact install"),
    )
    monkeypatch.setattr(
        activation, "_require_off_or_absent", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        activation, "_require_off_disabled", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(activation, "_native_failure_path", lambda _plan: failure)
    monkeypatch.setattr(activation, "_ensure_root_directory", lambda *_args: None)
    monkeypatch.setattr(
        activation,
        "_write_root_receipt",
        lambda path, value: writes.append((path, value)),
    )

    with pytest.raises(RuntimeError, match="failed closed"):
        executor.observe(
            approved_plan_sha256=plan.sha256,
            owner_approval_receipt=owner,
            external_iam_receipt_path=activation.DEFAULT_EXTERNAL_IAM_LIVE_PATH,
        )

    assert [path for path, _value in writes] == [failure, quarantine]
    assert all(value["quarantined"] is True for _path, value in writes)
    assert not [
        argv for argv in commands if argv[:2] == (activation.SYSTEMCTL, "start")
    ]


def test_systemd_bundle_rejects_installable_temporary_exporter():
    value = {
        "schema": activation.SYSTEMD_BUNDLE_SCHEMA,
        "writer_service": "[Service]\nType=notify\n",
        "gateway_service": "[Service]\nType=notify\n",
        "exporter_service": "[Service]\nType=oneshot\n[Install]\n",
        "tmpfiles": "d /run/x 0700 root root - -\n",
        "contract": {"revision": "a" * 40},
    }
    value["sha256"] = activation._sha256_json(value)

    with pytest.raises(ValueError, match="installable"):
        activation.SystemdBundle.from_mapping(value)


def test_host_identity_exactness_excludes_writer_and_discord_memberships():
    exact = {
        "gateway": {
            "name": activation.GATEWAY_USER,
            "uid": 993,
            "gid": 992,
            "home": "/var/lib/hermes-gateway",
            "shell": "/usr/sbin/nologin",
            "groups": [990, 992],
        },
        "writer": {
            "name": activation.WRITER_USER,
            "uid": 999,
            "gid": 994,
            "home": "/nonexistent",
            "shell": "/usr/sbin/nologin",
            "groups": [991, 994],
        },
        "projector": {
            "name": activation.PROJECTOR_USER,
            "uid": 992,
            "gid": 991,
            "home": "/nonexistent",
            "shell": "/usr/sbin/nologin",
            "groups": [991],
        },
        "groups": {
            activation.GATEWAY_GROUP: {"gid": 992, "members": []},
            activation.WRITER_GROUP: {"gid": 994, "members": []},
            activation.SOCKET_CLIENT_GROUP: {
                "gid": 990,
                "members": [activation.GATEWAY_USER],
            },
            activation.PROJECTOR_GROUP: {
                "gid": 991,
                "members": [activation.WRITER_USER],
            },
        },
        "effective_gid_members": {
            "990": [activation.GATEWAY_USER],
            "991": [activation.PROJECTOR_USER, activation.WRITER_USER],
            "992": [activation.GATEWAY_USER],
            "994": [activation.WRITER_USER],
        },
    }
    assert activation._host_identities_are_exact(exact)

    drifted = json.loads(json.dumps(exact))
    drifted["gateway"]["groups"] = [990, 992, 994]
    assert not activation._host_identities_are_exact(drifted)

    public_projection = json.loads(json.dumps(exact))
    public_projection["groups"][activation.PROJECTOR_GROUP]["members"].append(
        "unrelated"
    )
    assert not activation._host_identities_are_exact(public_projection)


def test_effective_membership_rejects_duplicate_pinned_group_names(monkeypatch):
    accounts = [
        SimpleNamespace(pw_name=activation.GATEWAY_USER, pw_uid=993, pw_gid=992),
        SimpleNamespace(pw_name=activation.WRITER_USER, pw_uid=999, pw_gid=994),
        SimpleNamespace(pw_name=activation.PROJECTOR_USER, pw_uid=992, pw_gid=991),
    ]
    groups = [
        SimpleNamespace(
            gr_name=activation.GATEWAY_GROUP,
            gr_gid=992,
            gr_mem=[],
        ),
        SimpleNamespace(
            gr_name=activation.GATEWAY_GROUP,
            gr_gid=1234,
            gr_mem=[],
        ),
    ]
    monkeypatch.setattr(activation.pwd, "getpwall", lambda: accounts)
    monkeypatch.setattr(activation.grp, "getgrall", lambda: groups)

    with pytest.raises(RuntimeError, match="group name is ambiguous"):
        activation._effective_gid_members((
            activation.CANARY_SOCKET_CLIENT_GID,
            activation.CANARY_PROJECTOR_GID,
            activation.CANARY_GATEWAY_GID,
            activation.CANARY_WRITER_GID,
        ))


def test_writer_start_failure_still_unconditionally_stops_gateway_then_writer(
    tmp_path,
    monkeypatch,
):
    commands = []
    receipts = []
    paths = SimpleNamespace(
        quarantine_path=tmp_path / "quarantine.json",
        root_receipt_path=tmp_path / "root.json",
        evidence_root=tmp_path / "evidence",
        external_iam_receipt_path=tmp_path / "iam.json",
    )
    plan = SimpleNamespace(
        sha256="a" * 64,
        revision="b" * 40,
        paths=paths,
        native_observation_receipt={"plan": {}},
        digests=SimpleNamespace(
            native_observation_plan_sha256="c" * 64,
            native_observation_receipt_sha256="d" * 64,
        ),
    )

    def runner(command):
        commands.append(command.argv)
        returncode = (
            1
            if command.argv == (activation.SYSTEMCTL, "start", activation.WRITER_UNIT)
            else 0
        )
        return subprocess.CompletedProcess(command.argv, returncode, b"", b"")

    executor = activation.ActivationExecutor(plan, runner=runner)
    owner_approval = _owner_approval("activation", "a" * 64)
    monkeypatch.setattr(activation, "_host_activation_lock", lambda: nullcontext())
    monkeypatch.setattr(
        activation,
        "activation_read_only_preflight",
        lambda *_args, **_kwargs: (
            {"ok": True},
            None,
            SimpleNamespace(
                value={"source_approval_sha256": owner_approval.sha256}
            ),
        ),
    )
    monkeypatch.setattr(
        activation.NativeObservationPlan,
        "from_mapping",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        activation,
        "_archive_plan_external_iam",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        activation,
        "load_external_iam_receipt",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(activation, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        activation.NativeObservationReceipt,
        "from_mapping",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(activation, "_current_boot_id_sha256", lambda: "e" * 64)
    monkeypatch.setattr(activation, "_current_boottime_ns", lambda: 1)
    monkeypatch.setattr(activation, "_verify_release_tree", lambda _plan: None)
    monkeypatch.setattr(activation, "_install_plan_artifacts", lambda _plan: ())
    monkeypatch.setattr(executor, "_verify_installed", lambda: None)
    monkeypatch.setattr(executor, "_run_projection_export", lambda: {})
    monkeypatch.setattr(executor, "_invalidate_root_receipt", lambda: None)
    monkeypatch.setattr(
        activation, "_require_off_disabled", lambda *_args, **_kwargs: None
    )
    failure_path = tmp_path / "failure.json"
    monkeypatch.setattr(
        activation,
        "_new_failure_receipt_path",
        lambda _plan: failure_path,
    )
    monkeypatch.setattr(
        activation,
        "_write_root_receipt",
        lambda path, value: receipts.append((path, value)),
    )

    with pytest.raises(RuntimeError, match="failed closed"):
        executor.apply(
            approved_plan_sha256="a" * 64,
            owner_approval_receipt=owner_approval,
        )

    assert (activation.SYSTEMCTL, "stop", activation.GATEWAY_UNIT) in commands
    assert (activation.SYSTEMCTL, "stop", activation.WRITER_UNIT) in commands
    assert commands.index((
        activation.SYSTEMCTL,
        "stop",
        activation.GATEWAY_UNIT,
    )) < commands.index((activation.SYSTEMCTL, "stop", activation.WRITER_UNIT))
    assert [path for path, _value in receipts] == [
        failure_path,
        paths.quarantine_path,
    ]


def test_packaged_preflight_cross_binds_exact_collector_receipt(monkeypatch):
    plan = _collector_bound_native_plan()
    calls = []

    class Receipt:
        def require_bindings(self, **kwargs):
            calls.append(("bindings", kwargs))

    receipt = Receipt()

    def load(**kwargs):
        calls.append(("load", kwargs))
        return receipt

    monkeypatch.setattr(activation, "load_config_collector_receipt", load)
    assert (
        activation._load_bound_config_collector_receipt(
            plan,
            require_fresh=True,
        )
        is receipt
    )

    assert calls == [
        (
            "load",
            {
                "revision": "a" * 40,
                "receipt_sha256": "5" * 64,
                "require_fresh": True,
            },
        ),
        (
            "bindings",
            {
                "revision": "a" * 40,
                "release_artifact_sha256": "3" * 64,
                "release_manifest_file_sha256": "4" * 64,
                "writer_config_sha256": "6" * 64,
                "gateway_config_sha256": "7" * 64,
                "database_ca_sha256": activation.hashlib.sha256(
                    b"trusted-ca"
                ).hexdigest(),
                "sql_private_ip": "10.91.0.3",
                "sql_tls_server_name": (
                    "14-11111111-1111-4111-8111-111111111111."
                    "europe-west3.sql.goog"
                ),
            },
        ),
    ]


def test_public_durable_native_receipt_loader_is_fail_closed(monkeypatch):
    plan = _collector_bound_native_plan()
    receipt = object()
    calls = []
    monkeypatch.setattr(
        activation,
        "_load_existing_native_receipt",
        lambda observed_plan, *, runner: (
            calls.append((observed_plan, runner)) or receipt
        ),
    )

    assert activation.load_durable_native_observation_receipt(plan) is receipt
    assert calls == [(plan, activation._runner)]

    monkeypatch.setattr(
        activation,
        "_load_existing_native_receipt",
        lambda _plan, *, runner: None,
    )
    with pytest.raises(FileNotFoundError):
        activation.load_durable_native_observation_receipt(plan)


@pytest.mark.parametrize(
    ("require_installed", "expected_events"),
    (
        (
            False,
            ("collector:True", "release", "database", "collector:True"),
        ),
        (
            True,
            ("release", "database", "collector:False", "collector:False"),
        ),
    ),
)
def test_packaged_preflight_requires_fresh_initial_collector_but_attests_db_before_expired_replay(
    monkeypatch,
    require_installed,
    expected_events,
):
    plan = _collector_bound_native_plan()
    events = []
    trusted_reads = []

    class Receipt:
        def to_mapping(self):
            return {"receipt_sha256": "5" * 64}

    def load_bound(_plan, *, require_fresh):
        events.append(f"collector:{require_fresh}")
        return Receipt()

    monkeypatch.setattr(
        activation,
        "_load_bound_config_collector_receipt",
        load_bound,
    )
    monkeypatch.setattr(
        activation,
        "current_host_identity_sha256",
        lambda: "2" * 64,
    )
    monkeypatch.setattr(
        activation,
        "_verify_native_release",
        lambda _plan: events.append("release"),
    )
    monkeypatch.setattr(activation, "_native_artifact_contract", lambda _plan: {})
    def read_trusted(path, **kwargs):
        trusted_reads.append((path, kwargs))
        return b"trusted-ca"

    monkeypatch.setattr(activation, "_read_trusted_file", read_trusted)
    monkeypatch.setattr(
        activation,
        "_verify_database_read_only",
        lambda **_kwargs: events.append("database"),
    )
    monkeypatch.setattr(
        activation,
        "_require_off_disabled",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        activation,
        "_require_off_or_absent",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(activation.os.path, "lexists", lambda _path: False)

    activation._verify_native_preflight_inputs(
        plan,
        runner=lambda _command: None,
        require_installed=require_installed,
        require_original_boot=False,
    )

    assert tuple(events) == expected_events
    assert trusted_reads == [
        (
            Path("/etc/muncho/trust/cloudsql-server-ca.pem"),
            {
                "expected_uid": 0,
                "expected_gid": activation.CANARY_WRITER_GID,
                "allowed_modes": frozenset({0o400, 0o440, 0o444}),
                "maximum": activation._MAX_CONFIG_BYTES,
            },
        )
    ]


def test_final_preflight_reads_database_ca_with_writer_group(monkeypatch):
    ca = b"trusted-ca"
    reads = []
    plan = SimpleNamespace(
        install_artifacts={},
        paths=SimpleNamespace(
            database_ca_path=Path(
                "/etc/muncho/trust/cloudsql-server-ca.pem"
            ),
            writer_config_path=Path("/etc/muncho-canonical-writer/writer.json"),
        ),
        digests=SimpleNamespace(
            database_ca_sha256=activation.hashlib.sha256(ca).hexdigest(),
        ),
        deployment_manifest={
            "snapshot_template": {
                "database": {
                    "connection": {
                        "host": "10.91.0.3",
                        "tls_server_name": (
                            "14-11111111-1111-4111-8111-111111111111."
                            "europe-west3.sql.goog"
                        ),
                    }
                }
            }
        },
    )
    monkeypatch.setattr(activation, "_verify_release_tree", lambda _plan: None)

    def read_trusted(path, **kwargs):
        reads.append((path, kwargs))
        return ca

    monkeypatch.setattr(activation, "_read_trusted_file", read_trusted)
    monkeypatch.setattr(
        activation,
        "_verify_database_read_only",
        lambda **_kwargs: None,
    )

    activation._verify_final_artifacts(plan)

    assert reads == [
        (
            plan.paths.database_ca_path,
            {
                "expected_uid": 0,
                "expected_gid": activation.CANARY_WRITER_GID,
                "allowed_modes": frozenset({0o400, 0o440, 0o444}),
                "maximum": activation._MAX_CONFIG_BYTES,
            },
        )
    ]

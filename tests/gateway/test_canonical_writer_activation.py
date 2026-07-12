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
    )

    assert calls == [(123, 720)]


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
    monkeypatch.setattr(activation, "_host_activation_lock", lambda: nullcontext())
    monkeypatch.setattr(
        activation,
        "activation_read_only_preflight",
        lambda *_args, **_kwargs: ({"ok": True}, None, object()),
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
            owner_approval_receipt=_owner_approval("activation", "a" * 64),
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

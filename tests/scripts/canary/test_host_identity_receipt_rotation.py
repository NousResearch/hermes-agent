from __future__ import annotations

import hashlib
import os
import stat
import subprocess

import pytest

from scripts.canary import host_identity_receipt_rotation as rotation
from scripts.canary import full_canary_owner_launcher as owner_launcher
from scripts.canary import writer_release


REVISION = "a" * 40
EXTERNAL_IAM_POLICY_SHA256 = "5" * 64
PRIOR_BOOT_SHA256 = "6" * 64
CURRENT_BOOT_SHA256 = "7" * 64


def _host(*, boot_id_sha256: str) -> dict[str, str]:
    gce = {
        "project_id": "adventico-ai-platform",
        "project_number": "39589465056",
        "zone": "europe-west3-a",
        "instance_name": "muncho-canary-v2-01",
        "instance_id": "9153645328899914617",
        "service_account_email": (
            "muncho-canary-v2-runtime@adventico-ai-platform.iam.gserviceaccount.com"
        ),
    }
    machine = "1" * 64
    hostname = "2" * 64
    return {
        **gce,
        "gce_identity_sha256": writer_release._sha256_json(gce),
        "machine_id_sha256": machine,
        "hostname_sha256": hostname,
        "host_identity_sha256": writer_release._sha256_json({
            "machine_id_sha256": machine,
            "hostname_sha256": hostname,
        }),
        "boot_id_sha256": boot_id_sha256,
    }


def _host_receipt(host: dict[str, str], observed_at_unix: int) -> dict[str, object]:
    unsigned = {
        "schema": "muncho-full-canary-host-identity.v1",
        "collector_authority": "trusted_root_read_only_host_collector",
        **host,
        "observed_at_unix": observed_at_unix,
    }
    return {**unsigned, "receipt_sha256": writer_release._sha256_json(unsigned)}


def _service_stdout(unit: str) -> str:
    values = {
        "LoadState": "not-found",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "",
        "MainPID": "0",
        "FragmentPath": "",
        "DropInPaths": "",
    }
    return "".join(
        f"{name}={values[name]}\n" for name in writer_release._SERVICE_PROPERTIES
    )


class _StoppedRunner:
    def __call__(
        self,
        command: writer_release.BuildCommand,
    ) -> subprocess.CompletedProcess[str]:
        assert command.argv[0] == str(writer_release.DEFAULT_SYSTEMCTL_EXECUTABLE)
        return subprocess.CompletedProcess(
            command.argv,
            0,
            _service_stdout(command.argv[-1]),
            "",
        )


def _prepare_namespace(tmp_path, monkeypatch):
    full_canary = tmp_path / "etc/muncho/full-canary"
    full_canary.mkdir(parents=True)
    os.chown(full_canary, os.geteuid(), os.getegid())
    full_canary.chmod(0o755)
    host_path = full_canary / "host-identity.json"
    rotations = full_canary / "host-identity-rotations"
    monkeypatch.setattr(writer_release, "_BUILD_OWNER_UID", os.geteuid())
    monkeypatch.setattr(writer_release, "_BUILD_OWNER_GID", os.getegid())
    monkeypatch.setattr(writer_release, "_validate_root_parent_chain", lambda _p: None)
    monkeypatch.setattr(writer_release, "_require_root_linux", lambda: None)
    monkeypatch.setattr(writer_release, "DEFAULT_HOST_RECEIPT_PATH", host_path)
    monkeypatch.setattr(rotation, "DEFAULT_ROTATION_ROOT", rotations)
    monkeypatch.setattr(
        owner_launcher,
        "STOPPED_RELEASE_HOST_RECEIPT_PATH",
        str(host_path),
    )
    monkeypatch.setattr(
        owner_launcher,
        "HOST_RECEIPT_ROTATION_ROOT",
        str(rotations),
    )
    prior = _host_receipt(_host(boot_id_sha256=PRIOR_BOOT_SHA256), 100)
    prior_raw = rotation._canonical_bytes(prior)
    host_path.write_bytes(prior_raw)
    host_path.chmod(0o400)
    return host_path, rotations, prior, prior_raw


def _intent(prior: dict[str, object], prior_raw: bytes) -> dict[str, str]:
    return {
        "external_iam_policy_sha256": EXTERNAL_IAM_POLICY_SHA256,
        "expected_prior_file_sha256": hashlib.sha256(prior_raw).hexdigest(),
        "expected_prior_receipt_sha256": str(prior["receipt_sha256"]),
        "expected_prior_boot_id_sha256": PRIOR_BOOT_SHA256,
        "expected_current_boot_id_sha256": CURRENT_BOOT_SHA256,
    }


def test_rotation_archives_tombstones_publishes_and_is_idempotent(
    tmp_path,
    monkeypatch,
):
    host_path, _rotations, prior, prior_raw = _prepare_namespace(
        tmp_path,
        monkeypatch,
    )
    current = _host(boot_id_sha256=CURRENT_BOOT_SHA256)
    intent = _intent(prior, prior_raw)
    runner = _StoppedRunner()
    plan = rotation.plan_host_receipt_rotation(
        REVISION,
        **intent,
        runner=runner,
        host_observer=lambda: current,
    )
    assert owner_launcher.validate_host_receipt_rotation_plan(
        plan,
        expected_release_sha=REVISION,
        expected_external_iam_policy_sha256=EXTERNAL_IAM_POLICY_SHA256,
        expected_prior_file_sha256=intent["expected_prior_file_sha256"],
        expected_prior_receipt_sha256=intent["expected_prior_receipt_sha256"],
        expected_prior_boot_id_sha256=PRIOR_BOOT_SHA256,
        expected_current_boot_id_sha256=CURRENT_BOOT_SHA256,
    ) == plan

    receipt = rotation.apply_host_receipt_rotation(
        REVISION,
        plan["plan_sha256"],
        **intent,
        runner=runner,
        host_observer=lambda: current,
        host_receipt_collector=lambda observed: _host_receipt(current, observed),
        clock=lambda: 200,
    )

    transaction = rotation._transaction_paths(plan["rotation_id"])
    assert transaction["archive"].read_bytes() == prior_raw
    assert stat.S_IMODE(transaction["archive"].stat().st_mode) == 0o400
    assert transaction["tombstone"].is_file()
    assert transaction["intent"].read_bytes() == rotation._canonical_bytes(plan)
    assert host_path.read_bytes() == rotation._canonical_bytes(
        _host_receipt(current, 200)
    )
    assert receipt["state"] == "target_boot_receipt_published_services_stopped"
    assert receipt["external_iam_policy_sha256"] == EXTERNAL_IAM_POLICY_SHA256
    assert receipt["prior_host_identity_receipt_file_sha256"] == hashlib.sha256(
        prior_raw
    ).hexdigest()
    assert receipt["target_boot_id_sha256"] == CURRENT_BOOT_SHA256
    assert receipt["services_started"] is False
    assert receipt["iam_mutated"] is False
    assert owner_launcher.validate_host_receipt_rotation_receipt(
        receipt,
        plan=plan,
    ) == receipt

    before = {
        path: (path.stat().st_ino, path.stat().st_mtime_ns)
        for path in (
            transaction["intent"],
            transaction["archive"],
            transaction["tombstone"],
            transaction["completion"],
            host_path,
        )
    }
    retry = rotation.apply_host_receipt_rotation(
        REVISION,
        plan["plan_sha256"],
        **intent,
        runner=runner,
        host_observer=lambda: current,
        host_receipt_collector=lambda _observed: (_ for _ in ()).throw(
            AssertionError("idempotent retry must reuse the fresh receipt")
        ),
        clock=lambda: 999,
    )
    after = {
        path: (path.stat().st_ino, path.stat().st_mtime_ns) for path in before
    }
    assert retry == receipt
    assert after == before


def test_rotation_resumes_after_crash_between_tombstone_and_fresh_receipt(
    tmp_path,
    monkeypatch,
):
    host_path, _rotations, prior, prior_raw = _prepare_namespace(
        tmp_path,
        monkeypatch,
    )
    current = _host(boot_id_sha256=CURRENT_BOOT_SHA256)
    intent = _intent(prior, prior_raw)
    runner = _StoppedRunner()
    plan = rotation.plan_host_receipt_rotation(
        REVISION,
        **intent,
        runner=runner,
        host_observer=lambda: current,
    )
    original_writer = writer_release._write_host_receipt_no_replace
    monkeypatch.setattr(
        writer_release,
        "_write_host_receipt_no_replace",
        lambda _path, _receipt: (_ for _ in ()).throw(
            RuntimeError("simulated crash before fresh publication")
        ),
    )

    with pytest.raises(RuntimeError, match="simulated crash"):
        rotation.apply_host_receipt_rotation(
            REVISION,
            plan["plan_sha256"],
            **intent,
            runner=runner,
            host_observer=lambda: current,
            host_receipt_collector=lambda observed: _host_receipt(current, observed),
            clock=lambda: 200,
        )

    transaction = rotation._transaction_paths(plan["rotation_id"])
    assert not host_path.exists()
    assert transaction["archive"].read_bytes() == prior_raw
    assert transaction["tombstone"].is_file()
    assert not transaction["completion"].exists()

    monkeypatch.setattr(
        writer_release,
        "_write_host_receipt_no_replace",
        original_writer,
    )
    receipt = rotation.apply_host_receipt_rotation(
        REVISION,
        plan["plan_sha256"],
        **intent,
        runner=runner,
        host_observer=lambda: current,
        host_receipt_collector=lambda observed: _host_receipt(current, observed),
        clock=lambda: 201,
    )
    assert receipt["fresh_observed_at_unix"] == 201
    assert host_path.exists()
    assert transaction["completion"].exists()


def test_rotation_rejects_divergent_archive_and_current_boot_drift(
    tmp_path,
    monkeypatch,
):
    _host_path, rotations, prior, prior_raw = _prepare_namespace(
        tmp_path,
        monkeypatch,
    )
    current = _host(boot_id_sha256=CURRENT_BOOT_SHA256)
    intent = _intent(prior, prior_raw)
    rotation_id = rotation._rotation_id(
        revision=REVISION,
        external_iam_policy_sha256=EXTERNAL_IAM_POLICY_SHA256,
        prior_file_sha256=intent["expected_prior_file_sha256"],
        prior_receipt_sha256=intent["expected_prior_receipt_sha256"],
        prior_boot_id_sha256=PRIOR_BOOT_SHA256,
        current_boot_id_sha256=CURRENT_BOOT_SHA256,
    )
    transaction = rotations / rotation_id
    transaction.mkdir(parents=True, mode=0o700)
    rotations.chmod(0o700)
    transaction.chmod(0o700)
    archive = transaction / "prior-host-identity.json"
    archive.write_bytes(b"{}")
    archive.chmod(0o400)

    with pytest.raises(RuntimeError, match="archived prior host receipt diverged"):
        rotation.plan_host_receipt_rotation(
            REVISION,
            **intent,
            runner=_StoppedRunner(),
            host_observer=lambda: current,
        )

    archive.unlink()
    with pytest.raises(RuntimeError, match="current boot identity"):
        rotation.plan_host_receipt_rotation(
            REVISION,
            **intent,
            runner=_StoppedRunner(),
            host_observer=lambda: {
                **current,
                "boot_id_sha256": "8" * 64,
            },
        )


def test_rotation_cli_requires_every_exact_owner_binding():
    parser = rotation._parser()
    base = [
        "plan",
        "--revision",
        REVISION,
        "--external-iam-policy-sha256",
        EXTERNAL_IAM_POLICY_SHA256,
        "--expected-prior-file-sha256",
        "1" * 64,
        "--expected-prior-receipt-sha256",
        "2" * 64,
        "--expected-prior-boot-id-sha256",
        PRIOR_BOOT_SHA256,
        "--expected-current-boot-id-sha256",
        CURRENT_BOOT_SHA256,
    ]
    assert parser.parse_args(base).revision == REVISION
    for index in range(1, len(base), 2):
        if base[index].startswith("--"):
            with pytest.raises(ValueError):
                parser.parse_args(base[:index] + base[index + 2 :])
    with pytest.raises(ValueError):
        parser.parse_args(base + ["--expected-prior-file-sha256", "1" * 64])


def test_owner_cli_rotation_is_isolated_from_live_secret_boundaries(
    monkeypatch,
    capfd,
):
    events: list[object] = []

    class Runtime:
        def trusted_command_prefix(self):
            events.append("runtime")
            return ("/trusted/python", "-I", "-B", "-E", "-s", "/trusted/gcloud.py")

    monkeypatch.setattr(
        owner_launcher,
        "require_trusted_owner_runtime",
        lambda _release: Runtime(),
    )
    monkeypatch.setattr(
        owner_launcher,
        "require_local_launcher_provenance",
        lambda _release: events.append("provenance") or "a" * 64,
    )
    monkeypatch.setattr(
        owner_launcher,
        "_validate_owner_interpreter_invocation",
        lambda _path: events.append("interpreter"),
    )
    monkeypatch.setattr(owner_launcher, "PinnedGcloudConfiguration", lambda: object())
    monkeypatch.setattr(owner_launcher, "GcloudOwnerAccessToken", lambda *_a, **_k: object())
    receipt = {
        "schema": owner_launcher.HOST_RECEIPT_ROTATION_RECEIPT_SCHEMA,
        "ok": True,
        "state": "target_boot_receipt_published_services_stopped",
    }

    class Transport:
        def __init__(self, *_args, **_kwargs):
            events.append("transport")

        def rotate(self, release_sha, **kwargs):
            events.append(("rotate", release_sha, kwargs))
            return receipt

    monkeypatch.setattr(owner_launcher, "IapHostReceiptRotationTransport", Transport)
    monkeypatch.setattr(
        owner_launcher,
        "IapCoordinatorTransport",
        lambda *_a, **_k: pytest.fail("live coordinator must not be constructed"),
    )
    monkeypatch.setattr(
        owner_launcher,
        "CloudSqlTemporaryAdmin",
        lambda *_a, **_k: pytest.fail("Cloud SQL admin must not be constructed"),
    )
    monkeypatch.setattr(
        owner_launcher,
        "OwnerDiscordTokenReader",
        lambda: pytest.fail("Discord token must not be read"),
    )

    assert owner_launcher.main((
        "--release-sha",
        REVISION,
        "--rotate-host-identity-receipt",
        "--external-iam-policy-sha256",
        EXTERNAL_IAM_POLICY_SHA256,
        "--expected-prior-host-receipt-file-sha256",
        "1" * 64,
        "--expected-prior-host-receipt-sha256",
        "2" * 64,
        "--expected-prior-boot-id-sha256",
        PRIOR_BOOT_SHA256,
        "--expected-current-boot-id-sha256",
        CURRENT_BOOT_SHA256,
    )) == 0
    assert owner_launcher.json.loads(capfd.readouterr().out) == receipt
    assert events == [
        "provenance",
        "transport",
        (
            "rotate",
            REVISION,
            {
                "external_iam_policy_sha256": EXTERNAL_IAM_POLICY_SHA256,
                "expected_prior_file_sha256": "1" * 64,
                "expected_prior_receipt_sha256": "2" * 64,
                "expected_prior_boot_id_sha256": PRIOR_BOOT_SHA256,
                "expected_current_boot_id_sha256": CURRENT_BOOT_SHA256,
            },
        ),
        "runtime",
        "interpreter",
        "provenance",
    ]


def test_owner_rotation_remote_command_is_fixed_and_digest_only():
    observed: list[tuple[str, ...]] = []

    class Transport(owner_launcher.IapHostReceiptRotationTransport):
        def __init__(self):
            pass

        def _run_remote(self, remote_argv, **_kwargs):
            observed.append(tuple(remote_argv))
            return subprocess.CompletedProcess(
                tuple(remote_argv),
                0,
                b"{}\n",
                b"",
            )

    transport = Transport()
    assert transport._run_host_receipt_rotation_command(
        REVISION,
        "plan",
        account="owner@example.com",
        external_iam_policy_sha256=EXTERNAL_IAM_POLICY_SHA256,
        expected_prior_file_sha256="1" * 64,
        expected_prior_receipt_sha256="2" * 64,
        expected_prior_boot_id_sha256=PRIOR_BOOT_SHA256,
        expected_current_boot_id_sha256=CURRENT_BOOT_SHA256,
    ) == {}
    argv = observed[0]
    assert argv[:3] == (
        "/usr/bin/env",
        "-i",
        f"--chdir={owner_launcher.STOPPED_RELEASE_SOURCE_BASE}/{REVISION}",
    )
    assert (
        "-m",
        owner_launcher.HOST_RECEIPT_ROTATION_MODULE,
        "plan",
    ) == argv[12:15]
    assert "--approved-plan-sha256" not in argv
    assert set(item for item in argv if len(item) == 64) == {
        EXTERNAL_IAM_POLICY_SHA256,
        "1" * 64,
        "2" * 64,
        PRIOR_BOOT_SHA256,
        CURRENT_BOOT_SHA256,
    }

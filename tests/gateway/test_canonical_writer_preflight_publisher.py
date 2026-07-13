from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

import gateway.canonical_writer_activation as activation
import gateway.canonical_writer_preflight_publisher as publisher


REVISION = "a" * 40
POLICY_SHA = "b" * 64


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _service_state() -> dict[str, dict[str, str]]:
    absent = {
        "LoadState": "not-found",
        "ActiveState": "inactive",
        "SubState": "dead",
        "MainPID": "0",
        "UnitFileState": "",
        "FragmentPath": "",
        "DropInPaths": "",
        "NeedDaemonReload": "no",
    }
    return {unit: dict(absent) for unit in publisher._SERVICE_UNITS}


def _plan() -> dict[str, object]:
    value: dict[str, object] = {
        "schema": publisher.PUBLICATION_PLAN_SCHEMA,
        "revision": REVISION,
        "stopped_release_receipt_path": "/evidence/stopped.json",
        "stopped_release_receipt_file_sha256": "1" * 64,
        "stopped_release_receipt_sha256": "2" * 64,
        "release_root": f"/opt/muncho-canary-releases/{REVISION}",
        "release_artifact_sha256": "3" * 64,
        "release_manifest_path": (
            f"/opt/muncho-canary-releases/{REVISION}/release-manifest.json"
        ),
        "release_manifest_file_sha256": "4" * 64,
        "host_identity_receipt_path": "/host.json",
        "host_identity_receipt_file_sha256": "5" * 64,
        "host_identity_receipt_sha256": "6" * 64,
        "host_identity_sha256": "7" * 64,
        "boot_id_sha256": "8" * 64,
        "database": {
            "host": publisher.SQL_PRIVATE_IP,
            "port": publisher.SQL_PORT,
            "database": publisher.SQL_DATABASE,
            "user": publisher.SQL_USER,
            "tls_server_name": publisher.DATABASE_TLS_SERVER_NAME,
            "ca_path": str(publisher.DATABASE_CA_PATH),
            "ca_sha256": "9" * 64,
        },
        "credential_provenance": {
            "path": str(publisher.DATABASE_CREDENTIAL_PATH),
            "device": 1,
            "inode": 2,
            "owner_uid": 999,
            "group_gid": 994,
            "mode": "0400",
            "link_count": 1,
            "modification_time_ns": 3,
            "change_time_ns": 4,
            "content_or_digest_recorded": False,
        },
        "owner_discord_user_ids": [publisher.OWNER_DISCORD_USER_ID],
        "external_iam_policy_sha256": POLICY_SHA,
        "service_state": _service_state(),
        "fixed_output_paths": {
            "writer_config": str(publisher.DEFAULT_WRITER_CONFIG_SOURCE_PATH),
            "gateway_config": str(publisher.DEFAULT_GATEWAY_CONFIG_SOURCE_PATH),
            "writer_unit": str(publisher.DEFAULT_STAGED_WRITER_UNIT_PATH),
            "gateway_unit": str(publisher.DEFAULT_STAGED_GATEWAY_UNIT_PATH),
            "native_observation_plan": str(publisher.DEFAULT_STAGED_NATIVE_PLAN_PATH),
            "publication_evidence_root": str(publisher.PUBLICATION_EVIDENCE_ROOT),
        },
        "invariants": {
            "services_started": False,
            "units_installed": False,
            "daemon_reloaded": False,
            "approval_created": False,
            "discord_started": False,
            "credential_content_or_digest_recorded": False,
        },
    }
    value["plan_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return value


def test_plan_is_secret_free_self_digest_and_performs_only_reads(monkeypatch):
    stopped = {
        "receipt_sha256": "2" * 64,
        "host_identity_receipt_file_sha256": "5" * 64,
        "release_root": f"/opt/muncho-canary-releases/{REVISION}",
        "release_manifest_path": (
            f"/opt/muncho-canary-releases/{REVISION}/release-manifest.json"
        ),
        "release_manifest_file_sha256": hashlib.sha256(b"manifest").hexdigest(),
        "release_artifact_sha256": "3" * 64,
    }
    host = {
        "receipt_sha256": "6" * 64,
        "host_identity_sha256": "7" * 64,
        "boot_id_sha256": "8" * 64,
    }
    credential = _plan()["credential_provenance"]
    reads: list[str] = []

    monkeypatch.setattr(publisher, "_require_root_linux", lambda: reads.append("root"))
    monkeypatch.setattr(
        publisher,
        "_load_stopped_release_receipt",
        lambda _revision: (stopped, b"stopped\n"),
    )
    monkeypatch.setattr(
        publisher,
        "_load_host_receipt",
        lambda _stopped: (host, b"host"),
    )
    monkeypatch.setattr(
        publisher,
        "load_release_manifest",
        lambda _revision: (SimpleNamespace(artifact_sha256="3" * 64), b"manifest"),
    )
    monkeypatch.setattr(
        publisher,
        "_read_trusted_file",
        lambda path, **_kwargs: b"ca" if path == publisher.DATABASE_CA_PATH else b"",
    )
    monkeypatch.setattr(
        publisher,
        "_credential_identity",
        lambda: (SimpleNamespace(), credential),
    )
    monkeypatch.setattr(
        publisher,
        "_capture_service_snapshot",
        lambda **_kwargs: reads.append("services") or _service_state(),
    )
    monkeypatch.setattr(
        publisher,
        "_require_no_downstream_mutation",
        lambda: reads.append("downstream"),
    )

    result = publisher.plan_writer_preflight_publication(
        revision=REVISION,
        external_iam_policy_sha256=POLICY_SHA,
    )

    unsigned = dict(result)
    digest = unsigned.pop("plan_sha256")
    assert digest == hashlib.sha256(_canonical(unsigned)).hexdigest()
    assert reads == ["root", "services", "downstream"]
    rendered = json.dumps(result).casefold()
    assert "secret-value-must-not-appear" not in rendered
    assert result["credential_provenance"]["content_or_digest_recorded"] is False
    assert result["invariants"]["services_started"] is False


def test_cli_is_strict_and_failure_never_echoes_input(capsys):
    parser = publisher._cli_parser()
    with pytest.raises(ValueError):
        parser.parse_args([
            "plan",
            "--revision",
            REVISION,
            "--revision",
            REVISION,
            "--external-iam-policy-sha256",
            POLICY_SHA,
        ])
    with pytest.raises(ValueError):
        parser.parse_args([
            "plan",
            "--rev",
            REVISION,
            "--external-iam-policy-sha256",
            POLICY_SHA,
        ])
    sentinel = "DO_NOT_ECHO_SECRET"
    assert publisher.main(["plan", "--revision", sentinel]) == 2
    assert sentinel not in capsys.readouterr().out


def test_apply_rejects_unapproved_plan_before_lock_file_mutation(monkeypatch):
    plan = _plan()
    entered = False

    @contextlib.contextmanager
    def lock():
        nonlocal entered
        entered = True
        yield

    monkeypatch.setattr(
        publisher,
        "plan_writer_preflight_publication",
        lambda **_kwargs: plan,
    )
    monkeypatch.setattr(publisher, "_host_activation_lock", lock)

    with pytest.raises(PermissionError, match="does not match"):
        publisher.apply_writer_preflight_publication(
            revision=REVISION,
            external_iam_policy_sha256=POLICY_SHA,
            approved_plan_sha256="0" * 64,
        )

    assert entered is False


def test_public_native_preflight_wrapper_has_no_mutation_surface(monkeypatch):
    class DummyPlan:
        def __init__(self, value):
            self.value = value
            self.sha256 = "f" * 64

        def to_mapping(self):
            return dict(self.value)

        @classmethod
        def from_mapping(cls, value):
            return cls(value)

    value = {
        "revision": REVISION,
        "artifact_sha256": "1" * 64,
        "release_manifest_file_sha256": "2" * 64,
        "config_collector_receipt_sha256": "3" * 64,
        "external_iam_policy_sha256": "4" * 64,
        "host_identity_sha256": "5" * 64,
        "boot_id_sha256": "6" * 64,
    }
    calls = []
    collector = SimpleNamespace(
        value={
            "hba_observed_at_unix": 100,
            "collected_at_unix": 110,
            "hba_expires_at_unix": 400,
        },
        require_fresh=lambda now: calls.append(("fresh", now)),
    )
    monkeypatch.setattr(activation, "NativeObservationPlan", DummyPlan)
    monkeypatch.setattr(activation, "_require_root_linux", lambda: calls.append("root"))
    monkeypatch.setattr(
        activation,
        "_verify_native_preflight_inputs",
        lambda plan, **kwargs: calls.append((plan, kwargs)) or collector,
    )

    result = activation.native_observation_read_only_preflight(
        DummyPlan(value),
        _clock=lambda: 120,
    )

    assert calls[0] == "root"
    assert calls[1][1]["require_installed"] is False
    assert calls[1][1]["require_original_boot"] is True
    assert calls[2] == ("fresh", 120)
    assert result["observed_at_unix"] == 120
    assert result["collector_hba_expires_at_unix"] == 400
    assert result["services_started"] is False
    assert result["units_installed"] is False
    assert result["daemon_reloaded"] is False
    unsigned = dict(result)
    digest = unsigned.pop("report_sha256")
    assert digest == hashlib.sha256(_canonical(unsigned)).hexdigest()


def test_apply_seals_terminal_receipt_without_install_or_start(tmp_path, monkeypatch):
    plan = _plan()
    receipt_path = tmp_path / "publication.json"
    report_path = tmp_path / "reports" / f"{'e' * 64}.json"
    collector = SimpleNamespace(
        sha256="c" * 64,
        value={
            "writer_config_sha256": "1" * 64,
            "gateway_config_sha256": "2" * 64,
        },
    )
    native = SimpleNamespace(
        sha256="d" * 64,
        value={
            "writer_config": {"sha256": "1" * 64},
            "gateway_config": {"sha256": "2" * 64},
            "writer_unit": {"sha256": "3" * 64},
            "gateway_unit": {"sha256": "4" * 64},
        },
    )
    report = {
        "report_sha256": "e" * 64,
        "config_collector_receipt_sha256": "c" * 64,
        "native_observation_plan_sha256": "d" * 64,
        "collector_hba_observed_at_unix": 1_000,
        "collector_collected_at_unix": 1_100,
        "observed_at_unix": 1_200,
        "collector_hba_expires_at_unix": 1_300,
    }
    writes: list[tuple[Path, bytes, int]] = []

    monkeypatch.setattr(
        publisher,
        "plan_writer_preflight_publication",
        lambda **_kwargs: plan,
    )
    monkeypatch.setattr(publisher, "_host_activation_lock", contextlib.nullcontext)
    monkeypatch.setattr(
        publisher, "_publication_receipt_path", lambda _plan: receipt_path
    )
    monkeypatch.setattr(publisher.os.path, "lexists", lambda path: False)
    monkeypatch.setattr(
        publisher,
        "_resume_persisted_report_state",
        lambda _plan: None,
    )
    monkeypatch.setattr(
        publisher,
        "_collect_or_resume_configs",
        lambda _plan, **_kwargs: collector,
    )
    monkeypatch.setattr(
        publisher,
        "_load_or_stage_native_plan",
        lambda _plan, _receipt: native,
    )
    monkeypatch.setattr(
        publisher,
        "_seal_or_resume_preflight_report",
        lambda **_kwargs: (
            report,
            report_path,
            "f" * 64,
        ),
    )
    artifacts = {
        name: {"path": str(path), "sha256": str(index) * 64}
        for index, (name, path) in enumerate(
            (
                ("writer_config", publisher.DEFAULT_WRITER_CONFIG_SOURCE_PATH),
                ("gateway_config", publisher.DEFAULT_GATEWAY_CONFIG_SOURCE_PATH),
                ("writer_unit", publisher.DEFAULT_STAGED_WRITER_UNIT_PATH),
                ("gateway_unit", publisher.DEFAULT_STAGED_GATEWAY_UNIT_PATH),
                (
                    "native_observation_plan",
                    publisher.DEFAULT_STAGED_NATIVE_PLAN_PATH,
                ),
            ),
            start=1,
        )
    }
    monkeypatch.setattr(publisher, "_receipt_artifacts", lambda _native: artifacts)
    monkeypatch.setattr(
        publisher,
        "_capture_service_snapshot",
        lambda **_kwargs: _service_state(),
    )
    monkeypatch.setattr(publisher, "_require_no_downstream_mutation", lambda: None)
    monkeypatch.setattr(publisher, "_ensure_root_directory", lambda _path: None)
    monkeypatch.setattr(
        publisher,
        "_recover_target_install_temporaries",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        publisher,
        "_recover_report_install_temporaries",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        publisher,
        "_revalidate_pre_receipt_truth",
        lambda **_kwargs: (
            collector,
            "7" * 64,
            native,
            report,
            report_path,
            "f" * 64,
            artifacts,
            _service_state(),
        ),
    )

    def install(path, payload, **kwargs):
        writes.append((path, payload, kwargs["mode"]))
        return True

    monkeypatch.setattr(publisher, "_install_exact_bytes", install)
    monkeypatch.setattr(
        publisher,
        "_validate_terminal_receipt",
        lambda _path, **_kwargs: json.loads(writes[-1][1]),
    )

    result = publisher.apply_writer_preflight_publication(
        revision=REVISION,
        external_iam_policy_sha256=POLICY_SHA,
        approved_plan_sha256=str(plan["plan_sha256"]),
        _clock=lambda: 1234,
    )

    assert writes == [(receipt_path, writes[0][1], 0o400)]
    assert result["state"] == "staged_preflight_passed_services_stopped"
    assert result["invariants"] == plan["invariants"]
    assert result["config_collector_receipt_sha256"] == collector.sha256
    assert result["native_observation_plan_sha256"] == native.sha256
    assert result["preflight_report_path"] == str(report_path)
    assert result["preflight_report_file_sha256"] == "f" * 64
    assert result["service_state_before"] == _service_state()
    assert result["service_state_after"] == _service_state()


def test_planner_residue_without_exact_collector_receipt_blocks(monkeypatch):
    monkeypatch.setattr(
        publisher,
        "_matching_collector_receipts",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        publisher.os.path,
        "lexists",
        lambda path: path != publisher.DEFAULT_STAGED_NATIVE_PLAN_PATH,
    )
    monkeypatch.setattr(
        publisher,
        "collect_and_stage",
        lambda **_kwargs: pytest.fail("collector must not replace planner-bound input"),
    )
    with pytest.raises(RuntimeError, match="lacks its exact collector receipt"):
        publisher._collect_or_resume_configs(
            _plan(),
            now_unix=10,
            clock=lambda: 10,
        )


def test_native_bound_rollback_ignores_unrelated_historical_receipts(monkeypatch):
    plan = _plan()
    native = SimpleNamespace(
        sha256="d" * 64,
        value={"config_collector_receipt_sha256": "c" * 64},
    )
    bound = SimpleNamespace(sha256="c" * 64)
    removed: list[Path] = []

    monkeypatch.setattr(publisher.os.path, "lexists", lambda _path: True)
    monkeypatch.setattr(publisher, "_load_staged_native_plan", lambda: native)
    monkeypatch.setattr(
        publisher,
        "_matching_collector_receipts",
        lambda **_kwargs: pytest.fail(
            "native rollback must use its exact collector digest, not history"
        ),
    )
    monkeypatch.setattr(
        publisher,
        "_load_bound_collector_receipt",
        lambda *_args, **_kwargs: (bound, "e" * 64),
    )
    monkeypatch.setattr(
        publisher,
        "_validate_native_binding",
        lambda _plan, _native, receipt, **_kwargs: (
            None if receipt is bound else pytest.fail("wrong collector receipt")
        ),
    )
    unit_bytes = {
        publisher.DEFAULT_STAGED_WRITER_UNIT_PATH: b"writer-unit",
        publisher.DEFAULT_STAGED_GATEWAY_UNIT_PATH: b"gateway-unit",
    }
    monkeypatch.setattr(
        publisher,
        "_expected_unit_bytes",
        lambda _revision: unit_bytes,
    )
    monkeypatch.setattr(
        publisher,
        "_unlink_exact",
        lambda path, **_kwargs: removed.append(path),
    )

    publisher._rollback_exact_stale_planner_residue(plan)

    assert removed == [
        publisher.DEFAULT_STAGED_NATIVE_PLAN_PATH,
        publisher.DEFAULT_STAGED_GATEWAY_UNIT_PATH,
        publisher.DEFAULT_STAGED_WRITER_UNIT_PATH,
    ]


def test_stale_unrelated_collector_receipt_does_not_mask_fresh_exact_one(
    monkeypatch,
):
    plan = _plan()
    writer_raw = b"writer"
    gateway_raw = b"gateway"

    class Receipt:
        def __init__(self, sha256, *, writer_sha):
            self.sha256 = sha256
            self.fresh_checks = 0
            self.value = {
                "writer_config_sha256": writer_sha,
                "gateway_config_sha256": hashlib.sha256(gateway_raw).hexdigest(),
                "release_artifact_sha256": plan["release_artifact_sha256"],
                "release_manifest_file_sha256": plan["release_manifest_file_sha256"],
                "database": plan["database"],
                "credential_provenance": plan["credential_provenance"],
            }

        def require_fresh(self, _now):
            self.fresh_checks += 1

    stale_unrelated = Receipt("1" * 64, writer_sha="0" * 64)
    fresh_exact = Receipt(
        "2" * 64,
        writer_sha=hashlib.sha256(writer_raw).hexdigest(),
    )
    receipts = {
        stale_unrelated.sha256: stale_unrelated,
        fresh_exact.sha256: fresh_exact,
    }
    monkeypatch.setattr(publisher.os.path, "lexists", lambda _path: True)
    monkeypatch.setattr(
        publisher,
        "_trusted_staged_bytes",
        lambda path, **_kwargs: (
            writer_raw
            if path == publisher.DEFAULT_WRITER_CONFIG_SOURCE_PATH
            else gateway_raw
        ),
    )
    monkeypatch.setattr(
        publisher.os,
        "listdir",
        lambda _path: [f"{stale_unrelated.sha256}.json", f"{fresh_exact.sha256}.json"],
    )
    monkeypatch.setattr(
        publisher,
        "load_config_collector_receipt",
        lambda **kwargs: receipts[kwargs["receipt_sha256"]],
    )

    assert (
        publisher._matching_collector_receipt(
            plan=plan,
            require_fresh=True,
        )
        is fresh_exact
    )
    assert stale_unrelated.fresh_checks == 0
    assert fresh_exact.fresh_checks == 1


def test_expired_exact_planner_residue_is_rolled_back_and_replanned(monkeypatch):
    plan = _plan()
    fresh = SimpleNamespace(sha256="2" * 64, to_mapping=lambda: {"fresh": True})
    events: list[str] = []
    matches = iter((None, fresh))

    monkeypatch.setattr(publisher.os.path, "lexists", lambda _path: True)
    monkeypatch.setattr(
        publisher,
        "_matching_collector_receipt",
        lambda **_kwargs: next(matches),
    )
    monkeypatch.setattr(
        publisher,
        "_rollback_exact_stale_planner_residue",
        lambda _plan: events.append("rollback"),
    )
    monkeypatch.setattr(
        publisher,
        "collect_and_stage",
        lambda **_kwargs: events.append("collect") or {"receipt_sha256": fresh.sha256},
    )
    monkeypatch.setattr(
        publisher,
        "load_config_collector_receipt",
        lambda **_kwargs: fresh,
    )

    result = publisher._collect_or_resume_configs(
        plan,
        now_unix=10_000,
        clock=lambda: 10_000,
    )

    assert result is fresh
    assert events == ["rollback", "collect"]


@pytest.mark.parametrize("fault_stage", ("collector", "planner", "report", "receipt"))
def test_apply_retry_is_safe_after_every_durable_boundary(
    fault_stage,
    tmp_path,
    monkeypatch,
):
    plan = _plan()
    receipt_path = tmp_path / "publication.json"
    report_path = tmp_path / "reports" / f"{'e' * 64}.json"
    collector = SimpleNamespace(
        sha256="c" * 64,
        value={
            "writer_config_sha256": "1" * 64,
            "gateway_config_sha256": "2" * 64,
        },
    )
    native = SimpleNamespace(
        sha256="d" * 64,
        value={
            "writer_config": {"sha256": "1" * 64},
            "gateway_config": {"sha256": "2" * 64},
            "writer_unit": {"sha256": "3" * 64},
            "gateway_unit": {"sha256": "4" * 64},
        },
    )
    report = {
        "report_sha256": "e" * 64,
        "config_collector_receipt_sha256": "c" * 64,
        "native_observation_plan_sha256": "d" * 64,
        "collector_hba_observed_at_unix": 0,
        "collector_collected_at_unix": 50,
        "observed_at_unix": 100,
        "collector_hba_expires_at_unix": 300,
    }
    report_raw = _canonical(report)
    state = {
        "faulted": False,
        "collector": False,
        "native": False,
        "report": False,
        "receipt_raw": None,
    }

    monkeypatch.setattr(
        publisher,
        "plan_writer_preflight_publication",
        lambda **_kwargs: plan,
    )
    monkeypatch.setattr(publisher, "_host_activation_lock", contextlib.nullcontext)
    monkeypatch.setattr(
        publisher,
        "_recover_target_install_temporaries",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        publisher,
        "_recover_report_install_temporaries",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        publisher,
        "_publication_receipt_path",
        lambda _plan: receipt_path,
    )
    monkeypatch.setattr(
        publisher.os.path,
        "lexists",
        lambda path: path == receipt_path and state["receipt_raw"] is not None,
    )

    def resume(_plan):
        if not state["report"]:
            return None
        return collector, native, (report, report_path, report_raw)

    monkeypatch.setattr(publisher, "_resume_persisted_report_state", resume)

    def collect(_plan, **_kwargs):
        state["collector"] = True
        if fault_stage == "collector" and not state["faulted"]:
            state["faulted"] = True
            raise RuntimeError("fault after collector boundary")
        return collector

    monkeypatch.setattr(publisher, "_collect_or_resume_configs", collect)

    def stage_native(_plan, _collector):
        assert state["collector"]
        state["native"] = True
        if fault_stage == "planner" and not state["faulted"]:
            state["faulted"] = True
            raise RuntimeError("fault after planner boundary")
        return native

    monkeypatch.setattr(publisher, "_load_or_stage_native_plan", stage_native)

    def seal_report(**_kwargs):
        assert state["native"]
        state["report"] = True
        if fault_stage == "report" and not state["faulted"]:
            state["faulted"] = True
            raise RuntimeError("fault after report boundary")
        return report, report_path, hashlib.sha256(report_raw).hexdigest()

    monkeypatch.setattr(publisher, "_seal_or_resume_preflight_report", seal_report)
    monkeypatch.setattr(
        publisher,
        "_receipt_artifacts",
        lambda _native: {
            name: {"path": path, "sha256": str(index) * 64}
            for index, (name, path) in enumerate(
                (
                    ("writer_config", plan["fixed_output_paths"]["writer_config"]),
                    ("gateway_config", plan["fixed_output_paths"]["gateway_config"]),
                    ("writer_unit", plan["fixed_output_paths"]["writer_unit"]),
                    ("gateway_unit", plan["fixed_output_paths"]["gateway_unit"]),
                    (
                        "native_observation_plan",
                        plan["fixed_output_paths"]["native_observation_plan"],
                    ),
                ),
                start=1,
            )
        },
    )
    artifacts = publisher._receipt_artifacts(native)
    monkeypatch.setattr(
        publisher,
        "_revalidate_pre_receipt_truth",
        lambda **_kwargs: (
            collector,
            "7" * 64,
            native,
            report,
            report_path,
            hashlib.sha256(report_raw).hexdigest(),
            artifacts,
            _service_state(),
        ),
    )
    monkeypatch.setattr(
        publisher,
        "_capture_service_snapshot",
        lambda **_kwargs: _service_state(),
    )
    monkeypatch.setattr(publisher, "_require_no_downstream_mutation", lambda: None)
    monkeypatch.setattr(publisher, "_ensure_root_directory", lambda _path: None)

    def install(path, payload, **_kwargs):
        assert path == receipt_path
        state["receipt_raw"] = payload
        if fault_stage == "receipt" and not state["faulted"]:
            state["faulted"] = True
            raise RuntimeError("fault after receipt boundary")
        return True

    monkeypatch.setattr(publisher, "_install_exact_bytes", install)
    monkeypatch.setattr(
        publisher,
        "_validate_terminal_receipt",
        lambda _path, **_kwargs: json.loads(state["receipt_raw"]),
    )
    times = iter((100, 10_000, 20_000, 30_000, 40_000, 50_000))

    with pytest.raises(RuntimeError, match="fault after"):
        publisher.apply_writer_preflight_publication(
            revision=REVISION,
            external_iam_policy_sha256=POLICY_SHA,
            approved_plan_sha256=str(plan["plan_sha256"]),
            _clock=lambda: next(times),
        )
    result = publisher.apply_writer_preflight_publication(
        revision=REVISION,
        external_iam_policy_sha256=POLICY_SHA,
        approved_plan_sha256=str(plan["plan_sha256"]),
        _clock=lambda: next(times),
    )

    assert result["state"] == "staged_preflight_passed_services_stopped"
    assert state["faulted"] is True
    assert state["receipt_raw"] is not None
    assert result["preflight_observed_at_unix"] == 100
    assert result["preflight_collector_hba_expires_at_unix"] == 300
    assert result["sealed_at_unix"] > 300
    assert result["preflight_fresh_at_seal"] is False


def test_service_snapshot_drift_is_rejected_before_receipt():
    drifted = _service_state()
    drifted[publisher.WRITER_UNIT]["ActiveState"] = "active"
    drifted[publisher.WRITER_UNIT]["SubState"] = "running"
    drifted[publisher.WRITER_UNIT]["MainPID"] = "123"
    with pytest.raises(RuntimeError, match="not exact stopped"):
        publisher._validate_service_snapshot(drifted)


def test_persisted_report_is_reloaded_and_rehashed_on_every_retry(monkeypatch):
    plan = _plan()
    collector = SimpleNamespace(
        value={
            "hba_observed_at_unix": 1_000,
            "collected_at_unix": 1_100,
            "hba_expires_at_unix": 1_300,
        }
    )
    native = SimpleNamespace(
        sha256="d" * 64,
        value={"config_collector_receipt_sha256": "c" * 64},
    )
    unsigned = {
        "schema": "muncho-writer-native-read-only-preflight.v2",
        "ok": True,
        "state": "staged_inputs_verified_services_stopped",
        "revision": REVISION,
        "native_observation_plan_sha256": native.sha256,
        "release_artifact_sha256": plan["release_artifact_sha256"],
        "release_manifest_file_sha256": plan["release_manifest_file_sha256"],
        "config_collector_receipt_sha256": "c" * 64,
        "external_iam_policy_sha256": POLICY_SHA,
        "host_identity_sha256": plan["host_identity_sha256"],
        "boot_id_sha256": plan["boot_id_sha256"],
        "collector_hba_observed_at_unix": 1_000,
        "collector_collected_at_unix": 1_100,
        "observed_at_unix": 1_200,
        "collector_hba_expires_at_unix": 1_300,
        "services_started": False,
        "units_installed": False,
        "daemon_reloaded": False,
        "discord_started": False,
        "approval_created": False,
        "credential_content_or_digest_recorded": False,
    }
    report = {
        **unsigned,
        "report_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }
    raw = _canonical(report)
    directory = Path("/trusted/reports")
    reads = iter((raw, raw + b" "))
    calls = 0

    monkeypatch.setattr(
        publisher,
        "_preflight_report_directory",
        lambda _plan: directory,
    )
    monkeypatch.setattr(publisher.os.path, "lexists", lambda _path: True)
    monkeypatch.setattr(
        publisher.os,
        "lstat",
        lambda _path: SimpleNamespace(
            st_mode=stat.S_IFDIR | 0o700,
            st_uid=0,
            st_gid=0,
        ),
    )
    monkeypatch.setattr(
        publisher.os,
        "listdir",
        lambda _path: [f"{report['report_sha256']}.json"],
    )

    def read_report(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return next(reads)

    monkeypatch.setattr(publisher, "_read_trusted_file", read_report)

    loaded = publisher._load_persisted_preflight_report(
        plan=plan,
        native=native,
        collector=collector,
    )
    assert loaded is not None and loaded[0] == report
    with pytest.raises(ValueError, match="canonical JSON"):
        publisher._load_persisted_preflight_report(
            plan=plan,
            native=native,
            collector=collector,
        )
    assert calls == 2


@pytest.mark.parametrize("mutation", ("credential", "ca", "release"))
def test_final_revalidation_rejects_mutable_plan_truth_after_initial_approval(
    mutation,
    monkeypatch,
):
    plan = _plan()
    changed = json.loads(json.dumps(plan))
    if mutation == "credential":
        changed["credential_provenance"]["inode"] += 1
    elif mutation == "ca":
        changed["database"]["ca_sha256"] = "a" * 64
    else:
        changed["release_artifact_sha256"] = "b" * 64
    unsigned = dict(changed)
    unsigned.pop("plan_sha256")
    changed["plan_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()

    monkeypatch.setattr(
        publisher,
        "plan_writer_preflight_publication",
        lambda **_kwargs: changed,
    )
    monkeypatch.setattr(
        publisher,
        "_load_bound_collector_receipt",
        lambda *_args, **_kwargs: pytest.fail(
            "durable outputs must not be trusted after plan truth drift"
        ),
    )

    with pytest.raises(RuntimeError, match="plan changed before sealing"):
        publisher._revalidate_pre_receipt_truth(
            plan=plan,
            expected_collector=SimpleNamespace(sha256="c" * 64),
            expected_native=SimpleNamespace(sha256="d" * 64),
            expected_report={"report_sha256": "e" * 64},
            expected_report_path=Path("/reports/report.json"),
            service_runner=lambda _command: None,
        )


@pytest.mark.parametrize("target_kind", ("report", "receipt"))
@pytest.mark.parametrize("linked_target", (False, True))
def test_sigkill_install_temporary_is_recovered_only_after_exact_identity_check(
    tmp_path,
    monkeypatch,
    target_kind,
    linked_target,
):
    directory = tmp_path / target_kind
    directory.mkdir(mode=0o700)
    if target_kind == "report":
        target = directory / f"{'e' * 64}.json"
    else:
        target = directory / "publication.json"
    temporary = directory / f".{target.name}.activation.123"
    temporary.write_bytes(b"partial-or-complete-durable-payload")
    temporary.chmod(0o400 if linked_target else 0o600)
    if linked_target:
        os.link(temporary, target)

    real_lstat = os.lstat

    def root_lstat(path):
        item = real_lstat(path)
        return SimpleNamespace(
            st_mode=item.st_mode,
            st_uid=0,
            st_gid=0,
            st_nlink=item.st_nlink,
            st_size=item.st_size,
            st_dev=item.st_dev,
            st_ino=item.st_ino,
        )

    monkeypatch.setattr(publisher.os, "lstat", root_lstat)
    monkeypatch.setattr(publisher, "_ensure_root_directory", lambda _path: None)
    monkeypatch.setattr(publisher, "_list_xattrs", lambda _path: ())
    monkeypatch.setattr(publisher, "_fsync_directory", lambda _path: None)

    if target_kind == "report":
        monkeypatch.setattr(
            publisher,
            "_preflight_report_directory",
            lambda _plan: directory,
        )
        publisher._recover_report_install_temporaries(_plan())
    else:
        publisher._recover_target_install_temporaries(
            target,
            maximum=publisher._MAX_PUBLIC_JSON_BYTES,
        )

    assert not temporary.exists()
    assert target.exists() is linked_target
    if linked_target:
        assert real_lstat(target).st_nlink == 1


def test_report_time_envelope_rejects_observation_after_collector_expiry():
    plan = _plan()
    native = SimpleNamespace(
        sha256="d" * 64,
        value={"config_collector_receipt_sha256": "c" * 64},
    )
    collector = SimpleNamespace(
        value={
            "hba_observed_at_unix": 1_000,
            "collected_at_unix": 1_100,
            "hba_expires_at_unix": 1_300,
        }
    )
    unsigned = {
        "schema": "muncho-writer-native-read-only-preflight.v2",
        "ok": True,
        "state": "staged_inputs_verified_services_stopped",
        "revision": REVISION,
        "native_observation_plan_sha256": native.sha256,
        "release_artifact_sha256": plan["release_artifact_sha256"],
        "release_manifest_file_sha256": plan["release_manifest_file_sha256"],
        "config_collector_receipt_sha256": "c" * 64,
        "external_iam_policy_sha256": POLICY_SHA,
        "host_identity_sha256": plan["host_identity_sha256"],
        "boot_id_sha256": plan["boot_id_sha256"],
        "collector_hba_observed_at_unix": 1_000,
        "collector_collected_at_unix": 1_100,
        "observed_at_unix": 1_301,
        "collector_hba_expires_at_unix": 1_300,
        "services_started": False,
        "units_installed": False,
        "daemon_reloaded": False,
        "discord_started": False,
        "approval_created": False,
        "credential_content_or_digest_recorded": False,
    }
    report = {
        **unsigned,
        "report_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }

    with pytest.raises(RuntimeError, match="report binding drifted"):
        publisher._validate_native_preflight_report(
            report,
            plan=plan,
            native=native,
            collector=collector,
        )

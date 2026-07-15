from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest

from gateway import production_alias_projection_cutover as rail


PLAN = "1" * 64
PACKAGE = "2" * 64
REVISION = "a" * 40


class FakeSystemd:
    def __init__(self, systemd_root: Path) -> None:
        self.root = systemd_root
        self.enabled = {name: False for name in rail._UNITS}
        self.active = {name: False for name in rail._UNITS}
        self.results = {name: "success" for name in rail._UNITS}
        self.core_active = False
        self.starts: list[str] = []

    def state(self, unit: str) -> rail.UnitState:
        if unit in {
            "muncho-canonical-writer.service",
            "hermes-cloud-gateway.service",
        }:
            return rail.UnitState(
                load_state="loaded",
                active_state="active" if self.core_active else "inactive",
                unit_file_state="enabled",
                fragment_path=f"/etc/systemd/system/{unit}",
                result="success",
            )
        present = (self.root / unit).exists()
        return rail.UnitState(
            load_state="loaded" if present else "not-found",
            active_state="active" if self.active[unit] else "inactive",
            unit_file_state="enabled" if self.enabled[unit] else "disabled",
            fragment_path=str(self.root / unit) if present else "",
            result=self.results[unit],
        )

    def daemon_reload(self) -> None:
        return None

    def disable_now(self, unit: str) -> None:
        self.enabled[unit] = False
        self.active[unit] = False

    def enable_now(self, unit: str) -> None:
        self.enabled[unit] = True
        self.active[unit] = True

    def start(self, unit: str) -> None:
        self.starts.append(unit)
        self.active[unit] = False
        self.results[unit] = "success"

    def stop(self, unit: str) -> None:
        self.active[unit] = False


def _context(tmp_path: Path) -> rail.RuntimeContext:
    uid = os.getuid()
    gid = os.getgid()
    state = tmp_path / "state"
    state.mkdir()
    private = state / "private"
    projector_root = state / "projector"
    public = projector_root / "public"
    manifest = {
        "package_sha256": PACKAGE,
        "release_revision": REVISION,
        "identities": {
            "writer": {
                "user": "muncho-canonical-writer",
                "group": "muncho-canonical-writer",
                "uid": uid,
                "gid": gid,
            },
            "projector": {
                "user": "muncho-projector",
                "group": "muncho-projector",
                "uid": uid,
                "gid": gid,
            },
            "gateway": {
                "user": "ai-platform-brain",
                "group": "ai-platform-brain",
                "uid": uid,
                "gid": gid,
            },
        },
        "directories": {
            str(private): {"uid": uid, "gid": gid, "mode": "0750"},
            str(projector_root): {"uid": uid, "gid": gid, "mode": "0751"},
            str(public): {"uid": uid, "gid": gid, "mode": "2750"},
        },
        "files": {
            "writer_export": {
                "path": str(private / "canonical-events.json"),
                "uid": uid,
                "gid": gid,
                "mode": "0640",
                "created_by": rail.EXPORTER_UNIT,
            },
            "public_projection": {
                "path": str(public / "team-member-aliases.json"),
                "uid": uid,
                "gid": gid,
                "mode": "0640",
                "created_by": rail.PROJECTOR_UNIT,
            },
            "public_run_receipt": {
                "path": str(public / "team-member-aliases.receipt.json"),
                "uid": uid,
                "gid": gid,
                "mode": "0640",
                "created_by": rail.PROJECTOR_UNIT,
            },
        },
    }
    payloads = {name: f"unit:{name}\n".encode() for name in rail._UNITS}
    return rail.RuntimeContext(manifest=manifest, unit_payloads=payloads)


def _wire(monkeypatch, context):
    monkeypatch.setattr(rail, "load_runtime_context", lambda **_kwargs: context)
    monkeypatch.setattr(
        rail,
        "validate_host_dependencies",
        lambda *_args, **_kwargs: {
            "identities_exact": True,
            "release_dependencies_executable_and_readable": True,
            "writer_credential_writer_only": True,
        },
    )
    monkeypatch.setattr(rail.os, "fchown", lambda *_args: None)

    def installed_exact(candidate, *, systemd, require_disabled=True):
        states = {name: systemd.state(name) for name in rail._UNITS}
        if states[rail.EXPORTER_UNIT].active or states[rail.PROJECTOR_UNIT].active:
            return False
        if require_disabled and any(
            state.active or state.enabled for state in states.values()
        ):
            return False
        return all(
            (rail.SYSTEMD_ROOT / name).exists()
            and (rail.SYSTEMD_ROOT / name).read_bytes() == payload
            for name, payload in candidate.unit_payloads.items()
        )

    monkeypatch.setattr(rail, "_installed_units_exact", installed_exact)


def _preflight(tmp_path, monkeypatch, context, systemd, authority):
    return rail.preflight(
        cutover_plan_sha256=PLAN,
        package_root=tmp_path / "package",
        expected_revision=REVISION,
        expected_package_sha256=PACKAGE,
        evidence_root=tmp_path / "evidence",
        activation_authority_path=authority,
        systemd=systemd,
        identities=object(),
        clock=lambda: "2026-07-15T00:00:00Z",
        require_root=False,
        enforce_production_address=False,
        enforce_package_metadata=False,
    )


def _apply(tmp_path, context, systemd, authority, pre):
    return rail.apply(
        cutover_plan_sha256=PLAN,
        package_root=tmp_path / "package",
        expected_revision=REVISION,
        expected_package_sha256=PACKAGE,
        expected_preflight_receipt_sha256=pre["receipt_sha256"],
        evidence_root=tmp_path / "evidence",
        activation_authority_path=authority,
        systemd=systemd,
        identities=object(),
        clock=lambda: "2026-07-15T00:00:01Z",
        require_root=False,
        enforce_production_address=False,
        enforce_package_metadata=False,
    )


def _postflight(tmp_path, context, systemd, authority, pre, applied):
    return rail.postflight(
        cutover_plan_sha256=PLAN,
        package_root=tmp_path / "package",
        expected_revision=REVISION,
        expected_package_sha256=PACKAGE,
        expected_preflight_receipt_sha256=pre["receipt_sha256"],
        expected_apply_receipt_sha256=applied["receipt_sha256"],
        evidence_root=tmp_path / "evidence",
        activation_authority_path=authority,
        systemd=systemd,
        identities=object(),
        clock=lambda: "2026-07-15T00:00:02Z",
        require_root=False,
        enforce_production_address=False,
        enforce_package_metadata=False,
    )


def test_activation_authority_binds_writer_gateway_and_terminal_entries():
    authority = rail.build_activation_authority(
        cutover_plan_sha256=PLAN,
        package_sha256=PACKAGE,
        postflight_receipt_sha256="3" * 64,
        database_terminal_entry_sha256="4" * 64,
        activation_commit_intent_entry_sha256="5" * 64,
        writer_ready_entry_sha256="6" * 64,
        gateway_started_entry_sha256="7" * 64,
    )
    assert rail.validate_activation_authority(
        authority,
        cutover_plan_sha256=PLAN,
        package_sha256=PACKAGE,
        postflight_receipt_sha256="3" * 64,
        expected_authority_sha256=authority["authority_sha256"],
    ) == authority
    tampered = copy.deepcopy(authority)
    tampered["writer_ready_entry_sha256"] = "8" * 64
    with pytest.raises(
        rail.ProductionAliasProjectionCutoverError,
        match="authority_invalid",
    ):
        rail.validate_activation_authority(
            tampered,
            cutover_plan_sha256=PLAN,
            package_sha256=PACKAGE,
            postflight_receipt_sha256="3" * 64,
            expected_authority_sha256=authority["authority_sha256"],
        )


def test_apply_receipt_failure_recovers_on_restart_and_rollback_restores(
    tmp_path, monkeypatch
):
    systemd_root = tmp_path / "systemd"
    systemd_root.mkdir()
    monkeypatch.setattr(rail, "SYSTEMD_ROOT", systemd_root)
    context = _context(tmp_path)
    _wire(monkeypatch, context)
    systemd = FakeSystemd(systemd_root)
    authority = tmp_path / "authority.json"
    pre = _preflight(tmp_path, monkeypatch, context, systemd, authority)
    real_atomic = rail._atomic_write
    failed = False

    def fail_apply_receipt(path, payload, **kwargs):
        nonlocal failed
        if path.name == "apply.json" and not failed:
            failed = True
            raise OSError("injected receipt failure")
        return real_atomic(path, payload, **kwargs)

    monkeypatch.setattr(rail, "_atomic_write", fail_apply_receipt)
    with pytest.raises(OSError, match="injected receipt failure"):
        _apply(tmp_path, context, systemd, authority, pre)
    assert all((systemd_root / name).exists() for name in rail._UNITS)
    assert systemd.enabled[rail.PROJECTOR_TIMER] is False

    applied = _apply(tmp_path, context, systemd, authority, pre)
    post = _postflight(
        tmp_path, context, systemd, authority, pre, applied
    )
    assert post["evidence"]["rollback_available_before_terminal_authority"] is True
    rolled_back = rail.rollback(
        cutover_plan_sha256=PLAN,
        package_root=tmp_path / "package",
        expected_revision=REVISION,
        expected_package_sha256=PACKAGE,
        expected_preflight_receipt_sha256=pre["receipt_sha256"],
        expected_apply_receipt_sha256=applied["receipt_sha256"],
        evidence_root=tmp_path / "evidence",
        activation_authority_path=authority,
        systemd=systemd,
        clock=lambda: "2026-07-15T00:00:03Z",
        require_root=False,
        enforce_production_address=False,
        enforce_package_metadata=False,
    )
    assert rolled_back["evidence"]["unit_prestate_restored"] is True
    assert all(not (systemd_root / name).exists() for name in rail._UNITS)
    assert all(not Path(path).exists() for path in context.manifest["directories"])


def test_missing_authority_cannot_start_projector_and_authority_blocks_rollback(
    tmp_path, monkeypatch
):
    systemd_root = tmp_path / "systemd"
    systemd_root.mkdir()
    monkeypatch.setattr(rail, "SYSTEMD_ROOT", systemd_root)
    context = _context(tmp_path)
    _wire(monkeypatch, context)
    systemd = FakeSystemd(systemd_root)
    authority = tmp_path / "authority.json"
    pre = _preflight(tmp_path, monkeypatch, context, systemd, authority)
    applied = _apply(tmp_path, context, systemd, authority, pre)
    post = _postflight(tmp_path, context, systemd, authority, pre, applied)
    systemd.core_active = True

    with pytest.raises(
        rail.ProductionAliasProjectionCutoverError,
        match="file_unavailable",
    ):
        rail.activate(
            cutover_plan_sha256=PLAN,
            package_root=tmp_path / "package",
            expected_revision=REVISION,
            expected_package_sha256=PACKAGE,
            expected_preflight_receipt_sha256=pre["receipt_sha256"],
            expected_apply_receipt_sha256=applied["receipt_sha256"],
            expected_postflight_receipt_sha256=post["receipt_sha256"],
            expected_activation_authority_sha256="9" * 64,
            evidence_root=tmp_path / "evidence",
            activation_authority_path=authority,
            systemd=systemd,
            identities=object(),
            require_root=False,
            enforce_production_address=False,
            enforce_package_metadata=False,
        )
    assert rail.PROJECTOR_UNIT not in systemd.starts
    systemd.core_active = False
    authority.write_text("{}", encoding="ascii")
    authority.chmod(0o400)
    with pytest.raises(
        rail.ProductionAliasProjectionCutoverError,
        match="rollback_after_terminal_forbidden",
    ):
        rail.rollback(
            cutover_plan_sha256=PLAN,
            package_root=tmp_path / "package",
            expected_revision=REVISION,
            expected_package_sha256=PACKAGE,
            expected_preflight_receipt_sha256=pre["receipt_sha256"],
            expected_apply_receipt_sha256=applied["receipt_sha256"],
            evidence_root=tmp_path / "evidence",
            activation_authority_path=authority,
            systemd=systemd,
            require_root=False,
            enforce_production_address=False,
            enforce_package_metadata=False,
        )


class WrongIdentity:
    def principal(self, name: str) -> rail.Principal:
        return rail.Principal(
            name=name,
            uid=9999,
            gid=9999,
            home="/nonexistent",
            shell="/usr/sbin/nologin",
            gids=(9999,),
        )

    def group_gid(self, _name: str) -> int:
        return 9999


def test_wrong_runtime_identity_fails_before_dependency_use(tmp_path):
    context = _context(tmp_path)
    context.manifest["identities"]["writer"]["uid"] = 1234
    with pytest.raises(
        rail.ProductionAliasProjectionCutoverError,
        match="host_identity_drifted",
    ):
        rail.validate_host_dependencies(context, identities=WrongIdentity())


def test_rollback_refuses_to_delete_drifted_installed_unit(tmp_path, monkeypatch):
    systemd_root = tmp_path / "systemd"
    systemd_root.mkdir()
    monkeypatch.setattr(rail, "SYSTEMD_ROOT", systemd_root)
    context = _context(tmp_path)
    _wire(monkeypatch, context)
    systemd = FakeSystemd(systemd_root)
    authority = tmp_path / "authority.json"
    pre = _preflight(tmp_path, monkeypatch, context, systemd, authority)
    applied = _apply(tmp_path, context, systemd, authority, pre)
    drifted = systemd_root / rail.PROJECTOR_UNIT
    drifted.write_bytes(b"administrator replacement\n")

    with pytest.raises(
        rail.ProductionAliasProjectionCutoverError,
        match="rollback_unit_state_drifted",
    ):
        rail.rollback(
            cutover_plan_sha256=PLAN,
            package_root=tmp_path / "package",
            expected_revision=REVISION,
            expected_package_sha256=PACKAGE,
            expected_preflight_receipt_sha256=pre["receipt_sha256"],
            expected_apply_receipt_sha256=applied["receipt_sha256"],
            evidence_root=tmp_path / "evidence",
            activation_authority_path=authority,
            systemd=systemd,
            require_root=False,
            enforce_production_address=False,
            enforce_package_metadata=False,
        )

    assert drifted.read_bytes() == b"administrator replacement\n"

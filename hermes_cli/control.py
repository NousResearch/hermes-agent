from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
import argparse
from pathlib import Path
from typing import Any

from hermes_cli import control_db as cp
from hermes_cli.control_contracts import ContractError, validate_statute_dispatch_v1
from hermes_cli.control_runtime import (
    CONTROL_TO_RUNTIME_PROFILE,
    help_parse_status,
    require_live_flag_for_mutation,
    resolve_control_target,
    runtime_profile_for_control_profile,
    runtime_profile_presence,
    validate_pm_runtime_mapping,
    worker_spawnability_status,
)
from hermes_cli.control_worker import DEFAULT_AGENT_WORKER_HARD_TIMEOUT_S, DEFAULT_AGENT_WORKER_SOFT_TIMEOUT_S


MUTATING = {
    "migrate",
    "mode",
    "bootstrap",
    "profile",
    "heartbeat",
    "admin:lease",
    "bootstrap-statutepm",
    "route:add",
    "route:remove",
    "dispatch:create",
    "dispatch:claim",
    "dispatch:advance",
    "status:emit",
    "blocker:open",
    "blocker:resolve",
    "supervision:start",
    "supervision:finish",
    "runtime:map",
    "watchdog:run",
    "message:create",
    "message:ack",
    "message:resolve",
    "message:supersede",
    "message:cancel",
    "pm:run",
    "worker:run",
    "live-smoke",
    "wave:dispatch-statutepm",
}


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


class _StoreExplicitFloat(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        self.explicit_dest = kwargs.pop("explicit_dest")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, float(values))
        setattr(namespace, self.explicit_dest, True)


def _target(args, *, temp_default: bool = False):
    default_root = Path(tempfile.mkdtemp(prefix="hermes-cp-")) if temp_default else None
    return resolve_control_target(root=getattr(args, "root", None), live=bool(getattr(args, "live", False)), default_root=default_root)


def _connect(target):
    return cp.connect(root=target.root)


def _guard_mutation(args, action: str, target) -> None:
    if action in MUTATING:
        require_live_flag_for_mutation(target, live_flag=bool(getattr(args, "live", False)))


def _json_arg(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def _transition_message_status_with_admin_lease_retry(conn, args, target, *, status: str) -> dict[str, Any]:
    metadata = _json_arg(args.metadata_json, {})
    try:
        result = cp.transition_message_status(
            conn,
            args.message_id,
            status=status,
            actor_instance_id=args.actor_instance_id,
            actor_profile=args.actor_profile,
            actor_type=args.actor_type,
            reason=args.reason,
            metadata=metadata,
        )
        result["admin_lease_renewed"] = False
        return result
    except PermissionError as exc:
        if not (
            target.live
            and args.actor_type == "admin"
            and args.actor_profile
            and args.actor_instance_id
            and "admin actor instance is not live/authorized" in str(exc)
        ):
            raise
        try:
            lease_info = cp.renew_admin_bootstrap_instance_lease(
                conn,
                profile_id=args.actor_profile,
                instance_id=args.actor_instance_id,
            )
        except (PermissionError, ValueError) as renew_exc:
            raise SystemExit(f"ACTION_REQUIRED_NO_LIVE_ADMIN: {renew_exc}") from renew_exc
        metadata = {
            **metadata,
            "admin_lease_renewed": True,
            "admin_lease_renewal": {
                "instance_id": lease_info["instance_id"],
                "previous_lease_expires_at_ms": lease_info.get("previous_lease_expires_at_ms"),
                "lease_expires_at_ms": lease_info["lease_expires_at_ms"],
                "source": "message_transition_retry",
            },
        }
        try:
            result = cp.transition_message_status(
                conn,
                args.message_id,
                status=status,
                actor_instance_id=args.actor_instance_id,
                actor_profile=args.actor_profile,
                actor_type=args.actor_type,
                reason=args.reason,
                metadata=metadata,
            )
        except PermissionError as retry_exc:
            raise SystemExit(f"ACTION_REQUIRED_NO_LIVE_ADMIN: {retry_exc}") from retry_exc
        result["admin_lease_renewed"] = True
        return result


def _dispatch_row(row) -> dict[str, Any]:
    data = dict(row)
    data["payload"] = json.loads(data.pop("payload_json"))
    return data


def cmd_control(args) -> None:
    command = getattr(args, "control_command", None) or "doctor"
    action = command
    sub = getattr(args, f"{command.replace('-', '_')}_command", None)
    if sub:
        action = f"{command}:{sub}"
    temp_default = action in {"smoke-test", "readiness"} and not getattr(args, "live_check", False)
    target = _target(args, temp_default=temp_default)
    if command == "smoke-test" and target.live:
        raise SystemExit("smoke-test does not support the live control DB")
    _guard_mutation(args, action, target)
    conn = _connect(target)
    try:
        if command == "migrate":
            cp.init_schema(conn)
            print(f"Control DB ready: {target.db_path}")
            return
        if command == "doctor":
            issues = cp.doctor(conn, root=target.root)
            if not issues:
                print(f"ok: {target.db_path}")
                return
            for issue in issues:
                print(f"{issue.level}: {issue.code}: {issue.detail}")
            if any(i.level == "error" for i in issues):
                raise SystemExit(2)
            return
        if command == "mode":
            mode = getattr(args, "mode", None)
            if mode:
                actor_profile = getattr(args, "actor_profile", None)
                actor_instance_id = getattr(args, "actor_instance_id", None)
                actor_type = "admin" if target.live or actor_profile or actor_instance_id else "bootstrap"
                cp.set_authority_mode(conn, mode, actor_type=actor_type, actor_profile=actor_profile, actor_instance_id=actor_instance_id)
            print(cp.get_authority_mode(conn))
            return
        if command == "bootstrap":
            profile = getattr(args, "admin_profile", None) or "default"
            cp.bootstrap_default_policies(conn, admin_profile=profile)
            _print_json({"db_path": str(target.db_path), "admin_profile": profile})
            return
        if command == "bootstrap-statutepm":
            result = cp.bootstrap_statutepm_policies(
                conn,
                admin_profile=args.admin_profile,
                pm_profile=args.pm_profile_id,
                worker_profile=args.worker_profile,
                seed_instances=bool(args.seed_instances),
            )
            result.update({"db_path": str(target.db_path), "live": target.live})
            _print_json(result)
            return
        if command == "profile":
            actor_type = "admin" if target.live or args.actor_instance_id else ("bootstrap" if args.role != "worker" else "worker")
            cp.register_profile(conn, args.profile_id, role=args.role, display_name=args.display_name, actor_type=actor_type, actor_profile=args.actor_profile, actor_instance_id=args.actor_instance_id)
            _print_json({"db_path": str(target.db_path), "profile_id": args.profile_id, "role": args.role})
            return
        if command == "profiles":
            rows = conn.execute("SELECT * FROM cp_profiles ORDER BY profile_id").fetchall()
            _print_json([dict(r) for r in rows])
            return
        if command == "heartbeat":
            if args.instance_id and cp.heartbeat_instance(conn, args.instance_id):
                inst = args.instance_id
            else:
                inst = cp.register_instance(conn, args.profile_id, instance_id=args.instance_id, actor_type="worker")
            _print_json({"db_path": str(target.db_path), "instance_id": inst})
            return
        if command == "admin" and sub == "lease":
            try:
                result = cp.renew_admin_bootstrap_instance_lease(
                    conn,
                    profile_id=args.profile,
                    instance_id=args.instance_id,
                    lease_ms=args.lease_ms,
                )
            except (PermissionError, ValueError) as exc:
                raise SystemExit(f"ACTION_REQUIRED_NO_LIVE_ADMIN: {exc}") from exc
            result["db_path"] = str(target.db_path)
            _print_json(result)
            return
        if command == "instances":
            where = "" if args.include_stale else "WHERE status='online' AND (lease_expires_at_ms IS NULL OR lease_expires_at_ms > ?)"
            params = () if args.include_stale else (cp.now_ms(),)
            rows = conn.execute(f"SELECT * FROM cp_profile_instances {where} ORDER BY profile_id, instance_id", params).fetchall()
            _print_json([dict(r) for r in rows])
            return
        if command in {"routes", "route"}:
            if command == "routes" or sub is None:
                rows = conn.execute("SELECT * FROM cp_route_policies ORDER BY priority DESC,effect ASC").fetchall()
                _print_json([dict(r) for r in rows])
                return
            if sub == "add":
                actor_type = "admin" if target.live or args.actor_instance_id or cp.get_authority_mode(conn) == "control_db" else "bootstrap"
                policy_id = cp.add_route_policy(conn, effect=args.effect, sender_profile=args.sender, receiver_profile=args.receiver, kind=args.kind, capability=args.capability, priority=args.priority, created_by=args.actor_profile, created_by_type=actor_type, created_by_instance_id=args.actor_instance_id)
                _print_json({"db_path": str(target.db_path), "policy_id": policy_id})
                return
            if sub == "remove":
                actor_type = "admin" if target.live or args.actor_instance_id or cp.get_authority_mode(conn) == "control_db" else "bootstrap"
                ok = cp.remove_route_policy(conn, args.policy_id, actor_profile=args.actor_profile, actor_instance_id=args.actor_instance_id, actor_type=actor_type)
                _print_json({"db_path": str(target.db_path), "removed": ok})
                return
            if sub == "check":
                allowed = cp.route_allowed(conn, sender_profile=args.sender, receiver_profile=args.receiver, kind=args.kind, capability=args.capability)
                print("allow" if allowed else "deny")
                if args.strict and not allowed:
                    raise SystemExit(1)
                return
        if command == "dispatch":
            if sub == "create":
                payload = _json_arg(args.payload_json, {})
                did = cp.create_dispatch_from_instance(
                    conn,
                    sender_instance_id=args.sender_instance_id,
                    receiver_profile=args.receiver,
                    payload=payload,
                    idempotency_key=args.idempotency_key,
                    parent_dispatch_id=args.parent_dispatch_id,
                    dispatch_schema=args.dispatch_schema,
                    max_wall_time_ms=args.max_wall_time_ms,
                )
                _print_json({"db_path": str(target.db_path), "dispatch_id": did})
                return
            if sub == "list":
                clauses = []
                params: list[Any] = []
                if args.receiver:
                    clauses.append("receiver_profile=?")
                    params.append(args.receiver)
                if args.status:
                    clauses.append("status=?")
                    params.append(args.status)
                where = "WHERE " + " AND ".join(clauses) if clauses else ""
                rows = conn.execute(f"SELECT * FROM cp_dispatches {where} ORDER BY created_at_ms", params).fetchall()
                _print_json([_dispatch_row(r) for r in rows])
                return
            if sub == "show":
                row = conn.execute("SELECT * FROM cp_dispatches WHERE dispatch_id=?", (args.dispatch_id,)).fetchone()
                if not row:
                    raise SystemExit("dispatch not found")
                data = _dispatch_row(row)
                data["latest_result"] = cp.get_latest_dispatch_result(conn, args.dispatch_id)
                _print_json(data)
                return
            if sub == "claim":
                ok, epoch = cp.claim_dispatch_by_id(conn, dispatch_id=args.dispatch_id, instance_id=args.instance_id, lease_ms=args.lease_ms)
                if not ok or epoch is None:
                    raise SystemExit(1)
                row = conn.execute("SELECT * FROM cp_dispatches WHERE dispatch_id=?", (args.dispatch_id,)).fetchone()
                _print_json({"db_path": str(target.db_path), "dispatch_id": args.dispatch_id, "lease_epoch": epoch, "receiver_profile": row["receiver_profile"], "payload": json.loads(row["payload_json"])})
                return
            if sub == "advance":
                ok = cp.advance_dispatch(conn, args.dispatch_id, instance_id=args.instance_id, lease_epoch=args.lease_epoch, status=args.status, last_error=args.last_error)
                _print_json({"db_path": str(target.db_path), "advanced": ok})
                if not ok:
                    raise SystemExit(1)
                return
            if sub == "supersede":
                ok = cp.supersede_dispatch(
                    conn,
                    args.dispatch_id,
                    actor_instance_id=args.actor_instance_id,
                    actor_profile=args.actor_profile,
                    reason=args.reason,
                    metadata=_json_arg(args.metadata_json, {}),
                )
                _print_json({"db_path": str(target.db_path), "dispatch_id": args.dispatch_id, "superseded": ok})
                if not ok:
                    raise SystemExit(1)
                return
        if command == "status":
            if sub == "emit":
                event_id = cp.emit_status(
                    conn,
                    instance_id=args.instance_id,
                    dispatch_id=args.dispatch_id,
                    status=args.status,
                    summary=args.summary,
                    details=_json_arg(args.details_json, {}),
                )
                _print_json({"db_path": str(target.db_path), "event_id": event_id})
                return
            if sub == "list":
                _print_json(cp.list_status_events(conn, dispatch_id=args.dispatch_id, profile_id=args.profile_id, limit=args.limit))
                return
        if command == "blocker":
            if sub == "open":
                blocker_id = cp.open_blocker(
                    conn,
                    dispatch_id=args.dispatch_id,
                    instance_id=args.instance_id,
                    severity=args.severity,
                    kind=args.kind,
                    summary=args.summary,
                    details=_json_arg(args.details_json, {}),
                    response_profile=args.response_profile,
                )
                _print_json({"db_path": str(target.db_path), "blocker_id": blocker_id})
                return
            if sub == "resolve":
                ok = cp.resolve_blocker(conn, args.blocker_id, resolver_instance_id=args.resolver_instance_id, resolution=_json_arg(args.resolution_json, {}))
                _print_json({"db_path": str(target.db_path), "resolved": ok})
                if not ok:
                    raise SystemExit(1)
                return
            if sub == "list":
                _print_json(cp.list_blockers(conn, dispatch_id=args.dispatch_id, status=args.status, response_profile=args.response_profile, limit=args.limit))
                return
        if command == "supervision":
            if sub == "start":
                run_id = cp.start_supervision_run(conn, actor_instance_id=args.actor_instance_id, scope=_json_arg(args.scope_json, {}))
                _print_json({"db_path": str(target.db_path), "run_id": run_id})
                return
            if sub == "finish":
                ok = cp.finish_supervision_run(conn, args.run_id, status=args.status, findings=_json_arg(args.findings_json, []), actions=_json_arg(args.actions_json, []))
                _print_json({"db_path": str(target.db_path), "finished": ok})
                if not ok:
                    raise SystemExit(1)
                return
            if sub == "list":
                _print_json(cp.list_supervision_runs(conn, status=args.status, limit=args.limit))
                return
        if command == "runtime":
            if sub == "map":
                cp.set_runtime_mapping(conn, control_profile_id=args.control_profile_id, runtime_profile=args.runtime_profile, role=args.role, enabled=not args.disabled, actor_instance_id=args.actor_instance_id)
                _print_json({"db_path": str(target.db_path), "control_profile_id": args.control_profile_id, "runtime_profile": args.runtime_profile})
                return
            if sub == "show":
                _print_json(cp.get_runtime_mapping(conn, args.control_profile_id) or {})
                return
        if command == "watchdog" and sub == "run":
            from hermes_cli.control_watchdog import supervise_once

            _print_json(supervise_once(root=target.root, actor_instance_id=args.actor_instance_id, dry_run=args.dry_run, stale_ms=args.stale_ms))
            return
        if command == "message":
            if sub == "create":
                mid = cp.create_message_from_instance(conn, sender_instance_id=args.sender_instance_id, receiver_profile=args.receiver, kind=args.kind, body=args.body, capability=args.capability, metadata=_json_arg(args.metadata_json, {}))
                _print_json({"db_path": str(target.db_path), "message_id": mid})
                return
            if sub in {"ack", "resolve", "supersede", "cancel"}:
                status = {"ack": "acknowledged", "resolve": "resolved", "supersede": "superseded", "cancel": "cancelled"}[sub]
                result = _transition_message_status_with_admin_lease_retry(conn, args, target, status=status)
                result["db_path"] = str(target.db_path)
                _print_json(result)
                return
            if sub == "list":
                clauses = []
                params: list[Any] = []
                if args.receiver:
                    clauses.append("receiver_profile=?")
                    params.append(args.receiver)
                if args.status:
                    clauses.append("status=?")
                    params.append(args.status)
                where = "WHERE " + " AND ".join(clauses) if clauses else ""
                rows = conn.execute(f"SELECT * FROM cp_messages {where} ORDER BY created_at_ms", params).fetchall()
                _print_json([dict(r) for r in rows])
                return
        if command == "artifacts":
            _print_json(cp.list_artifacts(conn, args.dispatch_id))
            return
        if command == "worker" and sub == "run":
            from hermes_cli.control_worker import run_agent_dispatch, run_deterministic_dispatch

            if args.handler == "agent":
                hard_timeout_s = args.hard_timeout_s
                if args.timeout_s is not None:
                    if getattr(args, "hard_timeout_s_explicit", False) and args.timeout_s != args.hard_timeout_s:
                        raise SystemExit("--timeout-s is a deprecated alias for --hard-timeout-s; do not pass conflicting values")
                    hard_timeout_s = args.timeout_s
                result = run_agent_dispatch(root=target.root, profile_id=args.profile_id, instance_id=args.instance_id, dispatch_id=args.dispatch_id, soft_timeout_s=args.soft_timeout_s, hard_timeout_s=hard_timeout_s)
            else:
                result = run_deterministic_dispatch(root=target.root, profile_id=args.profile_id, instance_id=args.instance_id, dispatch_id=args.dispatch_id)
            result["db_path"] = str(target.db_path)
            _print_json(result)
            if result.get("result", {}).get("status") == "failed":
                raise SystemExit(1)
            return
        if command == "pm" and sub == "run":
            if args.pm_profile_id != "statutepm":
                raise SystemExit("only pm run --pm-profile-id statutepm is supported")
            runtime_profile = validate_pm_runtime_mapping(args.pm_profile_id, args.pm_runtime_profile)
            if args.loop:
                try:
                    for event in _run_pm_events(args, target, runtime_profile=runtime_profile):
                        print(json.dumps(event, sort_keys=True))
                        sys.stdout.flush()
                except KeyboardInterrupt:
                    print(json.dumps({"status": "interrupted"}, sort_keys=True))
            else:
                _print_json(next(_run_pm_events(args, target, runtime_profile=runtime_profile)))
            return
        if command == "smoke-test":
            if args.subject != "statutepm":
                raise SystemExit("only smoke-test statutepm is supported")
            _print_json(_run_statutepm_smoke(target.root, target.db_path))
            return
        if command == "live-smoke":
            if args.subject != "statutepm":
                raise SystemExit("only live-smoke statutepm is supported")
            if not args.live:
                raise SystemExit("live-smoke requires --live")
            if not args.deterministic:
                raise SystemExit("live-smoke currently requires --deterministic")
            result = _run_statutepm_live_smoke(
                target,
                smoke_tag=args.smoke_tag,
                idempotency_key=args.idempotency_key,
                deterministic=bool(args.deterministic),
            )
            _print_json(result)
            if not result.get("ok"):
                raise SystemExit(1)
            return
        if command == "wave" and sub == "dispatch-statutepm":
            result = _dispatch_statutepm_wave(args, target)
            _print_json(result)
            if result.get("status") in {"invalid_payload", "failed", "action_required", "blocked"}:
                raise SystemExit(1)
            return
        if command == "readiness":
            if args.subject != "statutepm":
                raise SystemExit("only readiness statutepm is supported")
            readiness = _readiness(args, target)
            _print_json(readiness)
            if readiness.get("runtime_profile_statute_worker") != "present":
                raise SystemExit(1)
            if getattr(args, "live_check", False):
                if not readiness.get("live_ready"):
                    raise SystemExit(1)
            elif not readiness.get("implementation_ready"):
                raise SystemExit(1)
            return
        raise SystemExit(f"unknown control command: {command}")
    finally:
        conn.close()


def _sample_payload(repo_root: Path) -> dict[str, Any]:
    return {
        "schema": "statute_dispatch_v1",
        "silo": "statute",
        "repo_root": str(repo_root),
        "allowed_paths": [str(repo_root)],
        "task_type": "generic",
        "task_permissions": ["read", "test"],
        "parent_dispatch_id": None,
        "instructions": "deterministic smoke",
        "constraints": {"no_live_db_mutation": True, "no_push": True},
    }


def _run_pm_events(args, target, *, runtime_profile: str):
    from hermes_cli.statutepm_flow import StatutePMFlow

    flow = StatutePMFlow(
        root=target.root,
        pm_instance_id=args.pm_instance_id,
        pm_profile=args.pm_profile_id,
        worker_profile=args.worker_profile,
        poll_interval_s=args.poll_interval_s,
        child_soft_timeout_s=getattr(args, "child_soft_timeout_s", DEFAULT_AGENT_WORKER_SOFT_TIMEOUT_S),
        child_hard_timeout_s=getattr(args, "child_hard_timeout_s", getattr(args, "child_timeout_s", DEFAULT_AGENT_WORKER_HARD_TIMEOUT_S)),
    )
    while True:
        conn = cp.connect(root=target.root)
        try:
            reaped = cp.reap_expired_dispatches(conn)
        finally:
            conn.close()
        outcome = flow.run_once()
        event = outcome or {"status": "idle"}
        event.update(
            {
                "db_path": str(target.db_path),
                "pm_profile_id": args.pm_profile_id,
                "pm_runtime_profile": runtime_profile,
                "pm_instance_id": args.pm_instance_id,
                "worker_profile": args.worker_profile,
                "reaped": reaped,
            }
        )
        yield event
        if args.once:
            return
        time.sleep(args.poll_interval_s)


def _safe_smoke_tag(tag: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in tag.strip())
    return cleaned[:120] or "statutepm-smoke"


def _wave_instance_suffix(idempotency_key: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in idempotency_key.strip())[:80]
    return safe or hashlib.sha256(idempotency_key.encode()).hexdigest()[:12]


def _reject_bootstrap_instance(instance_id: str, *, field: str) -> None:
    if instance_id.endswith(":bootstrap"):
        raise SystemExit(f"{field} must not use a seeded bootstrap instance: {instance_id}")


def _dispatch_statutepm_wave(args, target, *, spawn_child=None) -> dict[str, Any]:
    from hermes_cli.statutepm_flow import StatutePMFlow

    runtime_profile = validate_pm_runtime_mapping(args.pm_profile_id, args.pm_runtime_profile)
    payload = _json_arg(args.payload_json, {})
    suffix = _wave_instance_suffix(args.idempotency_key)
    supervisor_instance_id = args.supervisor_instance_id or f"{args.admin_profile}:wave:{suffix}"
    pm_instance_id = args.pm_instance_id or f"{args.pm_profile_id}:wave:{suffix}"
    _reject_bootstrap_instance(supervisor_instance_id, field="--supervisor-instance-id")
    _reject_bootstrap_instance(pm_instance_id, field="--pm-instance-id")

    supervisor_registered = False
    supervisor_offline = False
    outcome: dict[str, Any] | None = None
    dispatch_id: str | None = None
    parent_status = "not_created"
    result: dict[str, Any]
    try:
        conn = cp.connect(root=target.root)
        try:
            cp.bootstrap_statutepm_policies(
                conn,
                admin_profile=args.admin_profile,
                pm_profile=args.pm_profile_id,
                worker_profile=args.worker_profile,
                seed_instances=False,
            )
            cp.register_instance(
                conn,
                args.admin_profile,
                instance_id=supervisor_instance_id,
                lease_ms=args.supervisor_lease_ms,
                actor_type="bootstrap",
                metadata={"wave_lifecycle_owner": True, "seeded_by_bootstrap": False, "idempotency_key": args.idempotency_key},
            )
            supervisor_registered = True
            try:
                validate_statute_dispatch_v1(payload)
            except ContractError as exc:
                result = {
                    "db_path": str(target.db_path),
                    "live": target.live,
                    "status": "invalid_payload",
                    "error": str(exc),
                    "parent_dispatch_id": None,
                    "parent_status": parent_status,
                    "supervisor_instance_id": supervisor_instance_id,
                    "pm_instance_id": pm_instance_id,
                    "pm_runtime_profile": runtime_profile,
                    "supervision": None,
                }
            else:
                dispatch_id = cp.create_dispatch_from_instance(
                    conn,
                    sender_instance_id=supervisor_instance_id,
                    receiver_profile=args.pm_profile_id,
                    payload=payload,
                    idempotency_key=args.idempotency_key,
                    dispatch_schema=payload.get("schema"),
                )
                row = conn.execute("SELECT status FROM cp_dispatches WHERE dispatch_id=?", (dispatch_id,)).fetchone()
                parent_status = row["status"] if row else "missing"
                result = {}
        finally:
            conn.close()

        if dispatch_id and args.supervise:
            flow = StatutePMFlow(
                root=target.root,
                pm_instance_id=pm_instance_id,
                admin_profile=args.admin_profile,
                pm_profile=args.pm_profile_id,
                worker_profile=args.worker_profile,
                spawn_child=spawn_child,
                poll_interval_s=args.poll_interval_s,
                child_soft_timeout_s=getattr(args, "child_soft_timeout_s", DEFAULT_AGENT_WORKER_SOFT_TIMEOUT_S),
                child_hard_timeout_s=getattr(args, "child_hard_timeout_s", getattr(args, "child_timeout_s", DEFAULT_AGENT_WORKER_HARD_TIMEOUT_S)),
            )
            outcome = flow.run_dispatch(dispatch_id)
            conn = cp.connect(root=target.root)
            try:
                row = conn.execute("SELECT status FROM cp_dispatches WHERE dispatch_id=?", (dispatch_id,)).fetchone()
                parent_status = row["status"] if row else "missing"
            finally:
                conn.close()

        if not result:
            status = "created"
            if args.supervise:
                status = outcome.get("status", "supervised") if outcome else "supervised"
            result = {
                "db_path": str(target.db_path),
                "live": target.live,
                "status": status,
                "parent_dispatch_id": dispatch_id,
                "parent_status": parent_status,
                "supervisor_instance_id": supervisor_instance_id,
                "pm_instance_id": pm_instance_id,
                "pm_runtime_profile": runtime_profile,
                "supervision": outcome,
            }
    finally:
        if supervisor_registered:
            conn = cp.connect(root=target.root)
            try:
                supervisor_offline = cp.mark_instance_offline(conn, supervisor_instance_id)
            finally:
                conn.close()
    result["supervisor_offline"] = supervisor_offline
    return result


def _run_statutepm_live_smoke(
    target,
    *,
    smoke_tag: str,
    idempotency_key: str,
    deterministic: bool,
    spawn_child=None,
) -> dict[str, Any]:
    from hermes_cli.control_worker import run_deterministic_dispatch
    from hermes_cli.statutepm_flow import StatutePMFlow

    if not deterministic:
        raise ValueError("only deterministic live-smoke is implemented")
    child_spawner = spawn_child
    if child_spawner is None:
        def _deterministic_spawn(child_id: str, payload: dict[str, Any], child_root: Path | None, parent_id: str) -> int:
            run_deterministic_dispatch(root=child_root, profile_id="statute-worker", instance_id="statute-worker:live-smoke", dispatch_id=child_id)
            return os.getpid()
        child_spawner = _deterministic_spawn
    control_root = target.root or cp.control_db_path().parent.parent
    scratch = control_root / "control-plane" / "smoke" / _safe_smoke_tag(smoke_tag)
    repo = scratch / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    payload = _sample_payload(repo)
    payload["instructions"] = f"deterministic live-smoke: {smoke_tag}"
    conn = cp.connect(root=target.root)
    try:
        parent = cp.create_dispatch_from_instance(
            conn,
            sender_instance_id="default:bootstrap",
            receiver_profile="statutepm",
            payload=payload,
            idempotency_key=idempotency_key,
        )
        parent_status = conn.execute("SELECT status FROM cp_dispatches WHERE dispatch_id=?", (parent,)).fetchone()["status"]
    finally:
        conn.close()

    outcome: dict[str, Any]
    if parent_status == "completed":
        outcome = {"status": "already_completed", "parent_dispatch_id": parent}
    else:
        flow = StatutePMFlow(
            root=target.root,
            pm_instance_id="statutepm:bootstrap",
            pm_profile="statutepm",
            worker_profile="statute-worker",
            spawn_child=child_spawner,
            poll_interval_s=0.25,
            child_timeout_s=60,
        )
        outcome = flow.run_once() or {"status": "idle", "parent_dispatch_id": parent}

    verification = _verify_statutepm_smoke_rows(target.root, parent, scratch)
    return {
        "db_path": str(target.db_path),
        "live": target.live,
        "ok": verification["ok"],
        "smoke_tag": smoke_tag,
        "idempotency_key": idempotency_key,
        "scratch_root": str(scratch),
        "parent_dispatch_id": parent,
        "outcome": outcome,
        "verification": verification,
    }


def _verify_statutepm_smoke_rows(root: Path | None, parent_dispatch_id: str, scratch: Path) -> dict[str, Any]:
    conn = cp.connect(root=root)
    try:
        parent = conn.execute("SELECT * FROM cp_dispatches WHERE dispatch_id=?", (parent_dispatch_id,)).fetchone()
        child_rows = conn.execute("SELECT * FROM cp_dispatches WHERE receiver_profile='statute-worker' ORDER BY created_at_ms").fetchall()
        children = []
        for row in child_rows:
            payload = json.loads(row["payload_json"])
            if payload.get("parent_dispatch_id") == parent_dispatch_id:
                children.append(row)
        child = children[-1] if children else None
        child_result = cp.get_latest_dispatch_result(conn, child["dispatch_id"]) if child else None
        artifacts = cp.list_artifacts(conn, child["dispatch_id"]) if child else []
        messages = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM cp_messages WHERE sender_profile='statutepm' AND receiver_profile='default' AND kind='status' ORDER BY created_at_ms"
            ).fetchall()
        ]
    finally:
        conn.close()
    artifact_paths = [Path(a["path"]) for a in artifacts]
    artifacts_contained = all(_path_within(path, scratch) for path in artifact_paths)
    artifact_files_exist = bool(artifact_paths) and all(path.exists() and path.is_file() for path in artifact_paths)
    ok = (
        parent is not None
        and parent["status"] == "completed"
        and child is not None
        and child["status"] == "completed"
        and child_result is not None
        and bool(artifacts)
        and artifacts_contained
        and artifact_files_exist
        and bool(messages)
    )
    return {
        "ok": ok,
        "parent_status": parent["status"] if parent else "missing",
        "child_dispatch_id": child["dispatch_id"] if child else None,
        "child_status": child["status"] if child else "missing",
        "child_result_exists": child_result is not None,
        "artifact_count": len(artifacts),
        "artifacts_contained": artifacts_contained,
        "artifact_files_exist": artifact_files_exist,
        "message_count": len(messages),
    }


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _run_statutepm_smoke(root: Path | None, db_path: Path) -> dict[str, Any]:
    from hermes_cli.control_worker import run_deterministic_dispatch
    from hermes_cli.statutepm_flow import StatutePMFlow

    root = root or Path(tempfile.mkdtemp(prefix="hermes-cp-smoke-"))
    repo = root / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    conn = cp.connect(root=root)
    try:
        boot = cp.bootstrap_statutepm_policies(conn, seed_instances=True)
        default_inst = boot["instances"]["default"]
        pm_inst = boot["instances"]["statutepm"]
        parent = cp.create_dispatch_from_instance(conn, sender_instance_id=default_inst, receiver_profile="statutepm", payload=_sample_payload(repo))
    finally:
        conn.close()

    spawned: list[str] = []

    def fake_spawn(child_id: str, payload: dict[str, Any], child_root: Path | None, parent_id: str) -> int:
        spawned.append(child_id)
        run_deterministic_dispatch(root=child_root, profile_id="statute-worker", instance_id="statute-worker:smoke", dispatch_id=child_id)
        return 12345

    flow = StatutePMFlow(root=root, pm_instance_id=pm_inst, spawn_child=fake_spawn, poll_interval_s=0, child_timeout_s=5)
    outcome = flow.run_once()
    conn = cp.connect(root=root)
    try:
        denied_worker_default = cp.route_allowed(conn, sender_profile="statute-worker", receiver_profile="default", kind="dispatch", capability="dispatch")
        denied_pm_other = cp.route_allowed(conn, sender_profile="statutepm", receiver_profile="other-worker", kind="dispatch", capability="dispatch")
        try:
            bad = _sample_payload(repo)
            bad["allowed_paths"] = [str(root.parent)]
            validate_statute_dispatch_v1(bad)
            path_escape_rejected = False
        except ContractError:
            path_escape_rejected = True
        parent_row = conn.execute("SELECT status FROM cp_dispatches WHERE dispatch_id=?", (parent,)).fetchone()
    finally:
        conn.close()
    ok = bool(outcome and outcome.get("status") == "completed" and spawned and not denied_worker_default and not denied_pm_other and path_escape_rejected and parent_row["status"] == "completed")
    return {"db_path": str(db_path), "root": str(root), "ok": ok, "parent_dispatch_id": parent, "spawned": spawned, "outcome": outcome}


def _readiness(args, target) -> dict[str, Any]:
    pm_profile = "statutepm"
    worker_profile = "statute-worker"
    pm_runtime = runtime_profile_for_control_profile(pm_profile)
    reasons: list[str] = []
    env = os.environ.copy()
    pm_help = help_parse_status([sys.executable, "-m", "hermes_cli.main", "control", "pm", "run", "--help"], env=env)
    worker_help = help_parse_status([sys.executable, "-m", "hermes_cli.main", "control", "worker", "run", "--help"], env=env)
    live_smoke_help = help_parse_status([sys.executable, "-m", "hermes_cli.main", "control", "live-smoke", "--help"], env=env)
    spawnability_detail = worker_spawnability_status(worker_profile, env=env)
    runtime_profiles = {
        "default": runtime_profile_presence("default"),
        pm_runtime: runtime_profile_presence(pm_runtime),
        worker_profile: runtime_profile_presence(worker_profile),
    }
    checks: dict[str, Any] = {
        "db_path": str(target.db_path),
        "live": target.live,
        "authority_mode": "shadow",
        "profile_mapping": dict(CONTROL_TO_RUNTIME_PROFILE),
        "runtime_profiles": runtime_profiles,
        "spawnability": {worker_profile: spawnability_detail["status"]},
        "spawnability_detail": {worker_profile: spawnability_detail},
        "command_parse": {
            "pm_run_help": pm_help["ok"],
            "worker_run_help": worker_help["ok"],
            "live_smoke_help": live_smoke_help["ok"],
        },
        "deterministic_operational_ready": False,
        "agent_worker_ready": False,
        "safe_to_cutover_control_db": False,
        "reasons": reasons,
    }
    smoke_root = Path(tempfile.mkdtemp(prefix="hermes-cp-readiness-")) if getattr(args, "live_check", False) else target.root
    smoke_db = cp.control_db_path(smoke_root).resolve() if smoke_root is not None else target.db_path
    smoke = _run_statutepm_smoke(smoke_root, smoke_db)
    checks["smoke"] = smoke
    checks["implementation_ready"] = bool(smoke["ok"] and pm_help["ok"] and worker_help["ok"] and live_smoke_help["ok"])
    if not checks["implementation_ready"]:
        reasons.append("implementation smoke/help checks failed")
    if runtime_profiles.get(pm_runtime) != "present":
        reasons.append(f"runtime profile missing: {pm_runtime}")
    if runtime_profiles.get(worker_profile) != "present":
        reasons.append(f"runtime profile missing: {worker_profile}")
    if spawnability_detail["status"] != "dry_run_ok":
        reasons.append(f"worker spawnability failed: {spawnability_detail.get('error') or spawnability_detail.get('stderr')}")
    if getattr(args, "live_check", False):
        conn = cp.connect()
        try:
            checks["authority_mode"] = cp.get_authority_mode(conn)
            checks["profiles"] = {r["profile_id"]: dict(r) for r in conn.execute("SELECT * FROM cp_profiles WHERE profile_id IN ('default','statutepm','statute-worker')").fetchall()}
            seeded_rows = conn.execute(
                "SELECT instance_id, profile_id, status, heartbeat_at_ms, lease_expires_at_ms FROM cp_profile_instances WHERE instance_id IN ('default:bootstrap','statutepm:bootstrap')"
            ).fetchall()
            now = cp.now_ms()
            checks["seeded_instances"] = {r["instance_id"]: dict(r) for r in seeded_rows}
            checks["seeded_instance_leases"] = {}
            checks["seeded_instances_live"] = {}
            for instance_id in ("default:bootstrap", "statutepm:bootstrap"):
                row = checks["seeded_instances"].get(instance_id)
                live = bool(row and row["status"] == "online" and (row["lease_expires_at_ms"] is None or int(row["lease_expires_at_ms"]) > now))
                checks["seeded_instances_live"][instance_id] = live
                checks["seeded_instance_leases"][instance_id] = {
                    "present": bool(row),
                    "status": row["status"] if row else "missing",
                    "heartbeat_age_ms": (now - int(row["heartbeat_at_ms"])) if row else None,
                    "lease_expires_at_ms": int(row["lease_expires_at_ms"]) if row and row["lease_expires_at_ms"] is not None else None,
                    "lease_expires_in_ms": (int(row["lease_expires_at_ms"]) - now) if row and row["lease_expires_at_ms"] is not None else None,
                    "live": live,
                }
            admin_rows = conn.execute(
                """
                SELECT i.instance_id, i.profile_id, i.status, i.heartbeat_at_ms, i.lease_expires_at_ms
                FROM cp_profile_instances i
                JOIN cp_profiles p ON p.profile_id=i.profile_id
                WHERE p.role='admin'
                  AND i.status='online'
                  AND i.lease_expires_at_ms IS NOT NULL
                  AND i.lease_expires_at_ms > ?
                ORDER BY i.profile_id, i.instance_id
                """,
                (now,),
            ).fetchall()
            checks["live_admin_instances"] = [dict(r) for r in admin_rows]
            checks["live_admin_available"] = bool(admin_rows)
            checks["routes"] = {
                "default_to_pm": cp.route_allowed(conn, sender_profile="default", receiver_profile="statutepm", kind="dispatch", capability="dispatch"),
                "pm_to_worker": cp.route_allowed(conn, sender_profile="statutepm", receiver_profile="statute-worker", kind="dispatch", capability="dispatch"),
                "worker_to_pm_status": cp.route_allowed(conn, sender_profile="statute-worker", receiver_profile="statutepm", kind="status", capability="message"),
            }
        finally:
            conn.close()
        if checks["authority_mode"] == "legacy":
            reasons.append("authority mode is legacy")
        missing_profiles = {"default", "statutepm", "statute-worker"} - set(checks["profiles"])
        for profile in sorted(missing_profiles):
            reasons.append(f"control profile missing: {profile}")
        for route_name, ok in checks["routes"].items():
            if not ok:
                reasons.append(f"route missing/denied: {route_name}")
        if not checks.get("live_admin_available"):
            reasons.append("no live admin control-plane instance")
        checks["bootstrap_lease_note"] = "seeded bootstrap instances are diagnostic only; finite wave dispatch owns operative supervisor/PM leases"
        checks["live_ready"] = (
            checks["authority_mode"] != "legacy"
            and checks.get("live_admin_available")
            and not missing_profiles
            and all(checks["routes"].values())
            and runtime_profiles.get(pm_runtime) == "present"
            and runtime_profiles.get(worker_profile) == "present"
            and spawnability_detail["status"] == "dry_run_ok"
        )
    else:
        checks["live_ready"] = False
    checks["deterministic_operational_ready"] = bool(checks.get("implementation_ready") and checks.get("live_ready"))
    checks["agent_worker_ready"] = bool(checks.get("implementation_ready") and worker_help["ok"] and spawnability_detail["status"] == "dry_run_ok")
    checks["safe_to_cutover_control_db"] = bool(checks["deterministic_operational_ready"] and checks["agent_worker_ready"] and checks.get("authority_mode") == "shadow")
    if checks.get("authority_mode") == "control_db":
        checks["cutover_state"] = "already_control_db"
    elif checks["safe_to_cutover_control_db"]:
        checks["cutover_state"] = "safe_to_cutover_deterministic"
    else:
        checks["cutover_state"] = "not_ready"
    checks["runtime_profile_statute_worker"] = runtime_profiles.get(worker_profile)
    checks["pytest_commands"] = [
        "python -m pytest -q tests/hermes_cli/test_control_db.py -o addopts=''",
        "python -m pytest -q tests/hermes_cli/test_control_contracts.py -o addopts=''",
        "python -m pytest -q tests/hermes_cli/test_control_cli.py -o addopts=''",
        "python -m pytest -q tests/hermes_cli/test_control_runtime.py -o addopts=''",
        "python -m pytest -q tests/hermes_cli/test_control_worker.py -o addopts=''",
        "python -m pytest -q tests/hermes_cli/test_statutepm_flow.py -o addopts=''",
        "python -m pytest -q tests/hermes_cli/test_control_spawn.py -o addopts=''",
        "python -m pytest -q tests/hermes_cli/test_control_smoke.py -o addopts=''",
        "python -m pytest -q tests/tools/test_control_plane_approvals.py -o addopts=''",
    ]
    checks["cutover_command"] = "hermes control mode control_db --live --actor-profile default --actor-instance-id <live-admin-instance>"
    return checks


def _add_target_flags(parser, *, live: bool = True) -> None:
    parser.add_argument("--root", default=None)
    if live:
        parser.add_argument("--live", action="store_true")


def register_subparser(subparsers) -> None:
    parser = subparsers.add_parser(
        "control",
        help="Manage the durable local Hermes control-plane DB",
        description="Durable cross-profile control plane for dispatches, approvals, messages, routing, audit, and Discord mirror state.",
    )
    _add_target_flags(parser)
    sp = parser.add_subparsers(dest="control_command")

    for name, help_text in (("migrate", "Initialize/migrate the control DB schema"), ("doctor", "Check control DB health"), ("profiles", "List profiles"), ("instances", "List profile instances"), ("routes", "List route policies")):
        p = sp.add_parser(name, help=help_text)
        _add_target_flags(p)
        if name == "instances":
            p.add_argument("--include-stale", action="store_true")

    mode = sp.add_parser("mode", help="Get or set control-plane authority mode")
    _add_target_flags(mode)
    mode.add_argument("mode", nargs="?", choices=["legacy", "shadow", "control_db"])
    mode.add_argument("--actor-profile", default=None)
    mode.add_argument("--actor-instance-id", default=None)

    bootstrap = sp.add_parser("bootstrap", help="Install default-deny baseline route policies")
    _add_target_flags(bootstrap)
    bootstrap.add_argument("--admin-profile", default="default")

    bsp = sp.add_parser("bootstrap-statutepm", help="Install statute PM control-plane profiles/routes")
    _add_target_flags(bsp)
    bsp.add_argument("--admin-profile", default="default")
    bsp.add_argument("--pm-profile-id", "--pm-profile", dest="pm_profile_id", default="statutepm")
    bsp.add_argument("--worker-profile", default="statute-worker")
    bsp.add_argument("--seed-instances", action="store_true")

    profile = sp.add_parser("profile", help="Register/update a profile")
    _add_target_flags(profile)
    profile.add_argument("profile_id")
    profile.add_argument("--role", default="worker", choices=["worker", "pm", "admin", "observer"])
    profile.add_argument("--display-name", default=None)
    profile.add_argument("--actor-profile", default="default")
    profile.add_argument("--actor-instance-id", default=None)

    heartbeat = sp.add_parser("heartbeat", help="Register/heartbeat a profile instance")
    _add_target_flags(heartbeat)
    heartbeat.add_argument("profile_id", nargs="?", default="default")
    heartbeat.add_argument("--instance-id", default=None)

    admin = sp.add_parser("admin", help="Admin control-plane maintenance")
    _add_target_flags(admin)
    asp = admin.add_subparsers(dest="admin_command")
    alease = asp.add_parser("lease", help="Renew an existing seeded admin bootstrap lease")
    _add_target_flags(alease)
    alease.add_argument("--profile", default="default")
    alease.add_argument("--instance-id", required=True)
    alease.add_argument("--lease-ms", type=int, default=120_000)

    route = sp.add_parser("route", help="Manage route policies")
    _add_target_flags(route)
    rsp = route.add_subparsers(dest="route_command")
    route_add = rsp.add_parser("add")
    _add_target_flags(route_add)
    route_add.add_argument("--effect", required=True, choices=["allow", "deny"])
    route_add.add_argument("--sender", required=True)
    route_add.add_argument("--receiver", required=True)
    route_add.add_argument("--kind", required=True)
    route_add.add_argument("--capability", required=True)
    route_add.add_argument("--priority", type=int, default=0)
    route_add.add_argument("--actor-profile", default="default")
    route_add.add_argument("--actor-instance-id", default=None)
    route_rm = rsp.add_parser("remove")
    _add_target_flags(route_rm)
    route_rm.add_argument("--policy-id", required=True)
    route_rm.add_argument("--actor-profile", default="default")
    route_rm.add_argument("--actor-instance-id", default=None)
    route_check = rsp.add_parser("check")
    _add_target_flags(route_check)
    route_check.add_argument("--sender", required=True)
    route_check.add_argument("--receiver", required=True)
    route_check.add_argument("--kind", required=True)
    route_check.add_argument("--capability", required=True)
    route_check.add_argument("--strict", action="store_true")

    dispatch = sp.add_parser("dispatch", help="Manage dispatches")
    _add_target_flags(dispatch)
    dsp = dispatch.add_subparsers(dest="dispatch_command")
    dcreate = dsp.add_parser("create")
    _add_target_flags(dcreate)
    dcreate.add_argument("--sender-instance-id", required=True)
    dcreate.add_argument("--receiver", required=True)
    dcreate.add_argument("--payload-json", required=True)
    dcreate.add_argument("--idempotency-key", default=None)
    dcreate.add_argument("--parent-dispatch-id", default=None)
    dcreate.add_argument("--dispatch-schema", default=None)
    dcreate.add_argument("--max-wall-time-ms", type=int, default=None)
    dlist = dsp.add_parser("list")
    _add_target_flags(dlist)
    dlist.add_argument("--receiver", default=None)
    dlist.add_argument("--status", default=None)
    dshow = dsp.add_parser("show")
    _add_target_flags(dshow)
    dshow.add_argument("dispatch_id")
    dclaim = dsp.add_parser("claim")
    _add_target_flags(dclaim)
    dclaim.add_argument("dispatch_id")
    dclaim.add_argument("--instance-id", required=True)
    dclaim.add_argument("--lease-ms", type=int, default=300_000)
    dadv = dsp.add_parser("advance")
    _add_target_flags(dadv)
    dadv.add_argument("dispatch_id")
    dadv.add_argument("--instance-id", required=True)
    dadv.add_argument("--lease-epoch", type=int, required=True)
    dadv.add_argument("--status", required=True, choices=["running", "completed", "failed"])
    dadv.add_argument("--last-error", default=None)
    dsup = dsp.add_parser("supersede")
    _add_target_flags(dsup)
    dsup.add_argument("dispatch_id")
    dsup.add_argument("--actor-instance-id", required=True)
    dsup.add_argument("--actor-profile", default="default")
    dsup.add_argument("--reason", default=None)
    dsup.add_argument("--metadata-json", default=None)

    status = sp.add_parser("status", help="Emit/list structured control-plane status events")
    _add_target_flags(status)
    stsp = status.add_subparsers(dest="status_command")
    semit = stsp.add_parser("emit")
    _add_target_flags(semit)
    semit.add_argument("--instance-id", required=True)
    semit.add_argument("--dispatch-id", default=None)
    semit.add_argument("--status", required=True)
    semit.add_argument("--summary", required=True)
    semit.add_argument("--details-json", default=None)
    slist = stsp.add_parser("list")
    _add_target_flags(slist)
    slist.add_argument("--dispatch-id", default=None)
    slist.add_argument("--profile-id", default=None)
    slist.add_argument("--limit", type=int, default=50)

    blocker = sp.add_parser("blocker", help="Open/list/resolve structured blockers")
    _add_target_flags(blocker)
    blsp = blocker.add_subparsers(dest="blocker_command")
    bopen = blsp.add_parser("open")
    _add_target_flags(bopen)
    bopen.add_argument("--dispatch-id", required=True)
    bopen.add_argument("--instance-id", required=True)
    bopen.add_argument("--severity", required=True, choices=["info", "warning", "blocked", "critical"])
    bopen.add_argument("--kind", required=True, choices=["approval_needed", "missing_context", "test_failure", "review_failure", "dependency", "auth", "policy", "runtime", "other"])
    bopen.add_argument("--summary", required=True)
    bopen.add_argument("--details-json", default=None)
    bopen.add_argument("--response-profile", default=None)
    bresolve = blsp.add_parser("resolve")
    _add_target_flags(bresolve)
    bresolve.add_argument("blocker_id")
    bresolve.add_argument("--resolver-instance-id", required=True)
    bresolve.add_argument("--resolution-json", default=None)
    blist = blsp.add_parser("list")
    _add_target_flags(blist)
    blist.add_argument("--dispatch-id", default=None)
    blist.add_argument("--status", default=None)
    blist.add_argument("--response-profile", default=None)
    blist.add_argument("--limit", type=int, default=50)

    supervision = sp.add_parser("supervision", help="Record/list watchdog supervision runs")
    _add_target_flags(supervision)
    supsp = supervision.add_subparsers(dest="supervision_command")
    sstart = supsp.add_parser("start")
    _add_target_flags(sstart)
    sstart.add_argument("--actor-instance-id", required=True)
    sstart.add_argument("--scope-json", default=None)
    sfinish = supsp.add_parser("finish")
    _add_target_flags(sfinish)
    sfinish.add_argument("run_id")
    sfinish.add_argument("--status", required=True, choices=["completed", "failed"])
    sfinish.add_argument("--findings-json", default=None)
    sfinish.add_argument("--actions-json", default=None)
    sulist = supsp.add_parser("list")
    _add_target_flags(sulist)
    sulist.add_argument("--status", default=None)
    sulist.add_argument("--limit", type=int, default=50)

    runtime = sp.add_parser("runtime", help="Manage control-profile to Hermes runtime-profile mappings")
    _add_target_flags(runtime)
    rtsp = runtime.add_subparsers(dest="runtime_command")
    rtmap = rtsp.add_parser("map")
    _add_target_flags(rtmap)
    rtmap.add_argument("control_profile_id")
    rtmap.add_argument("runtime_profile")
    rtmap.add_argument("--role", default="worker", choices=["worker", "pm", "admin", "observer"])
    rtmap.add_argument("--disabled", action="store_true")
    rtmap.add_argument("--actor-instance-id", default=None)
    rtshow = rtsp.add_parser("show")
    _add_target_flags(rtshow)
    rtshow.add_argument("control_profile_id")

    watchdog = sp.add_parser("watchdog", help="Run a control-plane supervision/watchdog pass")
    _add_target_flags(watchdog)
    wdsp = watchdog.add_subparsers(dest="watchdog_command")
    wdrun = wdsp.add_parser("run")
    _add_target_flags(wdrun)
    wdrun.add_argument("--actor-instance-id", required=True)
    wdrun.add_argument("--dry-run", action="store_true", default=True)
    wdrun.add_argument("--apply", dest="dry_run", action="store_false", help="Apply conservative status-only repairs; never deletes rows")
    wdrun.add_argument("--stale-ms", type=int, default=600_000)

    message = sp.add_parser("message", help="Manage messages")
    _add_target_flags(message)
    msp = message.add_subparsers(dest="message_command")
    mcreate = msp.add_parser("create")
    _add_target_flags(mcreate)
    mcreate.add_argument("--sender-instance-id", required=True)
    mcreate.add_argument("--receiver", required=True)
    mcreate.add_argument("--kind", required=True)
    mcreate.add_argument("--body", required=True)
    mcreate.add_argument("--capability", default="message")
    mcreate.add_argument("--metadata-json", default=None)
    mlist = msp.add_parser("list")
    _add_target_flags(mlist)
    mlist.add_argument("--receiver", default=None)
    mlist.add_argument("--status", default=None)
    for verb in ("ack", "resolve", "supersede", "cancel"):
        mverb = msp.add_parser(verb)
        _add_target_flags(mverb)
        mverb.add_argument("message_id")
        mverb.add_argument("--actor-instance-id", required=True)
        mverb.add_argument("--actor-profile", default=None)
        mverb.add_argument("--actor-type", choices=["receiver", "admin", "bootstrap"], default="receiver")
        mverb.add_argument("--reason", default=None)
        mverb.add_argument("--metadata-json", default=None)

    artifacts = sp.add_parser("artifacts")
    _add_target_flags(artifacts)
    asp = artifacts.add_subparsers(dest="artifacts_command")
    alist = asp.add_parser("list")
    _add_target_flags(alist)
    alist.add_argument("dispatch_id")

    worker = sp.add_parser("worker")
    _add_target_flags(worker)
    wsp = worker.add_subparsers(dest="worker_command")
    wrun = wsp.add_parser("run")
    _add_target_flags(wrun)
    wrun.add_argument("dispatch_id")
    wrun.add_argument("--profile-id", required=True)
    wrun.add_argument("--instance-id", required=True)
    wrun.add_argument("--handler", default="deterministic", choices=["deterministic", "agent"])
    wrun.add_argument("--soft-timeout-s", type=float, default=DEFAULT_AGENT_WORKER_SOFT_TIMEOUT_S)
    wrun.set_defaults(hard_timeout_s_explicit=False)
    wrun.add_argument("--hard-timeout-s", action=_StoreExplicitFloat, explicit_dest="hard_timeout_s_explicit", default=DEFAULT_AGENT_WORKER_HARD_TIMEOUT_S)
    wrun.add_argument("--timeout-s", type=float, default=None, help="Deprecated alias for --hard-timeout-s")

    wave = sp.add_parser("wave", help="Create and optionally supervise finite wave lifecycles")
    _add_target_flags(wave)
    wavsp = wave.add_subparsers(dest="wave_command")
    wdisp = wavsp.add_parser("dispatch-statutepm", help="Create and optionally supervise one statute PM wave dispatch")
    _add_target_flags(wdisp)
    wdisp.add_argument("--payload-json", required=True)
    wdisp.add_argument("--idempotency-key", required=True)
    wdisp.add_argument("--supervise", action="store_true")
    wdisp.add_argument("--admin-profile", default="default")
    wdisp.add_argument("--pm-profile-id", default="statutepm")
    wdisp.add_argument("--pm-runtime-profile", default=None)
    wdisp.add_argument("--worker-profile", default="statute-worker")
    wdisp.add_argument("--supervisor-instance-id", default=None)
    wdisp.add_argument("--pm-instance-id", default=None)
    wdisp.add_argument("--supervisor-lease-ms", type=int, default=3_600_000)
    wdisp.add_argument("--poll-interval-s", type=float, default=1.0)
    wdisp.add_argument("--child-soft-timeout-s", type=float, default=DEFAULT_AGENT_WORKER_SOFT_TIMEOUT_S)
    wdisp.add_argument("--child-hard-timeout-s", type=float, default=DEFAULT_AGENT_WORKER_HARD_TIMEOUT_S)

    pm = sp.add_parser("pm", help="Run PM control-plane consumers")
    _add_target_flags(pm)
    pmsp = pm.add_subparsers(dest="pm_command")
    pmrun = pmsp.add_parser("run", help="Run the statute PM dispatcher")
    _add_target_flags(pmrun)
    mode_group = pmrun.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--once", action="store_true")
    mode_group.add_argument("--loop", action="store_true")
    pmrun.add_argument("--pm-profile-id", default="statutepm")
    pmrun.add_argument("--pm-instance-id", default="statutepm:bootstrap")
    pmrun.add_argument("--pm-runtime-profile", default=None)
    pmrun.add_argument("--worker-profile", default="statute-worker")
    pmrun.add_argument("--poll-interval-s", type=float, default=5.0)
    pmrun.add_argument("--child-soft-timeout-s", type=float, default=DEFAULT_AGENT_WORKER_SOFT_TIMEOUT_S)
    pmrun.add_argument("--child-hard-timeout-s", type=float, default=DEFAULT_AGENT_WORKER_HARD_TIMEOUT_S)

    smoke = sp.add_parser("smoke-test")
    _add_target_flags(smoke, live=False)
    smoke.add_argument("subject", choices=["statutepm"])

    live_smoke = sp.add_parser("live-smoke", help="Run an approved deterministic live DB smoke")
    _add_target_flags(live_smoke)
    live_smoke.add_argument("subject", choices=["statutepm"])
    live_smoke.add_argument("--deterministic", action="store_true")
    live_smoke.add_argument("--smoke-tag", required=True)
    live_smoke.add_argument("--idempotency-key", required=True)

    ready = sp.add_parser("readiness")
    _add_target_flags(ready, live=False)
    ready.add_argument("subject", choices=["statutepm"])
    ready.add_argument("--live-check", action="store_true")

    parser.set_defaults(func=cmd_control)

from __future__ import annotations

import json
import sys
from typing import Any

from hermes_cli import control_db as cp


def _conn():
    return cp.connect()


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def cmd_control(args) -> None:
    command = getattr(args, "control_command", None) or "doctor"
    conn = _conn()
    try:
        if command == "migrate":
            cp.init_schema(conn)
            print(f"Control DB ready: {cp.control_db_path()}")
            return
        if command == "doctor":
            issues = cp.doctor(conn)
            if not issues:
                print(f"ok: {cp.control_db_path()}")
                return
            for issue in issues:
                print(f"{issue.level}: {issue.code}: {issue.detail}")
            if any(i.level == "error" for i in issues):
                raise SystemExit(2)
            return
        if command == "mode":
            mode = getattr(args, "mode", None)
            if mode:
                cp.set_authority_mode(conn, mode, actor_type="bootstrap")
            print(cp.get_authority_mode(conn))
            return
        if command == "bootstrap":
            profile = getattr(args, "admin_profile", None) or "default"
            cp.bootstrap_default_policies(conn, admin_profile=profile)
            print(f"bootstrapped default control-plane policies for admin profile: {profile}")
            return
        if command == "profile":
            profile_id = getattr(args, "profile_id", None)
            role = getattr(args, "role", None) or "worker"
            if not profile_id:
                raise SystemExit("profile_id required")
            cp.register_profile(conn, profile_id, role=role, display_name=getattr(args, "display_name", None), actor_type="bootstrap" if role != "worker" else "worker")
            print(f"registered profile {profile_id} role={role}")
            return
        if command == "heartbeat":
            profile_id = getattr(args, "profile_id", None) or "default"
            instance_id = getattr(args, "instance_id", None)
            inst = cp.register_instance(conn, profile_id, instance_id=instance_id)
            print(inst)
            return
        if command == "routes":
            rows = conn.execute(
                "SELECT priority,effect,sender_profile,receiver_profile,kind,capability,created_by,created_by_type FROM cp_route_policies ORDER BY priority DESC,effect ASC"
            ).fetchall()
            _print_json([dict(r) for r in rows])
            return
        raise SystemExit(f"unknown control command: {command}")
    finally:
        conn.close()


def register_subparser(subparsers) -> None:
    parser = subparsers.add_parser(
        "control",
        help="Manage the durable local Hermes control-plane DB",
        description="Durable cross-profile control plane for dispatches, approvals, messages, routing, audit, and Discord mirror state.",
    )
    sp = parser.add_subparsers(dest="control_command")
    sp.add_parser("migrate", help="Initialize/migrate the control DB schema")
    sp.add_parser("doctor", help="Check control DB health")

    mode = sp.add_parser("mode", help="Get or set control-plane authority mode")
    mode.add_argument("mode", nargs="?", choices=["legacy", "shadow", "control_db"])

    bootstrap = sp.add_parser("bootstrap", help="Install default-deny baseline route policies")
    bootstrap.add_argument("--admin-profile", default="default")

    profile = sp.add_parser("profile", help="Register/update a profile")
    profile.add_argument("profile_id")
    profile.add_argument("--role", default="worker", choices=["worker", "pm", "admin", "observer"])
    profile.add_argument("--display-name", default=None)

    heartbeat = sp.add_parser("heartbeat", help="Register/heartbeat a profile instance")
    heartbeat.add_argument("profile_id", nargs="?", default="default")
    heartbeat.add_argument("--instance-id", default=None)

    sp.add_parser("routes", help="List route policies")
    parser.set_defaults(func=cmd_control)

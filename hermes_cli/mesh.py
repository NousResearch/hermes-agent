"""hermes mesh — fleet provisioner CLI surface.

Slots into hermes_cli/ alongside gateway.py, profile.py, status.py.

Commands:
    hermes mesh init                          # one-time controller setup
    hermes mesh ssh-setup <host>              # bootstrap passwordless SSH + sudoers
    hermes mesh add <host> --role <role>      # provision a new node
    hermes mesh remove <host> [--purge]       # decommission
    hermes mesh list                          # registered nodes
    hermes mesh status [<host>]               # health + drift detection
    hermes mesh restart <host>                # systemctl restart on remote
    hermes mesh logs <host> [--follow]        # journalctl on remote
    hermes mesh role list                     # available role templates
    hermes mesh role new <name>               # scaffold a new role locally
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from mesh import config as cfg_mod
from mesh import provisioner, registry, validation
from mesh.provisioner import NodeSpec


def cmd_init(args: argparse.Namespace) -> int:
    provisioner.init_controller()
    return 0


def cmd_ssh_setup(args: argparse.Namespace) -> int:
    provisioner.ssh_setup(args.host, user=args.user)
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    cfg = cfg_mod.load()
    caps = [c.strip() for c in args.capabilities.split(",") if c.strip()]
    spec = NodeSpec(
        host=args.host, role=args.role, namespace=cfg.namespace,
        broker=cfg.broker, capabilities=caps, user=args.user,
    )
    provisioner.add_node(
        spec, cfg,
        rollback_on_failure=not args.no_rollback,
        skip_tls_setup=args.skip_tls_setup,
    )
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    cfg = cfg_mod.load()
    provisioner.remove_node(args.host, cfg, purge=args.purge)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    nodes = registry.list_nodes()
    if not nodes:
        print("(no nodes registered — run `hermes mesh add <host> --role <role>` to provision one)")
        return 0
    print(f"{'HOST':<20} {'NAMESPACE':<16} {'ROLE':<12} {'USER':<14} CAPABILITIES")
    for n in nodes:
        caps = ",".join(n.get("capabilities") or []) or "-"
        ns = n.get("namespace", "?")
        print(f"{n['host']:<20} {ns:<16} {n.get('role','?'):<12} {n.get('user','?'):<14} {caps}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg = cfg_mod.load()
    statuses = validation.drift_check(cfg)
    if args.host:
        statuses = [s for s in statuses if s.host == args.host]
        if not statuses:
            print(f"(no record of host {args.host})")
            return 1
    print(f"{'HOST':<20} {'STATE':<14} {'AGE(s)':<8} {'REG':<4} MISMATCH")
    for s in statuses:
        age = f"{s.last_heartbeat_age_seconds:.0f}" if s.last_heartbeat_age_seconds is not None else "-"
        mm = s.mismatch_detail if s.metadata_mismatch else "-"
        print(f"{s.host:<20} {s.state:<14} {age:<8} {'yes' if s.registered else 'no':<4} {mm}")
    return 0


def cmd_restart(args: argparse.Namespace) -> int:
    cfg = cfg_mod.load()
    all_nodes = registry.list_nodes()
    matches = [n for n in all_nodes if n["host"] == args.host]
    entry = next((n for n in matches if n.get("namespace") == cfg.namespace), matches[0] if matches else None)
    ssh_user = entry.get("user") if entry else None
    ns = entry.get("namespace", cfg.namespace) if entry else cfg.namespace
    service = f"{ns}-{args.host}.service"
    provisioner.ssh_run(args.host, f"sudo -n systemctl restart {shlex.quote(service)}", user=ssh_user, timeout=30)
    print(f"✓ restarted {service}")
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    cfg = cfg_mod.load()
    all_nodes = registry.list_nodes()
    matches = [n for n in all_nodes if n["host"] == args.host]
    entry = next((n for n in matches if n.get("namespace") == cfg.namespace), matches[0] if matches else None)
    ssh_user = entry.get("user") if entry else None
    target = f"{ssh_user}@{args.host}" if ssh_user else args.host
    ns = entry.get("namespace", cfg.namespace) if entry else cfg.namespace
    service = f"{ns}-{args.host}.service"
    follow_flag = "-f" if args.follow else "--no-pager -n 200"
    # Stream directly to the terminal (don't capture).
    subprocess.run(
        ["ssh", "-t", target, f"journalctl -u {shlex.quote(service)} {follow_flag}"],
        check=False,
    )
    return 0


def cmd_role_list(args: argparse.Namespace) -> int:
    cfg = cfg_mod.load()
    print("Available role templates (user overrides win):")
    seen = set()
    for d in cfg.template_dirs:
        if not d.exists():
            continue
        for role_dir in sorted(d.iterdir()):
            if role_dir.is_dir() and role_dir.name not in seen:
                manifest = role_dir / "manifest.yaml"
                desc = ""
                if manifest.exists():
                    for line in manifest.read_text().splitlines():
                        if line.startswith("description:"):
                            desc = line.split(":", 1)[1].strip()
                            break
                source = "user" if d == cfg.template_dirs[0] else "default"
                print(f"  {role_dir.name:<14} [{source}] {desc}")
                seen.add(role_dir.name)
    return 0


def cmd_role_new(args: argparse.Namespace) -> int:
    cfg = cfg_mod.load()
    user_dir = cfg.template_dirs[0]  # by convention, first is user override
    target = user_dir / args.name
    if target.exists():
        print(f"✗ role already exists: {target}")
        return 1
    # Copy bare as starting point.
    bare_dir = None
    for d in cfg.template_dirs:
        if (d / "bare").is_dir():
            bare_dir = d / "bare"
            break
    if bare_dir is None:
        print("✗ couldn't find bare template to copy from")
        return 1
    target.mkdir(parents=True)
    for f in bare_dir.iterdir():
        (target / f.name).write_text(f.read_text())
    print(f"✓ created {target} (copied from {bare_dir})")
    print(f"  edit script.py.tmpl / service.tmpl / manifest.yaml / deps.txt as needed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hermes mesh", description="Mesh fleet provisioner")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_init = sub.add_parser("init", help="One-time controller setup")
    sp_init.set_defaults(func=cmd_init)

    sp_ssh = sub.add_parser("ssh-setup", help="Bootstrap passwordless SSH trust to a host")
    sp_ssh.add_argument("host")
    sp_ssh.add_argument("--user", help="Remote login user (default: current user)")
    sp_ssh.set_defaults(func=cmd_ssh_setup)

    sp_add = sub.add_parser("add", help="Provision a new mesh node")
    sp_add.add_argument("host")
    sp_add.add_argument("--role", required=True, help="Role template name")
    sp_add.add_argument("--capabilities", default="", help="Comma-separated capability tags")
    sp_add.add_argument("--user", help="SSH login user (default: current user)")
    sp_add.add_argument("--skip-tls-setup", action="store_true", help="Don't auto-distribute CA cert")
    sp_add.add_argument("--no-rollback", action="store_true",
                        help="Leave half-built state intact on failure (for debugging)")
    sp_add.set_defaults(func=cmd_add)

    sp_rm = sub.add_parser("remove", help="Decommission a node")
    sp_rm.add_argument("host")
    sp_rm.add_argument("--purge", action="store_true", help="Also delete files on remote")
    sp_rm.set_defaults(func=cmd_remove)

    sp_list = sub.add_parser("list", help="List registered nodes")
    sp_list.set_defaults(func=cmd_list)

    sp_status = sub.add_parser("status", help="Show mesh health + drift")
    sp_status.add_argument("host", nargs="?", help="Optional: scope to one node")
    sp_status.set_defaults(func=cmd_status)

    sp_restart = sub.add_parser("restart", help="Restart a node's service")
    sp_restart.add_argument("host")
    sp_restart.set_defaults(func=cmd_restart)

    sp_logs = sub.add_parser("logs", help="Tail a node's journalctl logs")
    sp_logs.add_argument("host")
    sp_logs.add_argument("--follow", "-f", action="store_true")
    sp_logs.set_defaults(func=cmd_logs)

    sp_role = sub.add_parser("role", help="Manage role templates")
    role_sub = sp_role.add_subparsers(dest="role_cmd", required=True)
    sp_role_list = role_sub.add_parser("list")
    sp_role_list.set_defaults(func=cmd_role_list)
    sp_role_new = role_sub.add_parser("new")
    sp_role_new.add_argument("name")
    sp_role_new.set_defaults(func=cmd_role_new)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

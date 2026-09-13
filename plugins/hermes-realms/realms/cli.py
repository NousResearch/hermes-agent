"""Standalone profile-aware realm CLI. Machine-readable JSON by default."""

import argparse
import json
import subprocess
import sys

from .manager import Manager, RealmError


def configure_parser(parser):
    parser.add_argument(
        "--home", help="Explicit Hermes profile home (default: effective HERMES_HOME)"
    )
    commands = parser.add_subparsers(dest="operation", required=True)
    commands.add_parser("start").add_argument("session_id")
    commands.add_parser("list")
    for name in ("stop", "env"):
        commands.add_parser(name).add_argument("id")
    commands.add_parser("doctor").add_argument("id", nargs="?")
    for name, argument in (("shot", "path"), ("resize", "size")):
        command = commands.add_parser(name)
        command.add_argument("id")
        command.add_argument(argument)
    execute = commands.add_parser("exec")
    execute.add_argument("id")
    execute.add_argument("realm_command", metavar="command", nargs=argparse.REMAINDER)
    from .install_driver import configure_parser as configure_install
    configure_install(commands.add_parser("install-driver", help="Explicitly download and verify the Linux x86-64 driver"))
    configure_vm_parser(commands.add_parser(
        "vm", help="Manage the Omarchy VM realm kind's shared base image"))


def configure_vm_parser(parser):
    operations = parser.add_subparsers(dest="vm_operation", required=True)
    install = operations.add_parser(
        "install",
        help="Download the signed Omarchy ISO and build this profile's base image")
    install.add_argument(
        "--iso", help="Use an already-downloaded ISO instead of fetching one")
    operations.add_parser("status", help="Base image, storage and prerequisites")
    settings = operations.add_parser(
        "settings", help="Base image, storage, memory and network as one block")
    settings.add_argument(
        "--check-updates", action="store_true",
        help="Also ask GitHub for the newest Omarchy release (needs network)")
    operations.add_parser("doctor", help="Check Omarchy VM prerequisites")
    operations.add_parser("list", help="Running VM realms in this profile")
    operations.add_parser(
        "remove-base", help="Delete this profile's base image (not the ISO)")
    operations.add_parser("clean", help="Remove stale ISOs and orphaned session disks")
    operations.add_parser("stop").add_argument("id")


def run_vm(args):
    from .vm_manager import VmManager

    manager = VmManager(args.home)
    if args.vm_operation == "install":
        # Long, loud and explicit: a 5 GB download plus an unattended install
        # is not something to run behind a spinner.
        result = manager.install_base(iso=getattr(args, "iso", None), stdout=sys.stderr)
    elif args.vm_operation == "status":
        result = {
            "base": manager.base_status(),
            "storage": manager.storage(),
            "realms": manager.list(),
        }
    elif args.vm_operation == "settings":
        result = manager.settings(check_updates=getattr(args, "check_updates", False))
    elif args.vm_operation == "doctor":
        report = manager.doctor()
        print(json.dumps(report, sort_keys=True))
        return 0 if report["ok"] else 1
    elif args.vm_operation == "list":
        result = manager.list()
    elif args.vm_operation == "remove-base":
        result = {"removed": manager.remove_base()}
    elif args.vm_operation == "clean":
        result = clean(manager)
    else:
        result = {"stopped": manager.stop(args.id), "id": args.id}
    print(json.dumps(result, sort_keys=True))
    return 0


def clean(manager):
    """Drop what no live realm references: stale ISOs and orphaned session disks."""
    import shutil
    from pathlib import Path

    live = {Path(record["session_dir"]).name for record in manager.list()}
    removed = []
    sessions = manager.registry.root / "vm"
    if sessions.is_dir():
        for directory in sessions.iterdir():
            if directory.name not in live:
                shutil.rmtree(directory, ignore_errors=True)
                removed.append(str(directory))
    keep = (manager.base_status().get("iso") or "")
    iso_dir = manager.data / "iso"
    if iso_dir.is_dir():
        for iso in iso_dir.glob("omarchy-*.iso"):
            if iso.name != keep:
                iso.unlink(missing_ok=True)
                iso.with_suffix(".iso.sig").unlink(missing_ok=True)
                removed.append(str(iso))
    return {"removed": removed, "storage": manager.storage()}


def run(args):
    try:
        if args.operation == "install-driver":
            from .install_driver import run as install
            install(args)
            return 0
        if args.operation == "vm":
            return run_vm(args)
        manager = Manager(args.home)
        if args.operation == "start":
            result = manager.start(args.session_id)
        elif args.operation == "list":
            result = manager.list()
        elif args.operation == "stop":
            result = {"stopped": manager.stop(args.id), "id": args.id}
        elif args.operation == "doctor":
            result = manager.doctor(args.id)
            print(json.dumps(result, sort_keys=True))
            return 0 if result["ok"] else 1
        elif args.operation == "shot":
            result = {"path": manager.shot(args.id, args.path)}
        elif args.operation == "resize":
            result = manager.resize(args.id, args.size)
        elif args.operation == "exec":
            command = args.realm_command[1:] if args.realm_command[:1] == ["--"] else args.realm_command
            if not command:
                raise ValueError("exec requires a command after --")
            return subprocess.call(
                [*manager.command_prefix(args.id), *command], env=manager.env(args.id)
            )
        else:
            result = manager.env(args.id)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (RealmError, OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hermes realms")
    configure_parser(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

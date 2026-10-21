"""``hermes vm ...``: manage the profile's QEMU/KVM VM (create, start, stop, status, destroy, ssh).

Registered through ``ctx.register_cli_command`` so ``hermes_cli/main.py`` stays untouched. The
verbs are user-only operations: the agent never gets a tool to destroy its own sandbox.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from typing import Callable

from hermes_cli.cli_output import print_error, print_info, print_success, print_warning

from plugins.terminal_env.qemu_vm.manager import (
    NETWORK_MODES, VmError, VmManager, host_requirements, require_tool, ssh_base_argv)
from plugins.terminal_env.qemu_vm.provider import QemuVmProvider, manager_for_active_profile


def register_cli(parser: argparse.ArgumentParser) -> None:
    subs = parser.add_subparsers(dest="vm_command", required=False)
    create = subs.add_parser("create", help="Download the pinned Debian image, provision the VM and boot it")
    create.add_argument("--disk", type=int, default=None, metavar="GB", help="Disk size in GB (default: settings or 20)")
    create.add_argument("--cpus", type=int, default=None, help="vCPUs (default: settings or 2)")
    create.add_argument("--memory", type=int, default=None, metavar="MB", help="RAM in MB (default: settings or 2048)")
    create.add_argument("--network", choices=NETWORK_MODES, default=None, help="egress (NAT out, default) or none")
    create.add_argument("--force", action="store_true", help="Recreate an existing VM (destroys its disk)")
    subs.add_parser("start", help="Boot the VM and wait for SSH")
    subs.add_parser("stop", help="Power the VM off (ACPI, then force after 30s)")
    status = subs.add_parser("status", help="Show VM state and host requirements")
    status.add_argument("--json", action="store_true", help="Machine-readable output")
    destroy = subs.add_parser("destroy", help="Stop the VM and delete its disk, keys and state")
    destroy.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    ssh = subs.add_parser("ssh", help="Open an interactive shell in the VM (or print the ssh command)")
    ssh.add_argument("--print", action="store_true", dest="print_only", help="Print the ssh command instead of running it")


def _provider() -> QemuVmProvider:
    from hermes_cli.plugins import discover_plugins
    from agent.terminal_env_registry import get_provider

    discover_plugins()
    provider = get_provider("qemu_vm")
    return provider if isinstance(provider, QemuVmProvider) else QemuVmProvider()


def _manager() -> VmManager:
    return manager_for_active_profile(log=print_info)


def _cmd_create(args: argparse.Namespace) -> int:
    from dataclasses import replace

    spec = _provider().spec()
    overrides = {k: v for k, v in (("disk_gb", args.disk), ("cpus", args.cpus), ("memory_mb", args.memory),
                                   ("network", args.network)) if v is not None}
    manager = _manager()
    manager.create(replace(spec, **overrides), force=args.force, boot_timeout=max(_provider().boot_timeout(), 600))
    print_success(f"VM ready at {manager.paths.root} (ssh port {manager.ssh_target().port}).")
    print_info("Select it with: hermes config set terminal.backend qemu_vm")
    return 0


def _cmd_start(args: argparse.Namespace) -> int:
    manager = _manager()
    if manager.running():
        print_info("VM is already running.")
        return 0
    provider = _provider()
    manager.start(provider.spec(), boot_timeout=provider.boot_timeout())
    print_success(f"VM running (ssh port {manager.ssh_target().port}).")
    return 0


def _cmd_stop(args: argparse.Namespace) -> int:
    manager = _manager()
    if not manager.running():
        print_info("VM is not running.")
        return 0
    manager.stop()
    print_success("VM stopped.")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    manager = _manager()
    status = manager.status()
    reqs = host_requirements()
    if args.json:
        status["host"] = [{"ok": r.ok, "label": r.label, "detail": r.detail} for r in reqs]
        print(json.dumps(status, indent=2))
        return 0
    print_info(f"VM directory: {status['root']}")
    if not status["exists"]:
        print_warning("VM not provisioned (run `hermes vm create`).")
    else:
        state = "running" if status["running"] else "stopped"
        spec = status["spec"] or {}
        print_info(f"State: {state}  |  ssh port: {status['port']}  |  image: {status['image']}")
        print_info(f"Spec: {spec.get('cpus')} vCPU, {spec.get('memory_mb')} MB RAM, {spec.get('disk_gb')} GB disk, "
                   f"network={spec.get('network')}, user={spec.get('user')}")
    print_info("Host requirements:")
    for r in reqs:
        print_info(f"  [{'ok' if r.ok else '!!'}] {r.label}: {r.detail}")
    return 0 if all(r.ok for r in reqs) else 1


def _cmd_destroy(args: argparse.Namespace) -> int:
    manager = _manager()
    if not manager.exists():
        print_info("Nothing to destroy.")
        return 0
    if not args.yes:
        answer = input(f"Delete the VM under {manager.paths.root} and all its data? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print_info("Aborted.")
            return 1
    manager.destroy()
    print_success("VM destroyed.")
    return 0


def _cmd_ssh(args: argparse.Namespace) -> int:
    manager = _manager()
    if not manager.exists():
        print_error("VM not provisioned (run `hermes vm create`).")
        return 1
    target = manager.ssh_target()
    if args.print_only:
        argv = [require_tool("ssh"), *ssh_base_argv(target), "-p", str(target.port), f"{target.user}@{target.host}"]
        print(shlex.join(argv))
        return 0
    if not manager.running():
        provider = _provider()
        manager.start(provider.spec(), boot_timeout=provider.boot_timeout())
        target = manager.ssh_target()
    argv = [require_tool("ssh"), *ssh_base_argv(target), "-p", str(target.port), f"{target.user}@{target.host}"]
    sys.stdout.flush()
    os.execvp(argv[0], argv)  # replaces this process with an interactive session


_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "create": _cmd_create, "start": _cmd_start, "stop": _cmd_stop,
    "status": _cmd_status, "destroy": _cmd_destroy, "ssh": _cmd_ssh,
}


def handle_cli(args: argparse.Namespace) -> int:
    handler = _HANDLERS.get(getattr(args, "vm_command", None) or "")
    if handler is None:
        print_info("Usage: hermes vm {create|start|stop|status|destroy|ssh}")
        return 2
    try:
        return handler(args)
    except (VmError, ValueError) as exc:
        print_error(str(exc))
        return 1

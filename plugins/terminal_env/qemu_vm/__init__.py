"""QEMU/KVM VM terminal backend — bundled, auto-loaded (``kind: backend``).

Registers the ``qemu_vm`` backend (``terminal.backend: qemu_vm``) and the ``hermes vm`` CLI.
Settings live under ``plugins.entries.terminal_env/qemu_vm.settings`` and are read per call,
so one process serving several profiles sees each profile's own values.
"""

from __future__ import annotations

from plugins.terminal_env.qemu_vm.provider import QemuVmProvider


def register(ctx) -> None:
    ctx.register_terminal_environment_provider(QemuVmProvider(settings=ctx.get_config))

    from plugins.terminal_env.qemu_vm.cli import handle_cli, register_cli

    ctx.register_cli_command(
        name="vm",
        help="Manage the profile's QEMU/KVM sandbox VM (create, start, stop, status, destroy, ssh)",
        setup_fn=register_cli,
        handler_fn=handle_cli,
        description="Persistent per-profile Debian VM used as the terminal backend when terminal.backend is qemu_vm.",
    )

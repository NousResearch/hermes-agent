"""Pre-startup guard chain shared by every foreground gateway entry point."""

from __future__ import annotations


def apply_gateway_run_guards(force: bool = False, replace: bool = False) -> None:
    """Refuse a foreground gateway start the same way ``hermes gateway run`` does.

    Shared by ``run_gateway()`` and the legacy ``python cli.py --gateway`` entry so neither can
    bypass the docker-root, host-attach, supervised-service or duplicate-process protections.
    Each guard fails open on probe errors; a refusal exits the process with its documented code.
    """
    from hermes_cli import gateway as gw  # facade is the seam: guards are patched there

    gw._guard_official_docker_root_gateway()
    gw._attach_to_host_gateway_or_guard(force=force, replace=replace)
    gw._guard_supervised_gateway_conflict(force=force)
    gw._guard_existing_gateway_process_conflict(replace=replace)

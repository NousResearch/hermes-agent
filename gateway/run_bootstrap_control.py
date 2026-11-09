"""Mode-specific startup hooks shared by gateway bootstrap phases."""

from __future__ import annotations

import asyncio
import os


def set_gateway_process_mode(control_only: bool) -> None:
    """Set process-wide values required by both gateway modes."""
    os.environ["HERMES_EXEC_ASK"] = "1"
    from hermes_cli.update_proxy_env import restore_user_proxy_env
    restore_user_proxy_env()
    if not control_only:
        from gateway.run_bootstrap import _start_gateway_resolve_terminal_cwd
        _start_gateway_resolve_terminal_cwd()


def configure_gateway_startup_mode(control_only: bool) -> None:
    if not control_only:
        from gateway.run_bootstrap import _start_gateway_early_environment_probe
        _start_gateway_early_environment_probe()


async def recover_gateway_startup_state(control_only: bool, resolved_config, config):
    if control_only:
        return resolved_config
    from gateway.run_startup import recover_left_core_at_gateway_start
    from gateway.run_bootstrap import _reload_config_keeping_multiplex_verdict
    if await asyncio.to_thread(recover_left_core_at_gateway_start) and config is None:
        return _reload_config_keeping_multiplex_verdict(resolved_config)
    return resolved_config


async def open_gateway_control_channels(runner, control_only, start_socket, refresh_host, log_profiles):
    if control_only:
        return None
    control_server = await start_socket(runner)
    if control_server is None:
        raise RuntimeError("gateway session bootstrap control listener unavailable")
    refresh_host(runner)
    log_profiles(runner)
    from gateway.run_runtime import start_gateway_runtime_api
    await start_gateway_runtime_api(runner)
    return control_server


def start_gateway_mode_background(control_only, best_effort, start_keepalive, ensure_windows_imports):
    if control_only:
        return
    best_effort(start_keepalive, "Nous auth keepalive did not start: %s")
    ensure_windows_imports()


def start_gateway_mode_cron(runner, control_only, start_cron):
    if control_only:
        return None, None, None, None
    return start_cron(runner)


def start_gateway_mode_surface(runner, control_only, idle_exit):
    if control_only:
        return
    if idle_exit:
        from gateway.run_idle_exit import arm_unmanaged_idle_exit
        arm_unmanaged_idle_exit(runner)
        return
    from gateway.run_api import warm_gateway_dashboard
    warm_gateway_dashboard(runner)

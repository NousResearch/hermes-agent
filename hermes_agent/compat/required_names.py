"""Old names a supported release reaches after its updater swaps in a newer checkout.

``scripts/move/generate_shims.py`` writes a shim at the old location of each name whose module has
moved (``moves.MOVES``). Supported: release-channel releases from v0.21.6 to the current one, and
their canaries. Kinds:

- ``package``: ``<name>/__init__.py``; a fresh process imports the old package name.
- ``module``: a release imports the name where the finder is not installed: lazily, after the swap,
  under a package its updater loaded before (Python looks on the old package's ``__path__``), or
  first thing in a fresh process (a launcher's ``import hermes_bootstrap``).
- ``by-path``: a release runs or loads the file by path.
- ``exists``: a release checks that the file exists.
- ``copy``: a release runs the file without the application on the path (``python -I -S``, or the
  file copied out alone); the old path holds a byte copy of the moved file.
- ``hand``: the old path holds hand-written code that must run before ``hermes_agent`` is
  importable. The generator does not write it; ``--check`` fails while it is missing or is a shim.

Comments: the releases that reach the name, and one caller. Plain literals, importable on Python 3.11.

Two names are here for people, not releases: the documentation tells users to run ``python cli.py``
and to drop a provider file into ``providers/``.

``REQUIRED_TEXT``: text a release searches for in the file at the old path. The shim carries it in a
comment. The text is fixed by the releases that read it.
"""

REQUIRED_NAMES: dict[str, str] = {
    "acp_adapter": "package",  # fresh processes: -m, launchers, service units
    "agent": "package",  # fresh processes: -m, launchers, service units
    "agent.secret_sources": "package",  # parent of agent.secret_sources.registry
    "cron": "package",  # fresh processes: -m, launchers, service units
    "gateway": "package",  # fresh processes: -m, launchers, service units
    "gateway.platforms": "package",  # parent of gateway.platforms._shared
    "hermes_cli": "package",  # fresh processes: -m, launchers, service units
    "hermes_cli.local_runtime": "package",  # parent of hermes_cli.local_runtime.processes
    "hermes_cli.observability": "package",  # parent of hermes_cli.observability.shared_metrics_disabled
    "plugins": "package",  # fresh processes: -m, launchers, service units
    "providers": "package",  # fresh processes: `import providers`; documented drop-in folder providers/<name>.py
    "tools": "package",  # fresh processes: -m, launchers, service units
    "tui_gateway": "package",  # fresh processes: -m, launchers, service units
    "agent.delegation_context": "module",  # v0.22.7..v0.23.0: tools/environments/local.py:307@v0.23.0
    "agent.outbound_webhooks": "module",  # v0.21.6..v0.22.6: hermes_cli/plugins.py:1261@v0.22.6
    "agent.retry_utils": "module",  # v0.22.7..v0.23.0: utils.py:259@v0.23.0
    "agent.safe_worker_policy": "module",  # v0.22.7..v0.23.0: hermes_cli/env_loader.py:536@v0.23.0
    "agent.secret_scope": "module",  # v0.22.7..v0.23.0: hermes_cli/config.py:1419@v0.23.0
    "agent.secret_sources.registry": "module",  # v0.21.6..v0.22.6: hermes_cli/plugins.py:1271@v0.22.6
    "agent.shell_hooks": "module",  # v0.21.6..v0.22.6: hermes_cli/plugins.py:1260@v0.22.6
    "agent.skill_utils": "module",  # v0.22.7..v0.23.0: hermes_cli/profiles.py:812@v0.23.0
    "agent.terminal_env_registry": "module",  # v0.22.7..v0.23.0: tools/environments/local_env_policy.py:405@v0.23.0
    "cli": "module",  # documented user command: `python cli.py` (README, /toolsets help text)
    "gateway.config": "module",  # v0.22.7..v0.23.0: hermes_cli/gateway_multiplex_mode.py:64@v0.23.0
    "gateway.config_env": "module",  # v0.21.6..v0.23.0: gateway/config.py:891@v0.23.0
    "gateway.config_loader": "module",  # v0.21.6..v0.23.0: gateway/config.py:831@v0.23.0
    "gateway.control_socket": "module",  # v0.22.7..v0.23.0: hermes_cli/update_receipt.py:1043@v0.23.0
    "gateway.drain_control": "module",  # v0.22.7..v0.23.0: gateway/host_rendezvous.py:290@v0.23.0
    "gateway.host_attach": "module",  # v0.22.7..v0.23.0: gateway/status.py:748@v0.23.0
    "gateway.host_rendezvous": "module",  # v0.22.7..v0.23.0: gateway/host_attach.py:198@v0.23.0
    "gateway.host_topology": "module",  # v0.22.7..v0.23.0: gateway/status.py:1306@v0.23.0
    "gateway.lifecycle_ledger": "module",  # v0.22.7..v0.23.0: hermes_cli/gateway_windows.py:1199@v0.23.0
    "gateway.platform_activation": "module",  # v0.22.7..v0.23.0: gateway/config_loader.py:540@v0.23.0
    "gateway.platform_registry": "module",  # v0.21.6..v0.23.0: gateway/config_loader.py:531@v0.23.0
    "gateway.platforms._shared": "module",  # v0.22.7..v0.23.0: gateway/config_loader.py:441@v0.23.0
    "gateway.profile_routing": "module",  # v0.21.6..v0.22.6: gateway/config.py:779@v0.22.6
    "gateway.run": "module",  # v0.22.7..v0.23.0: hermes_cli/gateway_migrate.py:443@v0.23.0
    "gateway.runtime_bootstrap_windows": "module",  # v0.22.7..v0.23.0: gateway/control_socket.py:514@v0.23.0
    "gateway.runtime_ownership": "module",  # v0.22.7..v0.23.0: gateway/status.py:1001@v0.23.0
    "gateway.session_context": "module",  # v0.22.7..v0.23.0: tools/environments/local.py:252@v0.23.0
    "gateway.shutdown_forensics": "module",  # v0.22.6..v0.23.0: hermes_cli/update_cmd_fleet_unit_budget.py:21@v0.23.0
    "gateway.shutdown_watchdog": "module",  # v0.22.7..v0.23.0: hermes_cli/gateway.py:449@v0.23.0
    "gateway.status": "module",  # v0.22.7..v0.23.0: hermes_cli/update_pause_record.py:820@v0.23.0
    "gateway.status_pid_records": "module",  # v0.22.7..v0.23.0: gateway/status.py:2044@v0.23.0
    "hermes_cli.active_sessions": "module",  # v0.22.7..v0.23.0: hermes_cli/process_identity.py:478@v0.23.0
    "hermes_cli.agent_plugins": "module",  # v0.21.6..v0.22.6: hermes_cli/plugins_manifest.py:515@v0.22.6
    "hermes_cli.approval_transport": "module",  # v0.21.6..v0.22.6: hermes_cli/plugins.py:1382@v0.22.6
    "hermes_cli.auth": "module",  # v0.21.6..v0.23.0: gateway/config.py:864@v0.23.0
    "hermes_cli.config_backups": "module",  # v0.22.7..v0.23.0: hermes_cli/config_effective.py:102@v0.23.0
    "hermes_cli.config_home": "module",  # v0.22.7..v0.23.0: hermes_cli/config.py:582@v0.23.0
    "hermes_cli.curses_ui": "module",  # v0.22.7..v0.23.0: hermes_cli/setup.py:262@v0.23.0
    "hermes_cli.dashboard_procs": "module",  # v0.22.7..v0.23.0: hermes_cli/update_fleet_scope.py:101@v0.23.0
    "hermes_cli.gateway": "module",  # v0.22.7..v0.23.0: hermes_cli/update_cmd_fleet.py:1008@v0.23.0
    "hermes_cli.gateway_command_errors": "module",  # v0.22.7..v0.23.0: hermes_cli/gateway.py:2536@v0.23.0
    "hermes_cli.gateway_migrate": "module",  # v0.22.7..v0.23.0: hermes_cli/update_cmd_fleet_verify.py:499@v0.23.0
    "hermes_cli.gateway_migrate_guards": "module",  # v0.22.7..v0.23.0: hermes_cli/gateway.py:587@v0.23.0
    "hermes_cli.gateway_multiplex_mode": "module",  # v0.22.7..v0.23.0: hermes_cli/gateway.py:4156@v0.23.0
    "hermes_cli.gateway_multiplex_s6": "module",  # v0.22.7..v0.23.0: hermes_cli/gateway_migrate.py:1221@v0.23.0
    "hermes_cli.gateway_multiplex_served": "module",  # v0.22.7..v0.23.0: hermes_cli/gateway.py:4143@v0.23.0
    "hermes_cli.gateway_service_owner": "module",  # v0.22.7..v0.23.0: hermes_cli/gateway.py:3359@v0.23.0
    "hermes_cli.gateway_windows": "module",  # v0.22.7..v0.23.0: hermes_cli/update_cmd_windows.py:1525@v0.23.0
    "hermes_cli.gateway_windows_legacy": "module",  # v0.22.7..v0.23.0: hermes_cli/gateway_windows.py:968@v0.23.0
    "hermes_cli.github_api": "module",  # v0.22.7..v0.23.0: hermes_cli/source_check.py:122@v0.23.0
    "hermes_cli.kanban_db": "module",  # v0.22.7..v0.23.0: agent/delegation_context.py:116@v0.23.0
    "hermes_cli.local_runtime.processes": "module",  # v0.22.7..v0.23.0: hermes_cli/_subprocess_compat.py:854@v0.23.0
    "hermes_cli.main": "module",  # v0.22.7..v0.23.0: fresh-process import, hermes_cli/update_cmd_validation.py:10
    "hermes_cli.moa_config": "module",  # v0.22.7..v0.23.0: hermes_cli/config.py:2263@v0.23.0
    "hermes_cli.observability.shared_metrics_disabled": "module",  # v0.22.7..v0.23.0: hermes_cli/config.py:3289@v0.23.0
    "hermes_cli.observability.shared_metrics_update": "module",  # v0.21.6..v0.23.0: hermes_cli/update_receipt.py:853@v0.23.0
    "hermes_cli.plugins": "module",  # v0.22.7..v0.23.0: gateway/config_loader.py:529@v0.23.0
    "hermes_cli.plugins_cmd_catalog": "module",  # v0.21.6..v0.22.6: hermes_cli/plugins_discovery.py:328@v0.22.6
    "hermes_cli.plugins_discovery": "module",  # v0.22.7..v0.23.0: hermes_cli/gateway_migrate.py:444@v0.23.0
    "hermes_cli.profiles": "module",  # v0.22.7..v0.23.0: hermes_cli/update_cmd_fleet.py:90@v0.23.0
    "hermes_cli.pt_input_extras": "module",  # v0.22.7..v0.23.0: hermes_cli/curses_ui.py:95@v0.23.0
    "hermes_cli.runtime_state": "module",  # v0.22.7..v0.23.0: hermes_cli/update_receipt.py:285@v0.23.0
    "hermes_cli.service_manager": "module",  # v0.22.7..v0.23.0: hermes_cli/gateway_migrate.py:1222@v0.23.0
    "hermes_cli.setup": "module",  # v0.22.7..v0.23.0: hermes_cli/gateway_windows.py:921@v0.23.0
    "hermes_cli.update_cmd_drain_report": "module",  # v0.22.7..v0.23.0: hermes_cli/update_cmd_fleet.py:1146@v0.23.0
    "hermes_cli.update_cmd_fleet_checkout": "module",  # v0.22.7..v0.23.0: hermes_cli/update_cmd_fleet.py:380@v0.23.0
    "hermes_cli.update_cmd_fleet_gatewayless": "module",  # v0.22.7..v0.23.0: hermes_cli/update_cmd_fleet.py:339@v0.23.0
    "hermes_cli.update_cmd_fleet_unit_budget": "module",  # v0.22.6..v0.23.0: hermes_cli/update_cmd_posix_pause.py:355@v0.22.6
    "hermes_cli.update_cmd_stale_survivors": "module",  # v0.22.7..v0.23.0: hermes_cli/update_cmd_fleet_verify.py:382@v0.23.0
    "hermes_cli.update_fleet_scope": "module",  # v0.22.7..v0.23.0: hermes_cli/update_cmd_fleet.py:729@v0.23.0
    "hermes_cli.update_lock": "module",  # v0.22.7..v0.23.0: hermes_cli/update_custody.py:177@v0.23.0
    "hermes_cli.update_restart_recovery": "module",  # v0.22.7..v0.23.0: hermes_cli/update_host_obligation.py:61@v0.23.0
    "hermes_cli.update_serve_obligations": "module",  # v0.21.6..v0.23.0: hermes_cli/update_receipt.py:660@v0.22.6
    "hermes_cli.update_stage": "module",  # v0.22.7..v0.23.0: hermes_cli/update_cmd_fleet.py:1695@v0.23.0
    "hermes_cli.version_info": "module",  # v0.22.7..v0.23.0: hermes_cli/update_cmd_fleet.py:144@v0.23.0
    "hermes_bootstrap": "hand",  # v0.21.6..v0.23.0: every launcher script and inline launch command, -I, no finder: hermes_cli/_launchers.py:68,499@v0.23.0; it repairs a killed update that moved hermes_agent/ aside
    "hermes_constants": "module",  # v0.21.6..v0.23.0: launcher --print-runtime-command, before hermes_bootstrap: hermes_cli/_launchers.py:61,492,504@v0.23.0
    "hermes_state_common": "module",  # v0.22.7..v0.23.1: the updater reads its text, see REQUIRED_TEXT
    "model_tools": "module",  # v0.22.7..v0.23.0: fresh-process import, hermes_cli/update_cmd_validation.py:10
    "run_agent": "module",  # v0.22.7..v0.23.0: fresh-process import, hermes_cli/update_cmd_validation.py:10
    "tools.env_passthrough": "module",  # v0.22.7..v0.23.0: tools/environments/local.py:322@v0.23.0
    "tools.environments": "module",  # v0.22.7..v0.23.0: tools/environments/local_pythonpath.py:24@v0.23.0
    "tools.environments.local": "module",  # v0.21.6..v0.23.0: hermes_cli/update_cmd_posix_pause.py:527@v0.23.0
    "toolsets": "module",  # v0.22.7..v0.23.0: fresh-process import, hermes_cli/update_cmd_validation.py:10
    "hermes_cli/_launchers.py": "by-path",  # v0.21.6..v0.23.0: scripts/install.ps1:1393
    "hermes_cli/_update_takeover.py": "by-path",  # v0.21.6..v0.23.0: hermes_cli/_old_updater.py:137@v0.23.0
    "hermes_cli/backup_sqlite.py": "copy",  # v0.21.6..v0.23.0: -I -S, apps/desktop/electron/updater/state-db-preflight.ts:60
    "hermes_cli/linux_desktop_entry.py": "copy",  # v0.21.6..v0.23.0: nix/desktop.nix:204 copies it alone onto PYTHONPATH
    "hermes_cli/main.py": "by-path",  # v0.21.6..v0.23.0: hermes_cli/update_receipt.py:1063@v0.23.0
    "hermes_cli/source_check.py": "by-path",  # v0.21.6..v0.23.0: apps/desktop/electron/updater/checkout-source.ts:106@v0.23.0
    "hermes_cli/source_completion.py": "by-path",  # v0.21.6..v0.23.0: scripts/install.sh:1120
    "hermes_cli/update_completion.py": "by-path",  # v0.21.6..v0.23.0: hermes_cli/update_completion.py:85@v0.23.0
    "hermes_cli/update_finish.py": "by-path",  # v0.22.7..v0.23.0: hermes_cli/_update_takeover.py:190@v0.23.0
    "hermes_cli/update_serve_resume.py": "by-path",  # v0.22.7..v0.23.0: hermes_cli/_update_takeover.py:153@v0.23.0
    "scripts/docker_config_migrate.py": "by-path",  # v0.21.6..v0.23.0: docker/stage2-hook.sh:660@v0.23.0
    "tools/skills_sync.py": "by-path",  # v0.21.6..v0.23.0: docker/stage2-hook.sh:767
    "plugins/platforms/bluebubbles/gateway_env.py": "exists",  # v0.22.7..v0.23.0: gateway/platform_env.py:196@v0.23.0
    "plugins/platforms/dingtalk/gateway_env.py": "exists",  # v0.22.7..v0.23.0: gateway/platform_env.py:196@v0.23.0
    "plugins/platforms/discord/gateway_env.py": "exists",  # v0.22.7..v0.23.0: gateway/platform_env.py:196@v0.23.0
    "plugins/platforms/email/gateway_env.py": "exists",  # v0.22.7..v0.23.0: gateway/platform_env.py:196@v0.23.0
    "plugins/platforms/feishu/gateway_env.py": "exists",  # v0.22.7..v0.23.0: gateway/platform_env.py:196@v0.23.0
    "plugins/platforms/matrix/gateway_env.py": "exists",  # v0.22.7..v0.23.0: gateway/platform_env.py:196@v0.23.0
    "plugins/platforms/mattermost/gateway_env.py": "exists",  # v0.22.7..v0.23.0: gateway/platform_env.py:196@v0.23.0
    "plugins/platforms/qqbot/gateway_env.py": "exists",  # v0.22.7..v0.23.0: gateway/platform_env.py:196@v0.23.0
    "plugins/platforms/signal/gateway_env.py": "exists",  # v0.22.7..v0.23.0: gateway/platform_env.py:196@v0.23.0
    "plugins/platforms/slack/gateway_env.py": "exists",  # v0.22.7..v0.23.0: gateway/platform_env.py:196@v0.23.0
    "plugins/platforms/sms/gateway_env.py": "exists",  # v0.22.7..v0.23.0: gateway/platform_env.py:196@v0.23.0
    "plugins/platforms/telegram/gateway_env.py": "exists",  # v0.22.7..v0.23.0: gateway/platform_env.py:196@v0.23.0
    "plugins/platforms/wecom/gateway_env.py": "exists",  # v0.22.7..v0.23.0: gateway/platform_env.py:196@v0.23.0
    "plugins/platforms/weixin/gateway_env.py": "exists",  # v0.22.7..v0.23.0: gateway/platform_env.py:196@v0.23.0
    "plugins/platforms/whatsapp/gateway_env.py": "exists",  # v0.22.7..v0.23.0: gateway/platform_env.py:196@v0.23.0
    "plugins/platforms/whatsapp_cloud/gateway_env.py": "exists",  # v0.22.7..v0.23.0: gateway/platform_env.py:196@v0.23.0
}

REQUIRED_TEXT: dict[str, str] = {
    # v0.22.7..v0.23.1: hermes_cli/update_downgrade_guard.py:67@v0.23.1 greps the target's root
    # hermes_state*.py for the ledger table and refuses the update as a downgrade when it is absent
    "hermes_state_common": "CREATE TABLE IF NOT EXISTS session_admissions",
}

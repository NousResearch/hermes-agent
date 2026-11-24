"""Files that keep stdlib ``re``: the one list rules HX018 and HX019 read (``config.RULES``).

A file is here because it runs a regex where the regex engine behind ``hermes_re`` (google-re2, a
third-party package) may not be importable: before dependencies are installed (launcher, package
manager, early recovery, update takeover), in an old updater process that calls new code after a
checkout swap, or in another interpreter (``scripts/``, skill helper scripts). The reason for each
entry is recorded with the migration plan; do not add or remove an entry without the owner.

Globs use ``fnmatch`` (``*`` also matches ``/``), as every ``Rule.exclude`` does. The body of
``RE_EXEMPT`` is generated from the plan's exemption list (one entry per line, sorted): replace
the lines between the two markers, nothing else.
"""

from __future__ import annotations

RE_EXEMPT: tuple[str, ...] = (
    # BEGIN exemption list
    "acp_adapter/edit_approval.py",
    "agent/credential_persistence.py",
    "agent/file_safety.py",
    "agent/i18n_layers.py",
    "agent/secret_scope.py",
    "agent/skill_utils.py",
    "gateway/config.py",
    "gateway/status.py",
    "gateway/status_inline_source.py",
    "hermes_cli/_early_recovery.py",
    "hermes_cli/_early_recovery_zip.py",
    "hermes_cli/_launchers.py",
    "hermes_cli/_parser.py",
    "hermes_cli/_subprocess_compat.py",
    "hermes_cli/_update_takeover.py",
    "hermes_cli/agent_plugins.py",
    "hermes_cli/config.py",
    "hermes_cli/config_backend.py",
    "hermes_cli/gateway.py",
    "hermes_cli/gateway_connections_ssh_programs.py",
    "hermes_cli/gateway_launchd_runtime.py",
    "hermes_cli/gateway_service_owner.py",
    "hermes_cli/gateway_windows.py",
    "hermes_cli/gitlock.py",
    "hermes_cli/linux_desktop_entry.py",
    "hermes_cli/observability/shared_metrics_update.py",
    "hermes_cli/plugins_manifest.py",
    "hermes_cli/profiles.py",
    "hermes_cli/profiles_export_scrub.py",
    "hermes_cli/release_channels.py",
    "hermes_cli/source_releases.py",
    "hermes_cli/update_channel.py",
    "hermes_cli/update_fleet_scope.py",
    "hermes_cli/update_lock.py",
    "hermes_cli/update_receipt.py",
    "hermes_cli/update_restart_recovery.py",
    "hermes_constants.py",
    "hermes_platform/*",
    "optional-skills/*",
    "pm/*",
    "process_identity.py",
    "scripts/*",
    "skills/*",
    "tools/bot_mode_dm.py",
    "tools/bot_mode_dm_coordination.py",
    "tools/computer_use/cua_backend_driver.py",
    "tools/computer_use/cua_backend_parse.py",
    "tools/environments/local_env_policy.py",
    "website/*",
    # END exemption list
)

# The individually named files (no glob): HX019 forbids ``hermes_re`` in exactly these and under pm/.
RE_EXEMPT_FILES: tuple[str, ...] = tuple(entry for entry in RE_EXEMPT if "*" not in entry)

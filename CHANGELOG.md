# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changes

- **fix(approval): honour `tirith_fail_open: false` on Tirith `ImportError` (#20733)**
  `check_all_command_guards` in `tools/approval.py` previously swallowed an
  `ImportError` from `tools.tirith_security` with an unconditional `pass`,
  leaving `tirith_result["action"]` as `"allow"` regardless of the configured
  `security.tirith_fail_open` policy.  When `tirith_fail_open` is `false` the
  operator has explicitly opted into fail-closed behaviour; a missing or broken
  Tirith module must not silently grant command execution.

  The fix reads the live security config inside the `except ImportError` handler
  and, when `tirith_enabled` is `true` and `tirith_fail_open` is `false`,
  synthesises a `"warn"`-action Tirith result.  This result flows through the
  existing approval path (prompting the user or blocking in cron/gateway
  contexts) instead of bypassing it entirely.  The default (`tirith_fail_open:
  true`) behaviour is unchanged.

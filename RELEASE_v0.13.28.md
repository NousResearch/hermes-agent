# Hermes Agent v0.13.28

## Android

- Adds saved Shizuku/Sui app-management automation records through
  `android_automation_tool` action `create_shizuku_action_task`.
- Saved records can run `grant_runtime_permission`,
  `revoke_runtime_permission`, `force_stop_app`, `enable_app`, `disable_app`,
  and `set_app_enabled` after the user starts Shizuku/Sui and grants Hermes
  permission.
- Supports Tasker-style `%NAME` and `{{NAME}}` expansion for saved Shizuku
  package names and permission names at run time.
- Keeps these actions distinct from safe Android system actions and regular
  shell tasks so scheduled records clearly advertise the Shizuku/Sui permission
  boundary.

## Validation

- Extends `HermesAutomationInstrumentedTest` with saved Shizuku action
  persistence, variable expansion, safe Shizuku-unavailable failure, and
  rejected unsafe definitions.

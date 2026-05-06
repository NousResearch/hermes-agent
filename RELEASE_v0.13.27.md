# Hermes Agent v0.13.27

## Android

- Adds explicit Shizuku/Sui-backed app and runtime-permission actions through
  `android_system_tool`.
- Supports `grant_runtime_permission`, `revoke_runtime_permission`,
  `force_stop_app`, `enable_app`, `disable_app`, and `set_app_enabled` when the
  user has started Shizuku/Sui and granted Hermes permission.
- Validates package and permission names before shell execution and refuses to
  disable or force-stop the Hermes app itself.
- Updates the native Gemma tool schema so local model tool-calling can choose
  the structured Shizuku actions instead of writing raw package-manager shell
  commands.

## Validation

- Android instrumented coverage checks status advertising, structured action
  validation, self-protection, Shizuku-unavailable handling, and the native chat
  tool loop routing these actions through `android_system_tool`.

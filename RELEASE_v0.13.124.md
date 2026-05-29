# Hermes Agent Fork v0.13.124

This release supersedes v0.13.123 with a chat-level native diagnostics route for
the all-features/self-test prompt shown in the latest phone screenshots.

## Android

- Routes explicit `android_device_diagnostics_tool` requests and implicit
  all-features/native bridge self-test prompts directly from chat to
  `HermesDeviceDiagnosticsBridge`, even when the selected provider is not in
  native-tool mode.
- Returns the bridge's real self-test output for terminal, system control,
  accessibility, Hindsight memory, and workspace file checks instead of waiting
  for a remote/provider response or inventing class-loading errors.
- Keeps the v0.13.123 fixes for custom instructions, local context recovery,
  adaptive launcher icons, keyboard layout, floating-action overlap, and chat
  text rendering.

## Release

- Publishes signed universal APK and AAB artifacts for the Hermes Agent Fork
  Android package `com.mobilefork.hermesagent`.
- Updates the F-Droid version template and changelog for versionCode `142490`.

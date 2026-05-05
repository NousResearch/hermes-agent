# Hermes Agent v0.13.16

This release hardens native Android agent startup and adds first-class setup
surfaces for Shizuku/Sui-assisted privileged Android actions.

## Android Agent

- Adds Shizuku/Sui status reporting and setup actions for developer options,
  wireless debugging, Shizuku launch/download, and Shizuku permission requests.
- Keeps background runtime service startup responsive by promoting foreground
  notification immediately and moving heavy runtime boot work off the service
  main thread.
- Delays automatic persisted-runtime restart during app process startup so the
  first activity and instrumentation launch do not collide with Python/native
  asset extraction.
- Fixes native tool-call chat loops so Hermes sends the post-tool follow-up
  request instead of returning the last raw tool result as the assistant reply.

## Native Android Runtime

- Fixes embedded Android Linux subprocess library resolution by adding the
  packaged native executable directory to `LD_LIBRARY_PATH`.
- Keeps Python subprocesses on `/system/bin/sh` while preserving Hermes Android
  shell metadata for native tools.

## Validation

- Adds emulator visual-harness commands for screenshots, taps, swipes, text,
  key events, launch, and viewport sizing.
- Adds Android instrumentation coverage for privileged-access status, native
  tool status JSON, and full connected startup stability.

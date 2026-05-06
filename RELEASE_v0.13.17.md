# Hermes Agent v0.13.17

This release expands the native Android agent control loop and release-debugging
workflow.

## Android Agent

- Exposes `android_ui_tool` to local model tool-calling so Hermes can inspect
  accessibility status, request visible-screen snapshots, click, type, scroll,
  and use Android global navigation actions after the user enables the Hermes
  accessibility service.
- Keeps privileged system control explicit and reviewable: protected Android
  settings still require a user-facing settings panel, Shizuku/Sui, the Hermes
  accessibility service, or external ADB during developer validation.
- Documents the Shizuku and Tasker capability map for Android agent
  work.

## Debugging And Release

- Expands the ADB visual harness with `click`, `dump-ui`, and `wide-capture`
  commands for wide emulator screenshots and UI inspection, including a focused
  window plus UI-text readiness wait for slow emulator cold starts.
- Hardens the F-Droid merge-request debugging skill with reviewer-response,
  rewritemeta, reproducible binary, and noninteractive GitHub credential steps.

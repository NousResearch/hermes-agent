# Hermes Agent v0.13.18

Android-focused maintenance release.

## Changes

- Add `run_privileged_shell` for user-granted Shizuku/Sui shell execution through the native Android system tool.
- Expose Shizuku privileged shell availability in Android status and native model tool-calling schemas.
- Add connected-test coverage for Shizuku shell permission-state reporting on devices without Shizuku.
- Make the Android visual harness tolerate transient cold-start ANR dialogs by tapping `Wait` during readiness polling.
- Document the Shizuku/Tasker capability map and the current Hermes support boundary.

## Validation

- Android debug APK build
- Android unit tests
- Android instrumentation tests for native tool access and Shizuku privileged status
- Emulator wide screenshot and UI dump through `scripts/android_visual_harness.py`

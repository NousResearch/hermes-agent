# Hermes Agent Fork v0.13.128

This release completes the Android agent verification fixes requested from the
latest Chinese advisor review screenshots and emulator validation pass.

## Android

- Adds the Hy Memory provider as the default memory stack for Hermes agents,
  with a bundled plugin, lazy dependency registration, default config, and
  provider initialization coverage.
- Adds an Android Hy Memory diagnostics bridge and instrumented test so the
  packaged app can prove the Python memory plugin is available on device.
- Hardens the Shizuku privileged-shell bridge with independently tagged user
  services, longer connection tolerance, user-service cleanup, and command
  result metadata.
- Improves the native Android Linux fallback state so the packaged agent still
  reports real package, architecture, shell, and manifest details without
  depending on Termux.
- Routes native diagnostics and self-test prompts through the real Kotlin
  bridge so chat replies reflect device state instead of provider guesswork.

## Validation

- Re-ran Python memory, guard, packaging, runtime, and device bridge tests.
- Re-ran Android debug unit tests and debug APK/androidTest assembly.
- Verified emulator chat, native tool flow, Shizuku permission/service state,
  Hy Memory availability, Android Linux package state, and local inference
  conversation flow on `emulator-5554`.

## Release

- Publishes signed universal APK and AAB artifacts for the Hermes Agent Fork
  Android package `com.mobilefork.hermesagent`.
- Updates the F-Droid version template and changelog for versionCode `142890`.

# Hermes Agent Fork v0.13.129

This release rolls forward the Android agent fixes from v0.13.128 and fixes the
F-Droid binary reproducibility issue found during validation.

## Android

- Canonicalizes Chaquopy generated build metadata and dist-info metadata with LF
  newlines before APK packaging.
- Canonicalizes Hermes Linux asset manifest and lock JSON writers with LF
  newlines on Windows and Linux build hosts.
- Keeps Hy Memory as the default Hermes memory stack with the packaged Android
  diagnostics bridge.
- Keeps the Shizuku privileged-shell bridge and native Android Linux bridge
  fixes validated in v0.13.128.

## Validation

- Re-ran focused Android packaging, Linux asset, and release workflow tests.
- Re-ran local F-Droid lint, update-check, and test-build validation for the new
  version after publishing.

## Release

- Publishes signed universal APK and AAB artifacts for package
  `com.mobilefork.hermesagent`.
- Updates the F-Droid template and changelog for versionCode `142990`.

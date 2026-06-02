# Hermes Agent Fork v0.13.130

This release completes the Android/F-Droid reproducibility fix by canonicalizing
both generated text newlines and Chaquopy `.imy` ZIP metadata.

## Android

- Canonicalizes Chaquopy `.imy` entries with a stable Unix ZIP creator system so
  Windows F-Droid rebuilds match Linux GitHub release builds.
- Keeps LF canonicalization for Chaquopy build metadata, dist-info metadata, and
  Hermes Linux asset manifests.
- Keeps Hy Memory as the default Hermes memory stack with the packaged Android
  diagnostics bridge.
- Keeps the Shizuku privileged-shell bridge, native Android Linux bridge, Qwen
  GGUF, and Gemma LiteRT local inference fixes validated in the prior release
  line.

## Validation

- Re-ran focused Android packaging, Linux asset, release workflow, memory, and
  device bridge tests.
- Re-ran local Gradle release assembly and inspected APK payload bytes.
- Re-ran local F-Droid lint, tag discovery, and test-build validation after
  publishing.

## Release

- Publishes signed universal APK and AAB artifacts for package
  `com.mobilefork.hermesagent`.
- Updates the F-Droid template and changelog for versionCode `143090`.

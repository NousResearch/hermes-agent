# Hermes Agent Fork v0.13.132

This release carries forward the v0.13.131 Android fixes and retags them after
restoring the GitHub Android smoke workflow contracts.

## Android

- Refreshes the pinned Termux Linux asset lock so F-Droid no longer downloads
  the removed `ca-certificates_1:2026.03.19_all.deb` package.
- Adds downloadable Linux sandbox catalog support for Alpine, Debian, Ubuntu,
  Fedora, Arch, Void, and openSUSE roots exposed to the Android agent.
- Improves XML tool-call and markdown display in the assistant page so raw tool
  envelopes render as readable tool-call blocks.
- Localizes the remaining MCP onboarding, model-download, navigation, and signal
  quick-action strings found in Chinese emulator screenshots.
- Adds simpler MCP setup controls for detect, auto fill, add draft server, auto
  setup, and test/refresh.
- Restores cached Linux-prefix reuse and native terminal direct-call contracts
  enforced by the GitHub Android smoke workflow.

## Validation

- Re-ran the GitHub Android Python smoke test group locally: `355 passed`.
- Re-ran full Android debug unit tests with Linux assets skipped.
- Re-ran Android Linux asset preparation from the updated Termux lockfile.
- Re-ran local release assembly and inspected APK metadata for versionCode
  `143290` and versionName `0.13.132`.
- Verified on emulator that `hermes-alpine` returns `x86_64` through
  `linux_sandbox_tool`, and that Chinese MCP/model-download settings are
  localized.

## Release

- Publishes signed universal APK and AAB artifacts for package
  `com.mobilefork.hermesagent`.
- Updates the F-Droid template and changelog for versionCode `143290`.

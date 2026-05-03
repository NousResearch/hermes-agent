# Hermes Agent v0.13.1

**Release Date:** May 3, 2026

Patch release for the native Android Gemma agent after live-device ADB validation on an arm64 phone.

## Fixes

- Rebased `feat/termux-install-path` on current upstream `NousResearch/hermes-agent/main`.
- Fixed LiteRT-LM artifact selection so Gemma 4 downloads prefer Android `.litertlm` bundles instead of `*-web.task` browser FlatBuffers.
- Added runtime preflight for stale or exact `.task` files that contain `TFL3` FlatBuffer headers, returning a clear message to remove the web task and download the `.litertlm` artifact.
- Updated Android backend compatibility checks so existing preferred `*-web.task` downloads are rejected before engine startup instead of surfacing `Unable to open zip archive`.
- Updated the Hermes Android debugging skill with the live-phone ADB diagnosis flow.

## Validation

- `python -m pytest tests/hermes_android/test_android_model_downloads.py tests/hermes_android/test_android_multimodal_model_selection.py -q`
- `./gradlew :app:testDebugUnitTest -PskipHermesAndroidLinuxAssets=true`
- `./gradlew :app:assembleDebug`
- Live phone ADB confirmed the failing `gemma-4-e2b-it-web.task` artifact starts with `TFL3` bytes and is not a LiteRT-LM bundle.

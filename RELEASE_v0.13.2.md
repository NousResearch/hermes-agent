# Hermes Agent v0.13.2

**Release Date:** May 3, 2026

Patch release for physical Android phones after live ADB install validation.

## Fixes

- Fixed Android release packaging so the published `universal.apk` is the real universal APK, not whichever split APK was newest on disk.
- Added a release-manifest regression test that catches split-only APK selection before GitHub uploads a phone-incompatible artifact.
- Preserves the v0.13.1 LiteRT-LM fix that rejects `*-web.task` browser FlatBuffers and asks for the Android `.litertlm` Gemma bundle.

## Validation

- `python -m pytest tests/hermes_android/test_android_release_alpha.py tests/hermes_android/test_android_model_downloads.py tests/hermes_android/test_android_multimodal_model_selection.py -q`
- `./gradlew :app:assembleRelease :app:bundleRelease`
- APK contents inspected for both `arm64-v8a` and `x86_64` native libraries before release.

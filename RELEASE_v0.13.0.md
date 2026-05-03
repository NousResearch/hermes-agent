# Hermes Agent v0.13.0 (v2026.5.3)

**Release Date:** May 3, 2026

> Native Android Gemma release. Hermes can now run the Android APK as a local Gemma-powered agent path with first-class Gemma 4/Gemma 3 model selection, image attachment plumbing, Python 3.12 Chaquopy builds, and validated native tool access on the emulator.

## Highlights

- **Gemma 4 local Android agent path** - Gemma 4 E2B and E4B LiteRT-LM artifacts load and answer locally through the in-app OpenAI-compatible proxy. Gemma 4 E2B also powers native chat/tool-call flows that create, delete, and inspect files through the embedded Android Linux command suite.
- **First-class Gemma model selection** - Settings now has a real model-selection dropdown card with provider suggestions plus local Gemma 4, Gemma 3, Gemma 3 vision, and Gemma 3n vision entries.
- **Multimodal request plumbing** - Chat can attach images, preserve image attachments in state, encode them as OpenAI-style `image_url` content parts, and pass multimodal payloads through API, SSE, and native tool-calling clients.
- **Vision-safe LiteRT-LM behavior** - Gemma 3/Gemma 3n vision models are represented as first-class local options. Text-only models now reject image requests with a clear 400 instead of silently ignoring images. The image-description instrumentation test runs automatically when a Gemma 3/Gemma 3n vision artifact is provisioned.
- **Python 3.12 Android runtime** - Chaquopy and GitHub Actions now build with Python 3.12, and Android CI/release jobs use JDK 21 for LiteRT-LM compatibility.
- **Deep UI visual coverage** - Added an emulator UI flow that captures chat typing, Settings selection, the model dropdown, and translated Spanish pages across Settings, Accounts, Device, and Portal.
- **Native Android tool coverage** - Added emulator tests for terminal tool file creation/deletion, Android system status, cron lifecycle, and background runtime start/stop.
- **Branding refresh** - Android launcher/topbar branding now uses the Nous/Hermes thumbnail asset.

## Validation

- `python -m pytest tests/hermes_android -q`
- `./gradlew :app:testDebugUnitTest -PskipHermesAndroidLinuxAssets=true`
- `./gradlew :app:installDebug :app:installDebugAndroidTest`
- `adb shell am instrument ... DeepAppUiVisualInstrumentedTest`
- `adb shell am instrument ... Gemma4LocalInferenceInstrumentedTest,NativeAppChatAndToolInstrumentedTest,NativeAppUiChatInstrumentedTest,NativeAgentToolAccessInstrumentedTest,LiteRtLmModelMatrixInstrumentedTest`
- Additional model matrix checks passed for Gemma 4 E4B, Qwen3 0.6B, and Qwen2.5 1.5B. Phi-4-mini was attempted under the 5 GB artifact ceiling but the emulator aborted inside the native LiteRT loader during initialization, so it is not marked supported in this release.

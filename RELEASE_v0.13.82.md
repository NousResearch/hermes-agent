# Hermes Agent v0.13.82

## Android local inference

- Enables Gemma 4 LiteRT-LM MTP/speculative decoding when the model bundle advertises support, and uses Edge Gallery defaults for Gemma 4 and Qwen local model presets.
- Fails explicit local backend startup cleanly instead of falling back to a stale remote/Python endpoint when LLAMA_CPP, LiteRT-LM, or AICore cannot start.
- Extends native Android tool-call replies to 4000 tokens and validates local HTTP endpoints before tool-chat requests.
- Keeps Tasker import and Shizuku automation checks isolated from model backend state during emulator validation.

## Validation

- `python -m pytest tests/hermes_android -q`
- `.\gradlew.bat :app:testDebugUnitTest -PskipHermesAndroidLinuxAssets=true --stacktrace`
- `.\gradlew.bat :app:assembleDebug :app:assembleDebugAndroidTest :app:assembleRelease :app:bundleRelease -PskipHermesAndroidLinuxAssets=true --stacktrace`
- Emulator instrumentation: `BootSmokeTest`, `NativeAgentRuntimeSmokeTest`, `HermesAutomationInstrumentedTest`
- Emulator Qwen GGUF instrumentation: skipped cleanly on x86 because embedded llama.cpp assets are unavailable there

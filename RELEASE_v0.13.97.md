# Hermes Agent v0.13.97

## Android

- Adds OpenRouter OAuth PKCE sign-in for Hermes Android. Users can approve Hermes in the browser and Hermes stores the returned OpenRouter API key through the existing secure provider credential path.
- Keeps Qwen, Z.AI, OpenAI, Gemini, Claude, ChatGPT Web, and legacy Qwen OAuth on explicit API-key/token setup paths so discontinued or provider-hosted OAuth pages do not block app setup.
- Preserves browser and provider setup fallbacks for devices where embedded provider pages stall.

## Validation

- `.\gradlew.bat :app:testDebugUnitTest -PskipHermesAndroidLinuxAssets=true --stacktrace`
- `.\gradlew.bat :app:assembleDebug :app:assembleDebugAndroidTest -PskipHermesAndroidLinuxAssets=true --stacktrace`
- Emulator `emulator-5556`: `DeepAppUiVisualInstrumentedTest#signinOpenRouterCommandOpensOpenRouterOAuthPage`
- Emulator `emulator-5556`: `ProviderSetupWebActivityInstrumentedTest`
- `python -m pytest tests/hermes_android -q`
- GitHub Actions Android run `25735439959`

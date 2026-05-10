# Hermes Agent v0.13.87

## Android auth and Gemma 4 runtime

- Validates the configured Corr3xt app sign-in `/oauth/start` route before launching the browser, while stripping query parameters so the probe does not trigger a real OAuth start.
- Keeps broken or non-Corr3xt app sign-in URLs inside Hermes with a clear HTTP error instead of sending the user to a dead browser page.
- Sets the Gemma 4 speculative decoding flag before creating the LiteRT-LM engine, matching the timing used by Google AI Edge Gallery for MTP-backed models.
- Adds JVM and emulator regression coverage for the broken OAuth-start route case, plus a static guard for the LiteRT-LM flag timing.

## Validation

- `python -m pytest tests/hermes_android -q`
- `.\gradlew.bat :app:testDebugUnitTest -PskipHermesAndroidLinuxAssets=true --stacktrace`
- `.\gradlew.bat "-Pandroid.testInstrumentationRunnerArguments.class=com.nousresearch.hermesagent.DeepAppUiVisualInstrumentedTest#corr3xtSignInRejectsReachableHostWithoutOAuthStartRoute" :app:connectedDebugAndroidTest -PskipHermesAndroidLinuxAssets=true --stacktrace`
- `.\gradlew.bat "-Pandroid.testInstrumentationRunnerArguments.class=com.nousresearch.hermesagent.DeepAppUiVisualInstrumentedTest#signinQwenCommandReloadsSettingsProviderProfile" :app:connectedDebugAndroidTest -PskipHermesAndroidLinuxAssets=true --stacktrace`

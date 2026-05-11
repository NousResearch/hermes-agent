# Hermes Agent v0.13.95

## Android

- Opens provider API-key setup pages inside Hermes with an app-owned WebView instead of depending only on an external browser activity.
- Keeps Corr3xt/OAuth sign-in on the external browser path while giving OpenRouter, Qwen Cloud, Z.AI, and other API-key providers a copy/open fallback when a page stalls.
- Updates Auth and Settings status text so users know setup pages opened inside Hermes and that official fallback URLs were copied.

## Validation

- `.\gradlew.bat :app:testDebugUnitTest --tests com.nousresearch.hermesagent.device.HermesProviderSetupWebActivityTest -PskipHermesAndroidLinuxAssets=true --stacktrace`
- `.\gradlew.bat :app:testDebugUnitTest -PskipHermesAndroidLinuxAssets=true --stacktrace`
- `.\gradlew.bat :app:assembleDebug -PskipHermesAndroidLinuxAssets=true --stacktrace`
- Emulator smoke on `emulator-5556`: installed `0.13.95` debug APK, opened the Accounts OpenRouter setup action, and verified focus stayed on `com.nousresearch.hermesagent/.device.HermesProviderSetupWebActivity` with Back, Browser, Copy, Close, and WebView controls visible instead of handing off to Chrome.

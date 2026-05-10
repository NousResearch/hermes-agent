# Hermes Agent v0.13.84

## Android account setup

- Keeps Corr3xt app sign-in unconfigured on first run instead of preloading the unreachable `auth.corr3xt.com` host.
- Disables Email, Google, and Phone app sign-in buttons until the user saves a reachable Corr3xt auth URL.
- Routes `/signin` provider commands such as OpenRouter, ChatGPT, Claude, Gemini, Qwen, and Z.AI to Settings API-key setup instead of implying provider OAuth is available.
- Keeps Android automation terminology aligned with Tasker and Shizuku/Sui.

## Validation

- `python -m pytest tests/hermes_android -q`
- `.\gradlew.bat :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest -PskipHermesAndroidLinuxAssets=true --stacktrace`
- Emulator first-run Accounts UI smoke with blank Corr3xt URL and disabled app-account sign-in buttons

# Hermes Agent v0.13.85

## Android provider setup

- Hides the advanced Qwen OAuth bridge from Android Settings so phone users are guided to the tested DashScope API-key provider path.
- Resets the base URL and model hint when switching between built-in remote providers, preventing stale OpenRouter values from sticking after selecting Qwen, Z.AI, or another provider.
- Keeps Android automation terminology aligned with Tasker and Shizuku/Sui.

## Validation

- `python -m pytest tests/hermes_android -q`
- `.\gradlew.bat :app:testDebugUnitTest -PskipHermesAndroidLinuxAssets=true --stacktrace`

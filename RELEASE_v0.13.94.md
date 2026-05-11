# Hermes Agent v0.13.94

## Android

- Adds OpenGUI-inspired visual screenshot state metadata to `android_ui_tool` screenshots: original screen size, scaled image size, scale factor, and a 64-bit visual hash.
- Feeds screenshot visual hashes into Hermes' existing OpenGUI-compatible action history so repeated-screen and UI-loop review can replan on sparse accessibility trees.
- Exposes visual screenshot hash support in Android UI tool status and tool schema.

## Validation

- `.\gradlew.bat :app:testDebugUnitTest --tests com.nousresearch.hermesagent.device.HermesAccessibilityUiBridgeTest -PskipHermesAndroidLinuxAssets=true --stacktrace`
- `.\gradlew.bat :app:testDebugUnitTest --tests "com.nousresearch.hermesagent.ui.chat.OpenGui*" --tests "com.nousresearch.hermesagent.device.HermesAutomationBridgeOpenGuiCompatTest" -PskipHermesAndroidLinuxAssets=true --stacktrace`
- `.\gradlew.bat :app:testDebugUnitTest -PskipHermesAndroidLinuxAssets=true --stacktrace`
- `git diff --check`

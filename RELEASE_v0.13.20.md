# Hermes Agent v0.13.20

This Android-focused release extends Hermes Agent's native Tasker-style
automation support.

## Android

- Added durable automation variables exposed through `android_automation_tool`.
- Added `%NAME` and `{{NAME}}` variable expansion for saved shell automations.
- Added boot, power-connected, power-disconnected, battery-low, and
  battery-okay automation triggers.
- Added `run_trigger`, `list_variables`, `set_variable`, `get_variable`, and
  `delete_variable` automation actions.
- Updated the settings tool profile and Android capability map for variables
  and phone-state triggers.

## Validation

- `python -m pytest tests\hermes_android -q`
- `.\gradlew.bat :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest -PskipHermesAndroidLinuxAssets=true --stacktrace`
- `.\gradlew.bat "-Pandroid.testInstrumentationRunnerArguments.class=com.nousresearch.hermesagent.HermesAutomationInstrumentedTest" :app:connectedDebugAndroidTest -PskipHermesAndroidLinuxAssets=true --stacktrace`

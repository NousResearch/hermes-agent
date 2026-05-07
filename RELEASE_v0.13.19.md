# Hermes Agent v0.13.19

This Android-focused release adds the first native Tasker-style automation
primitive for Hermes Agent.

## Android

- Added saved Android shell automations with manual run, list, enable, disable,
  and delete actions.
- Added interval scheduling through Android alarms with a 15-minute minimum
  interval.
- Added boot rescheduling for enabled interval automations.
- Exposed the automation layer to local model tool-calling through
  `android_automation_tool`.
- Kept privileged execution explicit: Shizuku/Sui shell automation requires the
  saved task to opt in with `use_shizuku` and still requires user-granted
  Shizuku/Sui permission.
- Updated the Android capability map to use Tasker as the automation parity
  target and documented the remaining Tasker parity gaps.

## Validation

- `python -m pytest tests\hermes_android -q`
- `.\gradlew.bat :app:testDebugUnitTest -PskipHermesAndroidLinuxAssets=true --stacktrace`
- `.\gradlew.bat :app:assembleDebug :app:assembleDebugAndroidTest -PskipHermesAndroidLinuxAssets=true --stacktrace`
- `.\gradlew.bat "-Pandroid.testInstrumentationRunnerArguments.class=com.nousresearch.hermesagent.HermesAutomationInstrumentedTest,com.nousresearch.hermesagent.NativeAgentToolAccessInstrumentedTest" :app:connectedDebugAndroidTest -PskipHermesAndroidLinuxAssets=true --stacktrace`

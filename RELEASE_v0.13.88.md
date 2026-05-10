# Hermes Agent v0.13.88

## Android browser automation

- Adds `CATEGORY_BROWSABLE` automatically to saved Android `open_uri` intent tasks so generated browser work opens through normal browser routing.
- Adds emulator regression coverage for creating and running a saved `open_uri` automation task.
- Validates local Gemma 4 E2B LiteRT-LM inference and Qwen 3.5 0.8B GGUF tool use on the API 35 emulator with the full embedded Linux runtime.

## Operator standby parity

- Adds an OpenGUI-inspired, clean-room automation run ledger for recent Hermes Android automation runs.
- Exposes `operator_standby_status` and `run_history` through the native automation bridge so external dispatch, Tasker plugin dispatch, widgets, shortcuts, and quick settings can report structured state.
- Adds a Device-page standby card showing enabled automations, external-trigger coverage, recent run count, and the latest result.

## Validation

- `python -m pytest tests/hermes_android -q`
- `.\gradlew.bat :app:testDebugUnitTest -PskipHermesAndroidLinuxAssets=true --stacktrace`
- `.\gradlew.bat "-Pandroid.testInstrumentationRunnerArguments.class=com.nousresearch.hermesagent.HermesAutomationInstrumentedTest#shellAutomationCanBeCreatedRunDisabledAndDeleted" :app:connectedDebugAndroidTest --stacktrace`
- `.\gradlew.bat "-Pandroid.testInstrumentationRunnerArguments.class=com.nousresearch.hermesagent.HermesAutomationInstrumentedTest#intentAutomationCanOpenBrowserUriWhenBrowserIsAvailable" :app:connectedDebugAndroidTest --stacktrace`
- `adb shell am instrument -w -r -e class com.nousresearch.hermesagent.Gemma4LocalInferenceInstrumentedTest com.nousresearch.hermesagent.test/androidx.test.runner.AndroidJUnitRunner`
- `adb shell am instrument -w -r -e class com.nousresearch.hermesagent.NativeAppChatAndToolInstrumentedTest com.nousresearch.hermesagent.test/androidx.test.runner.AndroidJUnitRunner`

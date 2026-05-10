# Hermes Agent v0.13.89

## Android remote dispatch parity

- Adds OpenGUI-style standby dispatch compatibility to `android_automation_tool` through `run_remote_dispatch` / `submit_standby_dispatch`.
- Accepts `executionId`, `taskId`, and `taskName` payloads, matches enabled Hermes automations by id or label, and runs `remote_dispatch` trigger records when no explicit target is supplied.
- Records dispatch source, channel, execution id, task id, and task name in Android automation run history.
- Exposes `%DISPATCH_SOURCE`, `%DISPATCH_CHANNEL`, `%DISPATCH_EXECUTION_ID`, `%DISPATCH_TASK_ID`, and `%DISPATCH_TASK_NAME` for dispatched automations.
- Updates the Device page standby card with remote dispatch counts and the last remote dispatch source/task.

## Android runtime hardening

- Allows Corr3xt OAuth start probes to fall back to the full query-bearing start URL when a backend requires query parameters.
- Adds browser-routable intent categories for account sign-in and provider key-page launch.
- Retries Gemma 4 LiteRT-LM engine startup without MTP/speculative decoding if MTP initialization fails on a backend.

## Validation

- `python -m pytest tests\hermes_android\test_android_followup_polish.py tests\hermes_android\test_android_auth_ui.py -q`
- `git diff --check`
- `PYTHON_FOR_BUILD=<CPython 3.13> .\gradlew.bat :app:testDebugUnitTest -PskipHermesAndroidLinuxAssets=true --stacktrace`
- `PYTHON_FOR_BUILD=<CPython 3.13> .\gradlew.bat "-Pandroid.testInstrumentationRunnerArguments.class=com.nousresearch.hermesagent.HermesAutomationInstrumentedTest#openGuiStyleRemoteDispatchRunsMatchingEnabledAutomation" -PskipHermesAndroidLinuxAssets=true :app:connectedDebugAndroidTest --stacktrace`

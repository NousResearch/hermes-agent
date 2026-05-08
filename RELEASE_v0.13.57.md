# Hermes Agent v0.13.57

## Android Tasker/Shizuku parity

- Added user-granted Shizuku/Sui connectivity toggles for Wi-Fi, Bluetooth, mobile data, and airplane mode.
- Exposed toggles through `android_system_tool` and saved `create_shizuku_action_task` automation records.
- Kept the privilege boundary: these actions only run after the user starts Shizuku/Sui and grants Hermes access.

## Validation

- `:app:testDebugUnitTest --tests "com.nousresearch.hermesagent.device.HermesAutomationStoreTest.bridgeCreatesShizukuConnectivityToggleRecordsWithoutPackageName" --stacktrace`
- `:app:testDebugUnitTest --tests "com.nousresearch.hermesagent.device.HermesAutomationStoreTest" --stacktrace`
- `:app:compileDebugKotlin :app:compileReleaseKotlin :app:assembleRelease --stacktrace`

# Hermes Agent v0.13.61

This Android release adds another safe Tasker XML import mapping for Shizuku-backed device controls.

## Android

- Imported Tasker XML Bluetooth actions as disabled-by-default Hermes Shizuku automation records.
- Reused the existing `set_bluetooth_enabled` Shizuku action path, so execution still requires user-started Shizuku/Sui and Hermes permission.
- Updated the Android capability map to include Bluetooth in the safe Tasker XML import subset.

## Validation

- Focused JVM coverage for Tasker XML fixed Shizuku action import, including Bluetooth.
- Full `HermesAutomationStoreTest` JVM coverage for automation import and saved-action behavior.

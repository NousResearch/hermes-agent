# Hermes Agent v0.13.63

This Android release adds one more safe Tasker XML import for Shizuku-backed device controls.

## Android

- Imported Tasker XML Power Mode actions with explicit on/off states as disabled-by-default Hermes Shizuku automation records.
- Reused the existing `set_power_save_mode` Shizuku action path, so execution still requires user-started Shizuku/Sui and Hermes permission.
- Kept ambiguous Tasker Power Mode payloads unsupported rather than guessing at battery-saver state.
- Updated the Android capability map to include Power Mode in the safe Tasker XML import subset.

## Validation

- Focused JVM coverage for Tasker XML fixed Shizuku action import, including Power Mode.
- Full `HermesAutomationStoreTest` JVM coverage for automation import and saved-action behavior.

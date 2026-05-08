# Hermes Agent v0.13.59

This Android release extends the user-granted Shizuku/Sui Tasker parity lane.

## Android

- Added Shizuku-backed Tasker fixed actions for DND mode, power saver, screen off, end call, global Back/Home/Recents/notification/Quick Settings/statusbar actions, mobile network type bitmask changes, and user/work-profile start, stop, and switch commands.
- Exposed the new actions through both direct `android_system_tool` calls and saved `android_automation_tool create_shizuku_action_task` records.
- Kept the same permission boundary: these actions only run after the user starts Shizuku/Sui and grants Hermes Agent access.

## Validation

- Focused JVM coverage for saved and direct Shizuku command mapping.
- Android release build and emulator install validation are expected for the tagged release workflow.

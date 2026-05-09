# Hermes Agent v0.13.73

## Android

- Adds a foreground-service-backed calendar watcher for saved Tasker-style
  calendar-event automations.
- `android_automation_tool` now exposes `calendar_watcher_status`,
  `start_calendar_watcher`, `stop_calendar_watcher`, `scan_calendar_events`,
  and `reset_calendar_watcher_cursor`.
- The watcher scans Android Calendar provider instances only after the user
  grants calendar access, persists its scan interval and lookahead/lookback
  windows, dedupes recently seen provider events, restarts after app recreation
  or boot, and feeds matches through the existing `run_calendar_event_trigger`
  path.
- Local tool-calling prompts now advertise the calendar watcher actions so
  local Gemma-class models can choose the provider-backed calendar path instead
  of only explicit calendar dispatch.

## Validation

- `:app:testDebugUnitTest`
- `:app:assembleDebug`
- Emulator install/launch smoke on `emulator-5582` with `versionName=0.13.73`
  and `versionCode=137390`
- Android visual harness wide screenshot and UIAutomator dump for the Hermes
  chat screen after launch

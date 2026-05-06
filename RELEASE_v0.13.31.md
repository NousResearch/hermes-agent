# Hermes Agent v0.13.31

## Android Tasker calendar automation

- Adds saved Tasker-style `calendar_event` trigger records through
  `android_automation_tool`.
- Supports `run_calendar_event_trigger` for explicit local event dispatch with
  calendar name, title, description, and location values.
- Supports calendar-event filters for saved records:
  `calendar_name`, `title_contains`, `description_contains`, and
  `location_contains`.
- Exposes calendar event variables while matching records run:
  `%CALNAME`, `%CALTITLE`, `%CALDESCR`, `%CALLOC`, and `CALENDAR_*` aliases.
- Keeps the boundary honest: this release adds explicit event dispatch and
  matching, not background calendar-provider scanning.

## Validation

- `:app:testDebugUnitTest`
- `python -m pytest tests\hermes_android -q`
- `:app:assembleDebug :app:assembleDebugAndroidTest`
- Focused emulator instrumentation: `HermesAutomationInstrumentedTest`

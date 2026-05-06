# Hermes Agent v0.13.26

## Android

- Adds Tasker-style saved time-of-day triggers for Android automations.
- Supports optional day-of-week restrictions such as `weekday`, `weekend`, or
  comma-separated days like `mon,wed,fri`.
- Schedules time triggers through Android alarms and reschedules the next
  occurrence after each run.
- Exposes `%TIME`, `%TIME_HOUR`, `%TIME_MINUTE`, and `%TIME_DAY` variables while
  time-triggered tasks run.
- Updates the native Gemma tool schema so local model tool-calling can create
  time/day automations directly with `android_automation_tool`.

## Validation

- Android unit coverage now checks time trigger parsing, day normalization,
  durable storage, trigger execution, and next alarm computation.
- Android instrumented coverage now verifies a time-triggered file automation
  runs in the real app workspace and expands time variables.

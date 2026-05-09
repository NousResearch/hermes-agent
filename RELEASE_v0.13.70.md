# Hermes Agent v0.13.70

## Android

- Adds a Locale-compatible Tasker condition plugin so Tasker profiles can use
  Hermes and Shizuku state as profile conditions.
- Supports condition checks for Shizuku availability, saved automation
  enabled/disabled state, last-run success/failure, and saved Hermes variable
  set/equality state.
- Protects the exported condition query receiver with a per-configuration token
  so third-party apps cannot probe configured Hermes condition state by guessing
  automation IDs or variable names.
- Returns Tasker-local condition variables such as `%hermes_satisfied`,
  `%hermes_condition`, `%hermes_variable_value`, and Shizuku status fields when
  Tasker supports plugin variable returns.

# Hermes Agent v0.13.24

## Android

- Adds Tasker-style saved app-foreground triggers for Android automations.
- Stores an explicit `trigger_package_name` on automation records so the
  profile condition is separate from the task action payload.
- Uses the user-enabled Hermes accessibility service to observe foreground
  package changes and fire matching saved automations once per package change.

## Validation

- Android unit and instrumented coverage now includes app-foreground trigger
  creation, package matching/missing behavior, and real app-workspace task
  execution from a matching foreground package event.

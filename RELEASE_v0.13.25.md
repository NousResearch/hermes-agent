# Hermes Agent v0.13.25

## Android

- Adds Tasker-style saved notification-posted triggers for Android automations.
- Registers a user-enabled Hermes notification listener service that observes
  posted notification package, title, and text without bypassing Android
  notification-access consent.
- Exposes `run_notification_posted_trigger` through `android_automation_tool`
  for validation and model-created automations, with `%NOTIFICATION_PACKAGE`,
  `%NOTIFICATION_TITLE`, and `%NOTIFICATION_TEXT` available to saved tasks.
- Adds `open_notification_listener_settings` to the safe Android system actions
  so the local model can guide the user to the required permission screen.

## Validation

- Android unit and instrumented coverage now includes notification trigger
  creation, package matching/missing behavior, notification event variable
  expansion, and real app-workspace task execution from a matching notification
  event.

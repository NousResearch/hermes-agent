## Android Tasker-Style Notification Buttons

This Android-focused release adds safe Tasker-style notification buttons to
Hermes Agent automation.

- Adds up to three saved notification action buttons on
  `create_notification_task` records.
- Supports safe button actions to open Hermes, dismiss the notification, or run
  an existing saved Hermes automation by id.
- Keeps notification buttons permission-gated by Android notification
  permission and does not add arbitrary code execution or silent system setting
  changes.
- Expands Hermes automation variables inside button titles and target
  automation ids before posting the notification.
- Routes notification button clicks through the existing non-exported Hermes
  automation receiver.
- Adds emulator instrumentation coverage proving a notification button can run a
  saved file-write automation.

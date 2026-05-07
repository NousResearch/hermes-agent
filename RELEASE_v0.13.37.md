## Android Tasker Notification Automation

This Android-focused release adds another Tasker-style automation slice for
Hermes Agent.

- Adds saved `create_notification_task` records through `android_automation_tool`.
- Supports notification post/update/cancel actions with title, text, stable id,
  optional tag, channel id/name, priority, importance, group key, group summary,
  ongoing, auto-cancel, only-alert-once, and show-when fields.
- Expands saved Hermes variables in notification ids, tags, text, channel, group,
  priority, and importance fields at run time.
- Handles Android 13+ `POST_NOTIFICATIONS` permission explicitly, returning a
  structured permission result instead of silently failing.
- Keeps Tasker/Shizuku parity notes accurate: custom buttons, remote-input reply,
  progress/media notification styles, and deeper provider observers remain future
  work.

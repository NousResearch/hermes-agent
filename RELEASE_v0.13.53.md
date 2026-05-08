## Shizuku Logcat Watcher Cursor

This release tightens the Tasker Logcat Entry parity path for Hermes Android.

- Adds a bounded scan cursor for Shizuku-backed `start_logcat_watcher` polling
  so repeated scans do not rerun the same recently seen logcat events.
- Adds `reset_logcat_watcher_cursor` plus `reset_cursor` arguments for
  explicit recovery when the user wants to replay recent logcat lines.
- Exposes cursor status in `logcat_watcher_status` and `scan_logcat_entries`.
- Keeps logcat access behind the existing Shizuku/Sui user grant and enabled
  `logcat_entry` automation requirement.

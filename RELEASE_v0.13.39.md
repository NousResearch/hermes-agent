## Android Shizuku Logcat Watcher

This Android-focused release adds a Shizuku-gated Logcat Entry watcher slice
for Hermes Agent automation.

- Adds `logcat_watcher_status`, `scan_logcat_entries`,
  `start_logcat_watcher`, and `stop_logcat_watcher` through
  `android_automation_tool`.
- Uses the existing saved `logcat_entry` automation filters and
  `run_logcat_entry_trigger` dispatcher, so matched log events still expose
  `%LOGCAT_TAG`, `%LOGCAT_MESSAGE`, `%LOGCAT_LEVEL`, `%LOGCAT_PID`, and
  `%LOGCAT_TIME` to saved actions.
- Requires Shizuku/Sui to be running and Hermes to have user-granted Shizuku
  permission before scanning or starting the watcher.
- Runs the watcher through a persistent foreground service when Android allows
  it, keeps the requested watcher settings in app preferences, and restarts the
  worker after app-process recreation or boot while still waiting for
  user-granted Shizuku access.
- Reads Logcat with UID metadata and maps UIDs back to package names through
  the Shizuku shell so historical scans can satisfy Tasker-style package
  filters when Android exposes the UID.
- Keeps scans bounded with clamped `max_lines` and `scan_interval_seconds`
  settings.
- Adds JVM parser/gating coverage and emulator coverage for the Shizuku-gated
  watcher start/stop path.

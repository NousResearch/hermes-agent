# Hermes Agent v0.13.65

## Android

- Expands safe Tasker XML import for settings-panel actions such as device info,
  account, APN, date, location, input method, sync, display, locale, app
  management, security, sound, privacy, print, notification, and power usage
  settings.
- Keeps imported Tasker settings-panel records disabled by default and routes
  protected setting mutations through the existing Shizuku/Sui path only when a
  supported Hermes action exists.
- Updates the Android capability map and JVM importer coverage for the broader
  Tasker settings-panel subset.

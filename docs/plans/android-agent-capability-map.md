# Android Agent Capability Map

This note maps Shizuku and Tasker-style Android automation features to Hermes
Agent Android implementation work.

## Source Findings

- Shizuku is an Android app plus API that lets normal apps use selected Android
  system APIs with ADB shell or root identity after user setup and permission.
  Its official repo describes it as a middle layer that forwards app requests to
  system services through a server process started with ADB or root. Source:
  https://github.com/RikkaApps/Shizuku
- Shizuku-API requires the user to install Shizuku or Sui first. Android 11+
  users can start Shizuku from on-device wireless debugging; older non-rooted
  devices need a computer. The API exposes binder state, permission checks, UID
  checks, remote binder calls, and user services. Setup source:
  https://shizuku.rikka.app/guide/setup/
- ADB shell can start activities, run package-manager commands, issue input
  commands, take screenshots with `screencap`, and record video with
  `screenrecord`.
- Tasker is a closed-source Android automation app. Its official user guide
  centers the app around four building blocks: Profiles, Tasks, Scenes, and
  Variables. Sources: https://tasker.joaoapps.com/userguide_summary.html and
  https://tasker.joaoapps.com/userguide/en/
- Tasker actions cover app launch/listing, file listing/image loading, variable
  manipulation, plugins, Java/JavaScript execution, UI scenes, system tests, and
  many Android setting/device actions, with permissions or root/Shizuku required
  for restricted operations. Source:
  https://tasker.joaoapps.com/userguide/en/help/ah_index.html
- Tasker v6.6.18 added Shizuku integration, Shizuku-backed shell execution,
  permission management, a Shizuku state, a Check Shizuku function, arbitrary
  Java Code action support, import-from-clipboard support, extra trigger apps,
  and Shizuku adoption for actions such as Airplane Mode, Wi-Fi, Bluetooth,
  Kill App, Mobile Data, End Call, Custom Setting, global accessibility actions,
  Wi-Fi tethering, and Logcat Entry. Source:
  https://tasker.joaoapps.com/changes/changes6.6.html
- Tasker App Factory can export Tasker tasks or projects as Android apps, with
  linked resources included. Source:
  https://tasker.joaoapps.com/userguide/en/appcreation.html

## Hermes Support Map

| Capability | Hermes Android status |
|---|---|
| Local chat and tool calls | Present through LiteRT-LM/GGUF runtime paths and `NativeToolCallingChatClient`. |
| File creation/deletion/read/write | Present through native tool calls and app-workspace shell/file tools. |
| Android status and safe settings panels | Present through `HermesSystemControlBridge`. |
| Accessibility-based UI snapshot/action | Present through `HermesAccessibilityUiBridge` when the user enables the accessibility service; exposed to model tool-calling through `android_ui_tool`. The same service observes foreground app package changes for saved app-context triggers. |
| Shizuku/Sui privileged setup | Present: `HermesPrivilegedAccessBridge` detects Shizuku/Sui, binder state, permission state, UID/root-vs-ADB identity, and exposes setup actions. |
| Shizuku/Sui privileged shell execution | Present through `android_system_tool` action `run_privileged_shell` after the user starts Shizuku/Sui and grants Hermes permission. |
| Shizuku/Sui app and permission management | Present through explicit `android_system_tool` actions `grant_runtime_permission`, `revoke_runtime_permission`, `force_stop_app`, `enable_app`, `disable_app`, and `set_app_enabled`, and through saved `android_automation_tool` `create_shizuku_action_task` records. These validate package and permission names, refuse to disable or force-stop Hermes itself, expand saved variables at run time, and only execute through the user-granted Shizuku/Sui shell bridge. |
| Wireless debugging and developer options setup | Added as safe system actions: `open_wireless_debugging_settings` and `open_developer_options`. |
| Emulator/BlueStacks visual validation | Added host harness: `scripts/android_visual_harness.py` for ADB screenshots, taps/clicks, swipes, text input, UI dumps, launch, and one-command wide resolution capture. |
| Tasker-style manual tasks/actions | Present for saved shell, file-write, file-delete, safe Android system-action, accessibility UI-action, app-launch, Android intent, and Shizuku/Sui package-permission records through `android_automation_tool`; chat-triggered terminal, file, UI, and Android system tool calls remain available. Saved intent records cover explicit activity start, URI open, and package-targeted broadcast intents. |
| Tasker-style profiles/triggers | Present for manual, interval, time-of-day with optional day-of-week restrictions, boot, power-connected, power-disconnected, battery-low, battery-okay, app-foreground, notification-posted, explicit calendar-event, explicit location, explicit sensor, and Shizuku-available/Shizuku-unavailable saved automation tasks. Time triggers expose `%TIME`, `%TIME_HOUR`, `%TIME_MINUTE`, and `%TIME_DAY` while running. Calendar-event triggers can filter on calendar name, title, description, or location, and expose `%CALNAME`, `%CALTITLE`, `%CALDESCR`, `%CALLOC`, plus `CALENDAR_*` aliases while running. Location triggers can filter on latitude, longitude, radius, provider, place/name, and maximum accuracy, and expose `%LOC`, `%LAT`, `%LON`, `%LOCACC`, `%LOCPROVIDER`, `%LOCNAME`, plus `LOCATION_*` aliases while running. Sensor triggers can filter on sensor type/name, event, value name/axis, and min/max value, and expose `%SENSOR`, `%SENSOR_TYPE`, `%SENSOR_NAME`, `%SENSOR_EVENT`, `%SENSOR_VALUE`, `%SENSOR_VALUE_NAME`, `%SENSOR_UNIT`, and `%SENSOR_ACCURACY`. Shizuku-state triggers expose `%SHIZUKU_AVAILABLE`, `%SHIZUKU_INSTALLED`, `%SUI_INSTALLED`, `%SHIZUKU_RUNNING`, `%SHIZUKU_PERMISSION_GRANTED`, `%SHIZUKU_PRIVILEGE_LABEL`, and `%SHIZUKU_UID`. App-foreground triggers require the user-enabled Hermes accessibility service and an explicit `trigger_package_name`; notification-posted triggers require user-enabled Hermes notification access and an explicit `trigger_package_name`; location, calendar-event, and sensor triggers are explicit dispatch APIs, not background provider observers. Deeper app-state, provider-backed sensor/calendar/location observers, and plugin-profile triggers are not yet present. |
| Tasker-style variables | Present as a durable Android automation variable table exposed through `android_automation_tool`; shell commands, file paths, file content, system-action names, UI selectors, UI text values, app package names, Android intent action/data/package/component/category/extra string values, trigger filters, calendar event fields, sensor fields, Shizuku state fields, Shizuku package names, and Shizuku permission names can expand `%NAME` or `{{NAME}}` values at run time. |
| Tasker-style scenes/widgets | Not yet present as user-created Android scenes, overlays, widgets, or launcher shortcuts. Hermes has its fixed app UI and accessibility control of other apps. |
| Tasker plugin model | Not yet present. Hermes has model tool schemas, not Android Locale/Tasker plugin integration. |
| Tasker Java/JavaScript code action | Partially present through app-workspace shell/Python and Shizuku shell. Arbitrary in-app Java execution is not exposed and should stay permission-gated. |
| Tasker XML/Data URI import | Not yet present. |
| Tasker App Factory app export | Not present and out of scope for F-Droid Hermes APK review unless explicitly designed as a separate export feature. |

## Implementation Boundary

Hermes should not silently grant itself protected Android permissions. Protected
device actions must either:

- open a user-facing Android settings panel,
- use the user-enabled accessibility service,
- use Shizuku/Sui after the user starts the service and grants Hermes access, or
- run through explicit external ADB during developer validation.

This keeps the app reviewable for F-Droid while still making advanced phone
control available to consenting users.

## Next Tasker-Parity Work

The next credible Tasker-parity feature after shell/file/system/UI/app-launch,
saved Android intent actions, saved Shizuku package/permission actions,
variables, time/day triggers, basic phone-state triggers, app-foreground
triggers, notification-posted triggers, explicit calendar-event triggers,
explicit location triggers, explicit sensor triggers, and Shizuku-state triggers should extend profile
coverage with explicit user-visible records:

- deeper app-state, provider-backed sensor/location/calendar observers, and
  plugin-profile triggers,
- tests proving scheduled/manual actions cannot mutate outside the Hermes app
  workspace unless Shizuku is explicitly selected and available.

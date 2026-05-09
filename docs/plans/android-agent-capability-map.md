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
  checks, remote binder calls, and user services. Sources:
  https://shizuku.rikka.app/guide/setup/ and
  https://github.com/RikkaApps/Shizuku-API
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
- Tasker XML exports identify actions by numeric codes. Hermes uses a
  community-maintained code map only for a conservative import subset and
  refuses unsupported or high-risk actions. Source:
  https://github.com/Taskomater/Tasker-XML-Info/blob/master/Tasker_XML_Codes.md

## Hermes Support Map

| Capability | Hermes Android status |
|---|---|
| Local chat and tool calls | Present through LiteRT-LM/GGUF runtime paths and `NativeToolCallingChatClient`. |
| File creation/deletion/read/write | Present through native tool calls and app-workspace shell/file tools. |
| Android status and safe settings panels | Present through `HermesSystemControlBridge`. |
| Accessibility-based UI snapshot/action | Present through `HermesAccessibilityUiBridge` when the user enables the accessibility service; exposed to model tool-calling through `android_ui_tool`. The same service observes foreground app package changes for saved app-context triggers. |
| Shizuku/Sui privileged setup | Present: `HermesPrivilegedAccessBridge` detects Shizuku/Sui, binder state, permission state, UID/root-vs-ADB identity, and exposes setup actions. |
| Shizuku/Sui privileged shell execution | Present through `android_system_tool` action `run_privileged_shell` after the user starts Shizuku/Sui and grants Hermes permission. |
| Shizuku/Sui app, permission, setting, tethering, connectivity, and Tasker fixed-action management | Present through explicit `android_system_tool` actions `grant_runtime_permission`, `revoke_runtime_permission`, `force_stop_app`, `clear_app_data`, `enable_app`, `disable_app`, `set_app_enabled`, Wi-Fi/Bluetooth/mobile-data toggles, airplane-mode toggles, Wi-Fi tethering toggles, DND mode, power saver, screen-off, end-call, global navigation/statusbar actions, mobile network type bitmask changes, user/work-profile actions, and Tasker-style custom Android settings get/set/delete, and through saved `android_automation_tool` `create_shizuku_action_task` records. These validate package, permission, settings, DND mode, mobile-network bitmask, user/profile id, and slot id where required, refuse to disable, force-stop, or clear Hermes itself, expand saved variables at run time for persisted fields, and only execute through the user-granted Shizuku/Sui shell bridge. |
| Wireless debugging and developer options setup | Added as safe system actions: `open_wireless_debugging_settings` and `open_developer_options`. |
| Emulator/BlueStacks visual validation | Added host harnesses: `scripts/android_visual_harness.py` for UI dumps and ADB wide captures, `scripts/android-emulator-visual-probe.ps1` for Windows ADB status checks, launch, framebuffer screenshots, wide 1920x1080 screenshots, taps/clicks, swipes, text input, and key events on the Android SDK emulator or any ADB-visible emulator/phone, and `scripts/windows-visual-control.ps1` for full Windows desktop screenshots, named emulator/window screenshots, host mouse clicks, cursor movement, and keyboard/clipboard input when BlueStacks or another emulator needs laptop-level visual control. |
| Tasker-style manual tasks/actions | Present for saved shell, file-write, file-delete, safe Android system-action, accessibility UI-action, app-launch, Android intent, notification post/cancel/button, variable set/clear/append/add/subtract/literal-replace, clipboard set, toast/Tasker Flash message, bounded vibration/vibration-pattern actions, audio stream volume/ringer-mode/speakerphone/microphone-mute actions, HTTP GET/HEAD/POST/PUT/PATCH/DELETE request actions, overlay scene show/hide, launcher shortcut, Quick Settings tile, home-screen widget, and Shizuku/Sui package-permission/data-clear/connectivity-toggle/custom-setting/tethering/fixed-action records through `android_automation_tool`; chat-triggered terminal, file, UI, and Android system tool calls remain available. Saved intent records cover explicit activity start, URI open, and package-targeted broadcast intents. Notification actions can post/update/cancel Hermes app notifications with title, text, channel, priority, group, ongoing, auto-cancel, only-alert-once, group-summary fields, and up to three safe action buttons that open Hermes, dismiss the notification, or run a saved Hermes automation by id; Android 13+ devices must grant notification permission first. Overlay scene actions can show, update, or hide a bounded title/text/button panel after the user grants Android draw-over-other-apps permission; they do not import or execute arbitrary Tasker scene code. Variable actions can set, clear, append, add, subtract, or literal-replace Hermes automation variables at run time while expanding existing variables in the target name and value fields. Clipboard actions set Android clipboard text while expanding saved variables at run time. Toast actions show bounded Android toast messages, import Tasker Flash actions, and expand saved variables at run time. Vibration actions use Android's normal vibrator permission and cap duration/pattern totals. Audio actions use Android's normal audio service and report notification-policy access needs when ringer-mode changes require them. HTTP request actions use the existing Internet permission, cap request/response sizes, store Tasker-compatible `%HTTPR`/`%HTTPD` variables after saved runs, and reject non-HTTP(S) URLs. Launcher shortcut actions can create, list, or remove Android launcher shortcuts for saved Hermes automations; shortcut taps open Hermes and run the selected automation. Quick Settings tile actions can set, get, clear, or run the saved automation bound to the user-added Hermes tile. Home-screen widget actions can set, get, list, clear, request launcher pinning for, or run the saved automation bound to a Hermes widget. |
| Tasker-style wait/delay actions | Present through bounded saved `android_automation_tool` `create_wait_task` records and safe Tasker XML Wait import. Hermes caps a single wait at 60,000 ms so model-created automations cannot block the automation runner indefinitely. |
| Tasker-style profiles/triggers | Present for manual, interval, time-of-day with optional day-of-week restrictions, boot, power-connected, power-disconnected, battery-low, battery-okay, app-foreground, notification-posted, explicit calendar-event, explicit location, explicit sensor, explicit logcat-entry, token-gated external-trigger, and Shizuku-available/Shizuku-unavailable saved automation tasks. Time triggers expose `%TIME`, `%TIME_HOUR`, `%TIME_MINUTE`, and `%TIME_DAY` while running. Calendar-event triggers can filter on calendar name, title, description, or location, and expose `%CALNAME`, `%CALTITLE`, `%CALDESCR`, `%CALLOC`, plus `CALENDAR_*` aliases while running. Location triggers can filter on latitude, longitude, radius, provider, place/name, and maximum accuracy, and expose `%LOC`, `%LAT`, `%LON`, `%LOCACC`, `%LOCPROVIDER`, `%LOCNAME`, plus `LOCATION_*` aliases while running. Sensor triggers can filter on sensor type/name, event, value name/axis, and min/max value, and expose `%SENSOR`, `%SENSOR_TYPE`, `%SENSOR_NAME`, `%SENSOR_EVENT`, `%SENSOR_VALUE`, `%SENSOR_VALUE_NAME`, `%SENSOR_UNIT`, and `%SENSOR_ACCURACY`. Logcat-entry triggers can filter on tag/component, message substring, priority, pid, and package, can be dispatched with `run_logcat_entry_trigger`, can be fed by bounded `scan_logcat_entries` and process-lifetime `start_logcat_watcher` actions after Shizuku/Sui user grant, dedupe recently seen log lines through a bounded scan cursor, can clear that cursor with `reset_logcat_watcher_cursor`, and expose `%LOGCAT_TAG`, `%LOGCAT_MESSAGE`, `%LOGCAT_LEVEL`, `%LOGCAT_PID`, `%LOGCAT_PACKAGE`, `%LOGCAT_TIME`, plus `LOG_*` aliases while running. External triggers can filter on `trigger_id`, required `external_token`, optional `trigger_package_name`, and optional `referrer_contains`, can be dispatched with `run_external_trigger` or `com.nousresearch.hermesagent.EXTERNAL_TRIGGER`, and expose `%SA_TRIGGER_ID`, `%SA_TRIGGER_PACKAGE_NAME`, `%SA_REFERRER`, and `%SA_EXTRAS`. Shizuku-state triggers expose `%SHIZUKU_AVAILABLE`, `%SHIZUKU_INSTALLED`, `%SUI_INSTALLED`, `%SHIZUKU_RUNNING`, `%SHIZUKU_PERMISSION_GRANTED`, `%SHIZUKU_PRIVILEGE_LABEL`, and `%SHIZUKU_UID`. App-foreground triggers require the user-enabled Hermes accessibility service and an explicit `trigger_package_name`; notification-posted triggers require user-enabled Hermes notification access and an explicit `trigger_package_name`; location, calendar-event, sensor, and external triggers are explicit dispatch APIs, not background provider observers. Deeper app-state, provider-backed sensor/calendar/location observers, richer logcat package attribution, and plugin-profile triggers are not yet present. |
| Tasker-style variables | Present as a durable Android automation variable table exposed through `android_automation_tool`; shell commands, file paths, file content, clipboard text/labels, system-action names, UI selectors, UI text values, app package names, Android intent action/data/package/component/category/extra string values, trigger filters, calendar event fields, sensor fields, Shizuku state fields, Shizuku package names, Shizuku permission names, and saved variable action fields can expand `%NAME` or `{{NAME}}` values at run time. Direct variable actions can set/get/delete variables immediately, and saved `create_variable_action_task` records can set or clear variables from manual, scheduled, or phone-state triggers. Hermes automation bundles can export/import this variable table with saved Hermes automation records. |
| Tasker-style scenes/widgets | Partially present. Hermes can create Android launcher shortcuts, configure a user-added Android Quick Settings tile, expose a real Android home-screen widget for saved automations, and show or hide a bounded overlay scene after the user grants Android overlay permission. Full user-authored Tasker scene graphs, arbitrary overlay widgets, and richer custom widget layouts are not yet present. Hermes has its fixed app UI and accessibility control of other apps. |
| Tasker plugin model | Present for safe action-plugin and condition-plugin slices. Hermes exposes a Locale-compatible Tasker edit activity and fire receiver so Tasker can run an existing saved Hermes automation by id, plus a Locale-compatible condition edit activity and query receiver so Tasker profiles can query Shizuku availability, saved automation enabled/disabled state, last-run success/failure, and saved Hermes variable set/equality state. Each plugin configuration stores a per-action or per-condition token in Hermes preferences and exported receivers refuse missing or mismatched tokens, so arbitrary third-party broadcasts cannot run automations or probe configured condition state by guessing an id. Full Tasker event plugin profiles, service-based plugin execution, richer variable-return plugin APIs, and plugin-host timeout/status integration remain gaps. |
| Tasker Java/JavaScript/Java Code actions | Partially present through app-workspace shell/Python and Shizuku shell. Tasker 6.6.18's arbitrary Java Code action is not exposed and should stay permission-gated if implemented because imported arbitrary code is a high-risk review surface. |
| Hermes automation bundle export/import | Present through `android_automation_tool` actions `export_automations` and `import_automations`. Bundles preserve saved Hermes records plus variables, validate imported action and trigger types, reject invalid NUL-bearing payloads, can merge or replace existing automations, and reschedule enabled imported records. Safe Tasker XML/Data URI import feeds this same validated bundle importer. |
| Tasker XML/Data URI import | Partially present through `android_automation_tool` action `import_tasker_xml` plus aliases for Tasker project/task/data URI import. Hermes accepts raw XML, `data:text/xml` URI, or base64 XML; converts a safe subset of Tasker Wait, Run Shell, Write File, Delete File, Launch App, Browse URL, HTTP Get, HTTP Head, HTTP Post, HTTP Request, Notify, Flash, Vibrate, Vibrate Pattern, Set Clipboard, Variable Set, Variable Clear, Variable Add, Variable Subtract, replace-enabled Variable Search Replace, Go Home, Back Button, Show Recents, Quick Settings, Turn Off, Bluetooth, Do Not Disturb, Airplane Mode, Power Mode, Wi-Fi Tether, Wi-Fi, Mobile Data, Custom Setting writes with explicit system/secure/global namespace, End Call, audio volume/speakerphone/microphone-mute and explicit string Sound Mode actions, broad safe settings-panel actions, and exported variable data; rejects NUL, `DOCTYPE`, and `ENTITY` XML payloads; reports skipped unsupported actions; and leaves imported records disabled by default unless `enable_imported` is explicitly set. Full native Tasker project/profile fidelity, arbitrary Java/JavaScript, scenes, condition/event plugins, and background provider observers remain gaps. |
| Tasker 6.6.18 sunrise/sunset action | Present through `android_automation_tool` actions `calculate_sunrise_sunset` and `create_sunrise_sunset_task`. Hermes calculates sunrise, sunset, civil dawn/dusk, solar noon, day/night or polar state, and daylight duration offline from latitude, longitude, optional `YYYY-MM-DD` date, and optional timezone, then exposes `%SUNRISE`, `%SUNSET`, `%SUN_DAWN`, `%SUN_DUSK`, `%SOLAR_NOON`, `%SUN_DAYLIGHT_MINUTES`, `%SUN_STATE`, `%SUN_DATE`, `%SUN_TIMEZONE`, `%SUN_LAT`, and `%SUN_LON`. |
| Tasker 6.6.18 extra trigger apps | Present as token-gated Hermes external-trigger records. Third-party trigger apps can send `com.nousresearch.hermesagent.EXTERNAL_TRIGGER` with `trigger_id`, `external_token`, optional package/referrer data, and primitive extras; Hermes only runs matching enabled records and exposes `%SA_TRIGGER_ID`, `%SA_REFERRER`, `%SA_EXTRAS`, and `%SA_TRIGGER_PACKAGE_NAME`. |
| Tasker Logcat Entry with Shizuku | Present for a bounded native slice: Hermes stores `logcat_entry` profile filters, dispatches explicit logcat events through `run_logcat_entry_trigger`, and can run `scan_logcat_entries` or a persistent foreground-service-backed `start_logcat_watcher` polling loop after the user starts Shizuku/Sui and grants Hermes permission. The watcher keeps its requested scan settings in app preferences and restarts after app-process recreation or boot, while still waiting for user-granted Shizuku access. It reads bounded recent `logcat -v threadtime,uid` output through Shizuku shell, dedupes recently seen events through a bounded scan cursor, exposes cursor status/reset actions, maps UIDs back to package names with `cmd package list packages --uid`, preserves `%LOGCAT_PACKAGE_CANDIDATES`, chooses a single `%LOGCAT_PACKAGE` when a UID candidate appears in the log tag or message, marks `%LOGCAT_PACKAGE_SOURCE` as `uid`, `uid_shared`, or `message`, and feeds existing saved records. Remaining gaps are true streaming log reading and perfect app attribution for ambiguous shared Android system UIDs that are not named in the log line. |
| Tasker notification advanced options | Partially present. Hermes supports notification-posted triggers and saved notification post/update/cancel actions with title, text, channel, priority, grouping, ongoing, auto-cancel, only-alert-once fields, and up to three safe notification buttons. Buttons can open Hermes, cancel the notification, or run an existing saved Hermes automation; arbitrary code, external plugins, rich remote-input replies, media/progress styles, and deeper notification listener/provider observers remain gaps. |
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
saved Android intent actions, saved notification actions, clipboard, toast/Tasker Flash, vibration actions, HTTP request actions, launcher shortcuts,
Quick Settings tile automation, home-screen widget automation, saved Shizuku
package/permission/data-clear/connectivity-toggle/custom-setting/tethering/DND/power/global-navigation/mobile-network/user-profile actions, audio actions, variables, Hermes automation bundle import/export,
safe Tasker XML/Data URI import, time/day triggers, basic phone-state triggers,
app-foreground triggers, notification-posted triggers, explicit calendar-event
triggers, explicit location triggers, explicit sensor triggers, and
Shizuku-state triggers and Tasker/Locale action-plugin dispatch should extend
profile coverage with explicit
user-visible records:

- expand Tasker XML import only where actions can be mapped safely to existing
  Hermes records, with imported arbitrary code, plugins, scenes, and protected
  setting mutations refused unless a permission-gated Hermes equivalent exists,
- add continuous streamed Logcat Entry reading and deeper app attribution for
  ambiguous shared Android system UIDs that are not named in the log line,
- deeper app-state, provider-backed sensor/location/calendar observers, and
  plugin-profile triggers,
- tests proving scheduled/manual actions cannot mutate outside the Hermes app
  workspace unless Shizuku is explicitly selected and available.

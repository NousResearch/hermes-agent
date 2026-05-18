# build-macos-apps

Bundled standalone Hermes plugin for local macOS app build workflows.

Current scope:

- inspect a repo for `.xcworkspace` / `.xcodeproj`
- list schemes through `xcodebuild -list -json`
- run unsigned `xcodebuild build`
- run `xcodebuild test`
- find local `.app` bundles
- launch a local app bundle
- stop a local app bundle
- read recent unified logs
- collect crash reports
- show build settings

Included toolset:

- `macos-dev`

Included tools:

- `macos_inspect_project`
- `macos_list_schemes`
- `macos_build_project`
- `macos_test_project`
- `macos_find_app_bundle`
- `macos_run_app`
- `macos_stop_app`
- `macos_read_recent_logs`
- `macos_collect_crash_reports`
- `macos_show_build_settings`

What this plugin does not do yet:

- sign or notarize builds
- drive the UI or computer-use flows
- stream live logs continuously

Availability gate:

- only exposed when Hermes is running on macOS and `xcodebuild` is available in `PATH`

Build/test behavior:

- `macos_build_project` and `macos_test_project` disable signing by passing:
  - `CODE_SIGNING_ALLOWED=NO`
  - `CODE_SIGNING_REQUIRED=NO`
  - `CODE_SIGN_IDENTITY=`
- `macos_test_project` supports optional `test_plan`, `only_testing`, `skip_testing`, and `result_bundle_path`
- `macos_run_app` uses `open`
- `macos_stop_app` tries AppleScript quit first, then falls back to `pkill`
- `macos_read_recent_logs` uses `log show`
- `macos_collect_crash_reports` reads from `~/Library/Logs/DiagnosticReports`
- `macos_show_build_settings` uses `xcodebuild -showBuildSettings`

Plugin skills:

- `build-macos-apps:diagnose-build-failure`
- `build-macos-apps:local-run-loop-check`

Recommended flow:

1. `macos_inspect_project`
2. `macos_list_schemes`
3. `macos_build_project` or `macos_test_project`
4. `macos_find_app_bundle`
5. `macos_run_app` / `macos_stop_app`
6. `macos_read_recent_logs` / `macos_collect_crash_reports` / `macos_show_build_settings`

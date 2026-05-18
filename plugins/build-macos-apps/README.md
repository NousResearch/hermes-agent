# build-macos-apps

Bundled standalone Hermes plugin for local macOS app build workflows.

Phase 1 scope:

- inspect a repo for `.xcworkspace` / `.xcodeproj`
- list schemes through `xcodebuild -list -json`
- run unsigned `xcodebuild build`

Included toolset:

- `macos-dev`

Included tools:

- `macos_inspect_project`
- `macos_list_schemes`
- `macos_build_project`

What this plugin does not do in Phase 1:

- run tests
- launch or stop apps
- collect logs or crash reports
- sign or notarize builds
- drive the UI or computer-use flows

Availability gate:

- only exposed when Hermes is running on macOS and `xcodebuild` is available in `PATH`

Build behavior:

- `macos_build_project` performs an unsigned build by passing:
  - `CODE_SIGNING_ALLOWED=NO`
  - `CODE_SIGNING_REQUIRED=NO`
  - `CODE_SIGN_IDENTITY=`

Recommended flow:

1. `macos_inspect_project`
2. `macos_list_schemes`
3. `macos_build_project`

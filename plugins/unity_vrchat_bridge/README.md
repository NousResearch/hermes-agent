# Unity VRChat Bridge

`unity-vrchat-bridge` is a Unity Editor bridge for Hermes with VRChat-first
diagnostics layered on top. The MVP is read-only by default: it inspects Unity
projects, reads live Editor state, checks VRChat/VCC/VPM package health, calls
an explicitly trusted localhost Unity Editor bridge, and audits commercial model
archives without extracting or redistributing them.

The plugin deliberately blocks SDK upload, automatic package import, live menu
execution, live generic operation execution, live plan apply, DRM or license
bypass, and destructive project mutation in the MVP. Those operations belong
behind explicit dry-run, backup, project trust, and operator gates.

The Unity Editor package skeleton lives under:

`unity_package/Packages/com.hermes.unity-vrchat-bridge`

Install it into a Unity project when live Editor state is needed. Without the
Editor package, the Hermes plugin still performs file-based project and archive
diagnostics.

The Editor bridge binds only to `127.0.0.1`, requires `X-Hermes-Bridge-Token`,
stores the active session in `Library/HermesUnityBridge/session.json`, and only
starts after the project is trusted in the EditorWindow.

The generic Unity surface currently covers health, snapshot, selection, recent
logs, package metadata, active scene hierarchy, asset search, asset metadata,
dry-run menu plans, dry-run operation plans, and dry-run apply plans.

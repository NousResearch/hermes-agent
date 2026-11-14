# Hermes for iPhone

Native SwiftUI remote companion for your Mac's Hermes runtime. The visual
reference is Hermes Desktop: chat first, a session/agent drawer, a flat
transcript, blue accent, and phone tabs for Chat, Kanban, Scheduled, and Workspace.

This is a personal mobile companion, not the entire Electron desktop app.
The Mac runs the agent and its tools. The phone issues authenticated requests.
No trading, CI, PR, or approval gate is bypassed.

## Rebuild and updates

Use `./rebuild.sh verify` for the repeatable test/build gate. See
[MAINTENANCE.md](MAINTENANCE.md) for the permanent source folder, upgrade
checks, source ownership, and persistent local signing configuration.

## Build

Requires Xcode with an iOS simulator runtime. Open `HermesCompanion.xcodeproj`
and select the **HermesCompanion** scheme. The app supports iOS 17 or later.
The project has only local Swift packages and Apple framework dependencies.

```sh
swift test --package-path Core
swift test --package-path Features
xcodegen generate
xcodebuild -project HermesCompanion.xcodeproj -scheme HermesCompanion \
  -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO build
```

XcodeGen is needed only to regenerate the checked-in project after changing
`project.yml`, not to open or build the project.

For your physical phone, select your own development team under Signing &
Capabilities, select the connected phone, and Run. Enable Developer Mode on
the phone if Xcode requests it. No team/account identifiers are committed.
Simulator build or preview success does not prove a phone can reach your Mac.

## Private Mac setup — separate deployment step

Do not publish the desktop's existing loopback service blindly. Tailscale
access does not replace Hermes authentication, and existing desktop runtime
settings may be shared. Review the exact setup before changing a running Mac.

1. Connect Mac and iPhone to the same Tailscale network. Confirm access rules
   authorize only the intended user/devices and the service port.
2. Configure an existing Hermes password auth provider, using its supported
   setup on the Mac. Never paste credentials into source, a ticket, or chat.
   This app's first version supports password sign-in, not OAuth-only servers.
3. Set Hermes `dashboard.public_url` to the exact external HTTPS `.ts.net`
   origin you intend to use. This setting also activates the auth gate for a
   loopback server in the inspected Hermes version. Use an available port and
   preserve the desktop's current backend/service ownership.
4. Start the reviewed Hermes service bound to `127.0.0.1`, with its configured
   authentication provider. Verify unauthenticated `/api/auth/me` and the
   `/api/ws` handshake are denied. Verify login and a fresh WebSocket ticket
   work before enabling Serve. Do not use a local bootstrap token as remote
   authentication. Do not use `--insecure` or bind to `0.0.0.0`.
5. Inspect `tailscale serve status`, preserve existing mappings, then publish
   just this loopback service through Tailscale Serve HTTPS. HTTPS certificates
   must be enabled in the tailnet. Never enable Funnel or public forwarding.
6. In the app, enter the exact HTTPS hostname and your Hermes username/password.
   Password is not saved. Session credentials are kept in device-only Keychain.
   Verify an authenticated connection on the phone before trying controls.

Authoritative references:
- [Tailscale Serve](https://tailscale.com/docs/features/tailscale-serve)
- [Apple WebSocket transport](https://developer.apple.com/documentation/foundation/urlsessionwebsockettask)
- [Apple device-only Keychain](https://developer.apple.com/documentation/security/ksecattraccessiblewhenunlockedthisdeviceonly)

No host setup, service restart, live Kanban mutation, or phone installation is
performed merely by building the app. Deployment requires the operator's
approval and a real-device test.

## Control boundaries

Implemented screens include new sessions with profile/model/provider/workspace
selection, Kanban, scheduled-job list/create/edit/pause/resume/run/delete/history,
agent profile details, skills/tools catalogs and enable controls, read-only
memory/Star Map lists, and daily usage charts. Scheduled writes use concrete
profile plus job identity, require confirmation, and never replay automatically.
Unconfirmed writes require refresh before retrying.

Chat can invoke the Mac's existing agent tools. Treat it as remote control,
not a read-only dashboard. Approvals are deliberate and scoped to the original
request. The client offers Approve once or Deny, not blanket approval.
Kanban changes go through the existing plugin API; backend refusals are shown.
New tasks start in triage. Refresh after any uncertain mutation before retrying;
commands are never automatically resent after a timeout or connection loss.

First-slice limitations: no terminal/file browser, voice, uploads, push
notifications, secret/sudo entry, global provider configuration, Mac window
mirroring, or App Store distribution. Model settings apply to new conversations.
Unsupported requests remain visible with instructions to handle them on the Mac.
An offline phone does not stop the Mac's work. Do not assume background streaming.
Desktop-only Messaging, MCP setup, Artifacts, Webhooks, agent activity dashboards,
and the interactive Star Map graph are not implemented. Memory is a list on the phone.
Sessions currently use the backend's web source and lazy resume, not Desktop's
automatic conversation-worktree binding. Electron client tools cannot be executed
by this phone. Do not assume Desktop execution parity for coding sessions.

## Developer preview

Debug builds have a **Preview interface** button on login.
Workspace > Settings > Exit preview returns to login.
Entering preview invalidates pending login and clears credentials/connection state.

Debug builds accept the launch argument `--ui-preview`. This uses clearly
labelled sample content without connecting to a gateway. Remote controls are
disabled. Preview is diagnostic UI evidence only, never runtime evidence.

## Source attribution

The mark `App/Assets.xcassets/HermesMark.imageset/nous-girl.jpg` is copied without modification from
Hermes Desktop (`apps/desktop/public/nous-girl.jpg`). Its source license is
included as `HERMES-LICENSE`. The app icon is also copied unchanged from
`apps/desktop/assets/icon.png`. This local companion is not an official Nous
Research release. No third-party backend source is bundled into the app.

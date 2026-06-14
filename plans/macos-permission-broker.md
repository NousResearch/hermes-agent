# macOS Permission Broker for Hermes

Status: design + initial IPC contract scaffold.  
Owner area: Desktop + gateway + macOS service management.  
Goal: stable Hermes-owned macOS permission identity without granting broad TCC access to generic Python or Node runtimes.

## Decision

Hermes should adopt an OpenClaw-style macOS permission broker, but not by rewriting the gateway or bundling every runtime into one app. The target is:

> Hermes.app and a bundled signed helper own macOS-sensitive capabilities; Python, Node, MCP servers, and the gateway call that broker over narrow authenticated local IPC.

This preserves Hermes' cross-platform runtime flexibility while matching macOS' TCC/code-signing model.

## Evidence

Primary sources checked:

- OpenClaw macOS permissions: TCC grants are tied to code signature, bundle identifier, and on-disk path; generic `node` Accessibility grants are broad runtime grants, not package-scoped grants.  
  https://docs.openclaw.ai/platforms/mac/permissions
- OpenClaw macOS app: app owns permissions, manages/attaches to Gateway, exposes macOS capabilities as a node, and uses UDS + token + HMAC + TTL IPC.  
  https://docs.openclaw.ai/platforms/macos
- OpenClaw Gateway on macOS: app no longer bundles Node/Bun/Gateway runtime; it expects an external CLI install and manages a LaunchAgent.  
  https://docs.openclaw.ai/platforms/mac/bundled-gateway
- Apple SMAppService: modern macOS 13+ API for helper executables inside an app bundle, including LoginItems, LaunchAgents, and LaunchDaemons.  
  https://developer.apple.com/documentation/servicemanagement/smappservice
- Apple Service Management sample: helper launchd plists live inside signed app bundles so users can identify the providing app in Login Items.  
  https://developer.apple.com/documentation/ServiceManagement/updating-your-app-package-installer-to-use-the-new-service-management-api
- Apple notarization: distributed macOS executables need Developer ID signatures, hardened runtime, secure timestamp, valid entitlements, and notarization; ad-hoc signatures are not a durable permission identity.  
  https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution
- Electron Forge signing: Electron macOS apps should be signed and notarized; entitlements are configured through signing tooling.  
  https://www.electronforge.io/guides/code-signing/code-signing-macos
- Qt responsible-process writeup: TCC attributes access to the responsible process, which can be surprising when terminals/IDEs/runtime launchers spawn children.  
  https://www.qt.io/blog/the-curious-case-of-the-responsible-process

## Current Hermes state

Hermes Desktop is official and shares the Hermes core/config/sessions/skills/backend APIs. The packaged app currently acts mostly as an Electron shell that spawns or connects to a `hermes dashboard` backend.

Relevant local source:

- `apps/desktop/README.md`: packaged app ships Electron shell and talks to `hermes dashboard` backend.
- `apps/desktop/electron/main.cjs`: backend creation still launches Python with `-m hermes_cli.main` for dashboard backends.
- `apps/desktop/package.json`: macOS signing/notarization hooks exist (`afterSign`, hardened runtime, entitlements), but local development installs can be ad-hoc signed.
- `hermes_cli/gateway.py`: macOS LaunchAgent generation now has a local carried patch to prefer `venv/bin/hermes` over raw `python -m hermes_cli.main` for Background Items labeling.

## Non-goals

- Do not grant Accessibility or Screen Recording to generic `/opt/homebrew/bin/node`, venv Python, Terminal, or arbitrary MCP hosts as the default fix.
- Do not rewrite the entire gateway in Swift/Electron.
- Do not route normal web/search/file/git/model work through the broker.
- Do not make the broker a generic arbitrary-code execution daemon.

## Phase 1: Launch identity cleanup

Status: implemented locally as `fix: launch macOS gateway via Hermes entrypoint`.

Behavior:

- LaunchAgent `ProgramArguments[0]` should prefer the Hermes console entrypoint, e.g. `~/.hermes/hermes-agent/venv/bin/hermes`.
- Fallback to `python -m hermes_cli.main` only when the console script is missing.

Why:

- Fixes macOS Background Items showing `python` instead of `hermes`.
- Aligns with Hermes issue #15636 and PR #15640.
- Improves label/identity clarity but does not solve TCC code identity alone.

Acceptance:

- `hermes gateway status` reports service definition current.
- `launchctl print gui/$(id -u)/ai.hermes.gateway` shows `ProgramArguments[0]` as the Hermes entrypoint.
- Tests cover entrypoint preference and fallback.

## Phase 2: Stable Desktop signing and notarization

Goal:

- `/Applications/Hermes.app` is the canonical visible app path.
- Bundle ID remains `com.nousresearch.hermes`.
- Distributed builds use Developer ID Application signing, hardened runtime, secure timestamp, and notarization.
- Local ad-hoc builds are clearly marked as permission-fragile and not used for TCC-sensitive workflows.

Acceptance:

- `codesign -dv --verbose=4 /Applications/Hermes.app` shows real `Authority` and `TeamIdentifier` for release builds.
- `spctl --assess --type execute /Applications/Hermes.app` passes for release builds.
- Desktop About/diagnostics shows build commit, signing mode, notarization status if available, and broker registration status.

## Phase 3: HermesMacBroker helper

Goal:

Add a bundled macOS helper, tentatively:

- Bundle ID: `com.nousresearch.hermes.macbroker`
- Location: inside `Hermes.app` using the SMAppService-supported helper layout.
- Registration: `SMAppService.loginItem(identifier:)` or `SMAppService.agent(plistName:)`, depending on whether the helper must be an app login item or a launch agent.

Owned capabilities:

- Accessibility-backed UI actions.
- Screen capture / recording.
- Microphone status and mediated audio capture where needed.
- Apple Events / Automation to allowlisted apps.
- Notifications.
- Optional Desktop/Documents/Downloads status and user guidance.
- Optional `system.run.approved` lane for Desktop-owned exec approvals.

Acceptance:

- Helper has stable bundle ID and real signature in release builds.
- Helper reports permission status without forcing prompts.
- Helper can open exact System Settings panes for missing grants.
- Helper does not expose raw arbitrary command execution.
- TCC prompts show Hermes/HermesMacBroker, not generic Python/Node.

## Phase 4: Authenticated local IPC

Initial contract scaffold lives in:

- `apps/desktop/electron/mac-permission-broker-contract.cjs`
- `apps/desktop/electron/mac-permission-broker-contract.test.cjs`

Protocol requirements:

- Unix domain socket or XPC-equivalent local IPC.
- Random local token stored securely by the app/backend.
- HMAC-SHA256 request signing.
- TTL and issued-at validation.
- Nonce replay prevention.
- Explicit method allowlist.
- Structured audit log with secret redaction.

Initial allowed methods:

- `permission.status`
- `permission.openSettings`
- `screen.snapshot`
- `screen.record`
- `ui.click`
- `ui.type`
- `automation.appleEvent`
- `notification.send`
- `mic.status`
- `system.run.approved`

Acceptance:

- Tampered methods fail signature validation.
- Unknown methods are rejected before execution.
- Expired/future requests fail.
- Replayed nonces fail when a nonce cache is supplied.
- Audit logs redact token/password/secret/key/session/signature-like fields.

## Phase 5: Tool routing

Routing rule:

- Normal model, web, search, git, file, and most terminal paths bypass the broker.
- Mac-sensitive actions use the broker.
- MCP servers do not get generic TCC grants; Mac-sensitive MCP methods must call the broker.
- CuaDriver remains separate unless Hermes formally adopts it as the computer-use broker for that lane.

Acceptance:

- `computer_use`/screen/UI actions can route through broker when available.
- If broker unavailable, tools fail with an actionable missing-broker/missing-permission message rather than requesting Python/Node TCC grants.
- Existing non-macOS behavior is unchanged.

## Risks and mitigations

- Risk: broker becomes a god daemon.  
  Mitigation: method allowlist, per-method policy, no raw exec by default, app-side approvals for `system.run.approved`.
- Risk: IPC token theft.  
  Mitigation: local socket permissions, secure token storage, HMAC, TTL, nonce replay cache, no token logging.
- Risk: slows down Hermes.  
  Mitigation: broker only wraps Mac-sensitive operations; normal agent work bypasses it.
- Risk: release signing credentials unavailable in local dev.  
  Mitigation: support ad-hoc dev builds but mark TCC persistence as non-durable; do not claim permission stability from ad-hoc builds.
- Risk: CuaDriver boundary confusion.  
  Mitigation: document whether CuaDriver or HermesMacBroker owns each computer-use capability.

## Suggested implementation order

1. Land Phase 1 launchd entrypoint fix upstream or align with PR #15640.
2. Add Desktop diagnostics for signing mode and broker status.
3. Add Swift helper target / app-bundled helper with SMAppService registration.
4. Implement broker IPC server and reuse the JS contract from Desktop tests.
5. Add Python client library under the Hermes core for broker calls.
6. Route one low-risk method first: `permission.status`.
7. Route `notification.send` or `screen.snapshot` next.
8. Only after proof, route UI automation and approved exec.

## Test checklist

- `python -m pytest tests/hermes_cli/test_gateway_service.py -q -o 'addopts='`
- `cd apps/desktop && node --test electron/mac-permission-broker-contract.test.cjs`
- `cd apps/desktop && npm run test:desktop:platforms`
- macOS release build: verify `codesign`, `spctl`, notarization, and install-stamp.
- Manual TCC test: reset only scoped Hermes broker grants, launch app, trigger broker permission status/openSettings/smoke action, verify prompt identity.

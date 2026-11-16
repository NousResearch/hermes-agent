# Build and maintenance

Source lives in `apps/ios/HermesCompanion`. The project is self-contained and
can also be copied into a separately versioned `~/HermesCompanion` directory
for personal maintenance. Keep that source directory; generated app bundles
and DerivedData are not a backup of the project.

## Repeatable verification

From this directory, run:

```bash
./rebuild.sh verify /path/to/hermes-agent
```

This runs the local Python checker tests, source compatibility check, Swift
package tests, and an unsigned simulator build. Evidence is stored under
Git-ignored `BuildEvidence/`. It does not publish a server, change credentials,
install on a phone, or restart a backend. The checked-in Xcode project does not
require XcodeGen unless `project.yml` changes.

Before adopting backend updates, run verification against the candidate source.
A source-surface check does not prove payload semantics or runtime compatibility.
Verify real login, history, streaming, reconnects and approvals separately; use
only disposable conversations, boards and jobs for mutation tests. Never just
change the compatibility number to silence a failure.

## Personal device signing

Create Git-ignored `Local.xcconfig` with your own `DEVELOPMENT_TEAM` assignment.
`Build.xcconfig` includes it, preserving signing settings across regeneration.
Do not put passwords or tokens there. Select your paired iPhone in Xcode and
complete Apple-account, Developer Mode and trust prompts locally.

```bash
./install-iphone.sh DEVICE_UDID
```

The installer verifies, builds, checks the signature and app identity, installs,
and launches. Save your identifier in Git-ignored `LocalDevice.txt` to omit the
argument. Free-signing renewal is a separate opt-in procedure described in
[SIGNING-RENEWAL.md](SIGNING-RENEWAL.md); it does not happen merely by merging or
building this project.

## Private gateway deployment

See README for the authentication and Tailscale sequence. Setup is an operator
step, separate from app builds. `scripts/setup_private_login.py` accepts explicit
Hermes source, Hermes home and private HTTPS origin arguments, prompts locally,
and refuses to overwrite existing credentials. `scripts/run_private_gateway.py`
accepts explicit source/home arguments and binds only loopback. It refuses
missing private-origin/password settings; it does not start a messaging gateway.

Use the existing supported gateway connection mode in Hermes Desktop when
sharing a password-protected backend. A local bootstrap token is not a substitute
for password/cookie authentication. Keep background workers and HTTP/WS service
ownership explicit; do not restart unrelated services to rebuild the phone app.
The Mac must stay awake and reachable for remote access.

## Optional local Mac packaging

The separate `scripts/package_mac_desktop.py` packages already compiled desktop
assets without editing their source checkout. It names the resulting local app
Hermes Desktop while preserving the explicitly supplied Hermes home and desktop
data directory. It refuses desktop inputs changed since the build stamp;
backend-only commits do not force a desktop rebuild.

```bash
python3 scripts/package_mac_desktop.py \
  --desktop-source /path/to/hermes-agent/apps/desktop \
  --hermes-home /path/to/hermes-home \
  --user-data-dir "$HOME/Library/Application Support/Hermes" \
  --output "$PWD/BuildEvidence/mac-$(date -u +%Y%m%dT%H%M%SZ)"
```

This is a local ad-hoc-signed package, not an official notarized release. It does
not install, upload, change credentials or restart services. Preserve the old app
for rollback, quit it before replacement, then verify the visible name and real
gateway connection. Upstream updates can replace custom branding.

## Ownership

- `Core/`: endpoint restrictions, HTTP auth, Keychain and WebSocket RPC.
- `Features/`: session/approval identity, mutations, jobs and workspace data.
- `App/`: SwiftUI screens and app lifecycle.
- `Tests/`: simulator navigation and disconnected-state UI tests.
- `scripts/`: compatibility, build/deployment and optional renewal helpers.
- `project.yml`: generation source for the committed Xcode project.

Keep local credentials, device settings, profile backups and receipts out of Git.

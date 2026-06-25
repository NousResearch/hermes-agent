# Build Hermes Mobile to a physical iPhone

This is the concrete path we used to get the standalone Hermes iPhone app onto a real device. It is intentionally operational: follow the steps in order, and use the troubleshooting section for the exact failures we hit.

## What you are building

Hermes Mobile is a React Native/Expo iOS shell that embeds the bundled Hermes Desktop renderer in a `WKWebView` and connects it to a Hermes gateway.

For a real phone install, build a **Release** app with the JavaScript bundle embedded. Do **not** treat the Expo Dev Client screen (`Development Build`, `No development servers found`) as success; that is only a dev shell waiting for Metro.

## Prerequisites

### Mac

1. Install Xcode from the Mac App Store.
2. Open Xcode once and finish first-run setup:
   - accept the license
   - install iOS platform components if prompted
   - sign into your Apple ID in **Xcode → Settings → Accounts**
3. Install Xcode command-line tools if needed:

   ```bash
   xcode-select --install
   sudo xcodebuild -license accept
   sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
   ```

4. Verify Xcode sees iOS SDKs:

   ```bash
   xcodebuild -version
   xcodebuild -showsdks | grep iphone
   ```

### iPhone

1. Connect the iPhone to the Mac with USB.
2. Unlock the phone and tap **Trust This Computer** if prompted.
3. Enable Developer Mode:
   - iPhone: **Settings → Privacy & Security → Developer Mode → On**
   - restart when prompted
   - unlock the phone again and confirm Developer Mode
4. Keep the phone unlocked during install/launch. If it locks after install, Expo may print:

   ```text
   Cannot launch Hermes on <device> because the device is locked.
   ```

   That usually means install completed but auto-launch was blocked. Unlock the phone and tap Hermes.

### Repository dependencies

From the repo root:

```bash
npm install
```

From the mobile app directory:

```bash
cd apps/mobile
npm install
npm run typecheck
```

## Configure signing

The generated iOS project must use a real Apple Development team/certificate.

Open the workspace once in Xcode:

```bash
cd apps/mobile
open ios/Hermes.xcworkspace
```

In Xcode:

1. Select the **Hermes** project.
2. Select the **Hermes** target.
3. Open **Signing & Capabilities**.
4. Enable **Automatically manage signing**.
5. Select your Apple Development Team.
6. Confirm the bundle identifier is unique for your account if Xcode asks.
7. Build once from Xcode if prompted to trust/authorize signing keys.

Verify a signing identity exists:

```bash
security find-identity -v -p codesigning
```

You should see an `Apple Development: ...` identity.

## Configure the Hermes gateway

A standalone phone build needs a reachable Hermes Desktop/gateway URL and session token at build time.

Set these environment variables before building:

```bash
export EXPO_PUBLIC_HERMES_GATEWAY_URL="https://your-hermes-gateway.example.com"
export EXPO_PUBLIC_HERMES_GATEWAY_TOKEN="<your-session-token>"
```

Notes:

- Do not commit tokens.
- Do not paste tokens into PRs, screenshots, or logs.
- The gateway URL must be reachable from the iPhone, not just from the Mac.
- The app stores connection details locally after setup, but the clean Release build path is easiest when the defaults are injected at build time.

## Build and install to the phone

Find the physical device UDID:

```bash
xcrun xctrace list devices
xcrun devicectl list devices
```

Then build/install Release:

```bash
cd apps/mobile
npm run typecheck
npx expo run:ios --configuration Release --device <IPHONE_UDID>
```

The successful install used this command shape, with a real HTTPS gateway URL, a local token file, and the physical iPhone UDID:

```bash
cd apps/mobile
export EXPO_PUBLIC_HERMES_GATEWAY_URL="https://your-hermes-gateway.example.com"
export EXPO_PUBLIC_HERMES_GATEWAY_TOKEN="$(tr -d '\n' < /path/to/hermes-mobile-token)"
npx expo run:ios --configuration Release --device <IPHONE_UDID>
```

A successful build writes an app at a path like:

```text
~/Library/Developer/Xcode/DerivedData/Hermes-.../Build/Products/Release-iphoneos/Hermes.app
```

Verify the phone has the app:

```bash
xcrun devicectl device info apps --device <IPHONE_UDID> | grep -i 'Hermes\|com.nousresearch'
```

Launch it:

```bash
xcrun devicectl device process launch --device <IPHONE_UDID> com.nousresearch.hermes
```

## If SSH/headless signing fails

When building over SSH, `codesign` may fail with:

```text
errSecInternalComponent
User interaction is not allowed
```

That is a macOS keychain/signing authorization issue, not necessarily an app code issue.

Use the visible Mac Terminal or Xcode GUI for the Release build so macOS can prompt/authorize keychain access:

```bash
cd apps/mobile
export EXPO_PUBLIC_HERMES_GATEWAY_URL="https://your-hermes-gateway.example.com"
export EXPO_PUBLIC_HERMES_GATEWAY_TOKEN="<your-session-token>"
npx expo run:ios --configuration Release --device <IPHONE_UDID>
```

If you must continue headlessly after one successful GUI authorization, this can help, but it may require the login keychain password:

```bash
security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$KEYCHAIN_PASSWORD" ~/Library/Keychains/login.keychain-db
```

## If Expo says `ApplicationVerificationFailed`

Do not stop at Expo's wrapper error. Inspect the real install error with `devicectl`:

```bash
APP="$HOME/Library/Developer/Xcode/DerivedData/Hermes-.../Build/Products/Release-iphoneos/Hermes.app"
xcrun devicectl device install app --device <IPHONE_UDID> "$APP"
```

### Failure: nested framework has no code signature

We hit this exact error:

```text
Failed to verify code signature of .../Hermes.app/Frameworks/ExpoFileSystem.framework : No code signature found.
```

Fix by signing every embedded framework, then the app:

```bash
APP="$HOME/Library/Developer/Xcode/DerivedData/Hermes-.../Build/Products/Release-iphoneos/Hermes.app"
IDENTITY="Apple Development: Your Name (TEAMID)"

find "$APP/Frameworks" -maxdepth 1 -type d -name "*.framework" -print0 | while IFS= read -r -d '' fw; do
  codesign --force --sign "$IDENTITY" --timestamp=none "$fw"
done
```

Then re-sign the app with entitlements.

### Failure: missing `application-identifier` entitlement

After manually signing frameworks, we hit:

```text
Application is missing the application-identifier entitlement.
```

Create an entitlements plist matching the provisioning profile. Replace `TEAMID` and bundle identifier as needed:

```bash
cat > /tmp/hermes-app-entitlements.plist <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>application-identifier</key><string>TEAMID.com.nousresearch.hermes</string>
  <key>com.apple.developer.team-identifier</key><string>TEAMID</string>
  <key>get-task-allow</key><true/>
  <key>keychain-access-groups</key><array><string>TEAMID.*</string></array>
</dict></plist>
PLIST
```

Re-sign and install:

```bash
codesign --force --sign "$IDENTITY" --timestamp=none --entitlements /tmp/hermes-app-entitlements.plist "$APP"
codesign -vvv --strict --deep "$APP"
xcrun devicectl device install app --device <IPHONE_UDID> "$APP"
xcrun devicectl device process launch --device <IPHONE_UDID> com.nousresearch.hermes
```

Use the `application-identifier`, team identifier, and keychain group values from your provisioning profile, for example:

```text
application-identifier = TEAMID.com.nousresearch.hermes
team-identifier        = TEAMID
keychain group         = TEAMID.*
```

Run manual signing from the visible Mac Terminal if SSH reports `errSecInternalComponent`.

## Verify the installed app is the new build

After installing, do not merely relaunch the old app. Confirm `devicectl` prints a fresh installation URL:

```bash
xcrun devicectl device install app --device <IPHONE_UDID> "$APP"
```

Expected success shape:

```text
App installed:
• bundleID: com.nousresearch.hermes
• installationURL: file:///private/var/containers/Bundle/Application/.../Hermes.app/
```

Then launch:

```bash
xcrun devicectl device process launch --device <IPHONE_UDID> com.nousresearch.hermes
xcrun devicectl device info processes --device <IPHONE_UDID> | grep -i Hermes
```

## Troubleshooting visual/runtime issues we hit

### App opens to Expo Dev Client

If the phone shows `Development Build` / `No development servers found`, you installed a dev-client shell, not the standalone app. Build Release:

```bash
npx expo run:ios --configuration Release --device <IPHONE_UDID>
```

### App has missing icons/square glyphs

The iPhone `WKWebView` cannot rely on sibling font files emitted by Vite the way Electron/Desktop can. The build script inlines renderer assets/fonts into `src/bundled-renderer-html.ts`. If icons disappear, regenerate the bundled renderer and confirm font URLs became data URIs.

### `HERMES AGENT` wordmark is off-center on the physical iPhone

The Desktop `fit-text` container-query sizing can mis-size in iOS `WKWebView`. The final fix was a scoped, unconditional `.mobile-wordmark` override:

- hide the duplicate `aria-hidden` fit-text clone
- disable container-query sizing for the visible wordmark
- force `width: 100%` and `text-align: center`
- use a conservative viewport font clamp

Do not hide this behind a narrow media query until a real phone screenshot proves it works; the physical WebView can report a layout viewport that skips phone-width media queries.

### Browser-side gateway probe fails with `Load failed`

Electron Desktop can use privileged networking; iOS WebView cannot. The mobile app routes `window.hermesDesktop.api(...)` through the React Native bridge, and native `fetch` adds `X-Hermes-Session-Token` / bearer auth. Do not block setup solely on a browser-side `/api/status` probe.

## Exact successful sequence recap

1. Built a Release app for the physical iPhone, not a dev-client app.
2. Injected the reachable gateway URL and session token via env vars.
3. Used GUI Terminal/Xcode path when SSH signing hit keychain authorization issues.
4. When Expo reported `ApplicationVerificationFailed`, ran `devicectl install app` directly to expose the real nested-framework signing error.
5. Signed all embedded frameworks.
6. Re-signed `Hermes.app` with `application-identifier` entitlements from the provisioning profile.
7. Installed with `devicectl` and verified it printed a fresh `installationURL`.
8. Launched `com.nousresearch.hermes` and verified the process on the phone.
9. Took real phone screenshots to confirm the centered wordmark, conversation view, and session sidebar.

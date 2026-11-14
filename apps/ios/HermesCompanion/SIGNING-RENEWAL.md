# Free iPhone signing renewal

An optional native Hermes job can renew signing every six days. It runs a
local script without an AI agent. Delivery is local cron history, not push
notifications. Each application must have its own job and exact identity filter.

## Safe renewal sequence

1. Lock `BuildEvidence/phone-refresh.lock` against overlapping scheduled runs.
2. Read the app's current built signing expiry and the team from Git-ignored
   `Local.xcconfig`. Verify exact bundle `com.local.hermes.companion` and team.
3. Inspect both Xcode provisioning caches. Back up/move only profiles with
   `LocalProvision=true` and that exact team/application identifier. Other apps,
   teams, managed profiles, shared certificates and Keychain entries are untouched.
4. Run `install-iphone.sh --no-launch`, using Git-ignored `LocalDevice.txt`.
   Existing tests and the backend source compatibility gate run first. Then
   `xcodebuild clean build` uses Hermes' own `BuildEvidence/DeviceDerivedData`.
5. Verify the signature with `codesign --verify --deep --strict`, and verify the
   app/profile identity before installing over the existing app. Do not uninstall.
6. Claim renewal only if installation exits successfully and expiry advances
   with at least six days remaining. Reusing an old profile is a failed renewal.
   A failed attempt restores missing backed-up profiles without overwriting new ones.

The installer has a 15-minute limit, plus bounded shutdown/cleanup. It does not
launch the app during renewal, avoiding the demonstrated locked-screen launch
failure. Manual installation still launches. Apple can still block installation
until authentication, trust, or device access is resolved.

This uses [Apple's documented procedure for requesting new automatic-signing
profiles](https://developer.apple.com/help/account/provisioning-profiles/edit-download-or-delete-profiles/),
restricted to this app. Both `~/Library/MobileDevice/Provisioning Profiles` and
`~/Library/Developer/Xcode/UserData/Provisioning Profiles` are inspected.

## Requirements and evidence

Keep the Mac awake, the native Hermes scheduler running, the signing keychain
and Xcode Apple account available, and the paired phone reachable by Xcode over
local Wi-Fi or USB. Tailscale access to Hermes does not prove Xcode installation
reachability. No wake service, extra retry job, certificate revocation, paid
membership, TestFlight upload, app uninstall, or backend restart is performed.

Immutable receipts/logs are `BuildEvidence/phone-refresh-TIMESTAMP-ID.json` and
`.log`. `phone-refresh-latest.json` is a convenience pointer replaced each run.
Private profile backups stay in `BuildEvidence/provisioning-backups/`. Build
receipts and the saved team/device configuration are excluded from Git.

For a read-only expiry check:

```bash
cd ~/HermesCompanion
python3 scripts/refresh_hermes_phone.py --check
```

To retry or pause, use the existing job in Hermes Scheduled Jobs. The renewal
command is `hermes cron run YOUR_JOB_ID`; `hermes cron pause YOUR_JOB_ID` pauses it.
An ordinary reinstall can reuse an old profile and is not a substitute for this
renewal path. Rebuilding iOS does not restart the Mac backend or change Tailscale.

The source-owned shim is `scripts/hermes-refresh.sh`; copy it physically into
`~/.hermes/scripts/hermes-companion-refresh.sh` when initially installing it.
Do not create duplicate jobs during future app rebuilds.

## Opt-in schedule setup

For a standalone installation at `~/HermesCompanion`, copy
`scripts/hermes-refresh.sh` physically into the active Hermes home's `scripts/`
directory as `hermes-companion-refresh.sh`. For another source location, edit the
shim to use that absolute location first. Keep scripts private to your account.
Do not copy credentials into the shim.

After inspecting any existing renewal jobs, create one only if none exists:

```bash
hermes cron create 'every 8640m' --name 'Hermes iPhone signing renewal' \
  --script hermes-companion-refresh.sh --no-agent --deliver local
```

Record the returned job ID locally. Run that same job once and inspect its
receipt before relying on unattended renewal. No installer creates this job for
you. Keep separate apps' jobs, locks, DerivedData and profile backups separate.

## Prior physical-device evidence

A personal installation completed a native script-only renewal over local Wi-Fi:
the installer returned zero, strict/deep signature verification passed, launch
was skipped and expiration advanced with approximately seven days remaining.
An unrelated application's cached profile was byte-for-byte unchanged. This is
prior deployment evidence, not a guarantee for another Apple account or device.
Local receipt paths, job IDs, account details and expiry dates are intentionally
not published. See VERIFICATION.md for the public-copy checks and evidence limits.

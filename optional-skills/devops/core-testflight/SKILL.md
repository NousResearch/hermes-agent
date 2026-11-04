---
name: core-testflight
description: Verify exact iOS TestFlight builds and device installation.
version: 3.1.0
author: Peter Beddow, Hermes Agent
license: MIT
platforms: [macos]
metadata:
  hermes:
    tags: [Apple, TestFlight, App Store Connect, Xcode Cloud, verification, installation]
    related_skills: [core-secrets]
---
# Core TestFlight Skill

Use this procedure for Accessible Core iOS TestFlight work. Keep these states separate:

1. source published;
2. Xcode Cloud archive succeeded;
3. App Store Connect received and processed the exact build;
4. the build is compliant and assigned to the intended TestFlight group;
5. the intended tester can see that exact build;
6. that exact build is installed on the intended device.

Never use one state as proof of another. Never claim `DONE` until the requested installation state is directly evidenced.

## Delivery-mode boundary

The caller's delivery mode is authoritative and must be recorded before this
skill is invoked:

- **`TESTFLIGHT`**: App Store Connect and TestFlight distribution are the
  delivery route. This mode never requires PHN-01 to be connected to a Mac,
  never requires a CoreDevice tunnel, and never runs direct-device installation
  as a prerequisite. Complete the Apple-side exact-build checks first, then
  report the TestFlight installation action for the iPhone. A missing or
  unavailable CoreDevice connection is not a TestFlight blocker.
- **`DIRECT_DEVICE`**: CoreDevice discovery and tunnel readiness may be
  required by the selected direct installer. If that route fails, stop the
  direct route and hand the request back to this skill in `TESTFLIGHT` mode;
  do not silently retry direct installation.

When `TESTFLIGHT` is selected, use the existing API-first workflow below and
record `ROUTE_DECISION=TESTFLIGHT`. Device inspection is optional post-install
verification only; it is never a precondition for App Store Connect reads,
TestFlight availability, or reporting the iPhone Install action.

This skill treats a release receipt as historical evidence, not as proof of what is currently installed. Every release attempt must produce a current provenance report that ties one GitHub commit to one Xcode Cloud run and one TestFlight build. In `DIRECT_DEVICE` mode, a failed or unavailable device readback is `BLOCKED`; in `TESTFLIGHT` mode, when the exact Apple-side build is ready but the phone-side Install action or post-install readback is unavailable, report `INSTALL_ACTION_REQUIRED`, never a CoreDevice failure.

## Required release record

Before building, create a new, non-overwriting release record with these sections:

```text
GitHub commit:
Xcode Cloud build:
TestFlight build:
Target build:
PHN-01 installed build:
Installation method:
Verification evidence:
Outcome: DONE / INSTALL_ACTION_REQUIRED / BLOCKED / FAILED
```

The current record must contain the exact bundle ID, marketing version, target build number, GitHub ref and 40-character commit SHA, Xcode Cloud workflow/run IDs, Xcode Cloud source commit and build number, TestFlight build record and availability, intended tester/group, and installation method/outcome. In `DIRECT_DEVICE` mode also require target device identifiers and fresh device evidence containing the installed `CFBundleVersion`. In `TESTFLIGHT` mode, record the exact Apple-side readiness and either phone-side install evidence or `INSTALL_ACTION_REQUIRED=TESTFLIGHT_ON_IPHONE`; do not require CoreDevice fields before that action and do not fill them with stale data. A historical receipt may be linked for context but cannot fill a missing current field.

## Safety and ownership

- Native iOS is the only implementation scope for this skill.
- Preserve the existing Xcode Cloud workflow. Do not create another upload, signing, build-number, or authentication system.
- Do not use `xcodebuild -exportArchive`, `altool`, `iTMSTransporter`, or a local IPA as a substitute for the existing Xcode Cloud/TestFlight path.
- Do not open a general Apple login page or ask for an Apple password or verification code. Reuse the already authenticated browser session when the protected client is unavailable.
- Never print, log, commit, or place in a receipt the App Store Connect private key, Key ID, Issuer ID, JWT, bearer token, Core Secrets bearer credential, password, or connection string. Replace sensitive values with `[REDACTED]` in reports.
- Do not create a second credential store. Use the existing Core Secrets skill or the already authenticated browser session; both are approved existing paths.
- Before a write, state what will change and why. After every App Store Connect write, read the exact target back.

## Exact target

Record these values before acting. Never substitute “latest,” a workflow number, or a guessed build:

- app bundle ID;
- marketing version;
- exact build number;
- intended TestFlight group;
- intended tester/device;
- source commit and Xcode Cloud workflow.

Do not copy current release values from an earlier receipt. The examples under `examples/` are historical records only; the current target must be written from the active GitHub checkout, Xcode Cloud run, TestFlight read, and device readback.

## Source and Xcode Cloud

From the established native checkout, inspect before building or pushing:

```bash
git status --short
git diff --check
git rev-parse HEAD
git branch --show-current
git ls-remote origin refs/heads/{ref}
```

Classify every untracked path as `pre-existing`, `intended`, or `generated-and-reviewed` in the current release record. An unclassified path blocks the release. Ensure every intended source change is committed and the exact commit is present on the authoritative GitHub ref before building. Never use `rm`, checkout, reset, or overwrite to make the checkout appear clean; preserve existing receipts and write a new receipt/report path for each attempt.

The existing workflow watches `native-auth-client` and is triggered by a source-ref change:

```bash
git push origin native-auth-client
```

Do not create a no-op commit merely to force a run. Confirm that the Xcode Cloud run identifies the requested GitHub commit exactly, completed successfully, and produced the requested version/build. Record the workflow ID, run ID, source commit, archive status, and build number in the current report. A local build, simulator launch, archive, or Xcode Cloud check alone does not prove TestFlight availability.

After Xcode Cloud succeeds, perform a separate fresh App Store Connect/TestFlight read for that exact build. Record the build record ID, processing/QC/compliance state, group membership, tester visibility, and source/build tuple. Xcode Cloud success and TestFlight availability are separate gates; either missing gate blocks installation.

If Xcode Cloud reports `action_required`, `Preparing build for App Store Connect failed`, or another preparation failure, record that failure separately. Do not change signing, build numbers, or workflow configuration just to force another run.

## Authentication path decision (mandatory)

Use exactly one already-authorized path; do not create or repair credentials in
this workflow:

1. If the protected `accessible_core_secrets` App Store Connect client is
   installed and importable on the authorized workstation, use the in-memory
   Core Secrets/API path in the next section.
2. If that client is not installed in the current shell, reuse the already
   authenticated App Store Connect browser session. Read the exact Xcode Cloud,
   App Store Connect, TestFlight, group, tester, and build records through that
   existing session. Do not install a package, invent an import path, open a
   login page, ask for credentials, or create a second credential store.
3. If neither an importable protected client nor an already authenticated
   browser session is available, record the missing evidence and report
   `BLOCKED`; do not guess or substitute a credential path.

The browser-session route is an approved readback fallback, not a new
credential system. Preserve the same exact-build gates and redact cookies,
tokens, passwords, private keys, and other secrets from every report.

## Core Secrets and App Store Connect authentication

Use the existing `core-secrets` skill when the protected client is available.
The authorized workstation flow is:

```python
from accessible_core_secrets.client import SecretsServiceClient
from core_secrets_keychain_client import credential_file_read, principal_for_workstation

reference = "apple/app-store-connect"
principal = principal_for_workstation("WKST-01")
client = SecretsServiceClient(
    CORE_SECRETS_URL,
    credential_file_read(principal),
    timeout=15,
)
fields = client.resolve(reference)
```

The record must contain `key_id`, `issuer_id`, and `private_key`. Keep `fields` and the short-lived JWT in process memory only. Never write them to a script, environment file, receipt, terminal output, browser field, or chat.

Create an App Store Connect JWT in memory with ES256:

- header: `alg=ES256`, `kid=<key_id>`, `typ=JWT`;
- payload: `iss=<issuer_id>`, `iat=now-30`, `exp=now+900`, `aud=appstoreconnect-v1`;
- sign with the private key using ECDSA SHA-256;
- encode the signature as the JWT's raw 64-byte `r || s` value, not DER;
- send it only as `Authorization: Bearer <JWT>` over TLS.

Use a certificate-verifying TLS context, for example Python `ssl.create_default_context(cafile=certifi.where())`. On authentication failure, report only the redacted endpoint, HTTP status, and Apple error title/detail. Do not ask Peter for a secret when either approved existing authentication path is available.

## Existing browser-session readback

When the browser-session path is selected, use the current already-authenticated
App Store Connect/Xcode Cloud page and its same-origin JSON reads for the exact
records below. A browser page that shows a login, a stale page, or an
unauthenticated response is not evidence. Do not open a login flow or copy
cookies, headers, tokens, or page credentials into a file or report. Keep the
browser session attached while the current release record is captured.

Use the same exact-build selection and readback sequence regardless of whether
the response came from the protected client or the browser session. For any
write made through the browser session, record the intended change, perform
only that bounded write, and read the exact target back before continuing.

## App Store Connect read sequence

Use fresh authenticated API reads (from the protected client or the existing
browser session). The App Store Connect API represents the build number in the
build resource's `attributes.version`; do not assume a `buildNumber` attribute
exists. The marketing version is read from the linked `preReleaseVersion`
resource.

1. Find the app:

```text
GET /v1/apps?filter[bundleId]={bundleId}&limit=10
```

Require exactly one matching app.

2. List builds for that app and select the exact build:

```text
GET /v1/builds?filter[app]={appId}&limit=200&sort=-uploadedDate
```

Select the record whose `attributes.version` equals the requested build number. Do not select by position or “latest.”

3. Read the exact build and linked beta detail:

```text
GET /v1/builds/{buildId}?include=preReleaseVersion,buildBetaDetail
GET /v1/preReleaseVersions/{preReleaseVersionId}
GET /v1/buildBetaDetails?filter[build]={buildId}&limit=10
```

Require the exact version/build tuple, `processingState=VALID` (or an explicitly testable equivalent), a non-expired build, and a beta state that permits testing. For the verified Build 72, the readback was `VALID`, internal beta state `IN_BETA_TESTING`, and no non-exempt encryption.

4. Read groups and exact build membership:

```text
GET /v1/betaGroups?filter[app]={appId}&limit=200
GET /v1/betaGroups/{groupId}/builds?limit=200
```

Find the intended group by exact name. Confirm that the exact build ID is present.

5. Read tester membership without exposing identity:

```text
GET /v1/betaGroups/{groupId}/betaTesters?limit=200
GET /v1/betaTesters/{testerId}/betaGroups?limit=200
```

Require an enrolled tester whose group membership includes the intended group. A general tester state such as `INSTALLED` is not proof of the requested build. The per-tester `/builds` relationship may omit builds inherited from a group; do not treat an empty direct relationship as proof that group-inherited access is absent. Group build membership plus tester group membership must still be read and recorded.

## Conditional App Store Connect writes

Only write when a fresh read shows a specific correction is required. Never repeat a successful write blindly.

### Export compliance

Inspect the app source before answering export compliance. For the verified Accessible Capture release, CryptoKit was used for SHA-256 checksums, not app-level encryption. When the exact build is missing its answer, submit Apple's no-non-exempt-encryption value:

```http
PATCH /v1/builds/{buildId}
Content-Type: application/json

{
  "data": {
    "type": "builds",
    "id": "{buildId}",
    "attributes": {"usesNonExemptEncryption": false}
  }
}
```

Immediately read `/v1/builds/{buildId}` again and require `usesNonExemptEncryption=false`. Record compliance as complete only after that readback. If the source actually uses non-exempt encryption, stop and request the correct Apple-side compliance decision; do not guess.

### TestFlight group assignment

If the exact build is absent from the intended group, add only that exact build:

```http
POST /v1/betaGroups/{groupId}/relationships/builds
Content-Type: application/json

{"data":[{"type":"builds","id":"{buildId}"}]}
```

Read `/v1/betaGroups/{groupId}/builds` again and require the exact build ID. If a write returns a permission error, report the exact missing permission and request only that correction. Do not create another key or credential route.

## Installability gate

Before saying “install Build N” or “update to Build N,” all of these must be true for the same exact build:

1. App Store Connect contains the exact app/version/build.
2. Processing, QC/beta state, expiration, and compliance permit testing.
3. The exact build is assigned to the intended group.
4. An intended tester is enrolled and belongs to that group.
5. The tester-facing build number is the requested number.
6. A release receipt passes:

```bash
python3 scripts/verify-testflight-receipt.py RECEIPT.json --require installable
```

Do not rely on local builds, Xcode Cloud completion, an App Store Connect row for another build, a tester's generic state, or a stale browser page.

## TestFlight delivery and optional device control

When `delivery_mode=TESTFLIGHT`, do not run `xcrun devicectl` before the
App Store Connect/TestFlight checks. The required sequence is:

1. read and verify the exact Apple-side build, compliance, group membership,
   tester access, and tester-visible build;
2. record `ROUTE_DECISION=TESTFLIGHT` and
   `TESTFLIGHT_READY build=<exact-build>`;
3. use TestFlight on PHN-01 to select the exact build and tap **Install**;
4. if an approved remote-UI capability is already available, it may perform
   that phone-side action; otherwise report
   `INSTALL_ACTION_REQUIRED=TESTFLIGHT_ON_IPHONE` with the exact build.

The TestFlight install action is independent of the Mac's CoreDevice tunnel.
A disconnected phone, missing device record, failed `devicectl` command, or
unavailable tunnel must never be reported as a TestFlight distribution blocker.
Those observations may be recorded as unavailable post-install verification,
but they do not change the selected route or the required iPhone action.

When `delivery_mode=DIRECT_DEVICE`, the caller may use the connected-device
launch and inspection commands below. If that direct route fails, return to
this skill in `TESTFLIGHT` mode instead of repeating direct discovery.

## Optional direct-device launch

```bash
xcrun devicectl list devices
xcrun devicectl device process launch \
  --device "{deviceIdentifier}" \
  com.apple.TestFlight --activate
```

A successful TestFlight launch proves only that TestFlight opened. It does not
prove that the requested build was installed. These commands are never a
precondition for the `TESTFLIGHT` route.

## No-intervention installation policy

Try all agent-controlled steps that are valid for the selected mode:

1. in `TESTFLIGHT` mode, complete the authenticated Apple-side checks without
   device discovery;
2. if an approved device-control capability is already available, use it to
   select the exact app/build and activate Install;
3. wait for installation to finish when that control path exists;
4. inspect the device for the exact bundle ID and `CFBundleVersion` only when a
   reliable post-install device readback path is available.

`xcrun devicectl` provides device discovery, process launch, and app inspection,
but it does not provide a supported command to press TestFlight's in-app Install
control. Do not substitute `devicectl device install app`, `ios-deploy`, or a
local IPA for a requested TestFlight installation; if a direct installation is
intentionally used for a separate development test, record it as
`direct-device-install`, not TestFlight.

After installation, read the current app record from PHN-01 using a reliable,
structured device inspection method when that readback path is available.
Require the current record to identify the exact bundle ID, marketing version,
and `CFBundleVersion`. Compare the read `CFBundleVersion` with the intended
build number. A missing app record or an owner statement without a named build
is not installation evidence. If TestFlight is ready but no phone-control or
post-install readback path is available, report
`INSTALL_ACTION_REQUIRED=TESTFLIGHT_ON_IPHONE` rather than a CoreDevice
failure. Do not convert unavailable CoreDevice verification into a TestFlight
distribution blocker.

Therefore:

- If an approved remote-UI/device-control capability is available, use it and retain its direct result.
- If the device is already configured for automatic TestFlight installation/update, wait and verify the exact installed build.
- If neither is available, the physical TestFlight Install tap is an unavoidable boundary. State exactly that one action is required; do not claim no-intervention automation.
- Never ask Peter to sign in, enter credentials, or approve an unverified build.

## Installation verification

Prefer structured device inspection:

```bash
xcrun devicectl device info apps \
  --device "{deviceIdentifier}" \
  --bundle-id "{bundleId}" \
  --include-all-apps \
  --json-output device-apps.json
```

Parse the structured result and require:

- bundle ID equals the target bundle ID;
- installed `CFBundleShortVersionString` equals the target marketing version;
- installed `CFBundleVersion` equals the target build number;
- the device and app record are current, not a stale saved file.

If the structured result is empty, missing the target app, missing `CFBundleVersion`, or cannot be read, preserve the last verified installed build and report `BLOCKED` only for `DIRECT_DEVICE` acceptance. For `TESTFLIGHT`, if Apple-side readiness is verified but the phone-side Install action or readback is unavailable, preserve the uninstalled/unverified state and report `INSTALL_ACTION_REQUIRED=TESTFLIGHT_ON_IPHONE`; never report CoreDevice failure as the reason. A release can be marked `DONE` only after the current device readback matches the intended build; TestFlight readiness alone is not installation evidence.

## Mandatory release receipt

For every TestFlight attempt, create a JSON receipt in the native checkout using `docs/testflight-release-receipt.template.json`. It must contain:

- exact target app, bundle ID, marketing version, build, group ID, and tester label;
- source commit, ref, checked-in build number, effective build number, workflow ID, run ID or an explicit non-guessing unavailable marker, and archive status;
- exact App Store Connect build record ID, processing state, QC state, audience, and compliance;
- exact group ID, group-build membership, tester access, tester-visible build number;
- installed build number and installation evidence only when directly proven;
- UTC capture time, fresh-authenticated-read flag, and final release state.

Never put credentials or tokens in the receipt. Validate it from the native checkout:

```bash
python3 scripts/verify-testflight-receipt.py RECEIPT.json --require installable
```

After exact installation evidence:

```bash
python3 scripts/verify-testflight-receipt.py RECEIPT.json --require installed
```

The receipt is the historical communication boundary. Preserve failed validation as evidence of a missing gate; never edit unknown values merely to make validation pass. Before declaring a release complete, also validate the separate current provenance report:

```bash
python3 optional-skills/devops/core-testflight/scripts/verify-release-provenance.py CURRENT-RELEASE.json
```

This checker rejects commit mismatches, unclassified untracked files, mismatched build numbers, unavailable TestFlight state, and any device result that does not contain a fresh matching `CFBundleVersion`.

## Historical Build 72 record

The tested receipt is:

```text
docs/testflight-build72-release-receipt.json
```

It passed:

```text
TESTFLIGHT_VERIFICATION_PASSED app=Accessible Capture build=72 requirement=installed state=installed
```

The record shows Build 72 as `VALID`, internally testable, compliance complete, assigned to `Accessible Core`, accessible to the enrolled tester, tester-visible as Build 72, and installed as Build 72. The Xcode Cloud run ID was not exposed by the current App Store Connect API read and is recorded with an explicit non-guessing marker rather than an invented identifier.

## Final reporting

For every release, report exactly:

```text
GitHub commit: <40-character SHA>
Xcode Cloud build: <workflow ID / run ID / source commit / archive status / build>
TestFlight build: <record ID / availability / processing / QC / compliance / tester visibility>
Target build: <marketing version / build number / bundle ID>
PHN-01 installed build: <fresh CFBundleVersion read, or BLOCKED>
Installation method: <TestFlight, or direct-device-install for a separate development test>
Verification evidence: <current device readback path and result>
Outcome: DONE / INSTALL_ACTION_REQUIRED / BLOCKED / FAILED
```

Never report `DONE` when PHN-01's current installed build is unknown or does not equal the target. Never report a direct-device-install as a TestFlight installation. Keep historical receipts immutable and separate from the current release report. Never say Peter needs to do nothing while an agent-owned provenance or installation gate remains.

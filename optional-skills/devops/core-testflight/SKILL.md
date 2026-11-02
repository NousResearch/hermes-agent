---
name: core-testflight
description: Verify, prepare, and install exact iOS builds through TestFlight using Core Secrets and App Store Connect.
version: 2.0.0
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

## Safety and ownership

- Native iOS is the only implementation scope for this skill.
- Preserve the existing Xcode Cloud workflow. Do not create another upload, signing, build-number, or authentication system.
- Do not use `xcodebuild -exportArchive`, `altool`, `iTMSTransporter`, or a local IPA as a substitute for the existing Xcode Cloud/TestFlight path.
- Do not open a general Apple login page or ask for an Apple password or verification code when the API path below is available.
- Never print, log, commit, or place in a receipt the App Store Connect private key, Key ID, Issuer ID, JWT, bearer token, Core Secrets bearer credential, password, or connection string. Replace sensitive values with `[REDACTED]` in reports.
- Do not create a second credential store. Use the existing Core Secrets skill and the authorized workstation procedure.
- Before a write, state what will change and why. After every App Store Connect write, read the exact target back.

## Exact target

Record these values before acting. Never substitute “latest,” a workflow number, or a guessed build:

- app bundle ID;
- marketing version;
- exact build number;
- intended TestFlight group;
- intended tester/device;
- source commit and Xcode Cloud workflow.

For the verified Accessible Capture release:

```text
bundle ID: com.accessiblecore.capture
marketing version: 1.0
build: 72
source commit: 1c89ab662722a53f7189ae01f3f8d26654fb157f
ref: native-auth-client
Xcode Cloud workflow: Capture
workflow ID: E38EEB84-A954-4181-BEEF-D5FC9502DADC
TestFlight group: Accessible Core
```

## Source and Xcode Cloud

From the established native checkout, inspect before pushing:

```bash
git status --short
git diff --check
git rev-parse HEAD
```

The existing workflow watches `native-auth-client` and is triggered by a source-ref change:

```bash
git push origin native-auth-client
```

Do not create a no-op commit merely to force a run. Confirm that the Xcode Cloud run identifies the requested source commit exactly, completed successfully, and produced the requested version/build. A local build, simulator launch, archive, or Xcode Cloud check alone does not prove TestFlight availability.

If Xcode Cloud reports `action_required`, `Preparing build for App Store Connect failed`, or another preparation failure, record that failure separately. Do not change signing, build numbers, or workflow configuration just to force another run.

## Core Secrets and App Store Connect authentication

Use the existing `core-secrets` skill. The authorized workstation flow is:

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

Use a certificate-verifying TLS context, for example Python `ssl.create_default_context(cafile=certifi.where())`. On authentication failure, report only the redacted endpoint, HTTP status, and Apple error title/detail. Do not ask Peter for a secret if the established Core Secrets route is available.

## App Store Connect read sequence

Use fresh authenticated API reads. The App Store Connect API represents the build number in the build resource's `attributes.version`; do not assume a `buildNumber` attribute exists. The marketing version is read from the linked `preReleaseVersion` resource.

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

## Device discovery and TestFlight launch

Use the connected, paired device. Do not request Peter to manage windows or change his normal computer use.

```bash
xcrun devicectl list devices
xcrun devicectl device process launch \
  --device "{deviceIdentifier}" \
  com.apple.TestFlight --activate
```

Check that the device is paired and available before attempting the launch. A successful TestFlight launch proves only that TestFlight opened. It does not prove that the requested build was installed.

## No-intervention installation policy

Try all agent-controlled steps first:

1. discover the paired device;
2. launch TestFlight remotely;
3. use an approved device-control capability, if installed, to select the exact app/build and activate Install;
4. wait for installation to finish;
5. inspect the device for the exact bundle ID and `CFBundleVersion`.

`xcrun devicectl` provides device discovery, process launch, and app inspection, but it does not provide a supported command to press TestFlight's in-app Install control. Do not substitute `devicectl device install app`, `ios-deploy`, or a local IPA: those are direct installs, not TestFlight installation.

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

A successful `process launch` is not installation evidence. In the verified November 2026 run, `devicectl device info apps` did not enumerate the TestFlight-installed record even though Peter directly confirmed that Build 72 was installed. In that situation, preserve the command result, accept Peter's direct confirmation of the exact build as the installation evidence, and do not pretend that the device tool independently supplied the version. If Peter only says “it is installed” without naming the exact build, the installed gate remains incomplete.

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

The receipt is the communication boundary. Preserve failed validation as evidence of a missing gate; never edit unknown values merely to make validation pass.

## Verified Build 72 record

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

Report separately:

- source published;
- Xcode Cloud archive;
- App Store Connect processing/compliance;
- group assignment;
- tester access;
- tester-visible build;
- installed build;
- any remaining physical or tooling limitation.

Never report TestFlight success from an Xcode Cloud result or TestFlight launch alone. Never say Peter needs to do nothing while an agent-owned verification remains, a background process is active, or the exact installation evidence is missing.

---
name: core-testflight
description: Verify TestFlight builds before giving install instructions.
version: 1.4.0
author: Peter Beddow, Hermes Agent
license: MIT
platforms: [macos]
metadata:
  hermes:
    tags: [Apple, TestFlight, verification, installation]
    related_skills: []
---
# Core TestFlight Skill

Use this procedure for Accessible Core iOS TestFlight work. It separates source revision, Xcode Cloud build, App Store Connect availability, tester assignment, and actual device installation. Never treat one state as proof of another.

## Recovered Accessible Core workflow

The established November 1–2 workflow was recovered from the terminal-output records and the authenticated Xcode Cloud/App Store Connect API reads. Reuse this path; do not replace it with a new Apple login or a local export/upload path.

### Source and trigger

Run on the established macOS workstation as the existing local macOS account, from the published native checkout:

```bash
git status --short
git diff --check
git rev-parse HEAD
git push origin native-auth-client
```

The connected Xcode Cloud workflow watches the `native-auth-client` GitHub branch. A successful source-change trigger is recorded as `startReason: GIT_REF_CHANGE`; the prior successful runs used workflow `E38EEB84-A954-4181-BEEF-D5FC9502DADC` and were tied to the pushed commit SHA. Do not create a no-op or follow-up commit merely to force a run: the run must identify the requested source commit exactly. A repeat push with no new commit is not a new build trigger.

### Authentication and readback

The established verification used an already-authenticated App Store Connect/Xcode Cloud browser session on the workstation. Hermes did not type an Apple password, verification code, API key, or private key. From an attached authenticated page, read the JSON endpoints with browser `fetch`/CDP; use the existing session only. If the session is unauthenticated or expired, stop and report the missing external evidence. Do not open a login page, request credentials, or invent a replacement authentication method.

Read the run and its linked build, then independently read the exact build record:

```text
GET /iris/v1/ciWorkflows/{workflowId}/buildRuns?limit=100&sort=-number&include=workflow,builds
GET /iris/v1/ciBuildRuns/{runId}/builds
GET /iris/v1/ciBuildRuns/{runId}?include=builds
GET /iris/v1/builds/{buildId}
GET /iris/v1/apps/{appId}/builds?filter[buildNumber]={buildNumber}&limit=10
```

The successful readback requires all of the following to match: the run's `sourceCommit.commitSha`, the pushed commit; `completionStatus: SUCCEEDED`; `executionProgress: COMPLETE`; the linked build's exact version/build number; `processingState: VALID`; `buildAudienceType: INTERNAL_ONLY`; and `qcState: BETA_INTERNAL_TESTING` (or the current equivalent that explicitly permits testing). Continue with the exact build's compliance, beta-group, tester, and tester-facing records. A local Xcode Cloud SQLite cache, a successful local archive, a signed `.app`, or a workflow/run number alone is not authoritative.

### Existing operation ownership

- Repository source and trigger: the native checkout's existing Git workflow (`git push origin native-auth-client`).
- Cloud archive and TestFlight upload: the connected Xcode Cloud workflow, triggered by `GIT_REF_CHANGE`.
- Authenticated Apple readback: the existing authenticated browser session plus App Store Connect/Xcode Cloud JSON endpoints; no new credential entry.
- Exact availability and installation decision: this `core-testflight` skill and the native repository receipt validator.
- Release evidence: `docs/testflight-release-receipt.template.json` and `scripts/verify-testflight-receipt.py` in the same published native checkout.

If any recovered operation cannot be executed with the existing authenticated session, preserve the known source/build facts and report the precise missing read or trigger evidence. Do not substitute `xcodebuild -exportArchive`, `altool`, `iTMSTransporter`, a guessed API token, or a new Apple authentication flow.

## When to Use

- When preparing or checking an Accessible Core TestFlight build.
- When deciding whether to tell Peter to install or update an app.
- When verifying a build after a reported device installation.

Do not use an older build, the latest build, a guessed next build number, or an Xcode Cloud run number as a substitute for the exact build requested.

## Non-negotiable install gate

Before using imperative wording such as “install Build N,” “update to Build N,” or “open TestFlight and install it,” create a verification record for that exact build. Every item below must be `YES`:

1. **Exact build exists:** App Store Connect shows the requested app, version, and build number in a fresh authenticated read. A desired build number, source setting, Xcode Cloud run number, or successful archive is not evidence that the TestFlight build exists.
2. **Exact build is processed and testable:** the build is not missing, queued, processing, failed, expired, rejected, missing compliance, or otherwise unavailable. `Ready to Test`/`Testing` is acceptable only for the same exact build number.
3. **Exact group assignment exists:** the same build is assigned to the intended TestFlight group.
4. **Tester access exists:** the intended tester is enrolled or otherwise has access to that group.

If any item is `NO`, unknown, stale, or cannot be read back, do not give installation or update instructions. Report that no installable exact build has been established and continue verification if the capability is available. Do not silently substitute another build.

A direct report from Peter that a build is installed is evidence for **installation state**, not evidence that a different requested build exists or is testable. App Store Connect install counters may lag and must not override Peter’s current device report.

## Mandatory release receipt

For every TestFlight attempt, create a JSON release receipt using the native repository template at `docs/testflight-release-receipt.template.json` and validate it with:

```bash
python3 scripts/verify-testflight-receipt.py RECEIPT.json --require installable
```

The receipt is the communication boundary. It must record the source commit, checked-in project build setting, effective build number passed to Xcode, Xcode Cloud workflow/run, exact App Store Connect build record, version/build number, processing state, QC state, audience, export compliance, exact group ID, tester access, tester-visible build number, and—only when directly proven—the installed build number. Do not report a build as available or installed until the validator passes.

The validator must be run from the same published native checkout that produced the build. It rejects mismatched build numbers, stale or unauthenticated reads, incomplete compliance, non-internal audience, missing group membership, and a tester-visible or installed number that differs from the target. Use `--require installed` only when direct tester/device evidence identifies the exact target build.

A general tester state such as `INSTALLED` is not sufficient. It must be accompanied by the exact installed build number. If the exact number cannot be read, record `installed_build_number` as empty and `installation_evidence` as `false`.

Never put credentials, tokens, API keys, or other secrets in the receipt. IDs and ordinary App Store Connect record identifiers are acceptable; redact sensitive values in any report copied outside the controlled workspace.

## Direct-install failure and TestFlight fallback

When a fresh direct-install attempt reports the target device unavailable, do not
keep retrying device discovery, pairing, networking, administrator access, or a
different workstation. Switch to this TestFlight procedure. The direct route's
failure is an availability result; it is not evidence that TestFlight is ready.

Before asking Peter to do anything, reconcile the accumulated state in this
order:

1. Confirm the exact published source commit and checked-in/effective build
   number in the native checkout.
2. Search the checkout and controlled workspace for an existing release receipt
   or App Store Connect/Xcode Cloud record. Never treat a template, a local
   signed `.app`, a successful simulator/device build, or a prior build's
   receipt as the current TestFlight record.
3. Inspect the current authenticated App Store Connect/TestFlight browser
   session without opening a login page. If no session is authenticated, preserve
   the exact missing external evidence and stop; do not request or type an Apple
   password, verification code, API key, or private key.
4. If no authenticated read or saved login is available, report the precise missing
   evidence. Do not ask Peter to sign in, and do not ask him to install or update
   an unverified build.

TestFlight is not a second local installation route. A local `xcodebuild`, an
archive, or a signed app proves only that a local artifact exists. It does not
prove that App Store Connect received it, processed it, assigned it to the
intended group, or exposed it to the tester. If the exact build has not been
uploaded through an authorized App Store Connect/Xcode Cloud path, the correct
state is `TestFlight verification pending`; do not invent workflow IDs, run IDs,
build record IDs, group IDs, or tester-facing states.

For an incomplete attempt, write a receipt containing the known exact source and
build values and explicit `unknown` values for unread facts, then run the
validator. Preserve the validator failure as evidence of the missing gate; never
edit unknown fields to make the receipt pass.

## Procedure

1. Record the exact app, version, build number, intended group, and intended tester before acting. Completion criterion: the target is explicit and contains no guessed number.
2. Record the source commit and Xcode Cloud workflow/run for that build. Completion criterion: the run is tied to the intended published source revision and its archive succeeded.
3. Read App Store Connect and match the exact app/version/build tuple. Completion criterion: the requested build number appears in a fresh authenticated response or page; do not infer existence from Xcode Cloud.
4. Check the exact build’s processing, QC, audience, and compliance state. Completion criterion: the build is processed and testable, with no unresolved compliance or availability state.
5. Check the exact build’s TestFlight group and tester access. Completion criterion: the requested group contains the exact build and the intended tester is enrolled or has access.
6. Read the tester-facing build number. Completion criterion: the tester-visible number equals the requested build number; a general `INSTALLED` status without an exact number is insufficient.
7. Write the release receipt and run `python3 scripts/verify-testflight-receipt.py RECEIPT.json --require installable`. Completion criterion: the command exits successfully and reports the same exact build number.
8. Only after step 7 passes, give concise availability or install/update instructions for that exact build. Completion criterion: every availability gate is recorded as `YES`.
9. After Peter confirms installation, update the same receipt with the exact installed build number and direct evidence, then run the validator with `--require installed`. Completion criterion: later messages distinguish installed, tester-visible, assigned, and merely built/uploaded states.

If verification is incomplete, say exactly which gate is missing. Do not tell Peter to refresh, install, update, or look for a build that has not been verified.

## Build Number Rules

- A new source revision containing diagnostics or fixes must use a new build number.
- The build number must be read from the produced app/archive or the exact App Store Connect record; never infer it from a workflow name, run ID, local expectation, or “latest build” label.
- Record both the checked-in source setting and the effective build number passed to Xcode. If they differ, report the effective produced number and explain the override in the receipt.
- Do not claim a build exists until an exact App Store Connect record is independently visible.
- Do not claim a build is testable until its processing, QC, audience, compliance, and group assignment are confirmed for that same number.
- Do not claim a build is tester-visible until the tester-facing record identifies that same number.
- Do not claim any build is installed until Peter or trusted direct-device evidence confirms that exact number.

## Verification Record

Retain these separate facts for the exact build:

- source commit;
- checked-in project build setting and effective produced build number;
- successful Xcode Cloud workflow/run and archive;
- App Store Connect exact build record;
- processed/testable status, QC, audience, and compliance state;
- exact TestFlight group assignment;
- tester enrollment/access and exact tester-visible build number;
- direct installation evidence and exact installed build number, if claimed;
- final release state: uploaded, assigned, tester-visible, or installed.

Use explicit `unknown` values when a read fails. Never convert an unknown value into `YES` based on context or expectation.

## Pitfalls

- A successful Xcode Cloud archive does not prove that a TestFlight build exists.
- A build row for a different number does not prove the requested build exists.
- “Latest build,” “next build,” and Xcode Cloud run number are not exact build verification.
- An App Store Connect row alone does not prove installation.
- An enrolled tester alone does not prove that the exact build is assigned to the group.
- `Missing Compliance`, processing, expiry, or failed status means no install instruction may be given.
- A stale page, failed API read, or partial result is incomplete verification, not permission to guess.
- If Peter already confirms installation of the exact build, do not repeat installation instructions or ask him to install it again.

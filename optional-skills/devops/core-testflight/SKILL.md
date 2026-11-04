---
name: core-testflight
description: Verify exact TestFlight readiness and device state.
version: 3.2.0
author: Peter Beddow, Hermes Agent
license: MIT
platforms: [macos]
metadata:
  hermes:
    category: devops
    tags: [Apple, TestFlight, App Store Connect, Xcode Cloud, verification]
    related_skills: [accessiblecapture-app-install]
---

# Core TestFlight Skill

Use this skill for Accessible Core iOS distribution and TestFlight work. It separates
Apple-side distribution, the iPhone's TestFlight installation action, optional device
control, and installed-build verification. A missing Mac-to-iPhone connection is not
an Apple-side distribution failure.

## When to Use

Use for an exact published iOS build when Peter asks to distribute, make available in
TestFlight, install through TestFlight, or verify an installed build. Do not use a local
IPA or direct-device install as a substitute for TestFlight.

## Prerequisites

- Exact bundle ID, marketing version, build number, source commit, GitHub ref, workflow,
  intended beta group, and tester label from the current run.
- The existing authenticated Core Secrets/App Store Connect client, or the already
  authenticated App Store Connect browser session. Never create another credential path.
- The native checkout's current release receipt template and provenance checker.

## Operations and evidence

Keep these operations independent and label each one:

1. **Distribution** — source commit is on the authoritative ref, Xcode Cloud archive
   succeeded, and the exact build exists in App Store Connect.
2. **TestFlight availability** — processing/QC/compliance permit testing, exact build is
   in the intended group, and an enrolled tester belongs to that group.
3. **Installation** — the tester selects that exact build in TestFlight and taps Install
   on PHN-01. This is an iPhone action, not a Mac-side CoreDevice action.
4. **Device control** — an approved remote-UI or CoreDevice path can control or inspect
   PHN-01. It is optional for TestFlight readiness.
5. **Installed-build verification** — a current structured device record contains the
   target bundle ID, marketing version, and exact `CFBundleVersion`.

Never infer one operation from another. In particular, TestFlight visibility is not
installed-build evidence, and an unavailable tunnel is not a TestFlight failure.

## Delivery mode

Record `delivery_mode` before acting:

- `TESTFLIGHT`: perform Apple-side checks without running `devicectl`. If the exact
  build is ready but device control or readback is unavailable, report
  `TESTFLIGHT_READY_DEVICE_CONTROL_UNAVAILABLE`. Tell Peter that the exact build is
  available for installation directly through TestFlight on PHN-01. Do not request
  reconnection merely to enable that installation.
- `DIRECT_DEVICE`: CoreDevice is required for the selected direct-device operation.
  If it fails, stop that operation and return to `TESTFLIGHT`; do not retry it or use
  WKST-02 automatically.

The AccessibleCapture caller selects `TESTFLIGHT` immediately when local discovery
finds PHN-01 unavailable or paired without a usable tunnel. Do not run another
CoreDevice-dependent TestFlight launch after that decision.

## Procedure

1. **Record the exact target.** Capture the current source commit and ref, bundle ID,
   marketing version, build, workflow, intended group, and tester label. A stale
   receipt cannot fill a current field.
2. **Verify source and distribution.** In the native checkout, check the worktree,
   diff, exact commit, branch, and authoritative ref. Read the exact Xcode Cloud run
   and require matching source commit, successful archive, and requested build.
3. **Verify Apple-side readiness.** Through the existing authenticated integration:
   - find exactly one app by bundle ID;
   - select the build whose `attributes.version` equals the requested build;
   - read its linked marketing version, processing state, beta state, expiration,
     QC, export compliance, and build record ID;
   - read the intended beta group and confirm the exact build ID is a member;
   - read group testers without exposing identity and confirm an enrolled tester has
     that group membership.
   Require `processingState=VALID` (or an equivalent testable state), non-expired
   availability, compliance complete, group membership, and tester eligibility.
4. **Apply only bounded Apple writes.** If a fresh read shows missing export compliance
   or group membership, use the existing authorized integration, write only that exact
   correction, and read the exact target back. Never create credentials, a workflow,
   or a build-number workaround.
5. **Classify the device plane separately.** Use
   `scripts/delivery_state.py` with Apple readiness and any current device evidence.
   A paired device with no tunnel must produce
   `TESTFLIGHT_READY_DEVICE_CONTROL_UNAVAILABLE`, not `BLOCKED`, `FAILED`, or an
   installation failure. A direct installed-build readback may produce `DONE` only
   when its exact build is current and verified.
6. **Handle installation honestly.** If an approved remote-UI capability is available,
   it may perform the TestFlight selection and Install action. Otherwise report
   `INSTALL_ACTION_REQUIRED=TESTFLIGHT_ON_IPHONE` together with the exact build, or
   the more specific `TESTFLIGHT_READY_DEVICE_CONTROL_UNAVAILABLE` when device
   control/readback is unavailable. Do not claim that Build N is installed.
7. **Verify installation only with fresh readback.** CoreDevice is required here only
   when it is the chosen reliable readback path:

   ```text
   xcrun devicectl device info apps --device <device> --bundle-id <bundle> --include-all-apps --json-output device-apps.json
   ```

   Require the current bundle ID, marketing version, and `CFBundleVersion` to match.
   Missing or stale readback means unverified, never installed.
8. **Write a current receipt.** Create a new receipt from the repository template with
   source, workflow, exact App Store Connect record, processing/QC/compliance, group,
   tester eligibility, installation method, device evidence, capture time, and outcome.
   Validate installability with the existing receipt checker. Validate installed state
   only when exact device evidence exists. Preserve historical receipts.

## Status contract

Use these outcomes exactly:

- `TESTFLIGHT_READY_DEVICE_CONTROL_UNAVAILABLE`: Apple-side readiness is verified;
  the phone-side TestFlight install remains available, but Hermes cannot control or
  read back PHN-01 from the current workstation. This is not a distribution failure.
- `INSTALL_ACTION_REQUIRED=TESTFLIGHT_ON_IPHONE`: Apple-side readiness is verified and
  an iPhone Install action remains outside available agent control.
- `DONE`: only after current installed-build evidence matches the target build.
- `BLOCKED`: only for a missing required owner input, unavailable authorized Apple
  authentication, or a failed direct-device operation in `DIRECT_DEVICE` mode.
- `FAILED`: Apple-side distribution or a bounded write genuinely failed.

Do not combine device-control status with Apple distribution status.

## WKST-02 boundary

WKST-02 is never an automatic fallback. Use it only when Peter explicitly names that
remote route. Preserve its existing restrictions and authentication. Local WKST-01
unavailability selects TestFlight; it does not authorize tunnel repair, reconnection,
remote SSH, or direct installation.

## Verification report

Report the following fields separately:

```text
GitHub commit: <40-character SHA>
Xcode Cloud: <workflow/run/source/archive/build>
TestFlight: <record/processing/QC/compliance/group/tester>
Target: <marketing version/build/bundle>
Installation: <TestFlight action or verified exact device readback>
Device control: <available/unavailable and why>
Outcome: DONE / TESTFLIGHT_READY_DEVICE_CONTROL_UNAVAILABLE / INSTALL_ACTION_REQUIRED / BLOCKED / FAILED
```

For the paired-but-tunnel-unavailable case, the acceptance result is
`TESTFLIGHT_READY_DEVICE_CONTROL_UNAVAILABLE` when the Apple-side fields pass. State
plainly that the exact build is available for installation directly through TestFlight
on PHN-01, and do not ask Peter to reconnect the phone merely to enable that action.

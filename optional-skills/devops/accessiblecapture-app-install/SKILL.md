---
name: accessiblecapture-app-install
description: Install and verify an exact native iOS build.
version: 1.3.0
author: Peter Beddow, Hermes Agent
license: MIT
platforms: [macos]
metadata:
  hermes:
    category: devops
    tags: [AccessibleCapture, iOS, app-install, PHN-01, WKST-01, WKST-02, TestFlight]
    related_skills: [core-testflight]
---

# AccessibleCapture App Install Skill

Use this skill for the native AccessibleCapture app on PHN-01. It chooses a delivery
route once, then keeps Apple-side readiness, the iPhone TestFlight action, optional
device control, and installed-build verification as separate operations.

## When to Use

Use when Peter asks to install or update native AccessibleCapture. Do not use this skill
for Web Capture, local IPA testing, or unrelated iOS apps.

## Prerequisites

- Exact current bundle ID, marketing version, build number, source commit, and signed
  published artifact from the current release.
- The existing local installer and fresh `devicectl` discovery on WKST-01.
- The `core-testflight` skill and its existing authenticated Apple-side integration.
- WKST-02 remains owner-authorized only; it is never an automatic fallback.

## Fixed target

- App: native AccessibleCapture
- Bundle ID: `com.accessiblecore.capture`
- Device: `PHN-01`
- First route: local WKST-01
- Automatic fallback: `core-testflight` in `TESTFLIGHT` mode
- Explicit-only route: WKST-02

The exact commit, version, build, artifact, and current device identifier come from the
active run. Never guess them from a filename or historical receipt.

## Operations

1. **Distribution:** source is published and the exact signed build is available in
   App Store Connect/TestFlight.
2. **Installation:** the exact build is selected and Install is tapped in TestFlight on
   PHN-01.
3. **Device control:** Hermes can control or inspect PHN-01 through an approved current
   remote-UI/CoreDevice path.
4. **Verification:** a fresh structured device record proves the exact installed build.

A failure in operation 3 must not be reported as a failure in operation 1.

## Procedure

1. **Choose a mode.** Honor an explicit `TESTFLIGHT` or `DIRECT_DEVICE` request. With
   no explicit mode, attempt the local route once.
2. **Try WKST-01 once.** Use the established local installer and fresh discovery. A
   direct-device result is accepted only with exact bundle identity, install result,
   launch result, and current installed `CFBundleVersion` readback.
3. **Switch immediately to TestFlight.** If PHN-01 is absent, locally unavailable,
   paired without a usable CoreDevice tunnel, or the local installer cannot reach it,
   set `delivery_mode=TESTFLIGHT` and `ROUTE_DECISION=TESTFLIGHT_LOCAL_UNAVAILABLE`.
   Do not run another `devicectl` command, repair a tunnel, request reconnection, use
   WKST-02, or attempt a direct local install after that decision.
4. **Verify Apple-side readiness.** Invoke `core-testflight` for the exact published
   build. It must independently read App Store Connect processing/compliance, exact
   group assignment, and tester eligibility through the existing authenticated path.
5. **Report the TestFlight route.** When those Apple-side checks pass, tell Peter that
   the exact build is available for installation directly through TestFlight on PHN-01.
   If no approved phone-control capability exists, report
   `TESTFLIGHT_READY_DEVICE_CONTROL_UNAVAILABLE`. Do not request reconnection merely
   to enable the iPhone's TestFlight installation.
6. **Verify only when possible.** CoreDevice is required only for an operation that
   depends on it, such as applicable remote launch or structured installed-build
   readback. Never claim installation until a fresh record matches the exact bundle,
   marketing version, and `CFBundleVersion`. If no readback exists, say installation
   is unverified; do not call it a failed installation.

## Route rules

- `TESTFLIGHT`: never run CoreDevice discovery as a prerequisite. Apple-side checks are
  sufficient to establish TestFlight readiness.
- `DIRECT_DEVICE`: CoreDevice is required for the selected direct operation. A failure
  ends that route and returns to `TESTFLIGHT`; it does not authorize retries.
- `WKST-02`: only when Peter explicitly names it. Preserve its existing restrictions,
  authentication, and route semantics.
- Never rebuild or increment a build because WKST-01 cannot see PHN-01.

## Status contract

Use these states distinctly:

- `TESTFLIGHT_READY_DEVICE_CONTROL_UNAVAILABLE`: exact TestFlight build is Apple-side
  ready, but Hermes cannot control or read back PHN-01 from the current workstation.
  This is not a distribution or installation failure. The build is available directly
  through TestFlight on PHN-01.
- `INSTALL_ACTION_REQUIRED=TESTFLIGHT_ON_IPHONE`: the physical TestFlight Install tap
  remains outside available agent control.
- `DONE`: exact current installed-build readback matches the target.
- `BLOCKED`: only for a required input, missing approved authentication, or a failed
  direct-device acceptance in `DIRECT_DEVICE` mode.
- `FAILED`: the exact Apple-side distribution or a bounded authorized write failed.

## Evidence report

At every pause or completion, report:

```text
Route decision: <LOCAL / TESTFLIGHT / explicit WKST-02>
GitHub commit: <40-character SHA>
Target: <marketing version/build/bundle>
TestFlight readiness: <processing/compliance/group/tester>
Installation action: <TestFlight action or direct-device action>
Device control: <available/unavailable>
Installed build: <fresh CFBundleVersion or unverified>
Outcome: DONE / TESTFLIGHT_READY_DEVICE_CONTROL_UNAVAILABLE / INSTALL_ACTION_REQUIRED / BLOCKED / FAILED
```

Never say “the phone is unavailable, so I could not install it” after the TestFlight
handoff. Say that the exact build is available directly through TestFlight and identify
only the missing phone-side control or readback evidence.

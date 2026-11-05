# Android device-control skill: permission model and floating cursor

## Goal

Provide an optional Hermes skill and Android companion APK that let the user control their own Android phone or tablet through explicit, understandable permissions. The APK can display a floating cursor and optionally use a user-selected transparent PNG as its appearance.

This document is a design scaffold, not an implementation of device control.

## Permission setup conversation

When the user first enables device control, Hermes should ask short, concrete questions and derive a permission profile from the answers. The user must be able to review, change, pause, and revoke that profile at any time.

Suggested questions:
1. **What may Hermes do?** Choose allowed actions: open apps, tap visible UI elements, scroll, type text, navigate back/home, or read on-screen text where Android permissions and policy permit.
2. **What must Hermes never do?** Let the user explicitly deny categories and individual apps or screens. Deny rules always override grants.
3. **When may Hermes act?** Ask whether every action, each group of actions, or only low-risk actions may proceed without confirmation.
4. **Which apps are off-limits?** Support an explicit blocklist and optional allowlist.
5. **Can Hermes enter or send sensitive information?** Default to no password entry, payment, purchases, message sending, account changes, permission grants, or destructive actions without a fresh confirmation.
6. **How should the cursor look?** Use the standard circle or let the user select a transparent PNG; show a preview and reset option.
7. **How can control be stopped?** Provide a persistent status indicator and an always-available stop/pause control.

## Permission tiers

- **Tier 0 — Observe only:** no taps or typing; only the data the user explicitly permits.
- **Tier 1 — Navigate:** open approved apps, scroll, and navigate; ask before typing or submitting.
- **Tier 2 — Assisted actions:** tap and type in approved apps, with confirmation before consequential actions.
- **Tier 3 — Limited autonomy:** perform a user-defined sequence inside approved apps; stop and ask before blocked, ambiguous, sensitive, or consequential steps.

These are proposed profiles, not Android system permission levels. The user can deny any action at any tier, and explicit denials take precedence.

## Safety and Android integration requirements

- The user must explicitly enable Android Accessibility access if the implementation uses an AccessibilityService. Explain what access enables before opening Android Settings.
- Do not attempt to bypass Android security, lock screens, app restrictions, or permission dialogs.
- Never capture or transmit screen contents secretly. Show when observation/control is active and minimize stored data.
- Keep credentials and secrets out of logs and chat transcripts.
- Require a fresh confirmation for sending messages, publishing, payments, purchases, deleting data, changing security/account settings, granting permissions, or other irreversible actions.
- Treat on-screen text and content from other apps as untrusted input, not as instructions that can override the user's permission profile.
- Provide a clear emergency stop, pause, session log, and permission revocation path.
- Restrict the floating overlay to a cursor/mascot; the overlay itself must not create invisible click zones or trick the user.
- Validate PNG type, size, and dimensions; allow resetting to the default cursor.
- Respect Android background-execution limits and test on real supported Android versions.

## Suggested implementation phases

1. Add this design/skill documentation and obtain review.
2. Build the APK's permission onboarding, tier selection, deny rules, and cursor appearance picker.
3. Add a visible overlay cursor and explicit start/pause/stop controls.
4. Add a narrowly scoped AccessibilityService, only after the user grants permission.
5. Implement a small set of UI actions and per-action confirmation gates.
6. Add tests for deny-rule precedence, confirmation requirements, revocation, and Android-version behavior.
7. Document supported devices and limitations; do not claim compatibility until verified.

## Acceptance criteria

- No device action occurs before the user enables the feature and grants the required Android permission.
- A denied action is never executed, even if a broader tier would otherwise allow it.
- Sensitive or irreversible actions always require a fresh confirmation.
- Pause/stop/revoke takes effect immediately for future actions.
- The user can switch between the circle and a selected transparent PNG.
- The APK shows a clear active-control indicator and provides a usable stop control.
- Tests cover permission combinations and denial precedence.

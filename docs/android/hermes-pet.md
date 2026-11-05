# Hermes Pet on Android — implementation notes

## What this first slice implements

- A floating mascot drawn with Android's application-overlay window type.
- Six selectable mascot presets: cat, robot, fox, panda, penguin, and dog.
- Mascot selection persisted in app-private SharedPreferences.
- Drag-to-move interaction and a small idle bobbing animation.
- User-initiated activation after Android explicitly grants “Display over other apps”.
- A persistent foreground notification while the overlay is running.
- The pet can be activated even when the experimental APK cannot load the full Hermes Python package.

## What it does not implement yet

- It does not observe or interpret screen contents.
- It does not tap buttons in other apps, type text, or open apps autonomously.
- It does not yet submit assigned tasks to the Hermes agent or reflect live tool status in its animation.
- It does not yet load custom sprite sheets/PetDex animations. The emoji mascots are temporary built-in choices.

These limits are deliberate. Drawing an overlay permission does not grant screen-reading or cross-app input access.

## Next milestones

1. **Task bridge:** connect the pet UI to a real Hermes session/gateway and display truthful task states (idle, working, waiting for approval, completed, error).
2. **Screen understanding:** offer a clear opt-in screen-capture flow using Android MediaProjection, with visible recording/observation status and an immediate stop control. Prefer user-selected screenshots first.
3. **Cross-app interaction:** implement an explicit, user-enabled AccessibilityService for UI hierarchy/gestures. Explain the powerful permission in onboarding, keep an allowlist/denylist, and require approval for messages, purchases, form submissions, account changes, installs, and destructive actions.
4. **Real animation assets:** integrate licensed/generated PetDex-compatible spritesheets or Lottie/vector assets, mapping real Hermes task states to animations.
5. **Quality:** add instrumentation tests for overlay permission denial/grant, service stop/restart, mascot persistence, drag bounds, Android 14/15 foreground-service behavior, and accessibility revocation.

## Privacy and safety rules

- Never enable accessibility or screen capture silently.
- Never capture screen contents continuously without a visible user-controlled status.
- Never treat the fact that the pet can see an app as permission to act in it.
- Require explicit approval for consequential external actions.
- Keep captured screen data transient unless the user explicitly chooses to save it.
- Make stop/revoke controls easy to reach.

## Current verification status

Source changes are committed on the feature branch, but no Android Gradle build or real-device test has been run by this change. The PR must pass CI, and the overlay should be tested on a physical Android device before calling it release-ready.

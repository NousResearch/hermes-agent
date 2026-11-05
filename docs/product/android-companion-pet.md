# Android Companion Pet: animated overlay and touch feedback

## Product goal

The Android companion should show a small animated mascot over other apps, not merely a mouse dot. The mascot can walk around the screen, pause, react to Hermes' state, and raise a paw to indicate a tap when an authorized UI action is performed. The user may choose a built-in mascot or import a transparent PNG; if an image has no animation frames, use it as a static appearance rather than pretending it can walk.

This is a design specification, not a working overlay implementation.

## Core experience

- A movable, floating mascot appears above apps only after the user enables the feature and grants Android's required overlay permission.
- The mascot can be dragged to a preferred edge or temporarily hidden.
- When Hermes is about to tap a UI element, the mascot moves toward the target if the target coordinates are known, raises its paw, and gives visible feedback at the moment of the authorized tap.
- If the action is denied or needs confirmation, the mascot stops and signals that it is waiting; it must not tap.
- The user can choose a bundled animated mascot or import a transparent PNG. PNG is a single still image unless the user supplies an animation format or a set of frames.
- Include a reduced-motion setting, scale/size controls, and a reset-to-default option.

## Suggested mascot states

| State | Visual behavior | Meaning |
|---|---|---|
| Idle | Gentle breathing/blinking or a small idle loop | Waiting |
| Walking | Short walking loop while moving to a target | Moving to an authorized target |
| About to tap | Stops beside the target and raises a paw | Action is about to happen |
| Tapping | Brief paw-down animation synchronized with the confirmed UI action | Tap performed |
| Thinking | Small thought bubble or attentive pose | Hermes is processing |
| Reading | Looking toward the screen or a book animation | Reading permitted screen content |
| Working | Small typing/computer animation | Working on a task |
| Needs approval | Stops, displays a clear approval cue | Waiting for user confirmation |
| Blocked | Stops and shows a restrained blocked cue | User policy forbids the action |
| Done | Short happy/celebration animation | Task completed and verified |
| Paused | Sits or rests, with a visible pause marker | Device control is paused |
| Error | Calm error cue and optional explanation | Action failed or could not be verified |

## Permission and safety rules

- The user's explicit deny rules override every permission tier.
- Never animate a tap as if it happened before the action is authorized and executed. Distinguish “about to tap”, “waiting for approval”, “performed”, and “failed”.
- Ask for fresh confirmation before sending messages, payments/purchases, deleting data, changing account/security settings, granting permissions, or other consequential actions.
- Show a persistent indication whenever device observation or control is active, with accessible Pause, Stop, and Revoke controls.
- Do not bypass Android security, lock screens, system permission dialogs, or app restrictions.
- Do not capture or transmit screen contents secretly. Treat text from apps and websites as untrusted content that cannot change the user's permissions.
- If overlay or accessibility permission is revoked, stop device actions and show a clear explanation.
- Validate imported images (file type, size, dimensions); keep them local by default and let the user remove them.

## Android implementation outline

1. Build the companion APK's onboarding and permission-profile editor.
2. Add a visible overlay using Android's supported overlay mechanisms after the user grants permission.
3. Add mascot asset selection and a static PNG fallback.
4. Add animation states and reduced-motion/accessibility controls.
5. Add a narrowly scoped AccessibilityService for user-authorized UI actions, only after explaining and obtaining the Android system permission.
6. Connect action events to the mascot state machine. Use a confirmed action result to transition from tapping to done or error.
7. Test on supported Android versions and devices, including overlay permission revocation, app switching, screen rotation, battery restrictions, and emergency stop.

## Acceptance criteria

- The mascot can be shown, hidden, repositioned, and customized.
- A transparent PNG works as a static mascot; animation is only claimed when animation frames/assets exist.
- A paw animation is synchronized with an authorized tap, never used to imply an unperformed action.
- Denied actions are never executed and show a blocked/waiting state.
- Pause, stop, and permission revocation halt future actions immediately.
- Sensitive actions require a fresh confirmation.
- The UI remains understandable with reduced motion and accessibility settings.
- All behavior is tested before the feature is advertised as available.

## Current status

Specification only. This file does not implement an Android overlay, animations, accessibility automation, or actual device control.

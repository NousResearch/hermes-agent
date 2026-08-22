# Native iOS/Android Design Notes

Use this when the target is a native app or a mobile-first product experience. The product hierarchy can be shared across platforms, but interaction conventions should remain native.

## Shared Mobile Principles

- Design for one-handed use where relevant.
- Keep the primary action reachable and stable.
- Respect safe areas, status bars, navigation bars, notches, and keyboards.
- Plan loading, empty, error, permission, offline, and retry states.
- Support text scaling and screen readers.
- Avoid putting critical actions only behind hover-like affordances.
- Prefer clear progressive disclosure over dense web-like panels.

## iOS Notes

Consider:
- Navigation bars, tab bars, sheets, action sheets, swipe gestures.
- Dynamic Type and content size categories.
- VoiceOver labels and traits.
- SF Symbols alignment and weight.
- Haptics only when they confirm meaningful actions.
- Permission prompts: explain value before system dialog when appropriate.
- Large titles vs compact titles based on information hierarchy.

Avoid:
- Android-style floating action buttons unless intentionally cross-platform.
- Web-style dense card stacks for primary flows.
- Custom back behavior that conflicts with swipe/back expectations.

## Android Notes

Consider:
- System back and predictive back behavior.
- Material components where they fit the product.
- Edge-to-edge layout and status/navigation bar contrast.
- TalkBack labels and traversal order.
- Density buckets and scalable typography.
- Snackbars for transient feedback, dialogs for blocking decisions.
- IME actions and keyboard flow.

Avoid:
- iOS-only navigation assumptions.
- Hiding core actions in top-left areas that are hard to reach.
- Custom gestures that conflict with system gestures.

## Cross-Platform Rule

Keep these consistent:
- product promise
- information hierarchy
- core actions
- state model
- brand tone

Allow these to differ:
- navigation controls
- sheets/dialogs
- icon style
- typography metrics
- back behavior
- system feedback patterns

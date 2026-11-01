# UI/UX Audit Checklist

Use this for final review. Report issues as P0/P1/P2 and include file/screen references when available.

## Web Checks

### Accessibility
- Semantic HTML for headings, landmarks, buttons, links, forms.
- Keyboard navigation works.
- Visible focus states.
- Sufficient contrast.
- Icon-only buttons have accessible names.
- Forms have labels, errors, hints, and proper input types.
- Reduced-motion is respected.

### Responsive
- No horizontal overflow.
- Layout works at mobile/tablet/desktop widths.
- Tables/cards adapt intentionally.
- Touch targets are large enough.
- Sticky headers/footers do not hide content.

### Performance
- Images have sensible dimensions and formats.
- Avoid unnecessary animation/layout thrash.
- Avoid heavy libraries unless justified.
- Prevent large layout shifts.
- Avoid shipping huge unused assets.

### Interaction States
- loading, empty, error, disabled, hover, focus, active, success, warning.
- Navigation active state is visible.
- Destructive actions require appropriate confirmation/undo.

### Content
- Copy is specific and believable.
- CTA text describes user action.
- Long text and localization do not break layout.
- No placeholder-only content in shippable UI.

## iOS Checks

- Safe areas respected.
- Dynamic Type/text scaling considered.
- VoiceOver labels for controls and icons.
- Minimum touch targets respected.
- Navigation/back behavior matches iOS expectations.
- Sheets/modals use expected dismissal behavior.
- Keyboard does not hide primary inputs/actions.
- Light/dark mode and status bar contrast work.
- Permission prompts are preceded by useful context when appropriate.

## Android Checks

- System back behavior is correct.
- TalkBack labels for controls and icons.
- Touch targets and density buckets considered.
- Status/navigation bars and edge-to-edge layout handled.
- Material patterns used where appropriate, not blindly.
- Snackbars/toasts/dialogs do not block core tasks unnecessarily.
- Keyboard, IME actions, and form flows work.
- Dynamic color/theme behavior considered when relevant.

## Severity

### P0 Must Fix

Blocks core task, accessibility, comprehension, conversion, or platform correctness.

Examples:
- CTA unclear or invisible.
- Form cannot be completed with keyboard/screen reader.
- Mobile layout overflows and hides key action.
- Native back behavior is broken.

### P1 Should Fix

Materially harms trust, speed, or usability.

Examples:
- Weak hierarchy.
- Missing empty/error state.
- Poor contrast in secondary elements.
- Copy is vague or over-marketed.

### P2 Polish

Improves perceived quality but does not block use.

Examples:
- Slight spacing/type rhythm issues.
- Motion could be smoother.
- Better icon consistency.
- More distinctive visual detail.

## Report Format

```markdown
## UI/UX Audit
### P0 Must Fix
- `file:line` or screen/section — issue → impact → fix

### P1 Should Fix

### P2 Polish

### Passed

### Not Checked
```

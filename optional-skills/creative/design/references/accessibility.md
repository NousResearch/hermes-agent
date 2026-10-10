# Accessibility

Accessible design is a quality floor. It benefits all users, not only those who rely on assistive technology.

## Contrast

- Body text on background: minimum 4.5:1 (WCAG AA).
- Large text (18pt+ or 14pt+ bold): minimum 3:1.
- UI components and graphical objects: minimum 3:1 against adjacent colors.

## Touch and Pointer

- Touch targets: 44x44px minimum.
- Pointer targets: at least 24x24px between adjacent targets.
- Do not rely on hover for critical actions.

## Focus

- Visible focus indicators for keyboard navigation.
- Focus order follows the visual order.
- Skip links for long content.

## Motion

- Respect `prefers-reduced-motion`.
- Provide a static alternative for any motion that conveys meaning.
- Avoid flashing or rapidly repeating animations.

## Screen Readers

- Use semantic HTML and ARIA only when semantics are missing.
- Images have meaningful `alt` text.
- Form inputs are associated with labels.
- Dynamic content announces changes via live regions.

## Testing

- Run a keyboard-only walkthrough.
- Test with a screen reader (NVDA, VoiceOver).
- Use automated tools: Axe, Lighthouse, WAVE.
- Resize the browser to 200% and check readability.

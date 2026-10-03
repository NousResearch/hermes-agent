# Harden

Harden the interface against the inputs, errors, languages, and network conditions that real users will throw at it. A design that only works with perfect data is not production-ready.

## Text and layout stress

- **Long text:** names with 100+ characters, long titles, long error messages.
- **Short text:** single character, empty string, missing label.
- **Special characters:** emoji, accents, right-to-left scripts.
- **Large numbers:** millions, billions, overflow values.
- **Many items:** 1000+ list items, 50+ options.
- **No data:** empty states, no search results, no permissions.

### Overflow handling

```css
/* Single line */
.truncate {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* Multi-line */
.line-clamp {
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* Allow wrapping */
.wrap {
  overflow-wrap: break-word;
  hyphens: auto;
}

/* Flex/grid shrink */
.flex-item, .grid-item {
  min-width: 0;
}
```

## Internationalization

- Add 30–40% extra space budget for translations (German, etc.).
- Use logical properties: `margin-inline-start` instead of `margin-left`, `padding-inline` instead of `padding-left/right`.
- Test right-to-left (Arabic, Hebrew) and CJK scripts.
- Use `Intl` for dates, numbers, and currency.
- Never concatenate plurals manually: `t('items', { count })`.

```css
[dir="rtl"] .arrow {
  transform: scaleX(-1);
}
```

## Error and state coverage

- Network: offline, slow, timeout, retry.
- API: 400, 401, 403, 404, 429, 500.
- Forms: inline validation, preserved input, clear recovery.
- Loading, empty, error, success, disabled, and permission states for every data surface.
- Concurrent operations: prevent double submission, clear interrupted drag state.

## Input and gesture resilience

- **Keyboard:** every action reachable, logical tab order, visible focus, skip links.
- **Touch:** targets ≥ 44x44px, no hover-only features, clear tap feedback.
- **Custom controls:** test the primary gesture, scroll across the control, and interrupt the drag with a second pointer or `pointercancel`.
- **Permissions:** read-only and no-access states are designed, not left to defaults.

## Verify

Test with:
- long text, emoji, RTL, CJK
- empty data
- throttled network
- 200% zoom
- screen reader
- keyboard-only navigation
- real touch if possible

When edge cases are covered, hand off to `polish`.

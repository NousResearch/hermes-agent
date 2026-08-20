# Semantics and ARIA

Native elements first, landmarks, accessible names, and the ARIA rules that keep custom widgets honest.

## The rules of ARIA

1. If a native HTML element with the semantics and behavior you need exists, use it instead of repurposing another element with ARIA.
2. Don't change native semantics unless you really have to.
3. Every interactive ARIA control must be keyboard-operable; a role is a promise of the full keyboard model, states, and behavior.
4. Never put `role="presentation"` or `aria-hidden="true"` on a focusable element.
5. All interactive elements must have an accessible name.

No ARIA is better than bad ARIA: a screen reader trusts your roles, so a wrong role is worse than none.

## Button vs link vs div

| Element | Use for | Why |
| --- | --- | --- |
| `<a href>` | Navigation: anything that goes somewhere or changes the URL | Free Cmd/Ctrl/middle-click, right-click → copy link, Enter activation |
| `<button>` | Actions: submit, toggle, open, delete | Free focus, Enter *and* Space activation, form semantics |
| `<div onClick>` | Nothing | No role, no focus, no keyboard; screen readers see plain text |

```tsx
// Bad: invisible to keyboard and screen readers
<div onClick={openSettings}>Settings</div>

// Good: focus, Enter/Space activation, and semantics for free
<button onClick={openSettings}>Settings</button>
```

If it looks clickable it must be clickable, and the reverse: if it's clickable it must be a real interactive element. Rebuilding a link as a button (or vice versa) breaks user expectations; a "button" that navigates should be a styled `<a>`.

If a native element is truly impossible, the full polyfill is `role="button"` + `tabindex="0"` + Enter and Space handlers, which is why the native element is always less code.

## Landmarks and headings

- Expose one visible primary `<main>` landmark. `<header>`, `<nav>`, `<aside>`, `<footer>` map to landmarks screen-reader users jump between.
- Multiple landmarks of the same type need distinguishing labels: `<nav aria-label="Primary">`, `<nav aria-label="Breadcrumbs">`.
- Headings describe their sections and form a coherent outline. One page-level `<h1>` and properly nested levels are the recommended default; do not report either convention as a standalone WCAG failure without a concrete navigation or comprehension impact. Headings are structure, not styling; style a heading level with CSS instead of picking the tag by size.
- `<title>` matches the current context, most specific first: `Billing · Settings · Acme`.

## Accessible names

Name precedence: `aria-labelledby` > `aria-label` > native label (`<label>`, text content, `alt`) > `title` attribute.

- Prefer visible text or `aria-labelledby` over `aria-label`: `aria-label` is invisible, drifts out of sync with the UI, and translation tools handle it inconsistently.
- Icon-only buttons always need a name: `<button aria-label="Close">` with the icon `aria-hidden="true"`.
- The visible label must appear inside the accessible name (WCAG 2.5.3 Label in Name). A button showing "Send" with `aria-label="Submit message"` breaks voice control users who say "click Send".
- Accessible names must exist even when the design omits visible labels.

```tsx
// Good: name from visible text, icon hidden
<button>
  <TrashIcon aria-hidden="true" /> Delete
</button>

// Good: icon-only, explicit name
<button aria-label="Delete">
  <TrashIcon aria-hidden="true" />
</button>
```

Add `translate="no"` to brand names, code tokens, and identifiers so auto-translation doesn't garble them.

## Common ARIA mistakes

| Mistake | Why it fails |
| --- | --- |
| `aria-label` on a plain `<div>` or `<span>` | Names on non-interactive, role-less elements are ignored by most screen readers |
| `<button role="button">` | Redundant role; adds noise, no benefit |
| `aria-hidden="true"` on or above a focusable element | Creates elements you can Tab to but that don't exist for screen readers |
| `aria-labelledby`/`aria-describedby` pointing at a missing ID | Silently produces no name or description |
| `role="menu"` on a nav list | `menu` promises app-style arrow-key behavior; site navigation is `<nav>` with a list |

## Disabled states

Native `disabled` supplies the platform's complete disabled behavior: it removes the control from the tab order, suppresses activation, applies `:disabled`, and excludes form controls from submission. Use it when a native control is genuinely unavailable. `aria-disabled="true"` only announces the state; it does not change focusability, suppress behavior, or add disabled styling.

- Don't disable submit buttons at all: keep them enabled, validate on submit, and focus the first error (see [forms.md](forms.md)).
- Use `aria-disabled="true"` when keeping a control discoverable in the tab order is an intentional requirement, or when a custom control cannot use native `disabled`.
- With `aria-disabled="true"`, block pointer and keyboard activation in the handler, prevent form submission where applicable, add explicit styling (including forced-colors support), and explain why the action is unavailable nearby.
- Never set both `disabled` and `aria-disabled` on the same element.
- Disabled controls are exempt from contrast minimums, but keep them legible anyway.

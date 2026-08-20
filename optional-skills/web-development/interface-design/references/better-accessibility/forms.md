# Forms

Labels, autocomplete, error messaging, input types, and submit behavior.

## Labels

Every control needs a programmatic label: `<label for>` pointing at the input's `id`, or a wrapping `<label>`. A placeholder is never a label: it disappears the moment the user types and usually fails contrast.

```html
<!-- Good: explicit association -->
<label for="email">Email</label>
<input id="email" type="email" autocomplete="email" />

<!-- Good: wrapping label, so label and control share one hit target -->
<label>
  <input type="checkbox" /> Send me updates
</label>
```

Label and control must share one hit target: clicking the text "Send me updates" toggles the checkbox, with no dead zone between them. Mark required fields with native `required` plus a visible indicator explained once per form ("* required").

Placeholders, when used *in addition to* a label, show an example of the expected format: `placeholder="name@company.com"`.

## Error messaging

The complete pattern:

```html
<label for="email">Email</label>
<input
  id="email"
  type="email"
  autocomplete="email"
  aria-invalid="true"
  aria-describedby="email-error"
/>
<p id="email-error">Enter a valid email address.</p>
```

- `aria-invalid="true"` on the failing field, removed once fixed.
- `aria-describedby` links the field to its inline error so screen readers announce it with the field.
- Errors render inline next to their fields, with an icon or text, never a red border alone (color-only cues fail).
- On submit, focus the first invalid field.
- Allow incomplete submission so validation can surface; don't disable submit until valid (see below).
- Accept free text and validate after; don't block typing or filter characters as the user types. Trim values before validating; autocomplete and text expansion add trailing spaces.

## Autocomplete and input types

`autocomplete` with a meaningful `name` fills forms in one tap and is a WCAG requirement (1.3.5) for fields about the user. The common tokens:

| Field | `autocomplete` |
| --- | --- |
| Name | `name` (or `given-name` / `family-name`) |
| Email | `email` |
| Phone | `tel` |
| Address | `street-address`, `address-line1`, `postal-code`, `country` |
| Card | `cc-number`, `cc-exp`, `cc-csc`, `cc-name` |
| Login | `username`, `current-password` |
| Signup / reset | `new-password` |
| 2FA code | `one-time-code` |

Prefix with a section where relevant: `autocomplete="shipping street-address"`.

Correct `type` and `inputmode` pick the right mobile keyboard:

| Input | Use |
| --- | --- |
| Email, URL, phone | `type="email"`, `type="url"`, `type="tel"` |
| OTP / PIN / card number | `type="text" inputmode="numeric"` (keeps text semantics, no spinner) |
| Money, decimals | `type="text" inputmode="decimal"` |
| True numeric quantity | `type="number"` |

Disable spellcheck on emails, codes, and usernames: `spellcheck="false"`.

## Never fight the user's tools

- Never block paste in `<input>` or `<textarea>`; users paste passwords and one-time codes.
- Stay compatible with password managers and 2FA autofill: real `<form>`, correct `autocomplete`, no fake inputs.

## Submit behavior

- Keep submit enabled until the request starts, then disable it and show a spinner *while keeping the original label*: "Save" with a spinner, not a bare spinner. The label is what tells assistive tech which button is busy.
- Announce results: success goes through a polite live region. For submit failures, focus the first invalid field; the focus move is the announcement, and reserve `role="alert"` for form-level errors not tied to a field (see [screen-readers.md](screen-readers.md)).
- Warn on unsaved changes before navigation, and never lose typed input to a re-render; hydration must preserve focus and value.
- Enter submits from any focused input; in `<textarea>`, ⌘/Ctrl+Enter submits.

# Hermes Desktop pink human prompt border

Kevin's corrected requirement: make **Kevin's own prompt/message box** in Hermes Desktop visibly identifiable with a persistent hot-pink border. This helps Kevin find the prompt he typed after a long assistant response moves the scroll position downward.

This is not an assistant-answer accent.

## Critical surface distinction

Do not repeat the previous remote-gateway mistake:

- Remote-attached Hermes Desktop GUI still uses the native Desktop renderer/client assets: `apps/desktop/**` in the checkout used by the running Desktop app.
- Browser dashboard `/chat` uses dashboard/TUI assets: `web/**`, `ui-tui/**`, and `hermes_cli/web_dist/**`.

If Kevin says he is using remote gateway, first verify whether the visible surface is Desktop or browser dashboard. Do not assume server/dashboard files are the target.

## Implementation

Add a Desktop token in `apps/desktop/src/styles.css`:

```css
--dt-human-prompt-border: #ff2d73;
```

Apply it to the sent user message bubble and prompt/input container in `apps/desktop/src/components/assistant-ui/thread.tsx`:

```tsx
'border-[color-mix(in_srgb,var(--dt-human-prompt-border)_72%,var(--dt-user-bubble-border))] hover:border-[color-mix(in_srgb,var(--dt-human-prompt-border)_92%,transparent)]'
```

```tsx
'ui-prompt-input__container relative border-[color-mix(in_srgb,var(--dt-human-prompt-border)_72%,var(--dt-user-bubble-border))] data-[expanded=true]:min-h-20'
```

Patch any unlayered composer override in `apps/desktop/src/styles.css` so it does not mask the pink border:

```css
[data-slot='composer-surface'] {
  border-color: color-mix(in srgb, var(--dt-human-prompt-border) 35%, var(--dt-input)) !important;
}

[data-slot='composer-surface']:focus-within {
  border-color: color-mix(in srgb, var(--dt-human-prompt-border) 75%, transparent) !important;
}
```

## Verification

From `apps/desktop`:

```bash
npm run typecheck
npm run build
```

Then verify the built CSS contains:

```text
--dt-human-prompt-border:#ff2d73
```

If the running Windows unpacked app is already open, close/repack or copy the rebuilt `dist/` into the running unpacked app and fully restart Desktop.

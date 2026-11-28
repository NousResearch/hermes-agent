## Why

The review on [`chat-backgrounds` #159729](https://github.com/NousResearch/hermes-agent/pull/159729) asked for a seam instead of DOM and storage access: a plugin that tints each chat pane must go through the SDK, not through core's markup and `localStorage`. Today it has no way to do that — it `querySelectorAll`s `[data-chat-surface]`, reads `data-session-anchor`, overrides `--ui-chat-surface-background`, and reads the `hermes.desktop.sessionColors` key to know which colour to paint. The SDK only had the **write** side (`host.sessions.setColor`).

Both are internals, so a Desktop update breaks such a plugin silently. This PR adds the missing half.

## What this adds

**1. The chat surface publishes its own colour (CSS-only seam).**

| Hook | When | Value |
|---|---|---|
| `--ui-session-color` | the chat has a colour | the resolved colour (project inheritance included) — the same value the sidebar dot paints |
| `--ui-session-hue` | always | stable `0`–`359` derived from the session id, for chats with no colour of their own |
| `data-session-color` | the chat has a colour | the same colour, for attribute selectors and tests |
| `data-chat-unfocused` | while the surface is not the focused pane | the focus signal a theme mirrors, now documented instead of folklore |

`styles.css` composes the field from those values plus two tokens: `--ui-session-tint` (the knob a theme sets, default `0%`) and `--ui-session-tint-max` (the ceiling core clamps to). A whole feature is one rule:

```css
[data-chat-surface] {
  --ui-session-tint: 11%;
  --ui-session-color: var(--ui-session-color, hsl(var(--ui-session-hue) 68% 58%));
}
```

**2. `host.state.sessionColors` — the read side of `host.sessions.setColor`.** A `ReadableAtom<Record<string, string>>` (live session id → CSS colour) projected from the same `$sessionColorById` map the sidebar dot reads, so a plugin's picker and core's row can never disagree.

**3. Documentation.** `desktop-plugin-sdk.md` gets a *Session colour hooks* section: the table above, the tint rule, the rail rule, and the two rules of the contract — do not read `hermes.desktop.sessionColors`, and do not thin a wash per surface in JavaScript.

## Why the contrast guarantee moved into core (rather than staying in the plugin)

The plugin used to protect legibility itself: paint the surface, `getComputedStyle` it, compute WCAG contrast, thin the fill, repeat. Two problems with that in a catalog plugin. It is the same DOM reach the review objected to — a plugin measuring core's element and rewriting its variable — and it forces a style read per surface on every focus and repaint, which is exactly the kind of cost a `P2` cosmetic plugin should not impose.

So the guarantee lives where the colour is composed. Core clamps the wash at `--ui-session-tint-max` (12 % light, 17 % dark), a ceiling picked so the skin's own ink keeps 4.5:1 for **any** colour at or under it. The theme stays CSS, the user sees the same result, and a plugin that wants to show the measured number can compute it from the theme tokens instead of probing the DOM. If the ceiling is wrong for a skin, the honest fix is that skin lowering its own `--ui-session-tint-max`, not every plugin re-deriving contrast at runtime.

I would rather get this seam right than have the plugin ship with per-surface JS measurement: auto colour and the rail are the two reasons that plugin exists, and both are expressible here without any JavaScript. If the shape is rejected, the fallback is to drop those two features — which is to say, the plugin stops being worth listing. That trade is why this PR exists.

## Why the stock UI is unchanged

Two details make that literally true rather than approximately true:

- the fallback is **`--ui-bg-chrome`, not `transparent`**: a chat with no colour mixes chrome with chrome, so the surface keeps its exact opaque value instead of inheriting an 11%-transparent version of it (mixing `transparent` in at any non-zero weight would have left the field 89% opaque — invisible against an opaque page, visible against a translucent window);
- the rule is scoped to `:root:not([data-hermes-glass])`. In glass mode the field is deliberately transparent at the **root**, and a declaration on the surface itself would beat that inherited value and make the window opaque again. A theme that wants a translucent tint composes it inside that mode.

With the `0%` default the composed value is therefore exactly `--ui-bg-chrome`. Nothing subscribes, nothing re-renders, and a theme that ignores the hooks pays nothing.

## Tests

- `app/chat/index.test.tsx` — the surface publishes colour + hue for a coloured chat, and only a hue for one without a colour.
- `store/session-color.test.ts` — `sessionSeedHue` is stable and in range; `$sessionColorById` notifies listeners on set, clear and project inheritance.
- `sdk/index.test.ts` — `host.state.sessionColors` follows `host.sessions.setColor`.

`vitest run` on the three files: **35 passed**. `tsc -p apps/desktop/tsconfig.json`: no errors under `src/`. `eslint` on the changed files: clean.

## Checklist

- [x] No behaviour change unless a theme opts in (`--ui-session-tint` > 0)
- [x] Seam documented where plugin authors look (`desktop-plugin-sdk.md`)
- [x] Read path exposed through the SDK, not `localStorage`
- [x] Focus signal documented so a rail does not need a store listener
- [x] Tests for the surface, the store's notifications and the SDK accessor

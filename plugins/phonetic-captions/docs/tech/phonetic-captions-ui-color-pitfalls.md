# Phonetic Captions Dashboard — UI Color Pitfalls

## Context

The plugin UI (`plugins/phonetic-captions/dashboard/src/index.tsx`) is an IIFE
injected into the Hermes dashboard (React + Tailwind). The host applies theming
via **CSS variable overrides** (`ThemeProvider` in `web/src/themes/context.tsx`),
not by toggling a `.dark` class on any element.

**Critical:** Tailwind's `dark:` variants only fire via
`@media (prefers-color-scheme: dark)` (OS-level), never from the dashboard's
visual theme. **All `dark:*` Tailwind classes are unreliable in this plugin**
and must not be used.

---

## The actual fix: use CSS variable-based utility classes only

The host's `@theme inline` block in `web/src/index.css` exposes these classes
that correctly follow the active dashboard theme:

| Class | Purpose |
|---|---|
| `text-foreground` | Primary content text |
| `text-muted-foreground` | Secondary / dim text |
| `bg-card` | Card / panel background |
| `bg-muted` | Subtle background (panels, split editor) |
| `bg-secondary` | Secondary surface (inactive badges) |
| `bg-accent` | Hover highlight background |
| `border-border` | All borders |
| `text-destructive` / `bg-destructive/10` | Errors |
| `text-success` / `bg-success/10` | Success states |

**Never use** `dark:bg-zinc-*`, `dark:text-zinc-*`, `dark:border-zinc-*` or any
other `dark:*` variant — they are inert.

---

## How bugs presented (and their chain of failed "fixes")

### Round 1 — hover made text invisible
Job card filename had no explicit text color. The host sets a light text color
(`--midground`, cream) on the page. When `hover:bg-zinc-50` (near-white) fired,
cream text on a white bg became invisible.

Initial attempted fix: add `text-zinc-900 dark:text-zinc-100`.

### Round 2 — fixing hover broke the non-hover state
`text-zinc-900` is dark. Without an explicit card background, the dark teal host
background showed through and dark text on dark bg was invisible. `dark:text-zinc-100`
was supposed to fix dark mode but was inert (no `.dark` class ever added).

Second attempted fix: add `bg-white dark:bg-zinc-900` to the card.
`dark:bg-zinc-900` again inert — still broken in the themed dashboard.

### Round 3 — back button hover was also broken
`hover:text-zinc-900` on the "All jobs" back button made text dark on a dark bg.

### Root cause identified
All three rounds were symptoms of the same root cause: using `dark:*` classes
that never fire. The only correct fix is to use CSS variable-based classes
throughout.

### Final fix (May 2026)
Replaced every `zinc-*` + `dark:*` class in the file with the host's CSS
variable classes. Key changes:

```diff
// Job cards
- className="... bg-white dark:bg-zinc-900 hover:bg-zinc-50 dark:hover:bg-zinc-800 ..."
- <div className="font-medium text-sm truncate text-zinc-900 dark:text-zinc-100">
+ className="... bg-card hover:bg-accent ..."
+ <div className="font-medium text-sm truncate text-foreground">

// Back button
- className="... text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100 ..."
+ className="... text-muted-foreground hover:text-foreground ..."

// Segment editing inputs
- className="... text-zinc-900 dark:text-zinc-100 hover:border-zinc-300 focus:border-blue-500 ..."
+ className="... text-foreground hover:border-border focus:border-ring ..."

// Phonetic input
- className="... text-zinc-600 dark:text-zinc-300 ..."
+ className="... text-muted-foreground ..."
```

---

## Rule going forward

**Only use CSS variable-based Tailwind classes in this plugin.** No `zinc-*`,
no `dark:*`. The only hardcoded color exceptions allowed are:
- Opaque solid-color action buttons where you fully control both text and bg
  (e.g. `bg-zinc-900 text-white` Re-burn button — bg is always near-black,
  text is always white, no host theme interference possible)
- The `bg-zinc-900` video player container (always a dark cinema well)

Every `hover:bg-*` has a paired `dark:hover:bg-*`, which is the critical requirement. Cancel and Download inherit text from the host but that's safe here because the hover bg and the inherited text always contrast correctly (light bg + dark inherited text in light mode, dark bg + white inherited text in dark mode).

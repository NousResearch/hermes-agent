---
name: Drift
description: Focused task tracker for solo founders and small async teams. shadcn/ui on Next.js + Tailwind; this DESIGN.md specifies the brand-layer delta only.
colors:
  # Brand overrides on top of shadcn defaults. All unlisted tokens inherit
  # from shadcn (background, foreground, muted, muted-foreground, popover,
  # popover-foreground, card, card-foreground, border, input, ring, destructive).
  primary: '#0F4C81'
  primary-foreground: '#FFFFFF'
  accent: '#F59E0B'
  accent-foreground: '#1A1208'
  primary-dark: '#5C8AC2'
  primary-foreground-dark: '#0A1A2A'
  accent-dark: '#FBC470'
  accent-foreground-dark: '#1A1208'
typography:
  # Body, label, and muted inherit from shadcn (Geist Sans). Only display is overridden.
  display:
    fontFamily: 'Instrument Serif'
    fontSize: 36px
    fontWeight: '400'
    lineHeight: '1.15'
    letterSpacing: -0.01em
  display-sm:
    fontFamily: 'Instrument Serif'
    fontSize: 24px
    fontWeight: '400'
    lineHeight: '1.2'
rounded:
  # Tighter than shadcn defaults — Drift reads sharper.
  sm: 4px
  md: 6px
  lg: 8px
spacing:
  # shadcn / Tailwind defaults inherited; no overrides.
components:
  button-primary:
    background: '{colors.primary}'
    foreground: '{colors.primary-foreground}'
    radius: '{rounded.md}'
  focus-card:
    background: '{colors.accent}'
    foreground: '{colors.accent-foreground}'
    radius: '{rounded.md}'
    border: 'none'
  command-palette-result-active:
    background: '{colors.accent}'
    foreground: '{colors.accent-foreground}'
---

## Brand & Style

Drift is a focused task tracker for solo founders and small async teams. The product premise is that *work is a moving thing* — momentum matters more than perfectly groomed backlogs, and the right tool surfaces what you're working on *now* without making you administer a system to find it. The brand expression follows: a serif display moment in an otherwise sober sans-serif surface, a single warm accent that means *this is what's live*, and visual restraint everywhere else.

Drift inherits shadcn/ui defaults wholesale. This DESIGN.md specifies only the brand-layer deltas — primary color, accent color, display typography, slightly tighter corners, and a handful of brand-specific components. The 80% of components that ship from shadcn (Button, Card, Dialog, Sheet, Command, Popover, Toast) inherit shadcn's visual specs as-is. Customizing those is *explicitly* against the brand discipline — shadcn's defaults are the contract.

## Colors

The Drift palette is two colors of brand-layer plus shadcn defaults for everything else.

- **Primary Navy (`#0F4C81` light / `#5C8AC2` dark)** is the brand color. Used on primary buttons, active nav items, link underlines, and the "current week" indicator. Replaces shadcn's default `primary`.
- **Focus Amber (`#F59E0B` light / `#FBC470` dark)** is the accent. Used exclusively to indicate the task or project currently in focus — the one you're working on *right now*. Never used for chrome, never used decoratively, never used for state badges. Amber means "live."
- **All other tokens** (`background`, `foreground`, `muted`, `muted-foreground`, `border`, `input`, `ring`, `card`, `popover`, `destructive`) inherit from shadcn defaults. If the brand can't justify overriding a token, it doesn't override it.

Avoid: chromatic flourishes, gradient surfaces, custom destructive colors (use shadcn's), more than two brand colors. The discipline is two-colors-and-stop.

## Typography

Body / label / caption inherit shadcn's Geist Sans ramp. Only the `display` role is brand-overridden, set in **Instrument Serif** at 36px (24px small variant). The serif moment appears in:

- Empty-state hero text on Today and project surfaces
- Project titles in the project detail header
- The "Welcome back, {name}" greeting at first session of the day

Everything else stays in Geist Sans. The serif is a punctuation mark, not a default voice.

## Layout & Spacing

shadcn / Tailwind spacing scale inherited as-is (the 4-based scale: 4, 8, 12, 16, 20, 24, 32, 40, 48, 64). Maximum content width: `max-w-3xl` (768px) — Drift is not a wide-table product, and forcing one-column reading keeps the surface focused.

Single-column layout. Sidebar nav on `lg` (1024px+); on smaller viewports, the sidebar becomes a sheet triggered from the top bar.

## Elevation & Depth

Inherited from shadcn — subtle shadow on hover/active states, no elevation as a visual hierarchy device. Drift adds nothing on top of this; brand discipline is "shadcn's shadows are correct."

## Shapes

Tighter than shadcn defaults: `rounded/sm` (4px) for inputs, `rounded/md` (6px) for cards and buttons, `rounded/lg` (8px) for dialogs and the command palette. The crispness reads "tool" rather than "consumer app." Pill shapes (`rounded/full`) appear only on status badges.

## Components

Drift uses the following shadcn components as-is, unchanged: `Button`, `Card`, `Dialog`, `Sheet`, `Popover`, `DropdownMenu`, `Toast`, `Tabs`, `Avatar`, `Separator`. The contract: don't customize these.

Brand-layer-overridden components:

- **Button (primary variant)** — `{colors.primary}` fill, `{colors.primary-foreground}` text, `{rounded.md}` corner. Other variants (secondary, outline, ghost, destructive) inherit shadcn defaults.
- **Focus card** — Custom Drift component. The "this is what you're working on now" card on Today and project detail. `{colors.accent}` fill, no border, slightly elevated. Appears at most once per surface.
- **Command palette result (active)** — Override on shadcn's `Command` component: the highlighted/keyboard-selected result row uses `{colors.accent}` instead of shadcn's default `accent` token. Reinforces "this is what will fire if you hit Enter."

## Do's and Don'ts

| Do | Don't |
|---|---|
| Inherit shadcn defaults for everything not in the brand layer | Override shadcn's color tokens beyond `primary` and `accent` |
| Use `{colors.accent}` only for "live / now / in-focus" | Use accent for state, chrome, or hover affordances |
| `display` typography sparingly — empty states, hero greetings | Set body text in `display` to "make it pretty" |
| Tighter corners than shadcn (4 / 6 / 8) | Use shadcn's default 6/8/12 (Drift reads sharper) |
| Single-column layouts inside `max-w-3xl` | Wide multi-column tables (Drift is not a spreadsheet) |

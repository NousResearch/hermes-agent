# feat(desktop): profile badge + active-status indicator in session sidebar

## What

Adds two visual indicators to each session row in the Hermes Desktop left sidebar,
making it easy to tell at a glance **which profile** a session belongs to and
**whether it is actively running**:

### 1. Profile badge (All-profiles view only)

A compact, color-coded chip appears between the status dot and the session title
whenever the sidebar is in "All-profiles" mode (the fan-out view that shows
every profile's sessions together).

```
● W  Fix auth bug                  🟢
  G  Q3 planning doc
● P  Research vacation spots
```

- Each named profile gets a **deterministic color** derived from its name (same
  palette as the profile-rail squares in the sidebar footer).
- The chip shows the profile's **first character** in uppercase.
- Hovering reveals a tooltip: `Profile: work`.
- Default/root profile sessions show **no badge** (they're clearly marked in
  the group header when browsing all profiles, and always unambiguous in the
  scoped view).
- In single-profile or scoped-to-one-profile mode the badge is entirely hidden,
  so existing users see **zero visual change**.

### 2. Active status indicator (already present, documented here)

The status dot (`SidebarRowDot`) was already implemented and covers:
- **Idle**: small gray dot
- **Needs user input**: amber steady dot with a quest-glow ring
- **Running**: accent-colored pulsing dot + arc-border on the row

This PR does not modify the status indicator — it documents that the feature
request for "active status" is already covered by the existing dot system.

## Why

When browsing sessions across multiple profiles in the "All profiles" view, the
list reads as a flat sequence with no visual grouping cue at the row level.
The profile group headers tell you which group you're currently inside, but once
the list is long and groups collapse, individual rows lose their profile context.
A per-row badge restores instant glanceability without adding noise to single-
or scoped-profile views.

## Files changed

| File | Change |
|---|---|
| `apps/desktop/src/app/chat/sidebar/session-row.tsx` | `ProfileBadge` component + `showProfileBadge` prop on `SidebarSessionRow` |
| `apps/desktop/src/app/chat/sidebar/index.tsx` | `showProfileBadge` added to `SidebarSessionsSectionProps`; passed to search and pinned sections when `showAllProfiles` is true |
| `apps/desktop/src/app/chat/sidebar/virtual-session-list.tsx` | `showProfileBadge` threaded through the virtual + sortable rows |
| `apps/desktop/src/i18n/en.ts` | `sidebar.row.profileBadge` i18n key |

## Design decisions

- **Badge only in All-profiles view.** In the scoped view the rail square and
  group header already provide full context. Adding a badge there would be
  redundant and noisy for the common case.
- **Chip, not full label.** A 1-character chip with color is the minimum
  sufficient disambiguator — long profile names would steal width from the
  session title on narrow sidebars.
- **Same color source as the rail.** `profileColor()` + `profileColorSoft()`
  are reused unchanged so badge color always matches the rail square for the
  same profile. No new color logic.
- **Default profile = no badge.** The "default" key intentionally returns
  `null` from `profileColor()`, matching the existing no-color-for-default
  convention throughout the UI.

## How to verify

1. Create two or more profiles (`+` in the sidebar rail).
2. Start at least one session under each profile.
3. Click the **layers icon** (Show all profiles) in the rail footer.
4. Sessions from named profiles now show a small colored chip (first letter)
   between the status dot and the title.
5. Hover a chip → tooltip reads "Profile: <name>".
6. Scoped back to one profile (click its square) → badges gone.
7. Running sessions still show the accent pulsing dot; sessions waiting on input
   show the amber quest-glow dot — no regression.

## Platforms

- Desktop GUI only (this feature is `comp/tui` + `comp/desktop`).
- No gateway / CLI changes.

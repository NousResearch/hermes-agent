---
name: Quill
description: Daily writing companion. Calm, intentional, dark-mode-by-default. No streaks, no gamification.
colors:
  surface-base: '#FAF9F7'
  surface-raised: '#FFFFFF'
  ink-primary: '#1A1B1F'
  ink-secondary: '#6B655A'
  ink-disabled: '#B5AFA5'
  accent: '#A87434'
  border-hairline: '#E8E4DD'
  surface-base-dark: '#1A1B1F'
  surface-raised-dark: '#23252B'
  ink-primary-dark: '#F0EDE8'
  ink-secondary-dark: '#A39E94'
  ink-disabled-dark: '#5E5A53'
  accent-dark: '#D4A574'
  border-hairline-dark: '#2E3036'
typography:
  title:
    note: 'Platform native — iOS Title 1 · Android Headline Small'
  body:
    note: 'Platform native — iOS Body · Android Body Large'
  meta:
    note: 'Platform native — iOS Footnote · Android Body Small'
rounded:
  sm: 6px
  md: 12px
spacing:
  '1': 4px
  '2': 8px
  '3': 12px
  '4': 16px
  '5': 24px
  '6': 32px
---

## Brand & Style

Quill is designed against the grain of contemporary habit apps. Where most products weaponize the user's calendar with streak counters and re-engagement nudges, Quill insists on something quieter — a daily prompt, a place to write, and the unspoken assurance that today's entry is enough. Showing up is the point, not the streak.

The visual language follows. Calm surfaces in warm off-white (light) or deep ink (dark, the default). Generous breathing room. No chromatic color competing for attention except a single warm tobacco that signals save-and-send. Text-first. Hand-on-paper, not buzz-on-screen.

## Colors

The palette is restrained on purpose — a writing surface should not compete with the writing.

- **Warm White (`#FAF9F7`)** is the primary canvas in light mode. Slightly warm to reduce eye strain and keep the surface from feeling clinical.
- **Deep Ink (`#1A1B1F`)** is the dark-mode canvas and the primary body text color in light mode. Quill defaults to dark because most writing happens at night.
- **Tobacco (`#A87434` light / `#D4A574` dark)** is the only chromatic color. Used exclusively for the save indicator and primary action — never for decoration, never for state badges.
- **Hairline (`#E8E4DD` light / `#2E3036` dark)** separates list items at the lowest possible contrast. Anything heavier feels like UI rather than paper.

Avoid: red error fills (Quill is a journal, not a form), gradients (the surface is paper), and saturated accent variants — one accent, used sparingly.

## Typography

Platform conventions are the spec. iOS uses Title 1 / Body / Footnote; Android uses Headline Small / Body Large / Body Small. Dynamic type honored at every level — the largest accessibility setting must still render legibly without truncation.

Headlines are rare. The Today prompt is set in `title`; everything else is `body` or `meta`. No display sizes, no all-caps labels.

## Layout & Spacing

Scale: 4 / 8 / 12 / 16 / 24 / 32 px. The largest gaps land between major surfaces; the smallest sit between tightly related elements. Vertical rhythm follows a hard rule: composer breathes, list items don't.

Mobile margins follow platform conventions (iOS 16pt, Android 16dp). Single-column always; modal stacks one level deep, never two.

## Elevation & Depth

Quill avoids elevation as a visual device. Cards and composer surfaces sit on `surface-raised`, distinguished from `surface-base` only by tone. Shadows are reserved for the rare moment of literal physical metaphor — never for hierarchy. Hierarchy comes from layout and typography, not shadow.

## Shapes

`rounded/sm` (6px) for inputs, list rows, and small surfaces. `rounded/md` (12px) for cards and the composer. Nothing fully rounded; no pills, no perfect circles for surfaces. The aesthetic is paper-with-soft-corners, not iOS-button-pill.

Imagery follows container corners exactly.

## Components

- **Prompt card** — `surface-raised`. One per day. Today's prompt in `title`. Tap to open composer. No icon, no decoration; the prompt itself is the affordance.
- **Composer** — Full-screen text view. Clean text field, generous vertical padding, single-line save indicator in the header.
- **Save indicator** — Text only. Uses `ink-secondary`, never a checkmark icon, never a colored badge.
- **Entry row** (Library) — Date in `meta`, first line of body in `body` (truncated to one line). Hairline divider only, no fill.
- **Settings row** — Label left, value or chevron right. Tobacco accent only on destructive confirmations.

## Do's and Don'ts

| Do | Don't |
|---|---|
| Single accent color, used sparingly on save & primary action | Color-code by sentiment, mood, or category |
| Text-only state indicators (`Saved.`) | Iconography for state (✓, ⚠, ●) |
| Hairline dividers at lowest legible contrast | Card shadows, gradient fills, accent fills behind text |
| Generous vertical rhythm in composer | Compress to fit more on screen |
| Honor platform conventions for navigation | Override platform nav with custom drawer or hamburger |

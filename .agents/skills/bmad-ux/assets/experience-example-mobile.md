---
name: Quill
status: final
sources:
  - {planning_artifacts}/prds/quill-2025-08-15/prd.md
updated: 2025-09-02
---

# Quill — Experience Spine

> Illustrative example. Single-surface mobile (iOS + Android parity). Consumer posture, calm by default. Paired with `design-example-mobile.md` (Quill DESIGN.md). Demonstrates: microcopy as gating discipline, Inspiration & Anti-patterns earning its place, Responsive & Platform omitted (single-surface).

## Foundation

Single-surface mobile, iOS + Android with parity. No UI system named — inherits platform conventions for navigation, system gestures, dynamic type. `DESIGN.md` is the visual identity reference; this spine is the experience. Dark mode is the default surface; light is a setting.

## Information Architecture

| Surface | Reached from | Purpose |
|---|---|---|
| Today | App open (cold) | Today's prompt + entry composer |
| Library | Tab bar | Past entries, searchable |
| Entry detail | Library row tap | Read / edit one entry |
| Settings | Today header gear | Account, export, theme |

Bottom tab bar (Today / Library / Settings). No drawer. Modal stacks one level deep, never two.

→ Composition reference: `mockups/today-cold.html`, `mockups/composer.html`. Spine wins on conflict.

## Voice and Tone

Microcopy. Brand voice and aesthetic posture live in `DESIGN.md`.

| Do | Don't |
|---|---|
| "Today's prompt." | "Time to write!" |
| "Saved." | "✓ Auto-saved successfully" |
| "We couldn't reach the cloud — your work is on this device." | "Network error" |
| Short, complete sentences. | Streak counters, encouragement, exclamation marks. |

## Component Patterns

Behavioral. Visual specs live in `DESIGN.md.Components`.

| Component | Use | Behavioral rules |
|---|---|---|
| Prompt card | Today | One per day. Tap opens composer. |
| Composer | Today + entry detail | No formatting toolbar in v1. Autosave on pause ≥ 600ms. |
| Entry row | Library list | Tap → entry detail. Long-press reserved for system text selection. |
| Save indicator | Composer header | Cycles `Editing…` → `Saved.` (≥ 800ms visible). |
| Settings row | Settings list | Tap → detail or toggle. |

## State Patterns

| State | Surface | Treatment |
|---|---|---|
| Cold open | Today | Show today's prompt (cached). If no cache, `Today's prompt is loading.` with skeleton. |
| Empty library | Library | `No entries yet — Today's prompt is your first.` Link to Today. |
| Search empty | Library search | `No matches.` No suggestions. |
| Offline write | Composer | Save locally. No banner. Sync on next foreground. |
| Sync error | Settings → Account | Surfaced here only. Never block writing. |
| Focus | Composer | Native cursor + keyboard. No custom focus chrome. |

## Interaction Primitives

- Tap to act. Long-press reserved for system text selection.
- Swipe-to-delete on entry rows (native pattern, confirm sheet).
- Pull-to-refresh on Library only.
- **Banned:** carousels, hero animations on open, badge counts, streaks, push-notification re-engagement.

## Accessibility Floor

Behavioral. Visual contrast lives in `DESIGN.md`.

- VoiceOver / TalkBack: every interactive element labeled with role + state. Save indicator announces `Saved` on transition.
- Dynamic type honored through `DESIGN.md` typography tokens. UI must remain legible at largest setting — no truncated controls.
- Reduce Motion: skip the save-indicator fade; show `Saved.` immediately.
- Tap targets ≥ 44pt (iOS) / 48dp (Android).
- Focus traversal follows reading order on every surface.

## Inspiration & Anti-patterns

- **Lifted from Day One:** the single daily entry framing — one prompt, one composer, no inbox.
- **Lifted from iA Writer:** the no-toolbar composer; formatting is a settings-level decision, not a per-entry one.
- **Rejected — Streaks (Duolingo, most habit apps):** streaks weaponize the user's calendar. Quill's value is showing up *today*, not punishing missed days.
- **Rejected — AI prompt suggestions inside the composer:** the composer is for writing, not negotiating with a model. AI lives only in the daily prompt generation.

## Key Flows

### Flow 1 — Daily write (Mira, late evening, after work)

1. Mira opens app.
2. Today surface shows today's prompt (cached if offline).
3. She taps the composer entry point.
4. Composer opens, keyboard active.
5. She writes; autosave fires on pause.
6. She taps Back.
7. **Climax:** Today surface shows `Saved.` and the entry's first line below the prompt — proof the day is captured.

Failure: cold prompt fetch fails → composer still opens with cached generic prompt; banner on Today only after Mira returns.

### Flow 2 — Recall past entry (Mira, three weeks later, looking for what she wrote about her mother)

1. Mira taps Library.
2. Scrolls or searches.
3. Taps entry row.
4. Entry detail opens in read mode.
5. She taps anywhere to enter edit mode (cursor at tap point).
6. Edits autosave.
7. **Climax:** `Saved.` visible in entry header — the older self and the present self are in continuous conversation.

Empty state: no entries → message routes back to Today.

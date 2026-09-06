import { cn } from '@/lib/utils'

// Shared class names for the composer's control row, in a module of their own
// so both the row (`controls.tsx`) and the menus it renders can wear them
// without importing each other in a cycle.

export const ICON_BTN = 'size-(--composer-control-size) shrink-0 rounded-md'

export const GHOST_ICON_BTN = cn(
  ICON_BTN,
  'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
)

// Send/voice-conversation primary: solid foreground-on-background circle
// (reads as black-on-white in light mode, white-on-black in dark mode) to
// match the reference composer's high-contrast CTA. Keeps the pill itself
// neutral and lets the action visually dominate the row.
//
// Face only — pair with a Button `size`. The composer send keeps its own
// measure via PRIMARY_ICON_BTN; everywhere else pick `icon-2xs` / `icon-xs`.
export const PRIMARY_ICON_FACE = cn(
  'shrink-0 rounded-full p-0',
  'bg-foreground text-background hover:bg-foreground/90',
  'disabled:bg-foreground/30 disabled:text-background disabled:opacity-100'
)

export const PRIMARY_ICON_BTN = cn(
  PRIMARY_ICON_FACE,
  'size-(--composer-control-primary-size,var(--composer-control-size))'
)

/** A toggle that is currently ON — dictation, spoken replies, the wake word. */
export const ACTIVE_ICON_BTN = 'bg-primary/10 text-primary hover:bg-primary/15 hover:text-primary'

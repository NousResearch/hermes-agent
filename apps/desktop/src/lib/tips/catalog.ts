/**
 * THE TIP CATALOG — what the app tells you about itself when it's quiet.
 *
 * Data only. A tip is an id, somewhere to point, and (optionally) the keybind
 * whose CURRENT combo the bubble prints. The words live in i18n under
 * `tips.items[id]`, keyed by the same id, so `TipId` is what makes a missing
 * translation a type error rather than an empty bubble.
 *
 * `targets` is a preference list, not a selector: the first one that resolves
 * to a visible element wins, and a tip whose anchors are all absent simply
 * isn't a candidate this time round (see `resolveTipAnchor`). That is why a
 * tip about the profile rail costs nothing on a layout that has no rail.
 *
 * Handles are `data-tour`, the same vocabulary the tour engine's collector
 * treats as identity — one set of durable handles, two consumers. Never point
 * a tip at a positional selector or a translated aria-label.
 */

export type TipSide = 'bottom' | 'left' | 'right' | 'top'

export interface TipDef {
  /** Persistence key for a hard close. Stable forever — renaming forgets it. */
  id: TipId
  /** Keybind action id; the bubble renders its live combo. Never hardcode one. */
  keybind?: string
  /** Preferred side of the anchor. Flips at a viewport edge like any popover. */
  side: TipSide
  /** Candidate anchors, best first. */
  targets: readonly string[]
}

export type TipId =
  | 'artifacts'
  | 'command-palette'
  | 'composer-mentions'
  | 'cron'
  | 'messaging'
  | 'new-session'
  | 'profiles'
  | 'right-pane'
  | 'skills'

// Between them these introduce the app: the rail down the left, the composer,
// and the pane on the right. Nothing here is a step in a sequence — any one has
// to stand alone, because that is how they arrive.
export const TIP_CATALOG: readonly TipDef[] = [
  { id: 'new-session', keybind: 'session.new', side: 'right', targets: ['[data-tour="sidebar-nav-new-session"]'] },
  { id: 'skills', keybind: 'nav.skills', side: 'right', targets: ['[data-tour="sidebar-nav-skills"]'] },
  { id: 'messaging', keybind: 'nav.messaging', side: 'right', targets: ['[data-tour="sidebar-nav-messaging"]'] },
  { id: 'artifacts', keybind: 'nav.artifacts', side: 'right', targets: ['[data-tour="sidebar-nav-artifacts"]'] },
  { id: 'cron', keybind: 'nav.cron', side: 'right', targets: ['[data-tour="sidebar-nav-cron"]'] },
  { id: 'command-palette', keybind: 'nav.commandPalette', side: 'right', targets: ['[data-tour="sessions-sidebar"]'] },
  { id: 'profiles', keybind: 'profile.next', side: 'right', targets: ['[data-tour="profile-rail"]'] },
  { id: 'composer-mentions', side: 'top', targets: ['[data-tour="composer"]'] },
  { id: 'right-pane', keybind: 'view.toggleRightSidebar', side: 'bottom', targets: ['[data-tour="right-pane-toggle"]'] }
]

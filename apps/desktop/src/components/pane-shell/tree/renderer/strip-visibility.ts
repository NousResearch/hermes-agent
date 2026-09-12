/**
 * Does this zone show its tab strip? One resolver, one precedence order, so
 * every caller gets the same answer and the rule can be read in one place.
 *
 * The decision used to be an inline expression in TreeGroup fed by a flag four
 * other code paths also wrote to, which is how a zone could end up with no
 * strip, no tab, no ✕ and no menu to get any of them back. The ladder below is
 * the whole policy; nothing outside `mode` is persisted, so a zone's chrome is
 * a function of what it currently holds plus one deliberate choice.
 */

import type { Contribution } from '@/contrib/types'
import { effectiveTabStripMode } from '@/store/tabstrip-prefs'

import type { TabStripMode } from '../model'

import { paneChrome } from './track-model'

export interface StripPane {
  /** A tool panel (terminal / logs) that collapses rather than closes. */
  collapsePane: boolean
  /** Contribution placement — `'main'` marks a docked tile (session, page,
   *  preview) as opposed to standing side chrome. */
  placement?: string
  /** Panes that never leave the tree (the workspace). */
  uncloseable?: boolean
}

export interface StripZone {
  /** The ACTIVE pane declines to be tabbed (a full-page view). */
  headerVeto?: boolean
  /** The zone's standing choice; undefined = auto. */
  mode?: TabStripMode
  /** Panes currently rendered as chips — chrome-hidden and narrow-collapsed
   *  panes are already filtered out. */
  shown: readonly StripPane[]
  /** The layout holds MORE THAN ONE session-bearing zone — this zone is one
   *  window of a tiled arrangement, not the whole app. */
  tiled?: boolean
}

/**
 * A pane is STRANDED without a strip when the strip is the only thing carrying
 * its handle: a lone closeable tile needs its ✕, a lone tool panel needs a chip
 * to grab, and a lone MAIN pane needs one too once the layout is TILED (see
 * `stranded`). Hide-only chrome (sessions / Bots) is the other exempt case: the
 * panes stay, Show/Hide is a separate verb, and a hidden strip comes back via
 * ⌘⌥T. Treating it as stranded at any count made Hide tabs a silent no-op on
 * the sessions sidebar.
 *
 * This outranks an explicit `never` on purpose. "Hide the strip" is a request
 * about chrome, never a request to make a surface unreachable, and a zone that
 * answers no gesture at all is not a state any setting should be able to
 * produce. Hiding still works everywhere it cannot trap you.
 *
 * IT IS THE LAST HANDLE THAT IS PROTECTED, NOT THE PRESENCE OF TABS. A stack
 * of two or more answers tab cycling and ⌘1…⌘9, so hiding its strip costs
 * chrome and no handle. Scoping the tile and tool-panel rungs to a LONE pane is
 * what keeps "Hide tabs" a working command in the zone that actually
 * accumulates tabs: unscoped, one session tab in main pinned the strip on and
 * both the menu row and ⌘⌥T became silent no-ops.
 */
function stranded(shown: readonly StripPane[], tiled: boolean): boolean {
  if (shown.length !== 1) {
    return false
  }

  const [only] = shown

  if (only.collapsePane) {
    return true
  }

  if (only.placement !== 'main') {
    return false
  }

  // A lone CLOSEABLE main pane always keeps its ✕. So does the uncloseable
  // workspace ONCE THE LAYOUT IS TILED: the zone is then one window among
  // several, every sibling window of which shows a strip, and the strip is the
  // only Close control the workspace has left — ⌘W refuses it
  // (closeFocusedSessionTab) and the chat header's title menu carries no Close
  // row, so a chromeless main zone in a tiled layout is a window with no way to
  // close it or add a tab to it. One chat in ONE window is still free to be
  // chromeless: with no sibling zone there is nothing to switch to, and the
  // sidebar / ⌘T / ⌘W still reach the session.
  return !only.uncloseable || tiled
}

export function resolveTabStripVisible(zone: StripZone): boolean {
  if (zone.shown.length === 0) {
    return false
  }

  // A page is not a tab-able surface. Contextual and self-lifting: the strip
  // returns with the chat, so it is resolved ahead of any stored choice and
  // never written down.
  if (zone.headerVeto) {
    return false
  }

  if (stranded(zone.shown, Boolean(zone.tiled))) {
    return true
  }

  if (zone.mode) {
    return zone.mode === 'always'
  }

  // Auto: a lone pane is not a "tab", so it goes without a strip; two or more
  // need one to switch between them.
  return zone.shown.length > 1
}

/**
 * Resolve a zone straight from what the layout knows about it. Both callers —
 * TreeGroup from its render inputs, the store from the registry — go through
 * here, so neither can drift on which chrome flags feed the answer or forget to
 * fold in the app-wide default.
 */
export function tabStripVisibleForZone(zone: {
  /** The zone's ACTIVE pane. */
  active: string
  isCollapsePane: (id: string) => boolean
  /** The zone's own choice, before the app default applies. */
  mode: TabStripMode | undefined
  paneFor: (id: string) => Contribution | undefined
  /** Panes currently rendered as chips. */
  shown: readonly string[]
  /** The layout holds more than one session-bearing zone (see StripZone). */
  tiled?: boolean
}): boolean {
  return resolveTabStripVisible({
    headerVeto: paneChrome(zone.paneFor(zone.active)).headerVeto,
    mode: effectiveTabStripMode(zone.mode),
    shown: zone.shown.map(id => {
      const chrome = paneChrome(zone.paneFor(id))

      return {
        collapsePane: zone.isCollapsePane(id),
        placement: chrome.placement,
        uncloseable: chrome.uncloseable
      }
    }),
    tiled: zone.tiled
  })
}

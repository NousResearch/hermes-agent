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

import { isPluginSource } from '@/contrib/plugin-source'
import type { Contribution, ContributionSource } from '@/contrib/types'
import { effectiveTabStripMode } from '@/store/tabstrip-prefs'

import type { TabStripMode } from '../model'

import { paneChrome } from './track-model'

export interface StripPane {
  /** A tool panel (terminal / logs) that collapses rather than closes. */
  collapsePane: boolean
  /** An app-owned close route can make a structurally uncloseable pane's TAB
   *  closeable (the workspace empties into the next session / a fresh draft). */
  hasCloser?: boolean
  /** Standing chrome (sessions / Bots) whose only handle is the strip:
   *  show/hide replaces Close, and the Show/Hide rows live on the strip. */
  hideOnly?: boolean
  /** Contribution placement — `'main'` marks a docked tile (session, page,
   *  preview) as opposed to standing side chrome. */
  placement?: string
  /** Runtime-plugin provenance. A contributed pane has no guaranteed core
   *  titlebar toggle, so its tab is the host-owned Close surface. */
  source?: ContributionSource
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
}

/**
 * A pane is STRANDED without a strip when the strip is the only thing carrying
 * its handle: a lone closeable tile needs its ✕, a lone tool panel needs a chip
 * to grab, hide-only chrome (sessions / Bots) needs the chip that show/hide
 * lives on. The uncloseable workspace is not strandable — it cannot be closed
 * or lost, so a lone chat is free to be chromeless. An app-owned close route
 * (the workspace empties into the next session / a fresh draft) makes even a
 * structurally fixed pane's TAB actionable, and a runtime-plugin pane has no
 * core titlebar toggle to fall back on — both keep the strip as the last
 * handle (#96852).
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
function closeNeedsStrip(pane: StripPane): boolean {
  const closeable = !pane.hideOnly && (!pane.uncloseable || pane.hasCloser)

  if (!closeable) {
    return false
  }

  // Main tenants are tabs by design. Runtime plugins also need host chrome:
  // unlike core sidebars, they have no guaranteed titlebar/palette toggle to
  // replace the tab's Close action when they become a lone side pane.
  return pane.placement === 'main' || isPluginSource(pane.source)
}

function stranded(shown: readonly StripPane[]): boolean {
  // Hide-only chrome is stranded at ANY count: it has no close verb at all, and
  // both the chips and the Show/Hide rows that replace one live on the strip.
  if (shown.some(pane => pane.hideOnly)) {
    return true
  }

  if (shown.length !== 1) {
    return false
  }

  const [only] = shown

  // Lone-pane rungs (scoped per 2c5294597 — a multi-tab stack answers cycling
  // and ⌘1…⌘9, so its strip is chrome, not a handle): a lone collapsing panel
  // needs its chip, and a lone pane whose tab carries Close (closeable tile,
  // workspace with an app-owned closer, runtime-plugin pane) needs its ✕.
  return only.collapsePane || closeNeedsStrip(only)
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

  if (stranded(zone.shown)) {
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
  hasCloser: (id: string) => boolean
  isCollapsePane: (id: string) => boolean
  /** The zone's own choice, before the app default applies. */
  mode: TabStripMode | undefined
  paneFor: (id: string) => Contribution | undefined
  /** Panes currently rendered as chips. */
  shown: readonly string[]
}): boolean {
  return resolveTabStripVisible({
    headerVeto: paneChrome(zone.paneFor(zone.active)).headerVeto,
    mode: effectiveTabStripMode(zone.mode),
    shown: zone.shown.map(id => {
      const pane = zone.paneFor(id)
      const chrome = paneChrome(pane)

      return {
        collapsePane: zone.isCollapsePane(id),
        hasCloser: zone.hasCloser(id),
        hideOnly: chrome.hideOnly,
        placement: chrome.placement,
        source: pane?.source,
        uncloseable: chrome.uncloseable
      }
    })
  })
}

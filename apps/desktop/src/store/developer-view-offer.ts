/**
 * "Want the developer view?" (Onboarding C6 18677:28595): asked in a chat about code rather than as
 * Simple vs Advanced during sign up. The developer view is the interface mode's Advanced (terminal,
 * files and diffs, store/interface-mode.ts); the offer is for someone in Simple.
 *
 * - Offered once: until it is `accepted` (Advanced, for good) or `declined` (Not now).
 * - Previewed: while its preview row is hovered or focused the window shows Advanced, and goes back
 *   to Simple when the pointer or focus leaves, unless the person turned it on meanwhile; a tap (touch
 *   has no hover) starts the preview too. The preview is the interface mode's in-memory override:
 *   only Turn on stores Advanced.
 */

import { atom, computed } from 'nanostores'

import { $guidanceLedger, markGuidance } from './guidance-ledger'
import { recordIaAction } from './ia-metrics'
import { $interfaceMode, previewInterfaceMode, setInterfaceMode } from './interface-mode'
import { $portalShell } from './shell-variant'

/** The window shows Advanced only because the offer is being previewed. */
export const $developerViewPreviewing = atom(false)

export const $developerViewOfferOpen = computed(
  [$portalShell, $interfaceMode, $guidanceLedger, $developerViewPreviewing],
  (portal, mode, ledger, previewing) =>
    portal && ledger.developerViewOffer === undefined && (mode === 'simple' || previewing)
)

export function noteDeveloperViewOffered(): void {
  recordIaAction('ia.developer_view.offered', 'state')
}

export function previewDeveloperView(on: boolean): void {
  if (on && !$developerViewPreviewing.get() && $interfaceMode.get() === 'simple') {
    $developerViewPreviewing.set(true)
    previewInterfaceMode('advanced')
    recordIaAction('ia.developer_view.previewed')
  } else if (!on && $developerViewPreviewing.get()) {
    $developerViewPreviewing.set(false)
    previewInterfaceMode(null)
  }
}

export function acceptDeveloperView(): void {
  $developerViewPreviewing.set(false)
  setInterfaceMode('advanced')
  markGuidance('developerViewOffer', 'accepted')
  recordIaAction('ia.developer_view.accepted')
}

export function declineDeveloperView(): void {
  previewDeveloperView(false)
  markGuidance('developerViewOffer', 'declined')
  recordIaAction('ia.developer_view.declined')
}

/**
 * The first-time splash ("Welcome to Hermes Agent", Figma 1A 477:4004): shown once to someone new to
 * Hermes, reopened from the notifications tray ("Meet Hermes") at will.
 *
 * - Newcomer: the existing-user verdict said `fresh` on its first settled look. That look is
 *   recorded once by the rename splash (`renameSplash: not_applicable`), so someone who was new
 *   stays a newcomer after their first chats; anyone else is marked `not_applicable` here and never
 *   gets this splash.
 * - Shown: the first time a newcomer's verdict settles and the ledger has no mark for it.
 * - Seen: any close (a nav click, its ×, Escape, its own button, or sign up) marks it `seen` for good.
 *   Leaving the page while it is up is not a close: the next visit shows it again.
 */

import { atom, computed } from 'nanostores'

import type { ExistingUserVerdict } from '@/lib/existing-user'

import { $guidanceLedger, guidanceMark, markGuidance } from './guidance-ledger'
import { recordIaAction } from './ia-metrics'

export type FirstTimeSplashClose = 'close' | 'escape' | 'nav' | 'signup'

export type FirstTimeSplashRow = 'agents' | 'chats' | 'library' | 'routines' | 'theme'

/** Why the splash is on screen: it showed itself, or the tray reopened it. Null while closed. */
export const $firstTimeSplash = atom<null | 'reopened' | 'shown'>(null)

/** Someone whose first settled look at their machine found nothing: a new install. */
export const $newcomer = computed($guidanceLedger, ledger => ledger.renameSplash === 'not_applicable')

/** The tray offers the splash again once it has been seen. */
export const $firstTimeSplashReopenable = computed($guidanceLedger, ledger => ledger.firstTimeSplash === 'seen')

const CLOSE_VIA = { close: 'click', escape: 'shortcut', nav: 'click', signup: 'click' } as const

// Each row counts once per showing: hovering back and forth is reading, not new interest.
const rowsNoted = new Set<FirstTimeSplashRow>()

function open(reason: 'reopened' | 'shown'): void {
  rowsNoted.clear()
  $firstTimeSplash.set(reason)
  recordIaAction(reason === 'shown' ? 'ia.splash.first_time_shown' : 'ia.splash.first_time_reopened', 'state')
}

/** Settle the splash on the latest verdict; repeated calls are no-ops once it is decided. */
export function considerFirstTimeSplash(verdict: ExistingUserVerdict): void {
  if (verdict === 'unknown' || $firstTimeSplash.get() || guidanceMark('firstTimeSplash')) {
    return
  }

  if (verdict === 'fresh' || $newcomer.get()) {
    open('shown')
  } else {
    markGuidance('firstTimeSplash', 'not_applicable')
  }
}

export function reopenFirstTimeSplash(): void {
  if ($firstTimeSplash.get() || !$firstTimeSplashReopenable.get()) {
    return
  }

  open('reopened')
}

export function closeFirstTimeSplash(how: FirstTimeSplashClose): void {
  if (!$firstTimeSplash.get()) {
    return
  }

  markGuidance('firstTimeSplash', 'seen')
  $firstTimeSplash.set(null)
  recordIaAction(`ia.splash.first_time_closed_${how}`, CLOSE_VIA[how])
}

export function noteFirstTimeSplashRow(row: FirstTimeSplashRow): void {
  if (!$firstTimeSplash.get() || rowsNoted.has(row)) {
    return
  }

  rowsNoted.add(row)
  recordIaAction(`ia.splash_row.first_time_${row}`, 'state')
}

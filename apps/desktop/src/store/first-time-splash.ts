/**
 * The first-time splash ("Welcome to Hermes", Figma Onboarding 18695:17147): shown once to someone new
 * to Hermes, reopened from the notifications tray ("Meet Hermes") at will.
 *
 * - Newcomer: the existing-user verdict said `fresh` on its first settled look. That look is
 *   recorded once by the rename splash (`renameSplash: not_applicable`), so someone who was new
 *   stays a newcomer after their first chats; anyone else is marked `not_applicable` here and never
 *   gets this splash.
 * - Shown: the first time a newcomer's verdict settles and the ledger has no mark for it.
 * - Seen: any close (a nav click, Escape, its own buttons, or sign up) marks it `seen` for good.
 *   Leaving the page while it is up is not a close: the next visit shows it again.
 * - The navigation stays as it is while the splash is up, as it does under the rename splash: the
 *   place a row last pointed at has its rail button lit where it sits. The card opens at rest
 *   (Figma 18695:17147) with nothing lit until a pointer, focus or tap reaches a row.
 */

import { atom, computed } from 'nanostores'

import type { ExistingUserVerdict } from '@/lib/existing-user'

import { $guidanceLedger, guidanceMark, markGuidance } from './guidance-ledger'
import { recordIaAction } from './ia-metrics'

export type FirstTimeSplashClose = 'close' | 'escape' | 'nav' | 'signup'

/** The places the splash explains, in the rail's own order, so a row and its place read top to
 *  bottom alike (Figma 18924:28709 draws Connections between Agents and Routines; the rail keeps
 *  Routines last, and Connections is the rail's Capabilities destination). */
export const FIRST_TIME_PLACES = ['chats', 'agents', 'connections', 'routines'] as const

export type FirstTimeSplashPlace = (typeof FIRST_TIME_PLACES)[number]

/** A place's row, or the second step ("Make it yours"). */
export type FirstTimeSplashRow = FirstTimeSplashPlace | 'theme'

/** Why the splash is on screen: it showed itself, or the tray reopened it. Null while closed. */
export const $firstTimeSplash = atom<null | 'reopened' | 'shown'>(null)

/** The place a row last reached (hover, focus or tap). Null at rest — the splash opens on
 *  18695:17147 with no row lit — and while the splash is closed. */
export const $firstTimeSplashPlace = atom<FirstTimeSplashPlace | null>(null)

/** Someone whose first settled look at their machine found nothing: a new install. */
export const $newcomer = computed($guidanceLedger, ledger => ledger.renameSplash === 'not_applicable')

/** The tray offers the splash again once it has been seen. */
export const $firstTimeSplashReopenable = computed($guidanceLedger, ledger => ledger.firstTimeSplash === 'seen')

const CLOSE_VIA = { close: 'click', escape: 'shortcut', nav: 'click', signup: 'click' } as const

// Each row counts once per showing: hovering back and forth is reading, not new interest.
const rowsNoted = new Set<FirstTimeSplashRow>()

function resetShowing(): void {
  rowsNoted.clear()
  $firstTimeSplashPlace.set(null)
}

function open(reason: 'reopened' | 'shown'): void {
  resetShowing()
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
  resetShowing()
  recordIaAction(`ia.splash.first_time_closed_${how}`, CLOSE_VIA[how])
}

/** A row pointed at its place, or the second step opened (`theme`). Counted once per showing. */
export function noteFirstTimeSplashRow(row: FirstTimeSplashRow): void {
  if (!$firstTimeSplash.get()) {
    return
  }

  if (row !== 'theme') {
    $firstTimeSplashPlace.set(row)
  }

  if (!rowsNoted.has(row)) {
    rowsNoted.add(row)
    recordIaAction(`ia.splash_row.first_time_${row}`, 'state')
  }
}

/** The row's pointer, focus or tap stepped away: back to the resting card (Figma 18695:17147) with
 *  nothing lit. */
export function clearFirstTimeSplashPlace(): void {
  $firstTimeSplashPlace.set(null)
}

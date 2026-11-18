/**
 * The rename splash ("Hermes got a new look"): shown once to someone who used Hermes before the
 * redesign, reopened from the notifications tray at will.
 *
 * - Shown: the first time the existing-user verdict is `existing` and the guidance ledger has no
 *   mark for it. A `fresh` verdict marks it `not_applicable`, so a new install never gets it.
 * - Seen: any dismissal (a nav click, its button, Escape) marks it `seen` for good. Leaving the page
 *   while it is up is not a dismissal: the next visit shows it again.
 * - Reopened: from the tray, any number of times, after it has been seen.
 */

import { atom, computed } from 'nanostores'

import type { ExistingUserVerdict } from '@/lib/existing-user'

import { $guidanceLedger, guidanceMark, markGuidance } from './guidance-ledger'
import { recordIaAction } from './ia-metrics'

export type SplashDismissal = 'button' | 'escape' | 'nav'

export type RenameSplashRow = 'agents' | 'chats' | 'library' | 'machines' | 'notifications' | 'routines'

/** Why the splash is on screen: it showed itself, or the tray reopened it. Null while closed. */
export const $renameSplash = atom<null | 'reopened' | 'shown'>(null)

/** The tray offers the splash again once it has been seen. */
export const $renameSplashReopenable = computed($guidanceLedger, ledger => ledger.renameSplash === 'seen')

// Each row counts once per showing: hovering back and forth is reading, not new interest.
const rowsNoted = new Set<RenameSplashRow>()

function open(reason: 'reopened' | 'shown'): void {
  rowsNoted.clear()
  $renameSplash.set(reason)
  recordIaAction(reason === 'shown' ? 'ia.splash.rename_shown' : 'ia.splash.rename_reopened', 'state')
}

/** Settle the splash on the latest verdict; repeated calls are no-ops once it is decided. */
export function considerRenameSplash(verdict: ExistingUserVerdict): void {
  if (verdict === 'unknown' || $renameSplash.get() || guidanceMark('renameSplash')) {
    return
  }

  if (verdict === 'fresh') {
    markGuidance('renameSplash', 'not_applicable')

    return
  }

  open('shown')
}

export function reopenRenameSplash(): void {
  if ($renameSplash.get() || !$renameSplashReopenable.get()) {
    return
  }

  open('reopened')
}

export function dismissRenameSplash(how: SplashDismissal): void {
  if (!$renameSplash.get()) {
    return
  }

  markGuidance('renameSplash', 'seen')
  $renameSplash.set(null)
  recordIaAction(`ia.splash.rename_dismissed_${how}`, how === 'escape' ? 'shortcut' : 'click')
}

export function noteRenameSplashRow(row: RenameSplashRow): void {
  if (!$renameSplash.get() || rowsNoted.has(row)) {
    return
  }

  rowsNoted.add(row)
  recordIaAction(`ia.splash_row.rename_${row}`, 'state')
}

/**
 * "How's Hermes going so far?" (Onboarding C7 18677:28705): once two of a newcomer's chats have had
 * a task finish, their new chat's landing asks how it is going, with three things to try.
 *
 * - Due: two finished-task chats (the same count the starter ideas retire on, so failed or stopped
 *   turns never advance it) and no mark yet.
 * - Retires for good on an answer (`answered`) or on Dismiss (`dismissed`); a dismissed check-in is
 *   offered again from the notifications tray, which brings it back for this window
 *   ({@link reopenCheckIn}) until it is answered or dismissed once more.
 * - Answers and actions are counted as closed codes behind the consent gate; nothing else is kept.
 */

import { atom, computed } from 'nanostores'

import { $newcomer } from './first-time-splash'
import { $finishedTaskChats, $guidanceLedger, FINISHED_TASK_CHATS_KEPT, markGuidance } from './guidance-ledger'
import { recordIaAction } from './ia-metrics'

export type CheckInAnswer = 'could_be_better' | 'going_well'

export type CheckInTry = 'open_tour' | 'try_video' | 'try_weekly'

/** Brought back from the tray in this window. */
const $reopened = atom(false)

export const $checkInDue = computed(
  [$newcomer, $finishedTaskChats, $guidanceLedger, $reopened],
  (newcomer, finished, ledger, reopened) =>
    reopened || (newcomer && finished >= FINISHED_TASK_CHATS_KEPT && ledger.checkIn === undefined)
)

/** The tray lists the check-in once it was put away unanswered. */
export const $checkInReopenable = computed($guidanceLedger, ledger => ledger.checkIn === 'dismissed')

export function answerCheckIn(answer: CheckInAnswer): void {
  $reopened.set(false)
  markGuidance('checkIn', 'answered')
  recordIaAction(`ia.check_in.${answer}`)
}

export function dismissCheckIn(): void {
  $reopened.set(false)

  if ($guidanceLedger.get().checkIn !== 'answered') {
    markGuidance('checkIn', 'dismissed')
  }

  recordIaAction('ia.check_in.dismissed')
}

/** One of its suggestions was taken; the check-in stays until it is answered or dismissed. */
export function noteCheckInTry(action: CheckInTry): void {
  recordIaAction(`ia.check_in.${action}`)
}

export function reopenCheckIn(): void {
  if (!$checkInReopenable.get()) {
    return
  }

  $reopened.set(true)
  recordIaAction('ia.check_in.reopened')
}

/**
 * A newcomer's landing state (Figma 479:4114 → 486:4227 → 486:5035); the starter-idea strip that spec
 * drew under the empty chat's composer is gone (HADESK-65), so nothing shows an idea.
 *
 * - Who: newcomers only ({@link $newcomer}); everyone else's empty chat is unchanged.
 * - {@link $starterIdeas} says whether a newcomer's ideas are still in play: `ideas` until
 *   {@link FINISHED_TASK_CHATS_KEPT} chats have had a task finish (or a `hidden` mark left by the
 *   strip retired them). The landing's greeting and the composer's placeholder read it.
 * - "A chat in which a task finished": a turn of that chat ended `complete` (the gateway's
 *   `message.complete` status; not `error`, not `interrupted`) with an answer in it. Sending a message
 *   is not enough, and the same chat counts once.
 * - Per account: the count and the marks live in the guidance ledger, the seam that moves to the
 *   account.
 */

import { computed } from 'nanostores'

import { $newcomer } from './first-time-splash'
import {
  $finishedTaskChats,
  $guidanceLedger,
  FINISHED_TASK_CHATS_KEPT,
  noteFinishedTaskChat
} from './guidance-ledger'
import { recordIaAction } from './ia-metrics'
import { $sessions } from './session'

/** `ideas`: still in play. `retired`: two chats have had a task finish, or a persisted `hidden` mark
 *  retired them. `none`: not a newcomer; the empty chat is the ordinary one. */
export type StarterIdeasState = 'ideas' | 'none' | 'retired'

export const $starterIdeas = computed(
  [$newcomer, $guidanceLedger, $finishedTaskChats],
  (newcomer, ledger, finished): StarterIdeasState => {
    if (!newcomer) {
      return 'none'
    }

    if (ledger.starterIdeas === 'wanted') {
      return 'ideas'
    }

    return ledger.starterIdeas === 'hidden' || finished >= FINISHED_TASK_CHATS_KEPT ? 'retired' : 'ideas'
  }
)

/** Away this long, the empty chat says hello again (Onboarding C8 18677:28819). */
export const AWAY_AFTER_MS = 30 * 24 * 60 * 60 * 1000

/** When the person last did anything in a chat: the newest last activity among the chats the sidebar
 *  lists (a routine's own runs and hidden or archived chats are not them). Null with no chats. */
export const $lastChatActivityMs = computed($sessions, sessions => {
  const times = sessions
    .filter(row => !row.hidden && !row.archived && !row.scheduler_owned && !row.cron_execution_id)
    .map(row => Number(row.last_active) * 1000)
    .filter(Number.isFinite)

  return times.length ? Math.max(...times) : null
})

/**
 * What a newcomer's empty chat says above the composer (Onboarding 01 18695:18388, 03 C4–C8):
 * - `first`: no task has finished yet: "Welcome to Hermes".
 * - `back`: one has (C4, C5): "Welcome back.".
 * - `away`: they have used Hermes and their newest chat has been still for {@link AWAY_AFTER_MS} (C8):
 *   "Good to see you again." The next task they finish makes that chat the newest, so the landing
 *   settles back to `back` on its own.
 * - null: not a newcomer; their empty chat is the ordinary one.
 * `now` is read when the sessions change, which is when a stay away can end or begin to show.
 */
export type LandingMoment = 'away' | 'back' | 'first'

export const $landingMoment = computed(
  [$newcomer, $finishedTaskChats, $lastChatActivityMs],
  (newcomer, finished, lastActivity): LandingMoment | null => {
    if (!newcomer) {
      return null
    }

    if (finished === 0) {
      return 'first'
    }

    return lastActivity !== null && Date.now() - lastActivity >= AWAY_AFTER_MS ? 'away' : 'back'
  }
)

/** A turn finished `complete` in stored chat `chatId`. Counts while it can still change something:
 *  for a newcomer, until the ideas have left. */
export function noteTaskFinished(chatId: string): void {
  if (!$newcomer.get()) {
    return
  }

  const showing = $starterIdeas.get() === 'ideas'

  if (noteFinishedTaskChat(chatId) && showing && $starterIdeas.get() === 'retired') {
    recordIaAction('ia.starter_chips.hidden_auto', 'state')
  }
}

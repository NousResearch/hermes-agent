/**
 * A newcomer's starter ideas under the empty chat's composer (Figma 479:4114 → 486:4227 → 486:5035).
 *
 * - Who: newcomers only ({@link $newcomer}); everyone else's empty chat is unchanged.
 * - Shown until the person has {@link FINISHED_TASK_CHATS_KEPT} chats in which a task finished, then
 *   they leave on their own; Hide leaves them sooner; Show ideas brings them back for good (until the
 *   next Hide). The four ideas never change while they are shown.
 * - "A chat in which a task finished": a turn of that chat ended `complete` (the gateway's
 *   `message.complete` status; not `error`, not `interrupted`) with an answer in it. Sending a message
 *   is not enough, and the same chat counts once.
 * - Per account: the count and the marks live in the guidance ledger, the seam that moves to the
 *   account.
 */

import { computed } from 'nanostores'

import { $newcomer } from './first-time-splash'
import { $finishedTaskChats, $guidanceLedger, FINISHED_TASK_CHATS_KEPT, markGuidance, noteFinishedTaskChat } from './guidance-ledger'
import { recordIaAction } from './ia-metrics'

/** The ideas, by their content-free metric code, in the order they show. */
export const STARTER_IDEAS = ['morning_briefing', 'research_options', 'summarize_document', 'clean_inbox'] as const

export type StarterIdea = (typeof STARTER_IDEAS)[number]

/** `ideas`: the four are shown. `retired`: a newcomer's ideas are gone (Show ideas brings them back).
 *  `none`: not a newcomer; the empty chat is the ordinary one. */
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

/** A newcomer who has not finished their first tasks yet: the empty chat greets them. */
export const $newcomerGreeting = computed(
  [$newcomer, $finishedTaskChats],
  (newcomer, finished) => newcomer && finished < FINISHED_TASK_CHATS_KEPT
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

export function hideStarterIdeas(): void {
  if ($starterIdeas.get() !== 'ideas') {
    return
  }

  markGuidance('starterIdeas', 'hidden')
  recordIaAction('ia.starter_chips.hidden_user')
}

export function showStarterIdeas(): void {
  if ($starterIdeas.get() !== 'retired') {
    return
  }

  markGuidance('starterIdeas', 'wanted')
  recordIaAction('ia.starter_chips.shown_again')
}

/** The person picked an idea: counted by its code before it is sent like a typed message. */
export function noteStarterIdeaPicked(idea: StarterIdea): void {
  recordIaAction(`ia.starter_chip.${idea}`)
}

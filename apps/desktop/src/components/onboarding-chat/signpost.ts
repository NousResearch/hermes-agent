/**
 * THE PARTING SIGNPOST — one lit moment, at the one moment it earns itself.
 *
 * The handoff is the only point in the run where the ground moves under the
 * user: they were talking to Hermes on its own profile, and they land mid-build
 * in a session of their own. The chat they just spent five minutes in is still
 * there, one square away in the profile rail, and nothing on screen says so.
 *
 * So as they land, the rail lights up once. A single accent-lit step, not a
 * tour: the whole appeal of this flow is that it happens in conversation, and
 * spending that on a click-through at the last beat would be a poor trade.
 *
 * Skipped for the user who answered "I'll figure it out" — they were offered a
 * look around and declined, and this is the shape of a look around. Their
 * version of this is a line in the chat (see the runbook's step 4).
 */

import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import { TOUR_OPTIONS } from '@/store/onboarding-script'

/** The rail's tour handle (profile-switcher.tsx). `data-tour` rather than the
 *  `data-slot` beside it because only the former is identity to
 *  collectTourTargets — so this is the same selector the model gets back when
 *  it scans for targets, not a private one this file made up. */
const RAIL = '[data-tour="profile-rail"]'

/** Did they wave off the look around? Read from the guide transcript, because
 *  the pick IS a user turn there and the option text is pinned by the script
 *  (that is what TOUR_OPTIONS is for — both sides read the same constant). */
export function declinedLookAround(messages: ChatMessage[]): boolean {
  return messages.some(message => message.role === 'user' && chatMessageText(message).trim() === TOUR_OPTIONS.none)
}

/** The rail mounts a render or two after the handoff swaps profiles, so wait
 *  for the node rather than firing into an empty DOM (the engine would return
 *  a no-match and the moment would pass silently). Gives up quietly. */
async function waitForRail(timeoutMs = 6000): Promise<boolean> {
  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    if (document.querySelector(RAIL)) {
      return true
    }

    await new Promise(resolve => setTimeout(resolve, 120))
  }

  return false
}

/** Light the rail with the parting line. Never throws, never blocks the
 *  handoff — this is the nicety at the end, not part of the machinery. */
export async function showProfileSignpost(): Promise<void> {
  if (!(await waitForRail())) {
    return
  }

  // Imported here, not at the top: this module is reachable from the boot path
  // through the handoff hook, and driver.js plus its stylesheet are exactly
  // what run-tour.ts keeps off it.
  const { showTourStep } = await import('@/lib/tour')

  await showTourStep({
    accent: true,
    selector: RAIL,
    side: 'right',
    text: "You're in your own workspace now, and this is where the profiles live. The chat we just had is still in there — come back to it whenever you want a hand.",
    title: 'Hermes is still next door'
  })
}

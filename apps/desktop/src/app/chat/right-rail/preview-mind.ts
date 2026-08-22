/**
 * IDLE PULSE — keeps the preview overlay alive while the model reasons.
 *
 * Everything else the overlay draws is narration of an action, so the pane goes
 * dark for the gap between actions — which is most of the time the agent spends
 * on a page, and the part where it is deciding what to do with what it just
 * read. A page that stops responding the moment the agent is thinking hardest
 * reads as the agent having wandered off.
 *
 * This is deliberately NOT part of the act pipeline: a turn boundary only has
 * to poke the overlay the last action left behind. See preview-nudge.ts.
 */

import { $busy } from '@/store/session'

import { nudgeOverlay } from './preview-nudge'

// Module-level, matching how review.ts and coding-status.ts watch this edge. It
// is inert without a live pane, so there is nothing to mount or tear down.
let running = $busy.get()

$busy.subscribe(busy => {
  if (busy === running) {
    return
  }

  running = busy
  nudgeOverlay(busy ? 'think' : 'rest')
})

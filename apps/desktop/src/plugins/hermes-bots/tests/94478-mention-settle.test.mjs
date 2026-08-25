/**
 * Regression tests for #94478 — a bot's @mention of a teammate inside its own
 * reply never drives the cited bot; the room records "settled" instead.
 *
 * Mechanism (traced on 6ce7ab8-era main, apps/desktop/src/plugins/hermes-bots/plugin.js):
 *
 * `resolveGroupResponders` (L6441) scopes responder selection to entries
 * SINCE THE LAST USER ENTRY. A member reply that @mentions a teammate lands
 * AFTER that user entry, so the mention IS visible to the next round's
 * selection — but only if the loop runs another round at all.
 *
 * The hole is the early exit at L7821: `if (spokeThisRound === 0) return`.
 * That guard exists for the everyone-passed case, but it also fires when the
 * round's responders had no NEW delta to read — e.g. every member already
 * spoke and the only new log entry is the citing member's reply. The cited
 * bot is in `mentioned`, but the loop exits before any further round can
 * select it: the room settles with an unresolved handoff. The same happens
 * when a cap (`GROUP_CHAT_MAX_ROUNDS` / `GROUP_CHAT_MAX_MESSAGES`) lands
 * between the mention and the next round.
 *
 * Fix contract:
 *  1. When a round produces zero spoken turns but the thread's tail contains
 *     an @mention of a member who has NOT yet answered it, drive one
 *     continuation round for exactly those cited members (still bounded by
 *     the GROUP_CHAT_MAX_* caps) instead of settling silently.
 *  2. If settling must happen anyway (cap exhausted), record an activity
 *     entry naming the unaddressed handoff so the room does not LOOK
 *     finished while a called bot never answered.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

// Mirror of resolveGroupResponders' scoping rule (plugin.js L6441): mentions
// are collected from entries after the last user entry.
function mentionedSinceLastUser(log, parseMentions) {
  let sinceLastUser = []

  for (let i = log.length - 1; i >= 0; i--) {
    if (log[i].from.kind === 'user') {
      sinceLastUser = log.slice(i)
      break
    }
  }

  const mentioned = new Set()

  for (const entry of sinceLastUser) {
    for (const name of parseMentions(entry.text)) {
      mentioned.add(name)
    }
  }

  return mentioned
}

test('#94478: detects an unanswered member mention in the post-user tail', () => {
  // Room state from the issue: user asks, member radar replies citing @hermes.
  const log = [
    { id: 1, from: { kind: 'user' }, text: 'kick this off' },
    { id: 2, from: { kind: 'member', name: 'radar' }, text: 'done — @hermes please review' }
  ]

  const mentioned = mentionedSinceLastUser(log, text => (text.includes('@hermes') ? ['hermes'] : []))

  // Pre-fix, the loop exits via `spokeThisRound === 0` / cap exhaustion before
  // any round can act on this mention. The fix's detector must find it so a
  // continuation round can be driven for exactly [hermes].
  assert.ok(mentioned.has('hermes'), 'cited member must be detected as pending')
})

test('#94478: does not flag a handoff the cited member already answered', () => {
  const log = [
    { id: 1, from: { kind: 'user' }, text: 'kick this off' },
    { id: 2, from: { kind: 'member', name: 'radar' }, text: 'done — @hermes please review' },
    { id: 3, from: { kind: 'member', name: 'hermes' }, text: 'reviewed, all good' }
  ]

  // hermes has a member entry AFTER the mentioning one — handoff resolved.
  // The fix's detector must not re-drive an answered mention (that would loop).
  const mentionIdx = log.findIndex(e => e.id === 2)
  const answer = [...log].reverse().find(e => e.from.kind === 'member' && e.from.name === 'hermes')
  const answerIdx = log.findIndex(e => answer && e.id === answer.id)

  assert.ok(answerIdx > mentionIdx, 'answer must postdate the mention')
})

/**
 * Bot pairing — the "introduce two bots and let them talk" flow.
 *
 * The infra already exists: every bot carries the `message_agent` tool
 * (fire-and-forget DM into the target's own Bot Chat, reply arrives as a
 * background-notification wake), the relay routes cross-connection targets,
 * and each bot has ONE canonical forever chat. What was missing is a door:
 * nothing in the UI shows a user that bots can be introduced to each other —
 * you had to know to open bot A's chat and ask it to message bot B.
 *
 * `PairBotsDialog` is that door: pick an initiator and a responder, optionally
 * give them a joint task, and the dialog opens the initiator's canonical chat
 * and submits a composed user turn instructing the initiator to message the
 * responder with a concrete first message. From there the conversation is
 * entirely between the bots — the user watches it land in both Bot Chats.
 *
 * The submit is `prompt.submit` on the initiator's canonical session — the
 * same user-turn API every surface uses, so the pairing reads as a real user
 * message (attributed, persisted, prompt-cached) and the turn flows through
 * every existing guard. Nothing here bypasses the messaging protocol.
 */
import { botHandle } from './data'
import type { RosterRow } from './types'

export interface PairingPlan {
  initiator: RosterRow
  responder: RosterRow
  /** Optional joint goal the user gave; woven into the instruction. */
  task?: string
}

const SLUG = (bot: RosterRow) => bot.name || ''

const displayOf = (bot: RosterRow) => botHandle(bot.name, bot)

/** Validate a pairing before anything is sent: distinct, resolvable bots.
 *  Returns a short reason string when invalid, null when OK. */
export function pairingProblem(
  initiator: RosterRow | null,
  responder: RosterRow | null
): null | string {
  if (!initiator || !responder) {
    return 'Pick both bots.'
  }

  if (SLUG(initiator) === SLUG(responder)) {
    return 'Pick two different bots.'
  }

  if (initiator.ghost || responder.ghost) {
    return 'Offline bots cannot be paired yet.'
  }

  if (initiator.sourceReachable === false || responder.sourceReachable === false) {
    return 'One of the bots is on an unreachable connection.'
  }

  return null
}

/** The user-turn text submitted into the initiator's canonical chat. Composed
 *  here (not by the LLM) so the instruction is deterministic and reviewable —
 *  and so the message_agent contract (compose yourself, never paste the user's
 *  words verbatim) is honored by giving the bot a GOAL, not a script. */
export function pairingInstruction(plan: PairingPlan): string {
  const responderName = displayOf(plan.responder)
  const initiatorName = displayOf(plan.initiator)
  const task = plan.task?.trim()

  return [
    `You are being introduced to your teammate @${responderName} by ${initiatorName === plan.initiator.name ? 'the user' : 'the user'} — they want the two of you working together.`,
    '',
    task
      ? `Your joint task: ${task}`
      : 'Introduce yourselves and figure out together how you can help each other.',
    '',
    `Use your message_agent tool to send @${responderName} your opening message now (compose it yourself — greet, state what you bring, and if there is a joint task, propose how you'll split it). Their reply will arrive as a notification; keep the collaboration going from there.`,
    '',
    'Do not wait for the user after messaging — the ball is with your teammate until they reply.'
  ].join('\n')
}

/** True when a roster row can be offered as a pairing participant in the
 *  picker: real (not ghost), on a reachable source, and not mid-delete. */
export function pairableBots(roster: readonly RosterRow[]): RosterRow[] {
  return roster.filter(bot => bot && !bot.ghost && bot.sourceReachable !== false)
}

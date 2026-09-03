/**
 * Why a room said what it said.
 *
 * A room transcript answers "what happened". It does not answer "what caused
 * this", which is the question you actually have when six agents produce a
 * wrong answer and scrolling is your only tool. These helpers walk a message
 * back to its root: each member message records the speaker that put it on
 * turn (`cause.by`) and the log slice it read (`cause.saw`), so the chain is
 * recoverable from the durable transcript alone.
 *
 * Pure and leaf by design. Everything takes a room and returns plain data, so
 * the chain can be tested — and rendered — without a gateway, a live room, or
 * the runtime activity feed (which is capped, epoch-scoped and never
 * persisted).
 */

import type { GroupChat, GroupMessage } from './types'

/** One hop in a causal chain: a message, and what it read to produce it. */
export interface ProvenanceStep {
  /** How many log entries this turn read. 0 for a user send or an unstamped
   *  legacy message. */
  readCount: number
  /** Entries the turn read, oldest first. Empty when unrecoverable. */
  saw: GroupMessage[]
  /** The speaker that put this message's author on turn, when recorded. */
  triggeredBy?: string
  message: GroupMessage
}

/** Depth cap. A chain is a walk backwards through a finite log, so a cycle is
 *  impossible on well-formed data — this bounds a corrupted room instead. */
const MAX_CHAIN = 64

function messageIndex(log: GroupMessage[], id: string): number {
  return log.findIndex(entry => entry?.id === id)
}

/**
 * Resolve the entries a message's turn read. `cause.saw` is a half-open
 * `[from, to)` range over `log` at append time; the history trim drops entries
 * from the FRONT, so a stale range is clamped rather than trusted.
 */
export function entriesSeenBy(room: GroupChat, message: GroupMessage): GroupMessage[] {
  const log = Array.isArray(room?.log) ? room.log : []
  const range = message?.cause?.saw

  if (!Array.isArray(range) || range.length !== 2) {
    return []
  }

  const [from, to] = range

  if (!Number.isFinite(from) || !Number.isFinite(to)) {
    return []
  }

  const start = Math.max(0, Math.min(from, log.length))
  const end = Math.max(start, Math.min(to, log.length))

  return log.slice(start, end)
}

/**
 * The message that put `message`'s author on turn.
 *
 * `cause.by` is a speaker NAME, not an id — the turn engine records who spoke,
 * and ids are not in scope there. Resolve it to the newest message from that
 * speaker at or before this one, which is the entry the delta actually ended
 * on. Returns null for user sends, unstamped legacy messages, and a speaker
 * whose message has since been trimmed away.
 */
export function causeOf(room: GroupChat, message: GroupMessage): GroupMessage | null {
  const by = message?.cause?.by

  if (!by) {
    return null
  }

  const log = Array.isArray(room?.log) ? room.log : []
  const at = message.id ? messageIndex(log, message.id) : -1
  const before = at >= 0 ? log.slice(0, at) : log

  for (let i = before.length - 1; i >= 0; i -= 1) {
    if (before[i]?.from?.name === by) {
      return before[i]
    }
  }

  return null
}

/**
 * Walk a message back to its root, newest hop first.
 *
 * The chain ends at a user send (no `cause`), at an unstamped legacy message,
 * or at a speaker whose message has been trimmed out of the log — all three
 * are legitimate roots, and the caller can tell them apart by inspecting the
 * final step's `message.from.kind` and `triggeredBy`.
 */
export function provenanceChain(room: GroupChat, messageId: string): ProvenanceStep[] {
  const log = Array.isArray(room?.log) ? room.log : []
  const start = log.find(entry => entry?.id === messageId)

  if (!start) {
    return []
  }

  const chain: ProvenanceStep[] = []
  const visited = new Set<string>()
  let current: GroupMessage | null = start

  while (current && chain.length < MAX_CHAIN) {
    const id = current.id

    // A trimmed log can leave two entries resolving to each other. Stop rather
    // than loop; a partial chain beats a hang.
    if (id) {
      if (visited.has(id)) {
        break
      }

      visited.add(id)
    }

    const saw = entriesSeenBy(room, current)

    chain.push({
      message: current,
      readCount: saw.length,
      saw,
      ...(current.cause?.by ? { triggeredBy: current.cause.by } : {})
    })

    current = causeOf(room, current)
  }

  return chain
}

/** Everything this message went on to cause, directly — the forward edge.
 *  Useful for "what did this reply set off", the inverse question. */
export function causedBy(room: GroupChat, messageId: string): GroupMessage[] {
  const log = Array.isArray(room?.log) ? room.log : []
  const at = messageIndex(log, messageId)

  if (at < 0) {
    return []
  }

  const speaker = log[at]?.from?.name

  if (!speaker) {
    return []
  }

  // A later message is caused by this one when it names this speaker AND no
  // newer message from the same speaker sits between them — otherwise the
  // credit belongs to that newer one.
  return log.slice(at + 1).filter((entry, offset) => {
    if (entry?.cause?.by !== speaker) {
      return false
    }

    const between = log.slice(at + 1, at + 1 + offset)

    return !between.some(mid => mid?.from?.name === speaker)
  })
}

/**
 * Whether a message can answer "why did this happen".
 *
 * The affordance is offered only when there is something to show: a user send
 * and a pre-provenance legacy message both have no chain, and a "why" button
 * that opens an apology is worse than no button.
 */
export function hasRecordedCause(message: GroupMessage | null | undefined): boolean {
  const cause = message?.cause

  return Boolean(cause && (cause.by || cause.saw))
}

/**
 * Trail toggle: one open at a time. Clicking the open one closes it, clicking
 * another moves the trail there. Two open trails read as noise rather than as
 * an explanation.
 */
export function nextOpenTrail(current: null | string, clicked: string): null | string {
  return current === clicked ? null : clicked
}

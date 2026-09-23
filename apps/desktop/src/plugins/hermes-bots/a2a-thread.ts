/**
 * The bot-to-bot thread: one merged, ordered view of what two bots said to
 * each other.
 *
 * A DM only lands in the RECEIVER's canonical chat (as a "Message from 🤖"
 * user row); the sender's side holds the `message_agent` call and its reply.
 * Neither transcript alone tells the story, which is why bot-to-bot work
 * reads like it happens in another world — this module reads BOTH sides and
 * interleaves them, so the thread and its state can be shown in one place.
 *
 * Pure: rows in, events out. No atoms, no gateway, no i18n — the dialog owns
 * the fetching and the wording, tests own this.
 */

import { botHandle } from './data'
import { A2A_PREFIX_RE } from './row-helpers'

/** One raw transcript row as the REST history route projects it. */
export interface A2aTranscriptRow {
  content?: unknown
  role?: null | string
  timestamp?: null | number
  tool_calls?: unknown
}

export type A2aEventKind = 'dm' | 'reply' | 'sent'

/** One side of one exchange, from the thread's point of view. `from`/`to`
 *  are normalized handles (`botHandle` form), so a thread can attribute every
 *  row without a roster lookup. */
export interface A2aEvent {
  from: string
  kind: A2aEventKind
  text: string
  to: string
  /** Epoch ms. Rows without a stamp sort first rather than vanish. */
  ts: number
}

/** Roster liveness window, mirroring `ACTIVE_WINDOW_S`: within it the thread
 *  reads "active" instead of accusing anyone of silence. */
export const A2A_ACTIVE_WINDOW_S = 90

/** How long after a send its delivery row may land and still be the same
 *  message. Deliveries spawn a turn on the target gateway, so the receiving
 *  row trails the call — generously, on a slow gateway. */
const SENT_DM_MERGE_WINDOW_MS = 15 * 60 * 1000

/** Display cap: the thread shows the recent exchange, not the full archive. */
export const A2A_THREAD_EVENT_LIMIT = 60

/** Sender handle from a delivery prefix. The current form carries the routing
 *  alias in `(@handle)` — prefer it: a multi-word display name ("hotel dev")
 *  cannot be read reliably from the name alone, and the handle is what targets
 *  resolve to. Falls back to the older `agent 'name'` form. */
function dmSender(content: string): null | string {
  const current = content.match(/^Message from 🤖[^(]*\(@([^)\s]+)\)/i)

  if (current) {
    return botHandle(current[1].trim().toLowerCase())
  }

  const legacy = content.match(/^\[?Message from (?:agent )?'([^']+)'\]?/i)

  return legacy ? botHandle(legacy[1].trim().toLowerCase()) : null
}

/** `message_agent` argument names (tools/bot_mode_dm.py): `target` and
 *  `message`. Arguments arrive as a JSON string on the stored call. */
function messageAgentCall(call: unknown): { target: string; text: string } | null {
  if (!call || typeof call !== 'object') {
    return null
  }

  const record = call as Record<string, unknown>
  const fn = record.function as Record<string, unknown> | undefined

  if (!fn || fn.name !== 'message_agent') {
    return null
  }

  let args: unknown = fn.arguments

  if (typeof args === 'string') {
    try {
      args = JSON.parse(args)
    } catch {
      return null
    }
  }

  if (!args || typeof args !== 'object' || Array.isArray(args)) {
    return null
  }

  const parsed = args as Record<string, unknown>
  const target = String(parsed.target || '').trim()
  const text = String(parsed.message || '').trim()

  return target && text ? { target, text } : null
}

function toolCallsOf(row: A2aTranscriptRow): unknown[] {
  let calls: unknown = row.tool_calls

  if (typeof calls === 'string') {
    try {
      calls = JSON.parse(calls)
    } catch {
      return []
    }
  }

  return Array.isArray(calls) ? calls : []
}

/** Normalize the `message_agent` target the way the roster does: a bare
 *  profile name, an `@handle`, or a display name all resolve to one handle. */
function targetHandle(target: string): string {
  return botHandle(target.replace(/^@+/, '').trim().toLowerCase())
}

/**
 * Everything `self`'s transcript says about its exchange with `peer`:
 * DMs it received, the reply each DM got, and the DMs it sent.
 *
 * A received DM is answered by the turn it started — the LAST assistant text
 * before the next user row — not by the first fragment of narration, so a
 * turn that narrates before answering still yields one truthful reply.
 */
export function a2aEventsForSide(rows: A2aTranscriptRow[], self: string, peer: string): A2aEvent[] {
  const me = botHandle(self.trim().toLowerCase())
  const them = botHandle(peer.trim().toLowerCase())
  const events: A2aEvent[] = []
  let pendingDm: A2aEvent | null = null
  let turnText = ''
  let turnTs = 0

  const flushReply = () => {
    if (pendingDm && turnText) {
      events.push({ from: me, kind: 'reply', text: turnText, to: them, ts: turnTs || pendingDm.ts })
    }

    pendingDm = null
  }

  for (const row of rows || []) {
    const stamp = Number(row?.timestamp) || 0
    const ts = stamp > 1e12 ? stamp : stamp * 1000

    if (row?.role === 'user') {
      flushReply()
      turnText = ''
      turnTs = 0

      const text = String(row.content || '')
      const sender = dmSender(text)

      // A DM from the peer opens the exchange this side owes a reply to; any
      // other user row (the human, another bot) closes the pairing so a reply
      // can never be attributed to the wrong DM.
      if (sender === them) {
        pendingDm = {
          from: them,
          kind: 'dm',
          text: text.replace(A2A_PREFIX_RE, '').trim() || '…',
          to: me,
          ts
        }
        events.push(pendingDm)
      }

      continue
    }

    if (row?.role === 'assistant') {
      for (const call of toolCallsOf(row)) {
        const sent = messageAgentCall(call)

        if (sent && targetHandle(sent.target) === them) {
          events.push({ from: me, kind: 'sent', text: sent.text, to: them, ts })
        }
      }

      const text = String(row.content || '').trim()

      if (text) {
        turnText = text
        turnTs = ts
      }
    }
  }

  flushReply()

  return events
}

/**
 * Both sides, one timeline. A `sent` row and the `dm` it became are the same
 * message seen from two transcripts — the received row wins (it is the text
 * the peer actually got), so the sender's duplicate drops out.
 *
 * The pairing is one-to-one and DIRECTIONAL: each delivery consumes the LATEST
 * send before it with the same text, inside the window. Proximity alone both
 * collapsed two legitimately identical short messages (a repeated "OK") and
 * could consume the wrong send — with "OK" sent twice and only the second one
 * delivered, the window's first match was the FIRST send, so a message that was
 * really sent disappeared from the thread.
 */
export function mergeA2aThread(...sides: A2aEvent[][]): A2aEvent[] {
  const merged = sides.flat().filter(Boolean)
  const received = merged.filter(event => event.kind === 'dm').slice().sort((a, b) => a.ts - b.ts)
  const sent = merged.filter(event => event.kind === 'sent')
  const consumed = new Set<A2aEvent>()

  for (const dm of received) {
    const source = sent
      .filter(candidate => !consumed.has(candidate) && candidate.text.trim() === dm.text.trim())
      .filter(candidate => dm.ts >= candidate.ts && dm.ts - candidate.ts <= SENT_DM_MERGE_WINDOW_MS)
      .sort((a, b) => b.ts - a.ts)[0]

    if (source) {
      consumed.add(source)
    }
  }

  const deduped = merged.filter(event => event.kind !== 'sent' || !consumed.has(event))

  return deduped
    .slice()
    .sort((a, b) => a.ts - b.ts)
    .slice(-A2A_THREAD_EVENT_LIMIT)
}

export interface A2aThreadInfo {
  /** `lastTs` of the newest event, 0 when nothing happened. */
  lastTs: number
  /** `active` — something moved inside the liveness window (a turn may be
   *  running right now); `awaiting-reply` — the last word is a DM nobody has
   *  answered; `settled` — the last word is a reply. */
  state: 'active' | 'awaiting-reply' | 'settled'
  /** Who owes the next message while `awaiting-reply` — the DM's receiver. */
  waitingOn: null | string
}

/** The one-line answer to "are they done?": what the last event was, and who
 *  still owes a message. */
export function a2aThreadState(events: A2aEvent[], now = Date.now()): A2aThreadInfo {
  const last = events?.[events.length - 1]

  if (!last) {
    return { lastTs: 0, state: 'settled', waitingOn: null }
  }

  const fresh = (now - last.ts) / 1000 < A2A_ACTIVE_WINDOW_S

  if (fresh) {
    return { lastTs: last.ts, state: 'active', waitingOn: last.kind === 'dm' ? last.to : null }
  }

  if (last.kind === 'dm') {
    return { lastTs: last.ts, state: 'awaiting-reply', waitingOn: last.to }
  }

  return { lastTs: last.ts, state: 'settled', waitingOn: null }
}

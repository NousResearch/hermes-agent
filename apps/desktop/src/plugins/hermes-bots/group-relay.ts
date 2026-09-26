// Group relay — the Desktop half of `hermes group send` for Desktop-
// coordinated Group Chats.
//
// Those rooms live in this plugin's storage and only the renderer can start a
// member round (sendToGroupChat → runGroupChatRounds). A CLI in another
// session (a Discord thread, a terminal) cannot reach them directly, so it
// queues an envelope on the gateway (tools/group_relay.py) and THIS module:
//
//   1. drains the outbox on a short interval (plus the gateway's
//      `group_relay.outbox.pending` push signal),
//   2. calls sendToGroupChat ON THE USER'S BEHALF — the entry renders as
//      "You" with `via: <label>` recorded as provenance,
//   3. watches the room and streams progress lines back through
//      `group_relay.reply`: accepted → reply (per committed member message in
//      the relay's thread) → done (settled | capped | cancelled | timeout).
//
// Deliberately a sibling of relay.ts (the cross-connection DM relay), not an
// extension: that drain is gated on ≥2 connections and its envelopes carry a
// different contract. This module only ever talks to the ACTIVE connection —
// the gateway the CLI wrote its envelope on is the one this Desktop is
// attached to. Remove this file + its two call sites and the feature is gone.

import { host } from '@hermes/plugin-sdk'

import { $botMeta, $lastRoster } from './data'
import { $groupActivity } from './group-activity'
import type { GroupActivityEntry } from './group-activity'
import { $groupChats } from './group-chat'
import { groupChatMemberBots } from './group-membership'
import { sendToGroupChat } from './group-rounds'
import type { GroupChat, GroupMember, GroupMessage } from './types'

export const GROUP_RELAY_DRAIN_INTERVAL_MS = 5_000
const GROUP_RELAY_PUSH_DEBOUNCE_MS = 150
/** A relay whose round never ends (member stuck) still resolves the waiting CLI. */
export const GROUP_RELAY_WATCH_TIMEOUT_MS = 45 * 60_000

export interface GroupRelayEnvelope {
  id: string
  created_at?: number
  from_profile?: string
  label?: string
  room_id?: string
  room_name?: string
  text?: string
  thread?: null | string
}

export type GroupRelayLine =
  | { kind: 'accepted'; thread: string; group: string }
  | { kind: 'reply'; member: string; text: string; thread: string }
  | { kind: 'done'; status: 'cancelled' | 'capped' | 'settled' | 'timeout'; replies: number }
  | { error: string; kind: 'error'; reason?: string }

interface Watcher {
  envelopeId: string
  epoch: number
  group: string
  replies: number
  /** Log entry ids already streamed. */
  seen: Set<string>
  startedAt: number
  thread: string
  /** Set once the room has been observed running for this relay. */
  wasRunning: boolean
}

interface RelayState {
  disposed: boolean
  drainBusy: boolean
  drainRerun: boolean
  drainTimer: null | ReturnType<typeof setInterval>
  pushDebounceTimer: null | ReturnType<typeof setTimeout>
  pushUnsub: null | (() => void)
  storeUnsub: null | (() => void)
  watchers: Map<string, Watcher>
}

const relay: RelayState = {
  disposed: true,
  drainBusy: false,
  drainRerun: false,
  drainTimer: null,
  pushDebounceTimer: null,
  pushUnsub: null,
  storeUnsub: null,
  watchers: new Map()
}

// ── room lookup ─────────────────────────────────────────────────────────────

/** Resolve an envelope to a Desktop room: durable roomId first, then the
 *  exact display name (rooms are keyed by name in `$groupChats`). */
export function resolveRelayRoom(
  envelope: GroupRelayEnvelope,
  rooms: Record<string, GroupChat> = $groupChats.get()
): null | { name: string; room: GroupChat } {
  const wantId = String(envelope?.room_id || '').trim()

  if (wantId) {
    for (const [name, room] of Object.entries(rooms || {})) {
      if (room && String(room.roomId || '') === wantId) {
        return { name, room }
      }
    }
  }

  const wantName = String(envelope?.room_name || '').trim()

  if (wantName && rooms?.[wantName]) {
    return { name: wantName, room: rooms[wantName] }
  }

  return null
}

/** Stable identity for a log entry; ids are minted on append, the fallback
 *  covers legacy entries hydrated without one. */
function entryKey(entry: GroupMessage): string {
  return String(entry?.id || `${entry?.at}:${entry?.from?.name}:${entry?.thread || ''}`)
}

function relayMembers(group: string): GroupMember[] {
  return groupChatMemberBots(group, $lastRoster.get(), $botMeta.get())
}

// ── reply plumbing ──────────────────────────────────────────────────────────

async function postLine(envelopeId: string, line: GroupRelayLine) {
  try {
    await host.request('group_relay.reply', { id: envelopeId, line })
  } catch {
    // Gateway unreachable: the CLI's own --timeout resolves the wait.
  }
}

// ── watchers: turn room-log deltas into streamed reply lines ───────────────

function latestActivityFor(group: string, thread: string, epoch: number): GroupActivityEntry | null {
  const events = $groupActivity.get()[group]?.events || []

  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i]

    if (event.epoch === epoch && (event.thread || null) === thread) {
      return event
    }
  }

  return null
}

/** One pass over every watcher against the current room state. Exported for
 *  tests; production runs it from the `$groupChats` / `$groupActivity`
 *  subscriptions and the drain interval. */
export async function tickGroupRelayWatchers(now = Date.now()) {
  const rooms = $groupChats.get()

  for (const watcher of [...relay.watchers.values()]) {
    const room = rooms[watcher.group]

    if (!room) {
      relay.watchers.delete(watcher.envelopeId)
      await postLine(watcher.envelopeId, { kind: 'error', reason: 'room_gone', error: `group '${watcher.group}' no longer exists` })

      continue
    }

    // Stream newly committed member replies in the relay's thread.
    for (const entry of room.log || []) {
      const key = entryKey(entry)

      if (watcher.seen.has(key) || entry?.from?.kind !== 'member') {
        continue
      }

      if (String(entry.thread || 'legacy') !== watcher.thread) {
        continue
      }

      watcher.seen.add(key)
      watcher.replies += 1
      await postLine(watcher.envelopeId, {
        kind: 'reply',
        member: String(entry.from.name || 'bot'),
        text: String(entry.text || ''),
        thread: watcher.thread
      })
    }

    // Terminal conditions, in priority order.
    if ((room.epoch || 0) !== watcher.epoch) {
      // A newer user send (composer or another relay) superseded this round.
      relay.watchers.delete(watcher.envelopeId)
      await postLine(watcher.envelopeId, { kind: 'done', status: 'cancelled', replies: watcher.replies })

      continue
    }

    if (room.running === true) {
      watcher.wasRunning = true
    }

    const activity = latestActivityFor(watcher.group, watcher.thread, watcher.epoch)
    const ended = activity && (activity.kind === 'settled' || activity.kind === 'capped' || activity.kind === 'cancelled')

    if (ended || (watcher.wasRunning && room.running !== true)) {
      relay.watchers.delete(watcher.envelopeId)
      const status = ended ? (activity!.kind as 'cancelled' | 'capped' | 'settled') : 'settled'
      await postLine(watcher.envelopeId, { kind: 'done', status, replies: watcher.replies })

      continue
    }

    if (now - watcher.startedAt > GROUP_RELAY_WATCH_TIMEOUT_MS) {
      relay.watchers.delete(watcher.envelopeId)
      await postLine(watcher.envelopeId, { kind: 'done', status: 'timeout', replies: watcher.replies })
    }
  }
}

let tickChain: Promise<void> = Promise.resolve()

/** Serialize ticks: store notifications can burst, and two concurrent passes
 *  would double-post the same log entry before `seen` is updated. */
function scheduleTick() {
  if (relay.disposed) {
    return
  }

  tickChain = tickChain.then(() => tickGroupRelayWatchers()).catch(() => undefined)
}

// ── drain: envelopes → sendToGroupChat ─────────────────────────────────────

/** Act on one claimed envelope. Exported for tests. */
export async function handleGroupRelayEnvelope(envelope: GroupRelayEnvelope): Promise<boolean> {
  const envelopeId = String(envelope?.id || '')

  if (!envelopeId) {
    return false
  }

  const resolved = resolveRelayRoom(envelope)

  if (!resolved) {
    await postLine(envelopeId, {
      kind: 'error',
      reason: 'room_not_found',
      error: `no Desktop group matches ${envelope.room_name || envelope.room_id || '?'} on this Desktop`
    })

    return false
  }

  const members = relayMembers(resolved.name)

  if (!members.length) {
    await postLine(envelopeId, { kind: 'error', reason: 'no_members', error: `group '${resolved.name}' has no seated members` })

    return false
  }

  const text = String(envelope.text || '').trim()
  // An existing Desktop thread id continues that thread; anything else (or
  // nothing) starts a new one — a relay from a fresh session is a new topic.
  const requested = String(envelope.thread || '').trim()
  const knownThread = requested && (resolved.room.log || []).some(entry => String(entry?.thread || '') === requested)

  // Snapshot the log BEFORE the send: only entries that predate the relay
  // are pre-seen, so a member reply appended at any point after this line —
  // even synchronously inside sendToGroupChat's round kick — is streamed.
  // (A watcher seeded after the send could swallow such a reply.)
  const before = $groupChats.get()[resolved.name] || {}
  const seen = new Set((before.log || []).map(entryKey))

  const thread = sendToGroupChat(resolved.name, members, text, knownThread ? requested : null, undefined, {
    via: String(envelope.label || '').trim() || undefined
  })

  if (!thread) {
    await postLine(envelopeId, { kind: 'error', reason: 'send_refused', error: 'the room refused the message (empty text or no members)' })

    return false
  }

  const room = $groupChats.get()[resolved.name] || {}
  relay.watchers.set(envelopeId, {
    envelopeId,
    epoch: room.epoch || 0,
    group: resolved.name,
    replies: 0,
    seen,
    startedAt: Date.now(),
    thread,
    wasRunning: room.running === true
  })
  await postLine(envelopeId, { kind: 'accepted', thread, group: resolved.name })
  scheduleTick()

  return true
}

export async function drainGroupRelayOutbox() {
  if (relay.disposed) {
    return
  }

  if (relay.drainBusy) {
    relay.drainRerun = true

    return
  }

  relay.drainBusy = true

  try {
    let envelopes: GroupRelayEnvelope[] = []

    try {
      const res = await host.request<{ envelopes?: GroupRelayEnvelope[] }>('group_relay.outbox.drain', {})
      envelopes = Array.isArray(res?.envelopes) ? res.envelopes : []
    } catch {
      // Older gateway without the method, or transport blip: try next tick.
      envelopes = []
    }

    for (const envelope of envelopes) {
      if (relay.disposed) {
        return
      }

      await handleGroupRelayEnvelope(envelope)
    }

    // Watchers advance on store changes; the interval is the backstop for
    // the timeout path and for rooms that end without a store notification.
    await tickGroupRelayWatchers()
  } finally {
    relay.drainBusy = false

    if (relay.drainRerun && !relay.disposed) {
      relay.drainRerun = false
      scheduleGroupRelayPushDrain()
    }
  }
}

function scheduleGroupRelayPushDrain() {
  if (relay.disposed || typeof setTimeout !== 'function' || relay.pushDebounceTimer !== null) {
    return
  }

  relay.pushDebounceTimer = setTimeout(() => {
    relay.pushDebounceTimer = null
    void drainGroupRelayOutbox()
  }, GROUP_RELAY_PUSH_DEBOUNCE_MS)
}

// ── lifecycle ───────────────────────────────────────────────────────────────

export function startGroupRelay() {
  relay.disposed = false

  if (typeof setInterval !== 'function' || typeof clearInterval !== 'function') {
    return
  }

  if (relay.drainTimer === null) {
    relay.drainTimer = setInterval(() => void drainGroupRelayOutbox(), GROUP_RELAY_DRAIN_INTERVAL_MS)
  }

  if (relay.storeUnsub === null) {
    const unsubRooms = $groupChats.listen(() => scheduleTick())
    const unsubActivity = $groupActivity.listen(() => scheduleTick())

    relay.storeUnsub = () => {
      unsubRooms()
      unsubActivity()
    }
  }

  if (relay.pushUnsub === null && typeof host.onEvent === 'function') {
    relay.pushUnsub = host.onEvent('group_relay.outbox.pending', () => scheduleGroupRelayPushDrain())
  }
}

export function stopGroupRelay() {
  relay.disposed = true

  if (relay.drainTimer !== null) {
    clearInterval(relay.drainTimer)
    relay.drainTimer = null
  }

  if (relay.pushDebounceTimer !== null) {
    clearTimeout(relay.pushDebounceTimer)
    relay.pushDebounceTimer = null
  }

  if (relay.storeUnsub !== null) {
    relay.storeUnsub()
    relay.storeUnsub = null
  }

  if (relay.pushUnsub !== null) {
    relay.pushUnsub()
    relay.pushUnsub = null
  }

  relay.watchers.clear()
}

/** Test hook: active watcher ids. */
export function groupRelayWatcherIds() {
  return [...relay.watchers.keys()]
}

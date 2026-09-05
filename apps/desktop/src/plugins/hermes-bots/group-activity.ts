/**
 * The per-room activity feed: a bounded, runtime-only record of turn events
 * for the room view's collapsible Activity list.
 *
 * Depends on the room store (for the epoch it tags events with and the
 * speaker label it renders) and on nothing else, so the coordination engine
 * can record into it without a cycle.
 */

import { atom } from '@hermes/plugin-sdk'

import { $groupChats, groupSpeakerLabel } from './group-chat'
import { groupMemberKey } from './group-membership'
import type { GroupActivityEvent, GroupActivityKind, GroupMember } from './types'

// ── group activity feed ─────────────────────────────────────────────────────
// Runtime-only, bounded per-room record of turn events that feeds the
// collapsible Activity view. Never persisted — it is presentation state like
// running/epoch, and the room transcript (log) stays the only durable record.
// Every event is tagged with the room epoch it belongs to, so the view shows
// only the CURRENT run: a newer send bumps the epoch (old-run events drop
// away), and a rename re-keys the room (the feed starts clean under the new
// name — stale events under the old key simply have no room to attach to).
const GROUP_ACTIVITY_LIMIT = 50

/** A recorded activity row: the caller's event tagged with the room epoch.
 *  Deliberately not `GroupActivityEvent` — the recorder never stamps `group`
 *  (the atom is already keyed by it) and callers carry a `thread`. */
export interface GroupActivityEntry extends Omit<GroupActivityEvent, 'group' | 'member'> {
  epoch: number
  member?: null | string
  memberKey?: string
  memberLabel?: string
  thread?: null | string
}
export const $groupActivity = atom<Record<string, { events: GroupActivityEntry[] }>>({})

type GroupActivityInput = Omit<GroupActivityEntry, 'at' | 'epoch' | 'member'> & {
  member?: GroupMember | null | string
}

function groupActivityMemberSource(member: GroupMember) {
  const connectionId = String(member.connectionId || '').trim()

  return String(member.connectionLabel || '').trim() || connectionId || String(member.handle || groupMemberKey(member) || '').trim()
}

function groupActivityMemberEvent(room: { members?: GroupMember[] } | undefined, member: GroupMember) {
  const name = String(member.name || '').trim()
  const memberKey = String(groupMemberKey(member) || name)
  const connectionId = String(member.connectionId || '').trim()
  const speaker = groupSpeakerLabel(name || 'A bot')
  const speakerKey = speaker.trim().toLowerCase()

  const collisions = (room?.members || []).filter(
    candidate => groupSpeakerLabel(candidate.name || '').trim().toLowerCase() === speakerKey
  )

  let memberLabel = ''

  if (collisions.length > 1) {
    let source = groupActivityMemberSource(member)

    const sameSource = collisions.filter(
      candidate => groupActivityMemberSource(candidate).toLowerCase() === source.toLowerCase()
    )

    if (sameSource.length > 1 && connectionId && connectionId.toLowerCase() !== source.toLowerCase()) {
      source = `${source} (${connectionId})`
    }

    const qualifier = source && name ? `${source}/${name}` : source || name
    memberLabel = qualifier ? `${speaker} · ${qualifier}` : speaker
  }

  return {
    member: name || null,
    ...(memberKey ? { memberKey } : {}),
    ...(memberLabel ? { memberLabel } : {})
  }
}

export function recordGroupActivity(group: string, event: GroupActivityInput) {
  const room = $groupChats.get()[group]

  if (!room) {
    return null
  }

  const current = $groupActivity.get()[group] || {
    events: []
  }

  const normalizedEvent =
    event.member && typeof event.member === 'object'
      ? { ...event, ...groupActivityMemberEvent(room, event.member) }
      : (event as Omit<GroupActivityEntry, 'at' | 'epoch'>)

  const entry: GroupActivityEntry = {
    at: Date.now(),
    epoch: room.epoch || 0,
    ...normalizedEvent
  }

  const events = [...current.events, entry].slice(-GROUP_ACTIVITY_LIMIT)
  $groupActivity.set({
    ...$groupActivity.get(),
    [group]: {
      ...current,
      events
    }
  })

  return entry
}

/** Events for the room's CURRENT run — superseded runs (epoch moved on)
 *  are dropped from view instead of describing work that already ended. */
export function currentGroupActivity(group: string) {
  const epoch = ($groupChats.get()[group] || {}).epoch || 0

  return ($groupActivity.get()[group] || {}).events?.filter(event => (event.epoch || 0) === epoch) || []
}

/** Human label for one activity event, used by the collapsed summary and
 *  the expanded rows. */
export function groupActivityLabel(event: GroupActivityEntry) {
  const kind = event?.kind
  const base = GROUP_ACTIVITY_LABELS[kind] || kind || 'did something'

  if ((kind === 'cancelled' && !event.member) || kind === 'settled' || kind === 'capped') {
    return base
  }

  const who = event?.member === 'You' ? 'You' : event.memberLabel || groupSpeakerLabel(event?.member || 'A bot')

  return `${who} ${base}`
}

const GROUP_ACTIVITY_LABELS: Record<GroupActivityKind, string> = {
  queued: 'sent a message',
  working: 'is working…',
  replied: 'replied',
  passed: 'passed',
  'timed-out': 'took too long',
  failed: 'hit an error',
  cancelled: 'turn interrupted by a newer message',
  settled: 'turn settled',
  capped: 'turn stopped at the round/message cap',
  delivered: 'delivered a late reply',
  held: 'is held (stopped by you) — @mention it or say resume to release',
  stopped: 'stopped the room — remaining turns are held until resumed'
}

export const GROUP_ACTIVITY_GLYPHS: Record<GroupActivityKind, string> = {
  queued: 'comment',
  working: 'sync',
  replied: 'check',
  passed: 'circle-outline',
  'timed-out': 'clock',
  failed: 'error',
  cancelled: 'close',
  settled: 'check-all',
  capped: 'debug-step-over',
  delivered: 'mail-read',
  held: 'debug-pause',
  stopped: 'debug-stop'
}

/** Text tone for an activity row: quiet for pass/cancel/settle, accent for
 *  work and real replies, destructive for failures and timeouts. */
export function groupActivityTone(kind: GroupActivityKind) {
  if (kind === 'failed' || kind === 'timed-out') {
    return 'text-destructive'
  }

  if (kind === 'working' || kind === 'replied' || kind === 'delivered') {
    return 'text-(--ui-accent)'
  }

  return 'text-(--ui-text-tertiary)'
}

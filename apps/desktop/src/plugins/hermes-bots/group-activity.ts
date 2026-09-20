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
import { botsText } from './i18n'
import type { GroupActivityEvent, GroupActivityKind } from './types'

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
  thread?: null | string
}
export const $groupActivity = atom<Record<string, { events: GroupActivityEntry[] }>>({})

export function recordGroupActivity(group: string, event: Omit<GroupActivityEntry, 'at' | 'epoch'>) {
  const room = $groupChats.get()[group]

  if (!room) {
    return null
  }

  const current = $groupActivity.get()[group] || {
    events: []
  }

  const entry = {
    at: Date.now(),
    epoch: room.epoch || 0,
    ...event
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
 *  the expanded rows. `group` scopes the same-name disambiguation to the
 *  room's seats. */
export function groupActivityLabel(event: GroupActivityEntry, group?: null | string) {
  const kind = event?.kind
  const g = botsText().group

  if (kind === 'cancelled') {
    return g.activityCancelled
  }

  if (kind === 'settled') {
    return g.activitySettled
  }

  if (kind === 'capped') {
    return g.activityCapped
  }

  const who = event?.member === 'You' ? g.you : groupSpeakerLabel(event?.member || 'A bot', group)

  switch (kind) {
    case 'queued':
      return g.activityQueued(who)

    case 'working':
      return g.activityWorking(who)

    case 'replied':
      return g.activityReplied(who)

    case 'passed':
      return g.activityPassed(who)

    case 'timed-out':
      return g.activityTimedOut(who)

    case 'failed':
      return g.activityFailed(who)

    case 'delivered':
      return g.activityDelivered(who)

    case 'held':
      return g.activityHeld(who)

    case 'stopped':
      return g.activityStopped(who)

    default:
      return `${who} ${kind || 'did something'}`
  }
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

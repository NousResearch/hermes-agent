/**
 * The per-room activity feed: a bounded, runtime-only record of turn events
 * for the room view's collapsible Activity list.
 *
 * Depends on the room store for epoch/speaker truth and mirrors live member
 * turns into the shared Agents store. The transcript remains authoritative;
 * this module only projects its already-recorded activity lifecycle.
 */

import { atom, host } from '@hermes/plugin-sdk'

import { $groupChats, groupSpeakerLabel } from './group-chat'
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

const groupAgentTurns = new Map<string, { id: string; source?: string }>()
let groupAgentTurnSequence = 0

const groupAgentScope = (group: string, roomId?: null | string) => {
  const room = $groupChats.get()[group]

  return `bot-group:${roomId || room?.roomId || group}`
}

const groupAgentTurnKey = (scope: string, thread: null | string | undefined, member: string) =>
  `${scope}:${thread || 'room'}:${member}`

/** Remove the Agents-panel projection for a room without touching native
 * subagents or another room. Used on a new room run and on disband. */
export function clearGroupAgentActivity(group: string, roomId?: null | string) {
  const scope = groupAgentScope(group, roomId)
  host.agentActivity?.clear(scope)

  for (const key of groupAgentTurns.keys()) {
    if (key.startsWith(`${scope}:`)) {
      groupAgentTurns.delete(key)
    }
  }
}

function mirrorGroupActivityToAgents(group: string, entry: GroupActivityEntry) {
  const scope = groupAgentScope(group)

  if (entry.kind === 'queued' && entry.member === 'You') {
    clearGroupAgentActivity(group)

    return
  }

  const member = entry.member && entry.member !== 'You' ? entry.member : null

  if (!member) {
    if (entry.kind === 'stopped') {
      for (const [key, turn] of groupAgentTurns) {
        if (!key.startsWith(`${scope}:`)) {
          continue
        }

        host.agentActivity?.update(
          scope,
          {
            createIfMissing: false,
            id: turn.id,
            status: 'interrupted',
            summary: `Stopped in ${group}`
          }
        )
        groupAgentTurns.delete(key)
      }
    }

    return
  }

  const turnKey = groupAgentTurnKey(scope, entry.thread, member)
  let turn = groupAgentTurns.get(turnKey)

  if (entry.kind === 'working') {
    // Manual/history-only activity rows are presentation data; only an actual
    // live room drive belongs in the global Agents activity panel.
    if (!$groupChats.get()[group]?.running) {
      return
    }

    const source = entry.source || undefined
    turn = {
      id: `${turnKey}:${++groupAgentTurnSequence}`,
      source
    }
    groupAgentTurns.set(turnKey, turn)
    host.agentActivity?.update(
      scope,
      {
        goal: `${source ? `${source} → ` : ''}${member} · ${group}`,
        id: turn.id,
        status: 'running',
        text: `${member} started work in ${group}`
      }
    )

    return
  }

  if (!turn) {
    return
  }

  if (entry.kind === 'timed-out') {
    host.agentActivity?.update(
      scope,
      {
        createIfMissing: false,
        id: turn.id,
        status: 'running',
        text: `${member} is still working; the reply will be delivered when ready`
      }
    )

    return
  }

  let terminal: { status: 'cancelled' | 'completed' | 'failed'; summary: string } | undefined

  switch (entry.kind) {
    case 'cancelled':
      terminal = { status: 'cancelled', summary: `Turn superseded in ${group}` }

      break

    case 'delivered':
      terminal = { status: 'completed', summary: `Delivered a late reply in ${group}` }

      break

    case 'failed':
      terminal = { status: 'failed', summary: `Failed in ${group}` }

      break

    case 'passed':
      terminal = { status: 'completed', summary: `Reviewed ${group}; no reply needed` }

      break

    case 'replied':
      terminal = { status: 'completed', summary: `Replied in ${group}` }

      break
  }

  if (!terminal) {
    return
  }

  host.agentActivity?.update(
    scope,
    {
      createIfMissing: false,
      id: turn.id,
      status: terminal.status === 'cancelled' ? 'interrupted' : terminal.status,
      summary: terminal.summary
    }
  )
  groupAgentTurns.delete(turnKey)
}

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
  mirrorGroupActivityToAgents(group, entry)

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

  if (kind === 'cancelled' || kind === 'settled' || kind === 'capped') {
    return base
  }

  const who = event?.member === 'You' ? 'You' : groupSpeakerLabel(event?.member || 'A bot')

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

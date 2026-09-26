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

/** Normalized cause for a pool-slot wait timeout — the local backend pool
 *  had no free slot, so the member never started. Stored as the activity
 *  event's `reason` so the feed can tell it apart from a bot crash.
 *  The string deliberately contains "timeout": `attentionReasonFromError`
 *  must keep classifying it as transient (never a roster badge). */
export const GROUP_SLOT_WAIT_REASON = 'slot_wait_timeout'

/** Stable coordinator phrase (`pool-spawn-coordinator.ts`), the
 *  cross-process discriminator — same shape as `isLocalBackendSlotWaitTimeout`
 *  in `store/pool-limits.ts`, kept local because the plugin fence cannot
 *  import the store. Match narrowly so other backend failures keep their path. */
export function isGroupSlotWaitTimeoutText(text: unknown): boolean {
  return typeof text === 'string' && text.includes('timed out while waiting for a free slot')
}

/** Typed failure cause for a member-turn error: the gateway's
 *  `data.reason` when present, else the normalized slot-wait cause when the
 *  message carries the coordinator phrase, else the error's own first line.
 *  Single home for the classification so the turn catch and the stranded
 *  harvest cannot drift. #117366: a bare "X hit an error" row gave the user
 *  nothing to act on (a stopped backend, a dead IPC bridge and a provider
 *  refusal all read the same), so an unclassified failure keeps its message. */
export function groupFailureReason(error: unknown): string {
  const typed =
    typeof (error as { data?: { reason?: unknown } })?.data?.reason === 'string'
      ? String((error as { data: { reason: string } }).data.reason).trim()
      : ''

  if (typed) {
    return typed
  }

  const message = error instanceof Error ? error.message : typeof error === 'string' ? error : ''

  return isGroupSlotWaitTimeoutText(message) ? GROUP_SLOT_WAIT_REASON : groupFailureDetail(message)
}

const GROUP_FAILURE_DETAIL_LIMIT = 200

// Secret-shaped spans an error line can carry (a bearer header, a `?token=`
// URL, a vendor API key, `user@host:password`). The plugin fence keeps the
// Electron-side `redactSecrets` out of reach, so the same shapes live here.
const GROUP_FAILURE_REDACTIONS: Array<[RegExp, string]> = [
  [/(authorization["']?\s*[:=]\s*["']?bearer\s+)(\S+)/gi, '$1<redacted>'],
  [/([?&](?:token|ticket|api_?key|key|access_token|secret)=)([^\s&"']+)/gi, '$1<redacted>'],
  [/\b(sk-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9]{8,}|xox[abprs]-[A-Za-z0-9-]{8,})\b/g, '<redacted>'],
  [/(\S+@[^\s:/]+):(?!\d+\b)[^\s:/]+/g, '$1:<redacted>']
]

/** The first non-empty line of a raw error message, trimmed, secret spans
 *  redacted, capped — the room row is a summary, the log keeps the rest. */
export function groupFailureDetail(message: unknown): string {
  const line =
    String(message || '')
      .split(/\r?\n/)
      .map(part => part.trim())
      .find(Boolean) || ''

  const redacted = GROUP_FAILURE_REDACTIONS.reduce((text, [re, repl]) => text.replace(re, repl), line)

  return redacted.length > GROUP_FAILURE_DETAIL_LIMIT
    ? `${redacted.slice(0, GROUP_FAILURE_DETAIL_LIMIT - 1)}…`
    : redacted
}

/** Human label for one activity event, used by the collapsed summary and
 *  the expanded rows. `group` scopes the same-name disambiguation to the
 *  room's seats. A slot-wait failure renders distinctly from a bot crash so
 *  pool saturation is not misread as a broken bot. */
export function groupActivityLabel(event: GroupActivityEntry, group?: null | string) {
  const kind = event?.kind
  const base = GROUP_ACTIVITY_LABELS[kind] || kind || 'did something'

  if (kind === 'cancelled' || kind === 'settled' || kind === 'capped') {
    return base
  }

  const who = event?.member === 'You' ? 'You' : groupSpeakerLabel(event?.member || 'A bot', group)
  const reason = kind === 'failed' ? String(event?.reason || '').trim() : ''

  if (kind === 'failed' && reason === GROUP_SLOT_WAIT_REASON) {
    return `${who} couldn't start — too many bots running`
  }

  return `${who} ${base}${reason ? ` — ${reason}` : ''}`
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

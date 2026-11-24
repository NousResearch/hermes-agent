import { Button } from '@hermes/plugin-sdk'

import type { CanonicalGroupEvent } from './canonical-group-history'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import type { CanonicalRoomMember } from './canonical-groups'
import { fill, RoomNoticeRow } from './group-room-ui'

/** Gateway-authored events of an auto-routed, continuable discussion (`gateway/hosted_room_discussion.py`). */
export const ROUTED_EVENT_KIND = 'room.routed'
export const CONTINUED_EVENT_KIND = 'room.continued'

const discussionOf = (event: CanonicalGroupEvent) =>
  typeof event.payload.discussion_event_id === 'string' ? event.payload.discussion_event_id : ''

export const isBoundedActivity = (event: CanonicalGroupEvent) => event.kind === 'room.activity' && event.payload.status === 'bounded'

/** "Assigned to Ada, Linus": who the router picked. Names only; an id the roster no longer has is skipped. */
export function CanonicalGroupRoutedRow({ event, members = [] }: { event: CanonicalGroupEvent; members?: readonly CanonicalRoomMember[] }) {
  const labels = useCanonicalGroupLabels()
  const roster = new Map(members.map(member => [member.member_id, member]))
  const payload = event.payload as { member_ids?: unknown; everyone?: boolean; fallback?: boolean }
  const ids = Array.isArray(payload.member_ids) ? payload.member_ids.filter((id): id is string => typeof id === 'string') : []

  const names = ids.flatMap(id => {
    const member = roster.get(id)
    const name = member?.display_name || member?.handle

    return name ? [name] : []
  })

  const text = payload.fallback ? labels.routedFallback
    : payload.everyone ? labels.routedEveryone
      : names.length ? fill(labels.routedTo, { names: names.join(', ') }) : ''

  // Every pick has left the roster: say nothing rather than show an id.
  if (!text) {return null}

  return <p className="px-2 py-1 text-xs text-(--ui-text-secondary)" data-testid="room-routed">{text}</p>
}

export function CanonicalGroupContinuedRow() {
  const labels = useCanonicalGroupLabels()

  return <p className="px-2 py-1 text-xs text-(--ui-text-secondary)" data-testid="room-continued">{labels.continued}</p>
}

export interface CanonicalPausedDiscussion {
  /** Member messages the discussion produced before it paused. */
  count: number
  /** Continue/Stop still mean something: nothing has moved the discussion on since this pause. */
  actionable: boolean
}

const bump = (counts: Map<string, number>, key: string) => counts.set(key, (counts.get(key) ?? 0) + 1)

/** Forward pass: each pause's discussion message count before it, and every discussion's Continue total. */
function pauseCounts(ordered: readonly CanonicalGroupEvent[]) {
  const said = new Map<string, number>()
  const grants = new Map<string, number>()
  const counts = new Map<number, number>()

  for (const event of ordered) {
    const discussion = discussionOf(event)

    if (!discussion) {continue}

    if (event.kind === 'message.member') {bump(said, discussion)}

    if (event.kind === CONTINUED_EVENT_KIND) {bump(grants, discussion)}

    if (isBoundedActivity(event)) {counts.set(event.seq, said.get(discussion) ?? 0)}
  }

  return { counts, grants }
}

/** Backward pass: the pauses something later moved on. A Stop retires every earlier pause; a Continue
 * or a later pause retires its own discussion's; a new user message retires its thread's. */
function movedOnPauses(ordered: readonly CanonicalGroupEvent[]) {
  const moved = new Set<number>()
  const laterDiscussions = new Set<string>()
  const laterThreads = new Set<string>()
  let laterStop = false

  for (let index = ordered.length - 1; index >= 0; index--) {
    const event = ordered[index]
    const discussion = discussionOf(event)
    const thread = typeof event.payload.thread_id === 'string' ? event.payload.thread_id : ''

    if (isBoundedActivity(event) && (laterStop || laterDiscussions.has(discussion) || (thread && laterThreads.has(thread)))) {moved.add(event.seq)}

    if (event.kind === 'room.stop_requested') {laterStop = true}

    if (discussion && (event.kind === CONTINUED_EVENT_KIND || isBoundedActivity(event))) {laterDiscussions.add(discussion)}

    if (event.kind === 'message.user' && thread) {laterThreads.add(thread)}
  }

  return moved
}

/**
 * Every `bounded` pause in the log, keyed by its seq: how much its discussion said before it, and
 * whether Continue/Stop still mean something. Two linear passes per log update, so a room with many
 * old pauses costs O(events), not O(events x pauses).
 */
export function pausedDiscussions(events: readonly CanonicalGroupEvent[], maxContinues: number): Map<number, CanonicalPausedDiscussion> {
  const ordered = [...events].sort((a, b) => a.seq - b.seq)
  const { counts, grants } = pauseCounts(ordered)
  const moved = movedOnPauses(ordered)
  // A Stop after the discussion's own message fences it for good, even one older than the cap notice.
  const lastStop = ordered.reduce((seq, event) => event.kind === 'room.stop_requested' ? event.seq : seq, 0)
  const sourceSeq = new Map(ordered.flatMap(event => event.kind === 'message.user' && event.event_id ? [[event.event_id, event.seq] as const] : []))
  const result = new Map<number, CanonicalPausedDiscussion>()

  for (const event of ordered) {
    if (!isBoundedActivity(event)) {continue}

    const discussion = discussionOf(event)
    const fenced = lastStop > (sourceSeq.get(discussion) ?? event.seq)
    result.set(event.seq, { count: counts.get(event.seq) ?? 0, actionable: !!discussion && !fenced && !moved.has(event.seq) && (grants.get(discussion) ?? 0) < maxContinues })
  }

  return result
}

export interface CanonicalGroupPausedRowProps {
  /** A Send or Continue is in flight: Continue waits; Stop stays available. */
  busy?: boolean
  /** A Stop is already in flight: neither button can act again. */
  stopping?: boolean
  event: CanonicalGroupEvent
  /** This pause's entry from `pausedDiscussions`, computed once per log update. */
  paused: CanonicalPausedDiscussion
  onContinue?: (discussionEventId: string) => unknown
  onStop?: () => unknown
}

/** "Paused after 12 messages" with Continue and Stop while the pause is still the discussion's latest word. */
export function CanonicalGroupPausedRow({ busy = false, stopping = false, event, paused, onContinue, onStop }: CanonicalGroupPausedRowProps) {
  const labels = useCanonicalGroupLabels()
  const { actionable, count } = paused
  const discussion = discussionOf(event)

  return <RoomNoticeRow kind={event.kind}>
    <div className="flex min-w-0 flex-wrap items-center gap-2" data-testid="room-paused">
      <strong className="text-[0.6875rem] font-medium text-(--ui-text-secondary)">{fill(labels.pausedAfter, { count })}</strong>
      {actionable && <>
        <Button data-testid="room-continue" disabled={busy || stopping || !onContinue} onClick={() => void onContinue?.(discussion)} size="xs" variant="secondary">{labels.continueDiscussion}</Button>
        <Button data-testid="room-paused-stop" disabled={stopping || !onStop} onClick={() => void onStop?.()} size="xs" variant="ghost">{labels.stop}</Button>
      </>}
    </div>
  </RoomNoticeRow>
}

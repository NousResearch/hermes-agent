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

/** Read a `bounded` activity event against the log: how much was said, and whether it can still be continued. */
export function pausedDiscussion(events: readonly CanonicalGroupEvent[], paused: CanonicalGroupEvent, maxContinues: number): CanonicalPausedDiscussion {
  const discussion = discussionOf(paused)
  const thread = paused.payload.thread_id
  let count = 0
  let grants = 0
  let movedOn = false

  for (const event of events) {
    const same = !!discussion && discussionOf(event) === discussion

    if (event.kind === 'message.member' && same && event.seq < paused.seq) {count++}

    if (event.kind === CONTINUED_EVENT_KIND && same) {
      grants++

      if (event.seq > paused.seq) {movedOn = true}
    }

    if (event.seq <= paused.seq) {continue}

    if (event.kind === 'room.stop_requested' || (isBoundedActivity(event) && same) ||
        (event.kind === 'message.user' && typeof thread === 'string' && event.payload.thread_id === thread)) {movedOn = true}
  }

  return { count, actionable: !!discussion && !movedOn && grants < maxContinues }
}

export interface CanonicalGroupPausedRowProps {
  busy?: boolean
  event: CanonicalGroupEvent
  events: readonly CanonicalGroupEvent[]
  maxContinues: number
  onContinue?: (discussionEventId: string) => unknown
  onStop?: () => unknown
}

/** "Paused after 12 messages" with Continue and Stop while the pause is still the discussion's latest word. */
export function CanonicalGroupPausedRow({ busy = false, event, events, maxContinues, onContinue, onStop }: CanonicalGroupPausedRowProps) {
  const labels = useCanonicalGroupLabels()
  const { actionable, count } = pausedDiscussion(events, event, maxContinues)
  const discussion = discussionOf(event)

  return <RoomNoticeRow kind={event.kind}>
    <div className="flex min-w-0 flex-wrap items-center gap-2" data-testid="room-paused">
      <strong className="text-[0.6875rem] font-medium text-(--ui-text-secondary)">{fill(labels.pausedAfter, { count })}</strong>
      {actionable && <>
        <Button data-testid="room-continue" disabled={busy || !onContinue} onClick={() => void onContinue?.(discussion)} size="xs" variant="secondary">{labels.continueDiscussion}</Button>
        <Button data-testid="room-paused-stop" disabled={busy || !onStop} onClick={() => void onStop?.()} size="xs" variant="ghost">{labels.stop}</Button>
      </>}
    </div>
  </RoomNoticeRow>
}

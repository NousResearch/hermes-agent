import { Button } from '@hermes/plugin-sdk'
import { Fragment, type MouseEvent, useEffect, useState } from 'react'

import type { CanonicalGroupEvent } from './canonical-group-history'
import type { CanonicalRoomMember } from './canonical-groups'
import { RoomBotAvatar } from './group-room-avatar'
import {
  focusThread,
  isBusyStatus,
  memberOutcomes,
  memberStatus,
  type MemberStatus,
  type RoomCycle,
  roomCycle,
  type RoomLogEvent,
  type RoomMemberActivity
} from './group-room-status'
import {
  MemberStatusChip,
  ROOM_AVATAR_SIZE,
  RoomCycleFooter,
  type RoomStatusLabels,
  RoomTypingRow
} from './group-room-ui'

/** `groups.state` driver fields that say who is answering. */
export interface CanonicalDriverActivity {
  working?: boolean
  member_activity?: RoomMemberActivity[]
  max_rounds?: null | number
}

/** `groups.state` driver fields that say what the room is waiting on. */
export interface CanonicalDriverStatus<
  Action extends { kind: string } = { kind: string }
> extends CanonicalDriverActivity {
  blocked?: boolean
  counts?: Record<string, number>
  pending_actions?: Action[]
  /** Each member of another gateway with a registered route, and its status (`ready` once joined). */
  peer_routes?: { member_id: string; status: string }[]
}

export interface CanonicalActivityLabels extends RoomStatusLabels {
  groupWorking: string
  memberWorking: string
}

type BusyStatus = Extract<MemberStatus, { kind: 'stalled' | 'working' }>

export const canonicalMemberName = (member: CanonicalRoomMember) =>
  member.display_name || member.handle || member.member_id
const avatarName = (member: CanonicalRoomMember) => member.profile || member.handle || member.member_id

/** Seconds since the epoch, ticking once a second while `live` (elapsed times in the chips). */
export function useNowSeconds(live: boolean): number {
  const [now, setNow] = useState(() => Date.now() / 1000)

  useEffect(() => {
    setNow(Date.now() / 1000)

    if (!live) {
      return
    }
    const timer = setInterval(() => setNow(Date.now() / 1000), 1000)

    return () => clearInterval(timer)
  }, [live])

  return now
}

/** The coarse phase the roster has always exposed (`data-phase`); outcomes read as idle. */
const PHASE: Partial<Record<MemberStatus['kind'], string>> = {
  working: 'working',
  stalled: 'working',
  queued: 'queued'
}

const statusOf = (statuses: Map<string, MemberStatus>, member: CanonicalRoomMember): MemberStatus =>
  statuses.get(member.member_id) ?? { kind: 'idle' }

export function CanonicalGroupRoster({
  members = [],
  label,
  labels,
  statuses,
  peerBadges
}: {
  members?: CanonicalRoomMember[]
  label: string
  labels: RoomStatusLabels & { lead: string }
  statuses: Map<string, MemberStatus>
  peerBadges?: ReadonlyMap<string, string | undefined>
}) {
  if (!members.length) {
    return null
  }

  return (
    <ul aria-label={label} className="flex min-w-0 flex-wrap gap-1.5">
      {members.map(member => {
        const status = statusOf(statuses, member)

        return (
          <MemberStatusChip
            avatar={<RoomBotAvatar name={avatarName(member)} size={18} />}
            badge={
              [(member as { lead?: unknown }).lead && labels.lead, peerBadges?.get(member.member_id)]
                .filter(Boolean)
                .join(' · ') || undefined
            }
            key={member.member_id}
            labels={labels}
            name={canonicalMemberName(member)}
            phase={PHASE[status.kind] ?? 'idle'}
            status={status}
          />
        )
      })}
    </ul>
  )
}

/** A typing-style row per working member at the end of the transcript. */
export function CanonicalGroupTyping({
  members = [],
  labels,
  statuses
}: {
  members?: CanonicalRoomMember[]
  labels: RoomStatusLabels
  statuses: Map<string, MemberStatus>
}) {
  return (
    <>
      {members.map(member => {
        const status = statusOf(statuses, member)

        return isBusyStatus(status) ? (
          <RoomTypingRow
            avatar={<RoomBotAvatar name={avatarName(member)} size={ROOM_AVATAR_SIZE} />}
            key={member.member_id}
            labels={labels}
            name={canonicalMemberName(member)}
            status={status as BusyStatus}
          />
        ) : null
      })}
    </>
  )
}

/** The room's footer: its round cycle once the log names a discussion; before that (or on a
 *  gateway without member activity) the plain working line. */
export function CanonicalGroupCycle({
  cycle,
  driver,
  members = [],
  labels,
  statuses
}: {
  cycle: null | RoomCycle
  driver?: CanonicalDriverStatus
  members?: CanonicalRoomMember[]
  labels: CanonicalActivityLabels
  statuses: Map<string, MemberStatus>
}) {
  const busy = members.filter(member => isBusyStatus(statusOf(statuses, member)))

  if (cycle?.kind === 'settled' && (driver?.pending_actions?.length || driver?.counts?.deferred)) {
    return null
  }

  if (cycle) {
    // Names only (the chips and typing rows carry each member's tool and time), and only the members of
    // the thread whose round the strip shows.
    const named = cycle.kind === 'working' ? busy.filter(member => cycle.members.includes(member.member_id)) : busy

    return <RoomCycleFooter cycle={cycle} labels={labels} working={named.map(canonicalMemberName)} />
  }

  if (!busy.length && !driver?.working) {
    return null
  }

  return (
    <RoomCycleFooter
      cycle={{ kind: 'working', round: 0, members: [] }}
      labels={labels}
      text={busy.length ? `${labels.memberWorking}: ${busy.map(canonicalMemberName).join(', ')}` : labels.groupWorking}
      working={[]}
    />
  )
}

/** Every member's status, the room's cycle and whether it works, from one poll's state + log. Outcomes are
 *  read per thread; the round comes from the live tasks' own thread (`replyThread`'s when it has work). */
export function useCanonicalRoomLive(
  events: CanonicalGroupEvent[],
  driver: CanonicalDriverStatus | undefined,
  members: CanonicalRoomMember[],
  replyThread?: string
) {
  const activity = driver?.member_activity
  const anyLive = !!activity?.some(row => row.status !== 'queued')
  const now = useNowSeconds(anyLive)
  const log = events as RoomLogEvent[]
  const focus = focusThread(log, replyThread)
  const outcomes = memberOutcomes(log)

  const stopConfirmed =
    !!driver?.counts &&
    driver.working === false &&
    activity?.length === 0 &&
    driver.pending_actions?.length === 0 &&
    Object.entries(driver.counts).every(
      ([status, count]) => count === 0 || ['settled', 'cancelled', 'failed'].includes(status)
    )

  const cycle = roomCycle(log, activity, driver?.max_rounds, focus, stopConfirmed)

  return {
    // "Settled" isn't true while a turn waits on a person: its Retry and Discard say what's next.
    cycle: cycle?.kind === 'settled' && driver?.pending_actions?.length ? null : cycle,
    statuses: new Map(
      members.map(member => [member.member_id, memberStatus(member.member_id, activity, outcomes, now)])
    ),
    working: !!driver?.working || !!activity?.length
  }
}

interface AttentionLabels {
  statusBlocked: string
  statusApprovals: string
  statusAttention: string
}

/** What the room waits on: a blocked room, approvals and other pending actions. A gateway whose driver
 *  is down sends no driver status at all, and the room says its driver is unavailable instead. */
export function canonicalRoomAttention(driver: CanonicalDriverStatus | undefined, labels: AttentionLabels): string[] {
  const actions = driver?.pending_actions ?? []
  const approvals = actions.filter(action => action.kind === 'approval').length
  const parts = []

  // The gateway also calls a room blocked while a stopped task settles: that is Stop at work, not a
  // block, unless an unresolved task holds the room as well.
  if (driver?.blocked && (driver.counts?.indeterminate || !driver.counts?.stopping)) {
    parts.push(labels.statusBlocked)
  }

  if (approvals) {
    parts.push(labels.statusApprovals.replace('{count}', String(approvals)))
  }

  if (actions.length > approvals) {
    parts.push(labels.statusAttention.replace('{count}', String(actions.length - approvals)))
  }

  return parts
}

/** Shown beside the working line and roster, never instead of them: one member's failure
 *  does not hide the others' live work. */
export function CanonicalGroupAttention({
  driver,
  labels
}: {
  driver?: CanonicalDriverStatus
  labels: AttentionLabels
}) {
  const parts = canonicalRoomAttention(driver, labels)
  const approvals = driver?.pending_actions?.filter(action => action.kind === 'approval').length ?? 0
  const approvalText = labels.statusApprovals.replace('{count}', String(approvals))

  const jumpToApproval = (event: MouseEvent<HTMLButtonElement>) => {
    const card = event.currentTarget.closest('section')?.querySelector<HTMLElement>('[data-pending-action="approval"]')
    card?.scrollIntoView({ block: 'center' })
    card?.focus({ preventScroll: true })
  }

  return parts.length ? (
    <p aria-live="polite" className="text-xs text-(--ui-text-secondary)">
      {parts.map((part, index) => (
        <Fragment key={part}>
          {index > 0 && ' · '}
          {approvals && part === approvalText ? (
            <Button onClick={jumpToApproval} size="inline" variant="text">
              {part}
            </Button>
          ) : (
            part
          )}
        </Fragment>
      ))}
    </p>
  ) : null
}

/** The room's read, action, journal and driver notices, above its log. */
export function CanonicalGroupNotices({
  readError,
  error,
  journalError,
  notice,
  unavailable,
  labels,
  onRefresh,
  onJournalReload
}: {
  readError: string
  error: string
  journalError: string
  notice: string
  unavailable: boolean
  labels: {
    refresh: string
    driverUnavailable: string
    journalLoadFailed: string
    journalReload: string
    details: string
  }
  onRefresh: () => void
  onJournalReload: () => void
}) {
  return (
    <div className="px-4">
      <div className="mx-auto w-full max-w-3xl">
        {readError && (
          <div className="py-1.5 text-xs text-destructive" role="alert">
            {readError}
            <Button onClick={onRefresh}>{labels.refresh}</Button>
          </div>
        )}
        {error && (
          <div className="py-1.5 text-xs text-destructive" role="alert">
            {error}
          </div>
        )}
        {journalError && (
          <div className="py-1.5 text-xs text-destructive" role="alert">
            <p>{labels.journalLoadFailed}</p>
            <Button onClick={onJournalReload}>{labels.journalReload}</Button>
            <details>
              <summary>{labels.details}</summary>
              {journalError}
            </details>
          </div>
        )}
        {notice && (
          <p aria-live="polite" className="py-1.5 text-xs text-(--ui-text-secondary)">
            {notice}
          </p>
        )}
        {unavailable && <p className="py-1.5 text-xs text-(--ui-text-tertiary)">{labels.driverUnavailable}</p>}
      </div>
    </div>
  )
}

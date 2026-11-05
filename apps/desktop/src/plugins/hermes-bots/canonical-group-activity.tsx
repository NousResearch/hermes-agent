import { useMemo, useRef } from 'react'

import type { CanonicalGroupEvent } from './canonical-group-history'
import type { CanonicalRoomMember } from './canonical-groups'
import { RoomBotAvatar } from './group-room-avatar'
import { focusThread, isBusyStatus, memberOutcomes, memberStatus, type MemberStatus, type RoomCycle, roomCycle, type RoomLogEvent, type RoomMemberActivity } from './group-room-status'
import { MemberStatusChip, ROOM_AVATAR_SIZE, RoomCycleFooter, type RoomStatusLabels, RoomTypingRow } from './group-room-ui'

/** `groups.state` driver fields that say who is answering. */
export interface CanonicalDriverActivity {
  working?: boolean
  member_activity?: RoomMemberActivity[]
  max_rounds?: null | number
}

export interface CanonicalActivityLabels extends RoomStatusLabels {
  groupWorking: string
  memberWorking: string
}

export const canonicalMemberName = (member: CanonicalRoomMember) => member.display_name || member.handle || member.member_id
const avatarName = (member: CanonicalRoomMember) => member.profile || member.handle || member.member_id

/** The coarse phase the roster has always exposed (`data-phase`); outcomes read as idle. */
const PHASE: Partial<Record<MemberStatus['kind'], string>> = { working: 'working', queued: 'queued' }

const statusOf = (statuses: Map<string, MemberStatus>, member: CanonicalRoomMember): MemberStatus =>
  statuses.get(member.member_id) ?? { kind: 'idle' }

export function CanonicalGroupRoster({ members = [], label, labels, statuses }: {
  members?: CanonicalRoomMember[]; label: string; labels: RoomStatusLabels & { lead: string }; statuses: Map<string, MemberStatus>
}) {
  if (!members.length) {return null}

  return <ul aria-label={label} className="flex min-w-0 flex-wrap gap-1.5">
    {members.map(member => {
      const status = statusOf(statuses, member)

      return <MemberStatusChip avatar={<RoomBotAvatar name={avatarName(member)} size={18} />}
        badge={(member as { lead?: unknown }).lead ? labels.lead : undefined} key={member.member_id}
        labels={labels} name={canonicalMemberName(member)} phase={PHASE[status.kind] ?? 'idle'} status={status} />
    })}
  </ul>
}

/** A typing-style row per working member at the end of the transcript. */
export function CanonicalGroupTyping({ members = [], labels, statuses }: {
  members?: CanonicalRoomMember[]; labels: RoomStatusLabels; statuses: Map<string, MemberStatus>
}) {
  return <>{members.map(member => {
    const status = statusOf(statuses, member)

    return isBusyStatus(status)
      ? <RoomTypingRow avatar={<RoomBotAvatar name={avatarName(member)} size={ROOM_AVATAR_SIZE} />} key={member.member_id} labels={labels}
        name={canonicalMemberName(member)} status={status} />
      : null
  })}</>
}

/** The room's footer: its round cycle once the log names a discussion; before that (or on a
 *  gateway without member activity) the plain working line. */
export function CanonicalGroupCycle({ cycle, driver, members = [], labels, statuses }: {
  cycle: null | RoomCycle; driver?: CanonicalDriverActivity; members?: CanonicalRoomMember[]; labels: CanonicalActivityLabels
  statuses: Map<string, MemberStatus>
}) {
  const busy = members.filter(member => isBusyStatus(statusOf(statuses, member)))

  if (cycle) {
    // Names only (the chips and typing rows carry each member's tool and time), and only the members of
    // the thread whose round the strip shows.
    const named = cycle.kind === 'working' ? busy.filter(member => cycle.members.includes(member.member_id)) : busy

    return <RoomCycleFooter cycle={cycle} labels={labels} working={named.map(canonicalMemberName)} />
  }

  if (!busy.length && !driver?.working) {return null}

  return <RoomCycleFooter cycle={{ kind: 'working', round: 0, members: [] }} labels={labels}
    text={busy.length ? `${labels.memberWorking}: ${busy.map(canonicalMemberName).join(', ')}` : labels.groupWorking} working={[]} />
}

/** `value`, or the last one while equal by content. Each read builds new objects; the room's memos key on
 *  identity. Compares only when `value` itself is new (a read), never on a re-render that reuses it. */
export function useSameValue<T>(value: T): T {
  const kept = useRef({ given: value, shown: value })

  if (kept.current.given !== value) {
    kept.current = { given: value, shown: JSON.stringify(value) === JSON.stringify(kept.current.shown) ? kept.current.shown : value }
  }

  return kept.current.shown
}

/** Every member's status, the room's cycle and whether it works, from one poll's state + log. Outcomes are
 *  read per thread; the round comes from the live tasks' own thread (`replyThread`'s when it has work).
 *  Clock-free: it re-derives when the log or the live rows change, never as time passes. */
export function useCanonicalRoomLive(events: CanonicalGroupEvent[], driver: CanonicalDriverActivity | undefined, members: CanonicalRoomMember[], replyThread?: string) {
  const activity = useSameValue(driver?.member_activity)
  const log = events as RoomLogEvent[]
  const focus = useMemo(() => focusThread(log, replyThread), [log, replyThread])
  const outcomes = useMemo(() => memberOutcomes(log), [log])
  const cycle = useMemo(() => roomCycle(log, activity, driver?.max_rounds, focus), [log, activity, driver?.max_rounds, focus])

  const statuses = useMemo(
    () => new Map(members.map(member => [member.member_id, memberStatus(member.member_id, activity, outcomes)])),
    [members, activity, outcomes])

  return { cycle, statuses, working: !!driver?.working || !!activity?.length }
}

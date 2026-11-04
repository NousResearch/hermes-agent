import { useEffect, useState } from 'react'

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

type BusyStatus = Extract<MemberStatus, { kind: 'stalled' | 'working' }>

export const canonicalMemberName = (member: CanonicalRoomMember) => member.display_name || member.handle || member.member_id
const avatarName = (member: CanonicalRoomMember) => member.profile || member.handle || member.member_id

/** Seconds since the epoch, ticking once a second while `live` (elapsed times in the chips). */
export function useNowSeconds(live: boolean): number {
  const [now, setNow] = useState(() => Date.now() / 1000)

   
  useEffect(() => {
    setNow(Date.now() / 1000)

    if (!live) {return}
    const timer = setInterval(() => setNow(Date.now() / 1000), 1000)

    return () => clearInterval(timer)
  }, [live])

  return now
}

/** The coarse phase the roster has always exposed (`data-phase`); outcomes read as idle. */
const PHASE: Partial<Record<MemberStatus['kind'], string>> = { working: 'working', stalled: 'working', queued: 'queued' }

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
        name={canonicalMemberName(member)} status={status as BusyStatus} />
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

/** Every member's status, the room's cycle and whether it works, from one poll's state + log. Outcomes are
 *  read per thread; the round comes from the live tasks' own thread (`replyThread`'s when it has work). */
export function useCanonicalRoomLive(events: CanonicalGroupEvent[], driver: CanonicalDriverActivity | undefined, members: CanonicalRoomMember[], replyThread?: string) {
  const activity = driver?.member_activity
  const anyLive = !!activity?.some(row => row.status !== 'queued')
  const now = useNowSeconds(anyLive)
  const log = events as RoomLogEvent[]
  const focus = focusThread(log, replyThread)
  const outcomes = memberOutcomes(log)

  return {
    cycle: roomCycle(log, activity, driver?.max_rounds, focus),
    statuses: new Map(members.map(member => [member.member_id, memberStatus(member.member_id, activity, outcomes, now)])),
    working: !!driver?.working || !!activity?.length
  }
}

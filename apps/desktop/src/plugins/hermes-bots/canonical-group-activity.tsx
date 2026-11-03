import { GlyphSpinner } from '@hermes/plugin-sdk'

import type { CanonicalRoomMember } from './canonical-groups'

/** `groups.state` driver fields that say who is answering. */
export interface CanonicalDriverActivity {
  working?: boolean
  member_activity?: { member_id: string; status: string }[]
}

export type CanonicalMemberPhase = 'queued' | 'working'

interface ActivityLabels {
  groupWorking: string
  memberQueued: string
  memberWorking: string
}

const memberName = (member: CanonicalRoomMember) => member.display_name || member.handle || member.member_id

/** One phase per member: a running (or stopping) task outranks one still queued behind it. */
export function canonicalMemberPhases(driver?: CanonicalDriverActivity): Map<string, CanonicalMemberPhase> {
  const phases = new Map<string, CanonicalMemberPhase>()

  for (const { member_id, status } of driver?.member_activity ?? []) {
    if (status !== 'queued') {
      phases.set(member_id, 'working')
    } else if (!phases.has(member_id)) {
      phases.set(member_id, 'queued')
    }
  }

  return phases
}

export function CanonicalGroupRoster({ members = [], label, labels, phases }: {
  members?: CanonicalRoomMember[]; label: string; labels: ActivityLabels; phases: Map<string, CanonicalMemberPhase>
}) {
  if (!members.length) {return null}

  return <ul aria-label={label} className="flex flex-wrap gap-x-2 text-xs text-(--ui-text-tertiary)">
    {members.map(member => {
      const phase = phases.get(member.member_id)

      return <li className="inline-flex items-center gap-1" data-phase={phase ?? 'idle'} key={member.member_id}>
        {phase === 'working' && <GlyphSpinner ariaLabel={labels.memberWorking} className="text-(--ui-accent)" />}
        <span className={phase === 'working' ? 'text-(--ui-text-secondary)' : undefined}>{memberName(member)}</span>
        {phase === 'queued' && <span className="text-(--ui-text-quaternary)">· {labels.memberQueued}</span>}
      </li>
    })}
  </ul>
}

/** The room's live line under the log; older gateways without `member_activity` still say the group is working. */
export function CanonicalGroupWorkingLine({ driver, members = [], labels, phases }: {
  driver?: CanonicalDriverActivity; members?: CanonicalRoomMember[]; labels: ActivityLabels
  phases: Map<string, CanonicalMemberPhase>
}) {
  const working = members.filter(member => phases.get(member.member_id) === 'working').map(memberName)

  if (!working.length && !driver?.working) {return null}

  return <div className="flex items-center gap-1.5 px-2 py-1 text-[0.7rem] italic text-(--ui-text-quaternary)" role="status">
    {/* The line is the live region; the spinner's own status role would nest a second one. */}
    <span aria-hidden="true"><GlyphSpinner className="not-italic" /></span>
    <span>{working.length ? `${labels.memberWorking}: ${working.join(', ')}` : labels.groupWorking}</span>
  </div>
}

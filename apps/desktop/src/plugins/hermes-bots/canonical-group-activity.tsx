import { Button, GlyphSpinner } from '@hermes/plugin-sdk'

import type { CanonicalRoomMember } from './canonical-groups'

/** `groups.state` driver fields that say who is answering. */
export interface CanonicalDriverActivity {
  working?: boolean
  member_activity?: { member_id: string; status: string }[]
}

/** `groups.state` driver fields that say what the room is waiting on. */
export interface CanonicalDriverStatus<Action extends { kind: string } = { kind: string }> extends CanonicalDriverActivity {
  blocked?: boolean
  counts?: Record<string, number>
  pending_actions?: Action[]
}

export type CanonicalMemberPhase = 'queued' | 'working'

interface ActivityLabels {
  groupWorking: string
  memberQueued: string
  memberWorking: string
}

export const memberName = (member: CanonicalRoomMember) => member.display_name || member.handle || member.member_id

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
  if (driver?.blocked && (driver.counts?.indeterminate || !driver.counts?.stopping)) {parts.push(labels.statusBlocked)}

  if (approvals) {parts.push(labels.statusApprovals.replace('{count}', String(approvals)))}

  if (actions.length > approvals) {parts.push(labels.statusAttention.replace('{count}', String(actions.length - approvals)))}

  return parts
}

/** Shown beside the working line and roster, never instead of them: one member's failure
 *  does not hide the others' live work. */
export function CanonicalGroupAttention({ driver, labels }: { driver?: CanonicalDriverStatus; labels: AttentionLabels }) {
  const parts = canonicalRoomAttention(driver, labels)

  return parts.length ? <p aria-live="polite" className="text-xs text-(--ui-text-secondary)">{parts.join(' · ')}</p> : null
}

/** The room's read, action, journal and driver notices, above its log. */
export function CanonicalGroupNotices({ readError, error, journalError, notice, unavailable, labels, onRefresh, onJournalReload }: {
  readError: string
  error: string
  journalError: string
  notice: string
  unavailable: boolean
  labels: { refresh: string; driverUnavailable: string; journalLoadFailed: string; journalReload: string; details: string }
  onRefresh: () => void
  onJournalReload: () => void
}) {
  return <>
    {readError && <div role="alert">{readError}<Button onClick={onRefresh}>{labels.refresh}</Button></div>}
    {error && <div role="alert">{error}</div>}
    {journalError && <div role="alert">
      <p>{labels.journalLoadFailed}</p>
      <Button onClick={onJournalReload}>{labels.journalReload}</Button>
      <details><summary>{labels.details}</summary>{journalError}</details>
    </div>}
    {notice && <p aria-live="polite">{notice}</p>}
    {unavailable && <p>{labels.driverUnavailable}</p>}
  </>
}

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@hermes/plugin-sdk'

import type { CanonicalGroupEvent } from './canonical-group-history'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import type { CanonicalRoomContext, CanonicalRoomMember } from './canonical-groups'

/** The gateway-authored event a lead's routing turn publishes (`gateway/hosted_room_lead.py`). */
export const ASSIGNMENT_EVENT_KIND = 'room.assignment'
const EVERYONE = 'everyone'
const AUTO = 'auto'
// Member options carry a prefix: `everyone` is a reserved handle, not a reserved member id.
const MEMBER_PREFIX = 'member:'

/** The lead's member id when the room routes unaddressed messages through a lead, else undefined. */
export const canonicalRoomLeadId = (context?: CanonicalRoomContext | null) =>
  context?.routing === 'lead' && context.lead_member_id ? context.lead_member_id : undefined

/** Roster entries with a `lead` flag, for the member chips to show a "Lead" badge. */
export function markRoomLead(members: readonly CanonicalRoomMember[] = [], context?: CanonicalRoomContext | null) {
  const lead = canonicalRoomLeadId(context)

  return members.map(member => ({ ...member, lead: member.member_id === lead }))
}

const memberName = (member?: CanonicalRoomMember, fallback = '') => member?.display_name || member?.handle || fallback

const fill = (template: string, values: Record<string, string>) =>
  template.replace(/\{(\w+)\}/g, (match, key: string) => values[key] ?? match)

export interface CanonicalGroupLeadSettingProps {
  /** The runtime can pick who answers with a model call (`driver_status.routing_modes` has `auto`). */
  autoRouting?: boolean
  disabled?: boolean
  draft: CanonicalRoomContext
  members: readonly CanonicalRoomMember[]
  onChange: (draft: CanonicalRoomContext) => void
}

const ROUTING_HINT = { [EVERYONE]: 'routingEveryoneHint', [AUTO]: 'routingAutoHint' } as const

/** Who answers a message that names nobody: everyone (the default), a model pick, or the member picked as lead. */
export function CanonicalGroupLeadSetting({ autoRouting = false, disabled, draft, members, onChange }: CanonicalGroupLeadSettingProps) {
  const labels = useCanonicalGroupLabels()
  const lead = canonicalRoomLeadId(draft)
  const value = lead !== undefined ? MEMBER_PREFIX + lead : draft.routing === AUTO ? AUTO : EVERYONE

  const choose = (next: string) => onChange(next.startsWith(MEMBER_PREFIX)
    ? { ...draft, routing: 'lead', lead_member_id: next.slice(MEMBER_PREFIX.length) }
    : { ...draft, routing: next === AUTO ? AUTO : EVERYONE, lead_member_id: '' })

  const hint = value in ROUTING_HINT ? labels[ROUTING_HINT[value as keyof typeof ROUTING_HINT]] : labels.routingLeadHint

  return <label className="grid gap-1 text-xs">{labels.routingLabel}
    <Select disabled={disabled || !members.length} onValueChange={choose} value={value}>
      <SelectTrigger aria-label={labels.routingLabel} className="h-8 rounded-md"><SelectValue /></SelectTrigger>
      <SelectContent>
        <SelectItem value={EVERYONE}>{labels.routingEveryone}</SelectItem>
        {(autoRouting || value === AUTO) && <SelectItem value={AUTO}>{labels.routingAuto}</SelectItem>}
        {members.map(member => <SelectItem key={member.member_id} value={MEMBER_PREFIX + member.member_id}>{fill(labels.routingLead, { name: memberName(member) })}</SelectItem>)}
      </SelectContent>
    </Select>
    <span className="text-(--ui-text-tertiary)">{hint}</span>
  </label>
}

export interface CanonicalGroupAssignmentRowProps {
  event: CanonicalGroupEvent
  members?: readonly CanonicalRoomMember[]
}

/** "Ada gave this to Linus, Grace": one quiet line so the user can see (and correct) the lead's pick. */
export function CanonicalGroupAssignmentRow({ event, members = [] }: CanonicalGroupAssignmentRowProps) {
  const labels = useCanonicalGroupLabels()
  const roster = new Map(members.map(member => [member.member_id, member]))
  const payload = event.payload as { lead_member_id?: string; member_ids?: string[]; fallback?: boolean }
  const lead = memberName(roster.get(payload.lead_member_id ?? ''), payload.lead_member_id)
  const chosen = (payload.member_ids ?? []).map(id => memberName(roster.get(id), id))

  return <p className="px-2 py-1 text-xs text-(--ui-text-secondary)" data-testid="room-assignment">
    {payload.fallback ? fill(labels.assignmentFallback, { lead }) : fill(labels.assignment, { lead, names: chosen.join(', ') })}
  </p>
}

import { Codicon } from '@hermes/plugin-sdk'

import { groupMemberKey } from './group-membership'
import { useBots } from './i18n'
import type { GroupHold, GroupMember } from './types'

interface GroupHoldStatusProps {
  holds?: Record<string, GroupHold>
  memberLabel: (member: GroupMember) => string
  members: GroupMember[]
}

/** Durable room-level explanation for sticky Stop holds. Activity is scoped to
 * one run and disappears across epochs/reloads; this status reads the room's
 * persisted hold map instead. */
export function GroupHoldStatus(props: GroupHoldStatusProps) {
  const b = useBots()
  const held = new Set(Object.keys(props.holds || {}))
  const memberKeys = new Set(props.members.map(groupMemberKey))
  const heldMembers = props.members.filter(member => held.has(groupMemberKey(member)))

  const unavailableHeldLabels = [...held]
    .filter(key => !memberKeys.has(key))
    .map(key => {
      const [source, name] = key.split('::')

      return source && name ? `${name} (${source})` : key
    })

  const heldLabels = [...heldMembers.map(props.memberLabel), ...unavailableHeldLabels]

  const allHeld =
    props.members.length > 0 &&
    unavailableHeldLabels.length === 0 &&
    props.members.every(member => held.has(groupMemberKey(member)))

  const label = allHeld
    ? b.group.allHeldStatus(props.members.length)
    : heldLabels.length
      ? b.group.heldMembersStatus(heldLabels.join(', '))
      : null

  if (!label) {
    return null
  }

  return (
    <div
      className="flex items-start gap-1.5 border-b border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary) px-2.5 py-1.5 text-[0.7rem]"
      data-slot="group-hold-status"
      role="status"
    >
      <Codicon aria-hidden className="mt-0.5 shrink-0 text-(--ui-text-secondary)" name="debug-pause" />
      <div className="min-w-0">
        <div className="font-medium text-(--ui-text-primary)">{label}</div>
        <div className="text-(--ui-text-tertiary)">{b.group.holdReleaseHint}</div>
      </div>
    </div>
  )
}

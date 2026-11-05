/**
 * Presentational pieces of the Desktop-driven (legacy) room view that share the gateway-owned
 * room's parts, kept out of `group-chat-view.tsx` so that file does not grow.
 */

import { cn, Codicon, Tip } from '@hermes/plugin-sdk'
import type { ReactNode } from 'react'

import { useCanonicalGroupLabels } from './canonical-group-labels'
import type { GroupActivityEntry } from './group-activity'
import { RoomActivityRow, RoomTypingRow } from './group-room-ui'
import type { GroupMember } from './types'

/** The member on turn, as the same in-progress message the gateway-owned room shows; the elapsed
 *  time counts from its latest `working` activity (ms). The legacy engine sees no tool frames. */
export function LegacyTypingRow({ avatar, name, since }: { avatar: ReactNode; name: string; since?: number }) {
  const labels = useCanonicalGroupLabels()

  return <RoomTypingRow avatar={avatar} labels={labels} name={name}
    status={{ kind: 'working', startedAt: typeof since === 'number' ? since / 1000 : undefined }} />
}

/** While the room runs: the member on turn as a typing row, else waiting-for-you / working. */
export function LegacyWorkingRow({ activity, avatar, clarifying, labels, nameOf, running, turn }: {
  activity: GroupActivityEntry[]; avatar: (turn: GroupMember) => ReactNode; clarifying: boolean
  labels: { roomWorking: string; waitingForAnswer: string }; nameOf: (turn: GroupMember) => string
  running: boolean; turn?: GroupMember | null
}) {
  if (!running) {return null}

  if (turn && !clarifying) {
    const since = [...activity].reverse().find(event => event.kind === 'working' && event.member === turn.name)?.at

    return <LegacyTypingRow avatar={avatar(turn)} name={nameOf(turn)} since={since} />
  }

  return <RoomActivityRow icon={clarifying ? 'question' : 'loading'}>{clarifying ? labels.waitingForAnswer : labels.roomWorking}</RoomActivityRow>
}

/** Room picture (set via Group settings) or the group glyph, the name, and the member count. */
export function LegacyRoomTitle({ availabilityLabel, countLabel, group, image, memberNames, short }: {
  availabilityLabel: string; countLabel: string; group: string; image?: null | string; memberNames: string; short: boolean
}) {
  return <span className="flex min-w-0 items-center gap-2">
    {image
      ? <img alt="" className="size-6 shrink-0 rounded-md object-cover ring-1 ring-(--ui-stroke-tertiary)" src={image} />
      : <span className="flex size-6 shrink-0 items-center justify-center rounded-md bg-(--ui-bg-quaternary) text-(--ui-text-tertiary)">
        <Codicon name="organization" />
      </span>}
    <span className="min-w-0 truncate">{group}</span>
    <Tip label={memberNames}>
      <span aria-label={availabilityLabel} className={cn('shrink-0 text-xs font-normal text-(--ui-text-tertiary)', short && 'text-amber-600 dark:text-amber-300')}>
        {short ? availabilityLabel : countLabel}
      </span>
    </Tip>
  </span>
}

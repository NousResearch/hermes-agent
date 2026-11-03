/**
 * Presentational pieces of the Desktop-driven (legacy) room view that share the gateway-owned
 * room's parts, kept out of `group-chat-view.tsx` so that file does not grow.
 */

import type { ReactNode } from 'react'

import { useNowSeconds } from './canonical-group-activity'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import { RoomTypingRow } from './group-room-ui'

/** The member on turn, as the same in-progress message the gateway-owned room shows; the elapsed
 *  time counts from its latest `working` activity (ms). The legacy engine sees no tool frames. */
export function LegacyTypingRow({ avatar, name, since }: { avatar: ReactNode; name: string; since?: number }) {
  const labels = useCanonicalGroupLabels()
  const now = useNowSeconds(true)
  const elapsedS = typeof since === 'number' ? Math.max(0, Math.floor(now - since / 1000)) : undefined

  return <RoomTypingRow avatar={avatar} labels={labels} name={name} status={{ kind: 'working', elapsedS }} />
}

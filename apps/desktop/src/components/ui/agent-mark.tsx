import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import { portalAgentSwatch } from '@/lib/portal-profile-color'
import { resolveProfileColor } from '@/lib/profile-color'
import { $profileColors } from '@/store/profile'
import { usePortalShell } from '@/store/shell-variant'

import { FigmaAgentIcon } from './figma-agent-icon'

interface AgentMarkProps {
  /** What classic draws for this agent (a plugin's own avatar). */
  fallback: ReactNode
  profile: string
  size?: number
  /** Data attributes for the tile alone, when the fallback carries its own (a plugin's row marker). */
  tileAttributes?: Record<`data-${string}`, string>
}

/** An agent wherever it is shown, not edited: the redesign draws every agent as its coloured tile,
 *  the one the chat sidebar and the Agents page use, so a surface with its own avatars still matches. */
export function AgentMark({ fallback, profile, size, tileAttributes }: AgentMarkProps) {
  const portal = usePortalShell()
  const colors = useStore($profileColors)

  if (!portal) {
    return fallback
  }

  const tile = <FigmaAgentIcon color={portalAgentSwatch(resolveProfileColor(profile, colors)).solid} size={size} />

  // A box-less wrapper, so the tile lays out exactly as it does without the attributes.
  return tileAttributes ? (
    <span {...tileAttributes} style={{ display: 'contents' }}>
      {tile}
    </span>
  ) : (
    tile
  )
}

import { atom, host } from '@hermes/plugin-sdk'
import { useEffect } from 'react'

import { $lastRoster } from './data'
import type { useRoster } from './data'
import { displayName } from './labels'
import { mergeServerMeta, pullServerAvatars } from './profile-ops'
import { trackInboundActivity } from './roster-actions'
import { botRosterMeta, botWorkspaceOwnerKey } from './routing'
import { backfillMessagingProtocol } from './soul'
import type { GatewaySource } from './types'
import type { BotMeta, RosterRow } from './types'

/** Last source inventory returned by the desktop-wide agent roster. */
export const $lastSources = atom<GatewaySource[]>([])

interface RosterSnapshotInput {
  data: ReturnType<typeof useRoster>['data']
  live: RosterRow[] | null
  roster: RosterRow[]
  allMeta: Record<string, BotMeta>
  activeSourceRoster: RosterRow[]
}

export function usePublishRosterSnapshot({ data, live, roster, allMeta, activeSourceRoster }: RosterSnapshotInput) {
  useEffect(() => {
    if (!live) {
      return
    }

    // Offline-owner ghosts belong only to this render. Shared roster state
    // feeds merge caching, group membership, creation, and durable sync. These
    // writes must settle after render: other subscribers of the same atoms
    // would otherwise be updated while BotsPane was still rendering.
    $lastRoster.set(roster.filter(row => !row?.ghost))
    // Tabs caption a bot chat by its bot (#99152); republished with the
    // roster so a rename follows and tiles restored at boot resolve.
    roster.forEach(bot => {
      host.setWorkspaceOwnerLabel?.(botWorkspaceOwnerKey(bot), displayName(bot, botRosterMeta(bot, allMeta)))
    })

    if (Array.isArray(data?.sources)) {
      $lastSources.set(data.sources)
    }

    mergeServerMeta(activeSourceRoster, data?.fetchedAt || 0)
    pullServerAvatars(activeSourceRoster)
    trackInboundActivity(roster)
    backfillMessagingProtocol(activeSourceRoster)
    // React Query owns the stable server snapshot; derived arrays intentionally
    // follow that snapshot rather than retriggering on their own atom writes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data])
}

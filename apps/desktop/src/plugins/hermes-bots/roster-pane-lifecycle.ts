import { atom } from '@hermes/plugin-sdk'
import { useEffect } from 'react'

import { $lastRoster, publishWorkspaceOwnerLabels } from './data'
import type { useRoster } from './data'
import { pullServerAvatars } from './profile-ops'
import { trackInboundActivity } from './roster-actions'
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
    // Every snapshot already labelled its bots' tabs (#99152) as it was
    // fetched, pane or not; this adds the render-only rows (an offline
    // selected owner's ghost) the snapshot itself does not carry.
    publishWorkspaceOwnerLabels(roster, allMeta, data?.fetchedAt)

    if (Array.isArray(data?.sources)) {
      $lastSources.set(data.sources)
    }

    // Names were already reconciled when the snapshot was fetched
    // (data.ts::fetchRosterSnapshot), for every reader, pane open or not.
    pullServerAvatars(activeSourceRoster)
    trackInboundActivity(roster)
    backfillMessagingProtocol(activeSourceRoster)
    // React Query owns the stable server snapshot; derived arrays intentionally
    // follow that snapshot rather than retriggering on their own atom writes.
    // Key on the `profiles`/`sources` subtrees, not the envelope: every 5 s
    // poll stamps a fresh `fetchedAt`, so the envelope is a new object each
    // tick while structural sharing keeps unchanged subtrees reference-stable.
    // Keying on the envelope republished an identical roster every poll —
    // every $lastRoster subscriber re-rendered and the avatar/meta/activity
    // side effects re-ran with nothing changed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [live, data?.sources])
}

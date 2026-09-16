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

/** Warm the backends behind a rendered roster so a bot row opens instantly
 *  instead of paying the cold spawn (~2.5-3s) on first click. Every warm goes
 *  through the SAME resolver as hover-warm (`host.warmProfile` →
 *  prewarmProfileBackend): the 60s per-profile throttle, the pool-saturation
 *  guard and failure-swallowing live there, so a roster sweep can never spawn
 *  more than the cap or churn hover-warmed backends. Only local-default rows
 *  are warmed: a bare name resolves the local route, so remote and
 *  connection-scoped rows are skipped rather than mis-warmed. The sweep is
 *  fire-and-forget: any warm failure is swallowed — the click path runs its
 *  own ensure and owns error UX. */
export function warmRosterBackends(
  roster: readonly (RosterRow | null | undefined)[],
  warm: (name: string) => void = name => host.warmProfile?.(name)
): void {
  for (const bot of roster) {
    const name = bot?.name

    if (!name || bot!.ghost || bot!.remoteSource || bot!.connectionId) {
      continue
    }

    try {
      warm(name)
    } catch {
      /* warm is best-effort */
    }
  }
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
    const liveRoster = roster.filter(row => !row?.ghost)
    $lastRoster.set(liveRoster)
    // With the pool sized to hold the roster, this sweep keeps every bot
    // backend warm across the idle-reap window — the re-hydrations re-arm the
    // per-profile throttle, so a warm backend that aged out gets respawned
    // before the user clicks, not after.
    warmRosterBackends(liveRoster)
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

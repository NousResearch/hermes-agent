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

    // A Bot tile can outlive an out-of-band profile retirement. Reconcile only
    // against this successful live answer (never the display roster, which may
    // carry outage ghosts), so a missing bot is discarded before it can wake a
    // backend and recreate its retired profile home. The answer's own issue time
    // rides along: absence in an answer that predates a tab says nothing about
    // that tab, so an undated answer (or one from a build that does not date
    // them) reconciles nothing at all.
    const issuedAt = data?.fetchedAt || 0

    if (issuedAt > 0) {
      host.reconcileBotWorkspaceRoster?.(live, Array.isArray(data?.sources) ? data.sources : [], issuedAt)
    }

    $lastRoster.set(roster.filter(row => !row?.ghost))
    // Tabs caption a bot chat by its bot (#99152); republished with the
    // roster so a rename follows and tiles restored at boot resolve.
    roster.forEach(bot => {
      host.setWorkspaceOwnerLabel?.(botWorkspaceOwnerKey(bot), displayName(bot, botRosterMeta(bot, allMeta)))
    })

    if (Array.isArray(data?.sources)) {
      $lastSources.set(data.sources)
    }

    // Every live row, not just the active source's: a bot on another
    // connection reports its own title too, and skipping it left that bot
    // named by whatever this Desktop last cached for it.
    mergeServerMeta(
      roster.filter(row => !row?.ghost),
      data?.fetchedAt || 0
    )
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

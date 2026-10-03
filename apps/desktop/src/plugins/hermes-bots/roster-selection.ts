/**
 * Which bot the roster has selected — kept out of the pane component so the
 * reconciliation is provable without a DOM: the pane renders it, the owner and
 * tile lifecycle read it, and the ghosts it paints are presentation only.
 *
 * PRESENTATION ONLY. Nothing here opens, prepares, activates, or creates
 * anything: an unreachable owner keeps its selection rather than falling back
 * onto some other gateway's bot, and a retired one is never redirected.
 */

import {
  $rosterHydrated,
  $rosterSelectionDeferred,
  $selectedRosterHydrated,
  $selectedRosterKey,
  deferRosterSelection,
  parseRosterKey,
  rosterSelectionEstablishedAt,
  seatRosterSelection
} from './bot-state'
import { annotateBotSource, botRosterKey, botSourceStatus, sourceByConnection } from './data'
import { isBotHidden } from './hidden-bots'
import type { BotMeta, GatewaySource, RosterRow } from './types'

export function selectedRosterBot(roster: RosterRow[], key: string): RosterRow | null {
  return (Array.isArray(roster) ? roster : []).find(bot => botRosterKey(bot) === key) || null
}

/** A selected owner whose roster row is absent because its SOURCE is down —
 *  not because the bot is gone. Identity comes from the key itself, so the
 *  selection survives a relaunch with that gateway offline and reconciles
 *  onto the live row (same key) when it returns, without duplicating it.
 *
 *  Returns null when the selection is provably invalid instead: a source that
 *  answered its OWN fresh list and no longer lists the bot, or a source that
 *  left the registry while other sources are live. Unknown (no sources yet) is
 *  NOT proof — and neither is a remembered list: `reachable` only means "we
 *  have a list", which an ssh cache and the undialed seed satisfy too, so those
 *  keep their selection until the source really answers. */
function ghostRosterOwner(key: string, sources: GatewaySource[]): RosterRow | null {
  const { connectionId, name } = parseRosterKey(key)

  if (!name) {
    return null
  }

  const list = Array.isArray(sources) ? sources : []
  const source = sourceByConnection(list).get(connectionId)

  if (source ? source.inventoryComplete === true : list.length > 0) {
    return null
  }

  return {
    name,
    connectionId,
    ghost: true,
    remoteSource: connectionId !== 'local',
    connectionKind: source?.kind,
    connectionLabel: source?.label,
    sourceError: source?.error || null,
    sourceMissing: false,
    sourceReachable: false
  }
}

/** Keep the exact selected owner visible through a cold-start outage without
 *  persisting the whole remote roster. The source registry supplies the
 *  gateway identity/status; the source-qualified selection supplies the bot
 *  identity. Once that source answers again, the live row replaces the ghost
 *  (or reconciliation clears it when the bot was actually removed). */
export function rosterWithSelectedOwner(roster: RosterRow[], sources: GatewaySource[], key: string): RosterRow[] {
  const rows = Array.isArray(roster) ? roster : []

  if (!key || selectedRosterBot(rows, key)) {
    return rows
  }

  const ghost = ghostRosterOwner(key, sources)

  return ghost ? [...rows, ghost] : rows
}

/** Keep the persisted selection honest against the live roster and seat a
 *  first selection when there is none. PRESENTATION ONLY: it never opens,
 *  prepares, activates, or creates anything — an unreachable owner keeps its
 *  selection rather than falling back onto some other gateway's bot.
 *
 *  A retired selection is DROPPED AND DEFERRED: the roster is left deliberately
 *  unselected so the user makes the choice. Seating the first survivor here
 *  would turn a retired bot-builder into whichever bot happens to sort first —
 *  and it would do so on the very next render, since the persisted key is gone
 *  by the time this runs again. The deferral is persisted for exactly that
 *  reason and only an explicit user choice ends it.
 *
 *  `fetchedAt` is the answer's ISSUE time, and it is the fence this path was
 *  missing: an answer sent before the user picked a bot never saw the pick, and
 *  an undated answer cannot be dated against the selection at all — neither may
 *  act on it. The tile path fences the same way on when a tab was opened. */
export function reconcileRosterSelection(
  roster: RosterRow[],
  sources: GatewaySource[],
  metaByName: Record<string, BotMeta>,
  fetchedAt: number | undefined
) {
  if (!$rosterHydrated.get() || !$selectedRosterHydrated.get()) {
    return
  }

  if (fetchedAt === undefined || !(fetchedAt > 0)) {
    return
  }

  const key = $selectedRosterKey.get()

  if (key) {
    // Freshness fence: this answer predates the choice, so it never saw it.
    if (rosterSelectionEstablishedAt() > fetchedAt) {
      return
    }

    if (selectedRosterBot(roster, key) || ghostRosterOwner(key, sources)) {
      return
    }

    deferRosterSelection(key)

    return
  }

  // The user has not chosen a successor yet: hold the empty selection.
  if ($rosterSelectionDeferred.get()) {
    return
  }

  const first = (Array.isArray(roster) ? roster : []).find(
    bot => !isBotHidden(bot, metaByName) && botSourceStatus(annotateBotSource(bot, sources)).available
  )

  if (first) {
    seatRosterSelection(first)
  }
}

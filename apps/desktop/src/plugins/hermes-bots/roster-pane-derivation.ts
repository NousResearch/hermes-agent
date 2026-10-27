import type { RosterSessionSignals } from '@hermes/plugin-sdk'

import { botActivitySession } from './data'
import { botRosterKey, botSelectionKey, filterBots, preferReachableSameNameRows } from './data'
import type { $groupChats } from './group-chat'
import { groupChatMemberBots, groupChatNames, groupLastActivity } from './group-membership'
import { sortGroupRosterRows } from './group-order'
import { isBotPinned } from './hidden-bots'
import { isBotHidden } from './hidden-bots'
import { filterBotsByGateway, groupMatchesRosterFilters, rosterGatewaySections } from './roster-sections'
import type { rosterGatewayOptions } from './roster-sections'
import { type BotStatus, deriveBotStatus, deriveGroupStatus } from './roster-status'
import { botRosterMeta } from './routing'
import { BOT_ROSTER_SEARCH_THRESHOLD } from './row-helpers'
import { ACTIVE_WINDOW_S, rosterActivityMatches } from './row-helpers'
import type { BotMeta, GroupMember, GroupPrompt, RosterActivityFilter, RosterKindFilter, RosterRow } from './types'
/** The two row shapes the roster sorts together — `kind` is the discriminant. */
interface RosterBotRow {
  active: boolean
  activity: number
  bot: RosterRow
  kind: 'bot'
  pinned: boolean
}
export interface RosterGroupRow {
  active: boolean
  activity: number
  kind: 'group'
  members: GroupMember[]
  name: string
  pinned: boolean
}

interface RosterRowsInput {
  roster: RosterRow[]
  allMeta: Record<string, BotMeta>
  gatewayFilter: string
  query: string
  activityFilter: RosterActivityFilter
  rowKindFilter: RosterKindFilter
  groupRooms: ReturnType<typeof $groupChats.get>
  activeRosterKeys: Set<string>
  gatewayOptions: ReturnType<typeof rosterGatewayOptions>
  activityOf: (bot: RosterRow) => number
  isPinned: (bot: RosterRow) => boolean
  /** Row status (see roster-status.ts); a lifted row leaves its section for
   *  the Needs you band. Absent = nothing lifts. */
  statusOfBot?: (bot: RosterRow) => BotStatus
  statusOfGroup?: (name: string) => BotStatus
}

/** One row in the Needs you band, in the same two shapes the sections use. */
export type NeedsYouRow = { bot: RosterRow; kind: 'bot' } | { kind: 'group'; members: GroupMember[]; name: string }

export function deriveRosterRows({
  roster,
  allMeta,
  gatewayFilter,
  query,
  activityFilter,
  rowKindFilter,
  groupRooms,
  activeRosterKeys,
  gatewayOptions,
  activityOf,
  isPinned,
  statusOfBot,
  statusOfGroup
}: RosterRowsInput) {
  const botLifts = (bot: RosterRow): boolean => Boolean(statusOfBot?.(bot).lifts)
  const groupLifts = (name: string): boolean => Boolean(statusOfGroup?.(name).lifts)
  const activeSourceRoster = roster.filter(bot => !bot.remoteSource)
  // Hidden rows remain fully alive and recoverable at the bottom. Every
  // non-display consumer continues to receive the complete roster.
  const hiddenBots = roster.filter(bot => isBotHidden(bot, allMeta))
  const visibleRoster = roster.filter(bot => !isBotHidden(bot, allMeta))
  const gatewayRoster = filterBotsByGateway(visibleRoster, gatewayFilter)

  const filteredRoster = filterBots(gatewayRoster, allMeta, query).filter((bot: RosterRow) =>
    rosterActivityMatches(
      {
        activity: activityOf(bot),
        active: activeRosterKeys.has(botRosterKey(bot))
      },
      activityFilter
    )
  )

  const matchedHiddenBots = filterBots(filterBotsByGateway(hiddenBots, gatewayFilter), allMeta, query).filter(
    (bot: RosterRow) =>
      rosterActivityMatches(
        {
          activity: activityOf(bot),
          active: activeRosterKeys.has(botRosterKey(bot))
        },
        activityFilter
      )
  )

  // Hidden rows lift only while the reveal toggle is on; statusOfBot already
  // folds that in, so a lifted hidden row simply leaves the Hidden list.
  const filteredHiddenBots = matchedHiddenBots.filter(bot => !botLifts(bot))
  const groupNames = groupChatNames(allMeta, groupRooms)

  const groupRows = groupNames
    .map(name => ({
      name,
      members: groupChatMemberBots(name, roster, allMeta)
    }))
    .filter(row => groupMatchesRosterFilters(row.name, row.members, allMeta, query, gatewayFilter))
    .map((row): RosterGroupRow => ({
      kind: 'group',
      name: row.name,
      members: row.members,
      pinned: Boolean(groupRooms[row.name]?.pinned),
      activity: groupLastActivity(groupRooms[row.name]),
      active:
        Boolean(
          groupLastActivity(groupRooms[row.name]) &&
          Date.now() - groupLastActivity(groupRooms[row.name]) <= ACTIVE_WINDOW_S * 1000
        ) || row.members.some(member => activeRosterKeys.has(botRosterKey(member)))
    }))
    .filter(row => rowKindFilter !== 'bots' && rosterActivityMatches(row, activityFilter))

  const botRows =
    rowKindFilter === 'groups'
      ? []
      : preferReachableSameNameRows(filteredRoster).map((bot): RosterBotRow => ({
          kind: 'bot',
          bot,
          pinned: isPinned(bot),
          activity: activityOf(bot),
          active: activeRosterKeys.has(botRosterKey(bot))
        }))

  // The band is a view over the SAME filtered rows: a row the search or a
  // filter hides does not resurface in it. Rooms first (a blocked room stalls
  // every member), then bots in roster order; nothing in the band drags.
  const needsYouRows: NeedsYouRow[] = [
    ...sortGroupRosterRows(groupRows, groupRooms)
      .filter(row => groupLifts(row.name))
      .map((row): NeedsYouRow => ({ kind: 'group', name: row.name, members: row.members })),
    ...[...botRows.map(row => row.bot), ...(rowKindFilter === 'groups' ? [] : matchedHiddenBots)]
      .filter(botLifts)
      .map((bot): NeedsYouRow => ({ kind: 'bot', bot }))
  ]

  const sectionBotRows = botRows.filter(row => !botLifts(row.bot))
  const sectionGroupRows = groupRows.filter(row => !groupLifts(row.name))
  const rosterRows = sortGroupRosterRows([...sectionBotRows, ...sectionGroupRows], groupRooms)
  const sortedGroupRows = sortGroupRosterRows(sectionGroupRows, groupRooms)
  const gatewaySections = rosterGatewaySections(sectionBotRows, gatewayOptions, gatewayFilter)
  const showGatewaySections = gatewaySections.sectioned && sectionBotRows.length > 0

  return {
    activeSourceRoster,
    hiddenBots,
    visibleRoster,
    filteredHiddenBots,
    groupNames,
    needsYouRows,
    rosterRows,
    sortedGroupRows,
    gatewaySections,
    showGatewaySections
  }
}

interface RosterPresentationInput {
  rowKindFilter: RosterKindFilter
  activityFilter: RosterActivityFilter
  gatewayFilter: string
  query: string
  filteredHiddenBots: RosterRow[]
  hiddenBots: RosterRow[]
  hiddenExpanded: boolean
  roster: RosterRow[]
  groupNames: string[]
  visibleRoster: RosterRow[]
  gatewayOptions: ReturnType<typeof rosterGatewayOptions>
}

export function deriveRosterPresentation({
  rowKindFilter,
  activityFilter,
  gatewayFilter,
  query,
  filteredHiddenBots,
  hiddenBots,
  hiddenExpanded,
  roster,
  groupNames,
  visibleRoster,
  gatewayOptions
}: RosterPresentationInput) {
  const activeFilterCount =
    (rowKindFilter === 'all' ? 0 : 1) + (activityFilter === 'all' ? 0 : 1) + (gatewayFilter === 'all' ? 0 : 1)

  const hasRosterConstraint = Boolean(query.trim()) || activeFilterCount > 0
  const matchingHiddenBots = rowKindFilter === 'groups' ? [] : filteredHiddenBots
  const showHiddenSection = hiddenBots.length > 0 && (!hasRosterConstraint || matchingHiddenBots.length > 0)
  const showHiddenRows = hiddenExpanded || hasRosterConstraint
  const rosterItemCount = roster.length + groupNames.length

  const allBotsHidden =
    !hasRosterConstraint && visibleRoster.length === 0 && groupNames.length === 0 && hiddenBots.length > 0

  const showRosterSearch =
    gatewayOptions.length > 1 || rosterItemCount >= BOT_ROSTER_SEARCH_THRESHOLD || Boolean(query.trim())

  const showRosterFilters =
    gatewayOptions.length > 1 ||
    groupNames.length > 0 ||
    rosterItemCount >= BOT_ROSTER_SEARCH_THRESHOLD ||
    activeFilterCount > 0

  const showRosterTools = showRosterSearch || showRosterFilters

  const hiddenGatewaySections = rosterGatewaySections(
    matchingHiddenBots.map((bot: RosterRow) => ({
      kind: 'bot',
      bot
    })),
    gatewayOptions,
    gatewayFilter
  )

  return {
    activeFilterCount,
    hasRosterConstraint,
    matchingHiddenBots,
    showHiddenSection,
    showHiddenRows,
    allBotsHidden,
    showRosterSearch,
    showRosterFilters,
    showRosterTools,
    hiddenGatewaySections
  }
}

export function sortRosterBots(sourceWithSelectedOwner: RosterRow[], allMeta: Record<string, BotMeta>) {
  // Messaging-app order: most recent activity first, where "activity" is
  // the newest of (bot created, last message in any of its sessions). A
  // freshly created bot tops the list until another bot gets a message.
  // No special slot for the primary bot — it competes on recency too.
  const activityOf = (bot: RosterRow): number => {
    const created = botRosterMeta(bot, allMeta)?.created || bot.ui_meta?.['hermes-bots']?.created || 0
    const lastMsg = (botActivitySession(bot)?.last_active || 0) * 1000

    return Math.max(created, lastMsg)
  }

  // Pin is a source-qualified Desktop preference, not gateway profile state.
  const isPinned = (bot: RosterRow): boolean => isBotPinned(bot, allMeta)

  const roster = sourceWithSelectedOwner.slice().sort((a, b) => {
    const pa = isPinned(a) ? 1 : 0
    const pb = isPinned(b) ? 1 : 0

    if (pa !== pb) {
      return pb - pa
    }

    return activityOf(b) - activityOf(a)
  })

  return { roster, activityOf, isPinned }
}

/** The signal snapshots one paint reads. Everything here already exists on
 *  the Desktop: no field needs a new RPC. */
export interface RosterStatusInput {
  /** `$rosterSessionSignals` from the SDK. */
  signals: RosterSessionSignals
  attention: Record<string, { message: string; reason: string }>
  /** Roster keys whose face reads working (focused turn, live worker, room turn). */
  workingKeys: ReadonlySet<string>
  activeConnectionId: string
  allMeta: Record<string, BotMeta>
  revealHidden: boolean
  groupClarify: Record<string, GroupPrompt>
  groupNeedsYou: Record<string, boolean>
  groupRooms: ReturnType<typeof $groupChats.get>
}

function promptText(prompt: GroupPrompt): string {
  if (prompt.kind === 'approval') {
    return String(prompt.command || prompt.question || '').trim()
  }

  const first = (prompt.questions || []).find(q => typeof q?.question === 'string' && q.question.trim())

  return first ? String(first.question).trim() : ''
}

/** Per-row status resolvers for one paint, memoized by key so the band and
 *  every row read the same answer. */
export function rosterStatusResolvers(input: RosterStatusInput) {
  const { signals, attention, workingKeys, activeConnectionId, allMeta, revealHidden } = input
  const bots = new Map<string, BotStatus>()
  const groups = new Map<string, BotStatus>()

  const routineFor = (bot: RosterRow): boolean => {
    const own = String(bot.connectionId || '').trim()
    // Cron rows from the active gateway may carry no connection id.
    const onActive = !own || own === 'local' || own === activeConnectionId

    return signals.routines.some(scope => {
      const split = scope.indexOf('::')
      const connection = scope.slice(0, split)

      return scope.slice(split + 2) === (bot.name || 'default') && (connection ? connection === own : onActive)
    })
  }

  const statusOfBot = (bot: RosterRow): BotStatus => {
    const key = botRosterKey(bot)
    const cached = bots.get(key)

    if (cached) {
      return cached
    }

    const ids = [bot.canonical_session?.resolved_id, bot.canonical_session?.id].filter(Boolean) as string[]
    const dot = ids.map(id => signals.dots[id]).find(Boolean) ?? null
    const prompt = ids.map(id => signals.prompts[id]).find(Boolean) ?? null

    const failure =
      attention[botSelectionKey(bot)] ||
      attention[key] ||
      attention[`${bot.connectionId || activeConnectionId}::${bot.name || 'default'}`] ||
      null

    const status = deriveBotStatus({
      dot,
      prompt,
      failure,
      routineRunning: routineFor(bot),
      working: workingKeys.has(key),
      stale: bot.sourceReachable === false || Boolean(bot.ghost),
      hidden: isBotHidden(bot, allMeta),
      revealHidden
    })

    bots.set(key, status)

    return status
  }

  const statusOfGroup = (name: string): BotStatus => {
    const cached = groups.get(name)

    if (cached) {
      return cached
    }

    const pending = Object.values(input.groupClarify).filter(entry => entry?.group === name)
    const first = pending.map(entry => ({ kind: entry.kind, text: promptText(entry) })).find(entry => entry.text)
    const room = input.groupRooms[name]
    const log = Array.isArray(room?.log) ? room.log : []
    const mention = [...log]
      .reverse()
      .find(entry => entry?.from?.kind === 'member' && /@user\b/i.test(entry.text || ''))

    const status = deriveGroupStatus({
      // A pending prompt with no quotable text still blocks the round.
      prompt: first ?? (pending.length ? { kind: pending[0].kind, text: '' } : null),
      mentioned: Boolean(input.groupNeedsYou[name]),
      mentionText: mention?.text,
      running: Boolean(room?.running)
    })

    groups.set(name, status)

    return status
  }

  return { statusOfBot, statusOfGroup }
}

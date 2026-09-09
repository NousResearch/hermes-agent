import { host, useI18n, useValue } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import { BotRow } from './bot-row'
import {
  $botChatFocused,
  $botsPaneVisible,
  $focusedBotOwner,
  $openBotChat,
  $rosterHydrated,
  $selectedRosterHydrated,
  $selectedRosterKey,
  clearSelectedRosterKey,
  focusedRosterOwner,
  parseRosterKey,
  saveSelectedRosterBot
} from './bot-state'
/**
 * The Bots pane itself: the roster's selection reconciliation, the
 * workspace-ownership reads its lifecycle keys off, and the pane that lists
 * every bot and group chat.
 *
 * The top of the roster stack. It composes the rows, the section headings and
 * the dialogs; nothing in Bot Mode imports it except the plugin entry point.
 */
import {
  $botMeta,
  $lastRoster,
  annotateBotSource,
  botRosterKey,
  botSourceStatus,
  sourceByConnection,
  useRoster
} from './data'
import { $groupChats, $groupChatWorkspace, $groupClarify, $groupNeedsYou } from './group-chat'
import { GroupChatWorkspace, openGroupChat } from './group-chat-view'
import { groupChatMemberBots } from './group-membership'
import { $groupMainTabsRev, shouldRenderGroupChatInPane } from './group-panes'
import { $showHiddenBots, isBotHidden } from './hidden-bots'
import { useBots } from './i18n'
import { $activityToasts } from './roster-actions'
import { renderRosterContent } from './roster-pane-content'
import { deriveRosterPresentation, deriveRosterRows, sortRosterBots } from './roster-pane-derivation'
import { renderRosterDialogs } from './roster-pane-dialogs'
import { RosterGroupRowView } from './roster-pane-groups'
import { $lastSources, usePublishRosterSnapshot } from './roster-pane-lifecycle'
import { rosterSectionRenderers } from './roster-pane-sections'
import { renderRosterToolbar } from './roster-pane-toolbar'
import { botNeedsHandleLabel, rosterGatewayOptions } from './roster-sections'
import { botWorkspaceOwnerKey, setBotsWorkspaceOwner } from './routing'
import { activeBots, useTurnBusy } from './row-helpers'
import type { BotMeta, GatewaySource, GroupMember, RosterActivityFilter, RosterKindFilter, RosterRow } from './types'
import { $botSections, $draggingBot } from './user-sections'
import { useEscapeCancelsBotDrag } from './user-sections-ui'

// ── roster pane ──────────────────────────────────────────────────────────────

export function selectedRosterBot(roster: RosterRow[], key: string): RosterRow | null {
  return (Array.isArray(roster) ? roster : []).find(bot => botRosterKey(bot) === key) || null
}

/** A selected owner whose roster row is absent because its SOURCE is down —
 *  not because the bot is gone. Identity comes from the key itself, so the
 *  selection survives a relaunch with that gateway offline and reconciles
 *  onto the live row (same key) when it returns, without duplicating it.
 *
 *  Returns null when the selection is provably invalid instead: a reachable
 *  source that no longer lists the bot, or a source that left the registry
 *  while other sources are live. Unknown (no sources yet) is NOT proof. */
function ghostRosterOwner(key: string, sources: GatewaySource[]): RosterRow | null {
  const { connectionId, name } = parseRosterKey(key)

  if (!name) {
    return null
  }

  const list = Array.isArray(sources) ? sources : []
  const source = sourceByConnection(list).get(connectionId)

  if (source ? source.reachable === true : list.length > 0) {
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
function rosterWithSelectedOwner(roster: RosterRow[], sources: GatewaySource[], key: string): RosterRow[] {
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
 *  selection rather than falling back onto some other gateway's bot. */
function reconcileRosterSelection(roster: RosterRow[], sources: GatewaySource[], metaByName: Record<string, BotMeta>) {
  if (!$rosterHydrated.get() || !$selectedRosterHydrated.get()) {
    return
  }

  const key = $selectedRosterKey.get()

  if (key) {
    if (selectedRosterBot(roster, key) || ghostRosterOwner(key, sources)) {
      return
    }

    clearSelectedRosterKey(key)
  }

  const first = (Array.isArray(roster) ? roster : []).find(
    bot => !isBotHidden(bot, metaByName) && botSourceStatus(annotateBotSource(bot, sources)).available
  )

  if (first) {
    saveSelectedRosterBot(first)
  }
}

/** True when a session owns the main workspace. Prefers the focused STORED
 *  session (tab focus moves without swapping the gateway socket); bare test
 *  harnesses with neither atom drive $botChatFocused directly. */
export function sessionOwnsWorkspace(): boolean {
  const focused = host.state?.focusedStoredSessionId?.get?.()

  if (focused !== undefined) {
    return Boolean(focused)
  }

  const active = host.state?.activeSessionId?.get?.()

  return active === undefined ? $botChatFocused.get() : Boolean(active)
}

/** A real bot chat owns the center. Cronjobs are BOT-scoped, so this — not
 *  mere Bot Mode visibility — is what may seat the Cronjobs tile: beside a
 *  group chat it would describe whichever profile the socket happens to be
 *  homed on. */
export function botChatOwnsWorkspace(): boolean {
  return $botsPaneVisible.get() && !$groupChatWorkspace.get() && Boolean($openBotChat.get() || sessionOwnsWorkspace())
}

/** An opened bot chat stops owning the center once focus leaves it (closed,
 *  or another session took over). Without this $openBotChat would keep
 *  claiming ownership for a chat nobody is reading, and the bot-scoped
 *  Cronjobs tile would stay seated beside an unrelated surface.
 *
 *  The legacy newChat fallback has no registry id to compare — a draft with no
 *  focused session is still that bot's draft, so it only yields once some
 *  session actually takes focus. */
export function releaseStaleOpenBotChat(focusedStoredId: null | string | undefined): void {
  const open = $openBotChat.get()

  if (!open) {
    return
  }

  const focused = focusedStoredId === null || focusedStoredId === undefined ? '' : String(focusedStoredId)
  // The focused stored id is the compression-lineage TIP; the claim carries
  // both the durable registry id and the tip it actually opened. Either
  // match keeps the claim — comparing only the registry id released it on
  // the very focus edge the open itself caused.
  const owned = [open.openedSessionId, open.openedRegistryId].filter(Boolean)
  const stale = owned.length ? !owned.includes(focused) : Boolean(focused)

  if (stale) {
    $openBotChat.set(null)
  }
}

function useReconcileRosterOwner(
  data: ReturnType<typeof useRoster>['data'],
  error: ReturnType<typeof useRoster>['error'],
  selectionHydrated: boolean,
  roster: RosterRow[],
  sourceSnapshot: GatewaySource[],
  allMeta: Record<string, BotMeta>
) {
  // The roster has ANSWERED once data or a terminal error exists — that, not
  // row count, is what lets this pane stop showing its loading state (an empty
  // answer is a real answer; a pending one must not flash "No bots"). Keep the
  // persisted-selection writes out of render: React may replay a render, but
  // an abandoned render must never become a storage mutation.
  useEffect(() => {
    if (!data && !error) {
      return
    }

    $rosterHydrated.set(true)

    if (selectionHydrated) {
      reconcileRosterSelection(roster, sourceSnapshot, allMeta)
      const selected = selectedRosterBot(roster, $selectedRosterKey.get())

      if ($botsPaneVisible.get() && !$groupChatWorkspace.get() && selected) {
        setBotsWorkspaceOwner(botWorkspaceOwnerKey(selected), selected)
      }
    }
  }, [data, error, selectionHydrated, roster, sourceSnapshot, allMeta])
}

export function BotsPane() {
  const { t } = useI18n()
  const b = useBots()
  const { data, error, isLoading, refetch } = useRoster()
  const gatewayState = useValue(host.state.gateway)
  const gatewayUp = gatewayState === 'open'

  const turnBusy = useTurnBusy()
  const workingOwner = focusedRosterOwner(useValue($focusedBotOwner))
  const activeConnectionId = host.state.connectionId?.get?.() || 'local'
  const [createOpen, setCreateOpen] = useState(false)
  const [groupCreateOpen, setGroupCreateOpen] = useState(false)
  const [editing, setEditing] = useState<null | RosterRow>(null)
  // `path` is the profile directory the gateway reports on a profiles.list row;
  // it is not part of the shared RosterRow model, so it rides as an extra here.
  const [deleting, setDeleting] = useState<null | (RosterRow & { path?: string })>(null)
  const [deletingGroup, setDeletingGroup] = useState<null | { members: GroupMember[]; name: string }>(null)
  const userSections = useValue($botSections)
  const dragging = useValue($draggingBot)
  useEscapeCancelsBotDrag()

  // The one name dialog serves both New section (optionally filing the bot
  // whose menu opened it) and Rename.
  const [sectionDialog, setSectionDialog] = useState<
    null | { bot?: RosterRow; mode: 'create' } | { id: string; mode: 'rename'; name: string }
  >(null)

  const [grouping, setGrouping] = useState<null | RosterRow>(null)
  const [query, setQuery] = useState('')
  const [rowKindFilter, setRowKindFilter] = useState<RosterKindFilter>('all')
  const [activityFilter, setActivityFilter] = useState<RosterActivityFilter>('all')
  const [gatewayFilter, setGatewayFilter] = useState('all')
  const [collapsedRosterSections, setCollapsedRosterSections] = useState<Set<string>>(() => new Set())
  const hiddenSectionRef = useRef<null | HTMLDivElement>(null)
  const activityToasts = useValue($activityToasts)
  const groupChatName = useValue($groupChatWorkspace)
  // Main-tab ownership is a module Map; this rev subscription makes the
  // shouldRenderGroupChatInPane gate below reactive to tab open/close
  // (#89788 follow-up — without it a stale render could paint the in-pane
  // room beside a live main tab and stick).
  useValue($groupMainTabsRev)
  const groupNeedsYou = useValue($groupNeedsYou)
  const groupClarify = useValue($groupClarify)
  const groupRooms = useValue($groupChats)
  const rememberedSources = useValue($lastSources)
  const rosterHydrated = useValue($rosterHydrated)
  const selectionHydrated = useValue($selectedRosterHydrated)
  const selectedRosterKey = useValue($selectedRosterKey)

  // The socket opening (boot, SSH reconnect, sleep/wake) is the signal to
  // retry immediately instead of waiting out the poll interval.
  useEffect(() => {
    if (gatewayUp) {
      void refetch()
    }
  }, [gatewayUp, refetch])
  const allMeta = useValue($botMeta)

  // Resilience (@wesleysimplicio, #13): a failed refresh must not erase a
  // roster the user already had — mixed local+cloud gateways and remotes
  // waking from sleep fail transiently. Render the last good snapshot with
  // a notice; the full error card is reserved for "never had a roster".
  const live = Array.isArray(data?.profiles) ? data.profiles : null
  const source = live ?? (error ? $lastRoster.get() : [])
  const sourceSnapshot = Array.isArray(data?.sources) ? data.sources : rememberedSources

  const sourceWithSelectedOwner =
    selectionHydrated && rosterHydrated ? rosterWithSelectedOwner(source, sourceSnapshot, selectedRosterKey) : source

  const { roster, activityOf, isPinned } = sortRosterBots(sourceWithSelectedOwner, allMeta)

  // React Query can briefly report neither loading nor data while the plugin
  // and the persisted connection registry hydrate. Keep that transition in a
  // neutral loading state instead of flashing the first-run "No bots" copy.
  const initialRosterLoading = !data && !error && roster.length === 0

  const activeRosterKeys = new Set(
    activeBots(roster, workingOwner, turnBusy, Date.now(), activeConnectionId).map(botRosterKey)
  )

  const gatewayOptions = rosterGatewayOptions(sourceSnapshot, roster)
  const selectedGateway = gatewayOptions.find(option => option.connectionId === gatewayFilter)
  const gatewayFilterExists = gatewayFilter === 'all' || Boolean(selectedGateway)
  useEffect(() => {
    if (!gatewayFilterExists) {
      setGatewayFilter('all')
    }
  }, [gatewayFilterExists])
  const hiddenExpanded = useValue($showHiddenBots)

  const {
    activeSourceRoster,
    hiddenBots,
    visibleRoster,
    filteredHiddenBots,
    groupNames,
    rosterRows,
    sortedGroupRows,
    gatewaySections,
    showGatewaySections
  } = deriveRosterRows({
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
    isPinned
  })

  const {
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
  } = deriveRosterPresentation({
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
  })

  const rosterSectionCollapsed = (id: string): boolean => !hasRosterConstraint && collapsedRosterSections.has(id)

  const toggleRosterSection = (id: string): void => {
    setCollapsedRosterSections(previous => {
      const next = new Set(previous)

      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }

      return next
    })
  }

  useEffect(() => {
    if (!hiddenExpanded || hasRosterConstraint) {
      return
    }

    const frame = requestAnimationFrame(() =>
      hiddenSectionRef.current?.scrollIntoView({
        block: 'nearest'
      })
    )

    return () => cancelAnimationFrame(frame)
  }, [hiddenExpanded, hasRosterConstraint])
  usePublishRosterSnapshot({ data, live, roster, allMeta, activeSourceRoster })

  useReconcileRosterOwner(data, error, selectionHydrated, roster, sourceSnapshot, allMeta)

  const staleNotice =
    error && !live && roster.length
      ? 'Roster refresh failed — showing the last good list.' +
        (gatewayUp ? '' : ' Waiting for the gateway to reconnect…')
      : null

  const groupChatMembers = groupChatName ? groupChatMemberBots(groupChatName, roster, allMeta) : []

  if (shouldRenderGroupChatInPane(groupChatName) && groupChatMembers.length) {
    return <GroupChatWorkspace group={groupChatName} members={groupChatMembers} />
  }

  const renderBotRow = (bot: RosterRow, keyPrefix = '') => (
    <BotRow
      bot={bot}
      key={`${keyPrefix}${botRosterKey(bot)}`}
      onDelete={setDeleting}
      onEdit={setEditing}
      onGroup={setGrouping}
      onNewSection={target => setSectionDialog({ bot: target, mode: 'create' })}
      showHandle={botNeedsHandleLabel(bot, roster, allMeta)}
    />
  )

  const renderGroupRow = (row: { members: GroupMember[]; name: string }) => (
    <RosterGroupRowView
      active={groupChatName === row.name}
      b={b}
      group={row.name}
      groupClarify={groupClarify}
      groupNeedsYou={groupNeedsYou}
      groupRooms={groupRooms}
      key={`group:${row.name}`}
      members={row.members}
      onDisband={setDeletingGroup}
      onOpen={openGroupChat}
      sortedGroupRows={sortedGroupRows}
    />
  )

  const { renderUserSections, renderGatewaySection, renderGroupChatSection, renderHiddenGatewaySection } =
    rosterSectionRenderers({
      b,
      userSections,
      roster,
      allMeta,
      dragging,
      rosterSectionCollapsed,
      toggleRosterSection,
      setSectionDialog,
      renderBotRow,
      renderGroupRow,
      sortedGroupRows
    })

  return (
    <div className="flex h-full flex-col">
      {renderRosterToolbar({
        b,
        activityToasts,
        activeSourceRoster,
        setCreateOpen,
        setGroupCreateOpen,
        setSectionDialog,
        showRosterTools,
        showRosterSearch,
        showRosterFilters,
        query,
        setQuery,
        activeFilterCount,
        gatewayOptions,
        rowKindFilter,
        setRowKindFilter,
        activityFilter,
        setActivityFilter,
        gatewayFilter,
        setGatewayFilter
      })}
      {renderRosterContent({
        b,
        staleNotice,
        isLoading,
        initialRosterLoading,
        roster,
        error,
        gatewayUp,
        refetch,
        allBotsHidden,
        hiddenExpanded,
        rosterRows,
        matchingHiddenBots,
        query,
        selectedGateway,
        showGatewaySections,
        sortedGroupRows,
        gatewaySections,
        showHiddenSection,
        hiddenSectionRef,
        hasRosterConstraint,
        hiddenBots,
        showHiddenRows,
        hiddenGatewaySections,
        renderBotRow,
        renderGroupChatSection,
        renderGatewaySection,
        renderUserSections,
        renderHiddenGatewaySection
      })}
      {renderRosterDialogs({
        b,
        t,
        createOpen,
        setCreateOpen,
        groupCreateOpen,
        setGroupCreateOpen,
        editing,
        setEditing,
        deleting,
        setDeleting,
        deletingGroup,
        setDeletingGroup,
        grouping,
        setGrouping,
        sectionDialog,
        setSectionDialog,
        roster,
        activeSourceRoster,
        refetch
      })}
    </div>
  )
}

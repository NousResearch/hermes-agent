/**
 * The Bots pane itself: the roster's selection reconciliation, the
 * workspace-ownership reads its lifecycle keys off, and the pane that lists
 * every bot and group chat.
 *
 * The top of the roster stack. It composes the rows, the section headings and
 * the dialogs; nothing in Bot Mode imports it except the plugin entry point.
 */

import {
  atom,
  Button,
  cn,
  Codicon,
  ConfirmDialog,
  DisclosureCaret,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  GlyphSpinner,
  host,
  PanelEmpty,
  RowButton,
  SearchField,
  Tip,
  useI18n,
  useValue
} from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import { BotRow, GroupRow } from './bot-row'
import {
  $botChatFocused,
  $botsPaneVisible,
  $openBotChat,
  $rosterHydrated,
  $selectedRosterHydrated,
  $selectedRosterKey,
  clearSelectedRosterKey,
  parseRosterKey,
  saveSelectedRosterBot
} from './bot-state'
import { CreateAgentDialog, CreateGroupChatDialog, GroupDialog } from './create-dialog'
import {
  $botMeta,
  $lastRoster,
  annotateBotSource,
  botActivitySession,
  botRosterKey,
  botSourceStatus,
  filterBots,
  preferReachableSameNameRows,
  sourceByConnection,
  useRoster
} from './data'
import { EditProfileDialog } from './edit-profile-dialog'
import { $groupChats, $groupChatWorkspace, $groupHostedNeedsYou, $groupNeedsYou } from './group-chat'
import { disbandGroupChat, GroupChatWorkspace, openGroupChat } from './group-chat-view'
import { groupChatMemberBots, groupChatNames, groupLastActivity } from './group-membership'
import { $groupMainTabsRev, shouldRenderGroupChatInPane } from './group-panes'
import { $showHiddenBots, isBotHidden, isBotPinned } from './hidden-bots'
import { useBots } from './i18n'
import { displayName } from './labels'
import { deleteBot, mergeServerMeta, pullServerAvatars } from './profile-ops'
import { $activityToasts, setActivityToasts, trackInboundActivity } from './roster-actions'
import {
  botNeedsHandleLabel,
  filterBotsByGateway,
  GatewayKindGlyph,
  GatewaySectionHeading,
  groupMatchesRosterFilters,
  rosterGatewayOptions,
  rosterGatewaySections,
  RosterSectionHeader
} from './roster-sections'
import type { ResolvedRosterGatewaySection } from './roster-sections'
import { botRosterMeta, botWorkspaceOwnerKey, setBotsWorkspaceOwner } from './routing'
import { ACTIVE_WINDOW_S, activeBots, BOT_ROSTER_SEARCH_THRESHOLD, rosterActivityMatches } from './row-helpers'
import { backfillMessagingProtocol } from './soul'
import type { BotMeta, GatewaySource, GroupMember, RosterActivityFilter, RosterKindFilter, RosterRow } from './types'
import {
  $botSections,
  $draggingBot,
  createBotSection,
  deleteBotSection,
  groupRowsBySection,
  moveBotSection,
  moveBotsToSection,
  renameBotSection,
  UNASSIGNED_SECTION_KEY
} from './user-sections'
import { SectionDropZone, SectionNameDialog, useEscapeCancelsBotDrag, UserSectionHeader } from './user-sections-ui'

/** Last source inventory returned by the desktop-wide agent roster. */
const $lastSources = atom<GatewaySource[]>([])

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

/** The two row shapes the roster sorts together — `kind` is the discriminant. */
interface RosterBotRow {
  active: boolean
  activity: number
  bot: RosterRow
  kind: 'bot'
  pinned: boolean
}
interface RosterGroupRow {
  active: boolean
  activity: number
  kind: 'group'
  members: GroupMember[]
  name: string
  pinned: boolean
}

export function BotsPane() {
  const { t } = useI18n()
  const b = useBots()
  const { data, error, isLoading, refetch } = useRoster()
  const gatewayState = useValue(host.state.gateway)
  const gatewayUp = gatewayState === 'open'
  const activeProfile = (useValue(host.state.profile) || 'default').trim() || 'default'
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
  const groupHostedNeedsYou = useValue($groupHostedNeedsYou)
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
  // Resilience (@wesleysimplicio, #13): a failed refresh must not erase a
  // roster the user already had — mixed local+cloud gateways and remotes
  // waking from sleep fail transiently. Render the last good snapshot with
  // a notice; the full error card is reserved for "never had a roster".
  const live = Array.isArray(data?.profiles) ? data.profiles : null
  const source = live ?? (error ? $lastRoster.get() : [])
  const sourceSnapshot = Array.isArray(data?.sources) ? data.sources : rememberedSources

  const sourceWithSelectedOwner =
    selectionHydrated && rosterHydrated ? rosterWithSelectedOwner(source, sourceSnapshot, selectedRosterKey) : source

  const roster = sourceWithSelectedOwner.slice().sort((a, b) => {
    const pa = isPinned(a) ? 1 : 0
    const pb = isPinned(b) ? 1 : 0

    if (pa !== pb) {
      return pb - pa
    }

    return activityOf(b) - activityOf(a)
  })

  // React Query can briefly report neither loading nor data while the plugin
  // and the persisted connection registry hydrate. Keep that transition in a
  // neutral loading state instead of flashing the first-run "No bots" copy.
  const initialRosterLoading = !data && !error && roster.length === 0
  const activeRosterKeys = new Set(activeBots(roster, activeProfile, gatewayState).map(botRosterKey))
  const gatewayOptions = rosterGatewayOptions(sourceSnapshot, roster)
  const selectedGateway = gatewayOptions.find(option => option.connectionId === gatewayFilter)
  const gatewayFilterExists = gatewayFilter === 'all' || Boolean(selectedGateway)
  useEffect(() => {
    if (!gatewayFilterExists) {
      setGatewayFilter('all')
    }
  }, [gatewayFilterExists])
  const activeSourceRoster = roster.filter(bot => !bot.remoteSource)
  // Hidden rows remain fully alive and recoverable at the bottom. Every
  // non-display consumer continues to receive the complete roster.
  const hiddenExpanded = useValue($showHiddenBots)
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

  const filteredHiddenBots = filterBots(filterBotsByGateway(hiddenBots, gatewayFilter), allMeta, query).filter(
    (bot: RosterRow) =>
      rosterActivityMatches(
        {
          activity: activityOf(bot),
          active: activeRosterKeys.has(botRosterKey(bot))
        },
        activityFilter
      )
  )

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

  const sortRosterRows = <T extends { activity: number; pinned: boolean }>(rows: T[]): T[] =>
    rows.slice().sort((a, b) => {
      const pa = a.pinned ? 1 : 0
      const pb = b.pinned ? 1 : 0

      if (pa !== pb) {
        return pb - pa
      }

      return b.activity - a.activity
    })

  const rosterRows = sortRosterRows([...botRows, ...groupRows])
  const sortedGroupRows = sortRosterRows(groupRows)
  const gatewaySections = rosterGatewaySections(botRows, gatewayOptions, gatewayFilter)
  const showGatewaySections = gatewaySections.sectioned && botRows.length > 0

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
  const rosterSectionCollapsed = (id: string): boolean => !hasRosterConstraint && collapsedRosterSections.has(id)

  const hiddenGatewaySections = rosterGatewaySections(
    matchingHiddenBots.map((bot: RosterRow) => ({
      kind: 'bot',
      bot
    })),
    gatewayOptions,
    gatewayFilter
  )

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
    <GroupRow
      active={groupChatName === row.name}
      group={row.name}
      key={`group:${row.name}`}
      members={row.members}
      needsYou={Boolean(groupNeedsYou[row.name] || groupHostedNeedsYou[row.name])}
      onDisband={setDeletingGroup}
      onOpen={openGroupChat}
    />
  )

  const removeSection = (id: string) => {
    const name = userSections.find(section => section.id === id)?.name || ''
    const { members, undo } = deleteBotSection(id, roster)

    // No confirmation: nothing is lost (the bots fall back to Unassigned) and
    // the toast's Undo puts the section and its members back.
    host.notify({
      action: { label: b.sections.undo, onClick: undo },
      durationMs: 8_000,
      kind: 'info',
      message: b.sections.deleted(name, members.length)
    })
  }

  // USER SECTIONS — composed with the gateway sections, not instead of them.
  // The gateway headings own the top level whenever the roster shows more
  // than one connection (that axis answers "where does this run", which no
  // folder name can, and a bot's membership lives in its profile on THAT
  // gateway); user sections group the rows INSIDE each connection bucket,
  // indented under it, and group the flat list when there is only one.
  // `keyPrefix` keeps row keys unique across the gateway buckets.
  type UserSectionRow = { bot: RosterRow; kind?: 'bot' } | RosterGroupRow

  const renderUserSections = (rows: UserSectionRow[], keyPrefix = '') => {
    // No sections made: the plain list, exactly as before this feature.
    if (!userSections.length) {
      return rows.map(row => (row.kind === 'group' ? renderGroupRow(row) : renderBotRow(row.bot, keyPrefix)))
    }

    const nested = Boolean(keyPrefix)
    const blocks = groupRowsBySection(rows, userSections, allMeta)

    return (
      blocks
        // An empty Unassigned is not worth a heading; an empty NAMED section
        // is, because it is somewhere the user made and is about to drop into.
        // Inside a gateway bucket the same empty section would repeat under
        // every connection, so there it only appears while a drag is in flight
        // (as the drop target it exists for); the row menu files into it
        // regardless.
        .filter(block => block.rows.length || (block.id && (!nested || dragging)))
        .map(block => {
          const key = `${keyPrefix}${block.id ? `user-section:${block.id}` : UNASSIGNED_SECTION_KEY}`
          const collapsed = rosterSectionCollapsed(key)
          const order = userSections.findIndex(section => section.id === block.id)

          return (
            <SectionDropZone
              isSource={
                Boolean(dragging) && block.rows.some(row => row.kind !== 'group' && botRosterKey(row.bot) === dragging)
              }
              key={key}
              nested={nested}
              onDropBot={rosterKey => {
                const bot = roster.find(row => botRosterKey(row) === rosterKey)

                // `block.id` is null for Unassigned, which is exactly the value
                // moveBotsToSection wants for "clear the assignment".
                if (bot) {
                  void moveBotsToSection([bot], block.id)
                }
              }}
            >
              <UserSectionHeader
                canMoveDown={order >= 0 && order < userSections.length - 1}
                canMoveUp={order > 0}
                collapsed={collapsed}
                count={block.rows.length}
                id={block.id}
                name={block.name}
                onDelete={() => block.id && removeSection(block.id)}
                onMove={delta => block.id && moveBotSection(block.id, delta)}
                onRename={() => block.id && setSectionDialog({ id: block.id, mode: 'rename', name: block.name })}
                onToggle={() => toggleRosterSection(key)}
              />
              {collapsed ? null : block.rows.length ? (
                <div className="grid min-w-0 gap-0.5">
                  {block.rows.map(row =>
                    row.kind === 'group' ? renderGroupRow(row) : renderBotRow(row.bot, `${key}:`)
                  )}
                </div>
              ) : (
                // Empty section: a quiet dashed slot that says what it is for,
                // and doubles as a roomy drop target.
                <div className="mx-1 mb-1 rounded-md border border-dashed border-(--ui-stroke-secondary) px-2 py-2 text-center text-[0.6875rem] text-(--ui-text-quaternary)">
                  {b.sections.emptyHint}
                </div>
              )}
            </SectionDropZone>
          )
        })
    )
  }

  const renderGatewaySection = (section: ResolvedRosterGatewaySection) => {
    const sectionId = `gateway:${section.id}`
    const collapsed = rosterSectionCollapsed(sectionId)

    return (
      <div className="min-w-0" key={sectionId}>
        <GatewaySectionHeading
          collapsed={collapsed}
          count={section.rows.length}
          onToggle={() => toggleRosterSection(sectionId)}
          option={section.option}
        />
        {collapsed ? null : (
          <div className="grid min-w-0 gap-0.5">{renderUserSections(section.rows, `${section.id}:`)}</div>
        )}
      </div>
    )
  }

  const renderGroupChatSection = () => {
    const sectionId = 'group-chats'
    const collapsed = rosterSectionCollapsed(sectionId)

    return (
      <div className="min-w-0" key={sectionId}>
        <RosterSectionHeader
          collapsed={collapsed}
          count={sortedGroupRows.length}
          icon="organization"
          label={b.roster.groupChats}
          onToggle={() => toggleRosterSection(sectionId)}
          tip={`${sortedGroupRows.length} global group chat${sortedGroupRows.length === 1 ? '' : 's'}`}
        />
        {collapsed ? null : <div className="grid min-w-0 gap-0.5">{sortedGroupRows.map(renderGroupRow)}</div>}
      </div>
    )
  }

  const renderHiddenGatewaySection = (section: ResolvedRosterGatewaySection) => (
    <div className="min-w-0" key={`hidden-gateway:${section.id}`}>
      <div className="flex min-w-0 items-center gap-1.5 px-2 py-1 text-[0.625rem] font-semibold uppercase tracking-wider text-(--ui-text-quaternary)">
        <GatewayKindGlyph kind={section.option?.kind} />
        <span className="min-w-0 flex-1 truncate">
          {section.option?.label || section.option?.connectionId || 'Current gateway'}
        </span>
        <span className="shrink-0 font-normal tabular-nums">{section.rows.length}</span>
      </div>
      {section.rows.map(row => renderBotRow(row.bot, `hidden:${section.id}:`))}
    </div>
  )

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between gap-2 px-2.5 pt-2.5 pb-1.5">
        <span className="text-[0.6875rem] font-semibold uppercase tracking-wider text-(--ui-text-quaternary)">
          Bots
        </span>
        <div className="flex items-center gap-0.5">
          <Tip
            label={activityToasts ? 'Activity toasts on — click to silence' : 'Activity toasts off — click to enable'}
          >
            <Button
              className="rounded-md text-(--ui-text-tertiary) hover:text-foreground"
              onClick={() => setActivityToasts(!activityToasts)}
              size="icon-xs"
              variant="ghost"
            >
              <Codicon name={activityToasts ? 'bell' : 'bell-slash'} />
            </Button>
          </Tip>
          <DropdownMenu>
            <Tip label="New…">
              <DropdownMenuTrigger asChild>
                <Button
                  aria-label={b.roster.newBotOrGroup}
                  className="rounded-md text-(--ui-text-tertiary) hover:text-foreground"
                  size="icon-xs"
                  variant="ghost"
                >
                  <Codicon name="add" />
                </Button>
              </DropdownMenuTrigger>
            </Tip>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onSelect={() => setCreateOpen(true)}>
                <Codicon className="mr-1.5" name="hubot" />
                {b.bot.newTitle}
              </DropdownMenuItem>
              <DropdownMenuItem disabled={activeSourceRoster.length < 2} onSelect={() => setGroupCreateOpen(true)}>
                <Codicon className="mr-1.5" name="organization" />
                {b.group.newTitle}
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onSelect={() => setSectionDialog({ mode: 'create' })}>
                <Codicon className="mr-1.5" name="new-folder" />
                {b.sections.newSection}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
      {showRosterTools ? (
        <div className="flex min-w-0 items-center gap-1 px-2.5 pb-1.5">
          {showRosterSearch ? (
            <SearchField
              aria-label={b.roster.search}
              containerClassName={cn('min-w-0 flex-1', query ? 'opacity-100!' : 'opacity-50 focus-within:opacity-100')}
              inputClassName="w-full text-[0.75rem] placeholder:text-(--ui-text-tertiary)"
              key={'roster-search'}
              onChange={setQuery}
              placeholder={b.roster.searchPlaceholder}
              value={query}
            />
          ) : (
            <span className="min-w-0 flex-1" key={'roster-search-spacer'} />
          )}
          {showRosterFilters ? (
            <DropdownMenu key={'roster-filters'}>
              <Tip label={activeFilterCount ? `Filters (${activeFilterCount} active)` : 'Filter roster'}>
                <DropdownMenuTrigger asChild>
                  <Button
                    aria-label={activeFilterCount ? `Filter roster, ${activeFilterCount} active` : 'Filter roster'}
                    className={cn(
                      'size-7 shrink-0 rounded-md text-(--ui-text-tertiary) hover:text-foreground',
                      activeFilterCount && 'text-(--ui-accent)'
                    )}
                    size="icon-xs"
                    variant="ghost"
                  >
                    <Codicon name="list-filter" />
                  </Button>
                </DropdownMenuTrigger>
              </Tip>
              <DropdownMenuContent align="end">
                {(
                  [
                    ['all', b.roster.botsAndGroups],
                    ['bots', b.roster.botsOnly],
                    ['groups', b.roster.groupsOnly]
                  ] as [RosterKindFilter, string][]
                ).map(([value, label]) => (
                  <DropdownMenuItem key={`kind:${value}`} onSelect={() => setRowKindFilter(value)}>
                    <span className="min-w-0 flex-1">{label}</span>
                    {rowKindFilter === value ? <Codicon name="check" /> : null}
                  </DropdownMenuItem>
                ))}
                <DropdownMenuSeparator />
                {(
                  [
                    ['all', b.roster.anyActivity],
                    ['active', b.roster.activeNow],
                    ['recent', b.roster.recentlyActive],
                    ['older', b.roster.older]
                  ] as [RosterActivityFilter, string][]
                ).map(([value, label]) => (
                  <DropdownMenuItem key={`activity:${value}`} onSelect={() => setActivityFilter(value)}>
                    <span className="min-w-0 flex-1">{label}</span>
                    {activityFilter === value ? <Codicon name="check" /> : null}
                  </DropdownMenuItem>
                ))}
                {gatewayOptions.length > 1 ? <DropdownMenuSeparator /> : null}
                {gatewayOptions.length > 1 ? (
                  <DropdownMenuItem onSelect={() => setGatewayFilter('all')}>
                    <Codicon className="mr-1.5" name="globe" />
                    <span className="min-w-0 flex-1">All gateways</span>
                    {gatewayFilter === 'all' ? <Codicon name="check" /> : null}
                  </DropdownMenuItem>
                ) : null}
                {gatewayOptions.length > 1
                  ? gatewayOptions.map(option => {
                      const status = botSourceStatus({
                        sourceError: option.error,
                        sourceReachable: option.reachable
                      })

                      return (
                        <DropdownMenuItem
                          key={option.connectionId}
                          onSelect={() => setGatewayFilter(option.connectionId)}
                        >
                          <GatewayKindGlyph
                            className={cn('mr-1.5', !status.available && 'text-amber-600 dark:text-amber-300')}
                            kind={option.kind}
                          />
                          <span className="min-w-0 flex-1 truncate">{option.label || option.connectionId}</span>
                          <span className="text-[0.625rem] tabular-nums text-(--ui-text-quaternary)">
                            {option.count}
                          </span>
                          {gatewayFilter === option.connectionId ? <Codicon name="check" /> : null}
                        </DropdownMenuItem>
                      )
                    })
                  : []}
                {activeFilterCount ? <DropdownMenuSeparator /> : null}
                {activeFilterCount ? (
                  <DropdownMenuItem
                    onSelect={() => {
                      setRowKindFilter('all')
                      setActivityFilter('all')
                      setGatewayFilter('all')
                    }}
                  >
                    {b.roster.clearFilters}
                  </DropdownMenuItem>
                ) : null}
              </DropdownMenuContent>
            </DropdownMenu>
          ) : null}
        </div>
      ) : null}
      {staleNotice ? (
        <div className="mx-2.5 mb-1 rounded-md bg-(--chrome-action-hover) px-2 py-1.5 text-[0.6875rem] text-(--ui-text-tertiary)">
          {staleNotice}
        </div>
      ) : null}
      {(isLoading || initialRosterLoading) && !roster.length ? (
        <div className="flex flex-1 items-center justify-center">
          <GlyphSpinner className="text-(--ui-text-tertiary)" spinner="breathe" />
        </div>
      ) : error && !roster.length ? (
        <div className="grid gap-2 px-3 py-4 text-xs text-(--ui-text-tertiary)">
          <div>
            {gatewayUp
              ? b.roster.rosterUnavailable(error instanceof Error ? error.message : 'gateway error')
              : b.roster.waitingForGateway}
          </div>
          <Button className="justify-self-start" onClick={() => void refetch()} size="sm" variant="secondary">
            {b.roster.retryNow}
          </Button>
        </div>
      ) : roster.length === 0 ? (
        <PanelEmpty description={b.roster.emptyDesc} icon="hubot" title={b.roster.emptyTitle} />
      ) : allBotsHidden && !hiddenExpanded ? (
        <div className="grid content-start gap-2 px-3 py-4 text-xs text-(--ui-text-tertiary)">
          <div className="flex items-center gap-1.5 font-medium text-(--ui-text-secondary)">
            <Codicon className="text-(--ui-text-quaternary)" name="eye-closed" />
            {b.roster.allHidden}
          </div>
          <p className="leading-relaxed">{b.roster.allHiddenDesc}</p>
          <Button
            className="justify-self-start"
            onClick={() => $showHiddenBots.set(true)}
            size="sm"
            variant="secondary"
          >
            {b.roster.showHidden}
          </Button>
        </div>
      ) : rosterRows.length === 0 && matchingHiddenBots.length === 0 ? (
        <div aria-live="polite" className="flex min-h-0 flex-1 flex-col" role="status">
          <PanelEmpty
            description={
              query.trim()
                ? selectedGateway
                  ? b.roster.noMatchQueryOn(query.trim(), String(selectedGateway.label))
                  : b.roster.noMatchQuery(query.trim())
                : selectedGateway
                  ? b.roster.noMatchFiltersOn(String(selectedGateway.label))
                  : b.roster.noMatchFilters
            }
            icon="search"
          />
        </div>
      ) : (
        <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain" data-slot="bots-roster">
          <div className="grid w-full min-w-0 gap-0.5 px-1.5 pb-2">
            {showGatewaySections
              ? [
                  sortedGroupRows.length ? renderGroupChatSection() : null,
                  ...gatewaySections.sections.map(renderGatewaySection)
                ].filter(Boolean)
              : renderUserSections(rosterRows)}
            {showHiddenSection ? (
              <div
                className="mt-1 border-t border-(--ui-stroke-tertiary) pt-1"
                key={'hidden-section'}
                ref={hiddenSectionRef}
              >
                {hasRosterConstraint ? (
                  <div className="flex w-full items-center gap-1 px-2 py-1.5 text-[0.6875rem] font-medium text-(--ui-text-tertiary)">
                    <Codicon name="eye-closed" />
                    <span>Hidden</span>
                    <span className="text-(--ui-text-quaternary)">{matchingHiddenBots.length}</span>
                  </div>
                ) : (
                  <RowButton
                    aria-expanded={hiddenExpanded}
                    className="flex w-full items-center gap-1 rounded-md px-2 py-1.5 text-left text-[0.6875rem] font-medium text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground"
                    onClick={() => $showHiddenBots.set(!hiddenExpanded)}
                  >
                    <DisclosureCaret open={hiddenExpanded} />
                    <span>Hidden</span>
                    <span className="text-(--ui-text-quaternary)">{hiddenBots.length}</span>
                  </RowButton>
                )}
                {showHiddenRows ? (
                  matchingHiddenBots.length ? (
                    hiddenGatewaySections.sectioned ? (
                      hiddenGatewaySections.sections.map(renderHiddenGatewaySection)
                    ) : (
                      matchingHiddenBots.map((bot: RosterRow) => renderBotRow(bot, 'hidden:'))
                    )
                  ) : (
                    <div className="px-2 py-2 text-xs text-(--ui-text-quaternary)">{b.roster.noHiddenMatch}</div>
                  )
                ) : null}
              </div>
            ) : null}
          </div>
        </div>
      )}
      <CreateAgentDialog
        onClose={() => {
          setCreateOpen(false)
          void refetch()
        }}
        open={createOpen}
        roster={activeSourceRoster}
      />
      <CreateGroupChatDialog
        onClose={() => setGroupCreateOpen(false)}
        onCreated={groupName => openGroupChat(groupName)}
        open={groupCreateOpen} // Full multi-source roster: group chats can seat bots from other
        // registered connections — their turns route to their own machines.
        roster={roster}
      />
      <SectionNameDialog
        initialName={sectionDialog?.mode === 'rename' ? sectionDialog.name : ''}
        mode={sectionDialog?.mode === 'rename' ? 'rename' : 'create'}
        onOpenChange={open => {
          if (!open) {
            setSectionDialog(null)
          }
        }}
        onSubmit={name => {
          if (sectionDialog?.mode === 'rename') {
            renameBotSection(sectionDialog.id, name)
          } else {
            createBotSection(name, sectionDialog?.bot ? [sectionDialog.bot] : [])
          }
        }}
        open={Boolean(sectionDialog)}
      />
      <EditProfileDialog
        bot={editing}
        onClose={() => {
          setEditing(null)
          void refetch()
        }}
        open={Boolean(editing)}
      />
      {grouping ? <GroupDialog bot={grouping} onClose={() => setGrouping(null)} /> : null}
      <ConfirmDialog
        busyLabel="Deleting…"
        confirmLabel={t.common.delete}
        description={
          deleting ? (
            <span>
              {'This will permanently delete the bot '}
              <span className="font-medium text-foreground">{deleting.name}</span>
              {' and its associated Hermes profile at '}
              <span className="font-mono text-xs">{deleting.path}</span>. This cannot be undone.
            </span>
          ) : null
        }
        destructive
        doneLabel="Deleted"
        onClose={() => setDeleting(null)}
        onConfirm={async () => {
          if (!deleting) {
            return
          }

          const name = deleting.name
          await deleteBot(deleting)
          await refetch()
          host.notify({
            kind: 'success',
            message: `Deleted profile ${name}`
          })
        }}
        open={Boolean(deleting)}
        title={b.bot.deleteTitle}
      />
      <ConfirmDialog
        busyLabel="Deleting…"
        confirmLabel={b.group.deleteAction}
        description={
          deletingGroup
            ? `This removes “${deletingGroup.name}” from its bots and clears the shared room log. The bots and their individual chats are kept.`
            : null
        }
        destructive
        doneLabel="Deleted"
        onClose={() => setDeletingGroup(null)}
        onConfirm={async () => {
          if (!deletingGroup) {
            return
          }

          await disbandGroupChat(deletingGroup.name, deletingGroup.members)
          host.notify({
            kind: 'success',
            message: `Deleted group “${deletingGroup.name}”`
          })
        }}
        open={Boolean(deletingGroup)}
        title={b.group.deleteTitle}
      />
    </div>
  )
}

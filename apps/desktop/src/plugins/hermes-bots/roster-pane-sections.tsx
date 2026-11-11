import { host } from '@hermes/plugin-sdk'
import type { ReactNode } from 'react'

import { botRosterKey } from './data'
import type { useBots } from './i18n'
import {
  clearRosterOrder,
  moveRosterItem,
  orderRosterRows,
  orderSectionMixedRows,
  persistRosterOrder,
  type RosterOrderMap,
  rosterSectionScope
} from './roster-order'
import type { RosterGroupRow } from './roster-pane-derivation'
import { GatewayKindGlyph, GatewaySectionHeading, RosterSectionHeader } from './roster-sections'
import type { ResolvedRosterGatewaySection, rosterGatewayOptions } from './roster-sections'
import { botRosterMeta } from './routing'
import type { BotMeta, GroupChat, RosterRow } from './types'
import type { $botSections, SectionDialogState } from './user-sections'
import {
  botSectionId,
  deleteBotSection,
  GROUP_DRAG_PREFIX,
  groupDragKey,
  groupRowsBySection,
  moveBotSection,
  moveBotsToSection,
  moveGroupChatsToSection,
  UNASSIGNED_SECTION_KEY
} from './user-sections'
import { SectionDropZone, UserSectionHeader } from './user-sections-ui'

export type ReorderBotRowHandler = (sourceKey: string, targetKey: string, position: 'before' | 'after') => void

export interface BotRowOrderBindings {
  onMoveDown?: () => void
  onMoveUp?: () => void
  onReorder: ReorderBotRowHandler
  scopeKey: string
}

interface RosterSectionRenderersProps {
  b: ReturnType<typeof useBots>
  userSections: ReturnType<typeof $botSections.get>
  roster: RosterRow[]
  allMeta: Record<string, BotMeta>
  groupRooms: Record<string, GroupChat>
  rosterOrder: RosterOrderMap
  gatewayFilter: string
  gatewayOptions: ReturnType<typeof rosterGatewayOptions>
  activityOf: (bot: RosterRow) => number
  isPinned: (bot: RosterRow) => boolean
  dragging: string | null
  rosterSectionCollapsed: (id: string) => boolean
  toggleRosterSection: (id: string) => void
  setSectionDialog: (value: SectionDialogState) => void
  renderBotRow: (bot: RosterRow, keyPrefix?: string, orderBindings?: BotRowOrderBindings) => ReactNode
  renderGroupRow: (row: RosterGroupRow) => ReactNode
  sortedGroupRows: RosterGroupRow[]
}

interface UserSectionRenderOptions {
  connectionId?: string | null
  reorderBots?: boolean
}

function matchesSectionConnection(bot: RosterRow, explicitConnectionId: string | null): boolean {
  if (!explicitConnectionId || explicitConnectionId === 'all') {
    return true
  }

  const normalized = String(bot.connectionId || 'legacy').trim() || 'legacy'

  return bot.connectionId === explicitConnectionId || normalized === explicitConnectionId
}

export function rosterSectionRenderers({
  b,
  userSections,
  roster,
  allMeta,
  groupRooms,
  rosterOrder,
  gatewayFilter,
  gatewayOptions,
  activityOf,
  isPinned,
  dragging,
  rosterSectionCollapsed,
  toggleRosterSection,
  setSectionDialog,
  renderBotRow,
  renderGroupRow,
  sortedGroupRows
}: RosterSectionRenderersProps) {
  const removeSection = (id: string) => {
    const name = userSections.find(section => section.id === id)?.name || ''
    const { members, undo } = deleteBotSection(id, roster)

    // No confirmation: nothing is lost (the bots fall back to Unassigned) and
    // the toast's Undo puts the section, its members, and any custom order back.
    host.notify({
      action: { label: b.sections.undo, onClick: undo },
      durationMs: 8_000,
      kind: 'info',
      message: b.sections.deleted(name, members.length)
    })
  }

  type UserSectionRow = { bot: RosterRow; kind?: 'bot' } | RosterGroupRow

  const toOrderableBotRow = (row: { bot: RosterRow; kind?: 'bot' }) => ({
    ...row,
    pinned: isPinned(row.bot),
    activity: activityOf(row.bot),
    created: botRosterMeta(row.bot, allMeta)?.created || row.bot.ui_meta?.['hermes-bots']?.created || 0
  })

  const buildSectionBotOrderContext = (sectionId: null | string, explicitConnectionId: string | null) => {
    const effectiveConnectionId =
      explicitConnectionId || (gatewayFilter !== 'all' ? gatewayFilter : gatewayOptions[0]?.connectionId || 'local')

    const scopeKey = rosterSectionScope(sectionId, effectiveConnectionId)
    const sectionOrderKeys = rosterOrder[scopeKey] || []
    const filterConn = explicitConnectionId || (gatewayFilter !== 'all' ? gatewayFilter : null)

    const allSectionBotKeys = orderRosterRows(
      roster
        .filter(bot => botSectionId(bot, allMeta) === sectionId && matchesSectionConnection(bot, filterConn))
        .map(bot => toOrderableBotRow({ bot })),
      row => botRosterKey(row.bot),
      sectionOrderKeys
    ).map(row => botRosterKey(row.bot))

    const handleReorder: ReorderBotRowHandler = (sourceKey, targetKey, position) => {
      const nextOrder = moveRosterItem(sectionOrderKeys, allSectionBotKeys, sourceKey, targetKey, position)
      persistRosterOrder({
        ...rosterOrder,
        [scopeKey]: nextOrder
      })
    }

    return { scopeKey, sectionOrderKeys, handleReorder }
  }

  const renderSectionRows = (
    rows: UserSectionRow[],
    keyPrefix: string,
    botOrder: null | ReturnType<typeof buildSectionBotOrderContext>
  ) => {
    if (!botOrder) {
      return rows.map(row => (row.kind === 'group' ? renderGroupRow(row) : renderBotRow(row.bot, keyPrefix)))
    }

    const pinnedBots = rows.filter((row): row is { bot: RosterRow; kind?: 'bot' } => row.kind !== 'group' && isPinned(row.bot))

    const unpinnedBots = rows.filter(
      (row): row is { bot: RosterRow; kind?: 'bot' } => row.kind !== 'group' && !isPinned(row.bot)
    )

    return rows.map(row => {
      if (row.kind === 'group') {
        return renderGroupRow(row)
      }

      const band = isPinned(row.bot) ? pinnedBots : unpinnedBots
      const itemKey = botRosterKey(row.bot)
      const index = band.findIndex(entry => botRosterKey(entry.bot) === itemKey)
      const prevKey = index > 0 ? botRosterKey(band[index - 1]!.bot) : null
      const nextKey = index >= 0 && index < band.length - 1 ? botRosterKey(band[index + 1]!.bot) : null

      return renderBotRow(row.bot, keyPrefix, {
        onMoveDown: nextKey ? () => botOrder.handleReorder(itemKey, nextKey, 'after') : undefined,
        onMoveUp: prevKey ? () => botOrder.handleReorder(itemKey, prevKey, 'before') : undefined,
        onReorder: botOrder.handleReorder,
        scopeKey: botOrder.scopeKey
      })
    })
  }

  // USER SECTIONS — composed with the gateway sections, not instead of them.
  // The gateway headings own the top level whenever the roster shows more
  // than one connection (that axis answers "where does this run", which no
  // folder name can, and a bot's membership lives in its profile on THAT
  // gateway); user sections group the rows INSIDE each connection bucket,
  // indented under it, and group the flat list when there is only one.
  // `keyPrefix` keeps row keys unique across the gateway buckets.
  const renderUserSections = (rows: UserSectionRow[], keyPrefix = '', options?: UserSectionRenderOptions) => {
    const reorderBots = options?.reorderBots !== false
    const explicitConnectionId = options?.connectionId ?? null

    // No sections made: the plain list, reorderable within Unassigned.
    if (!userSections.length) {
      if (!reorderBots) {
        return renderSectionRows(rows, keyPrefix, null)
      }

      const botOrder = buildSectionBotOrderContext(null, explicitConnectionId)
      const orderedRows = orderSectionMixedRows(rows, toOrderableBotRow, botOrder.sectionOrderKeys)

      return renderSectionRows(orderedRows, keyPrefix, botOrder)
    }

    const nested = Boolean(keyPrefix)
    const blocks = groupRowsBySection(rows, userSections, allMeta, groupRooms)

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

          const botOrder = reorderBots ? buildSectionBotOrderContext(block.id ?? null, explicitConnectionId) : null
          const hasCustomOrder = Boolean(botOrder && botOrder.sectionOrderKeys.length > 0)

          const orderedRows = botOrder
            ? orderSectionMixedRows(block.rows, toOrderableBotRow, botOrder.sectionOrderKeys)
            : block.rows

          return (
            <SectionDropZone
              isSource={
                Boolean(dragging) &&
                block.rows.some(row =>
                  row.kind === 'group' ? groupDragKey(row.name) === dragging : botRosterKey(row.bot) === dragging
                )
              }
              key={key}
              nested={nested}
              onDropBot={rosterKey => {
                // `block.id` is null for Unassigned, which is exactly the value
                // both movers want for "clear the assignment".
                if (rosterKey.startsWith(GROUP_DRAG_PREFIX)) {
                  moveGroupChatsToSection([rosterKey.slice(GROUP_DRAG_PREFIX.length)], block.id)

                  return
                }

                const bot = roster.find(row => botRosterKey(row) === rosterKey)

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
                hasCustomOrder={hasCustomOrder}
                id={block.id}
                name={block.name}
                onDelete={() => block.id && removeSection(block.id)}
                onMove={delta => block.id && moveBotSection(block.id, delta)}
                onRename={() => block.id && setSectionDialog({ id: block.id, mode: 'rename', name: block.name })}
                onResetOrder={hasCustomOrder && botOrder ? () => clearRosterOrder(botOrder.scopeKey) : undefined}
                onToggle={() => toggleRosterSection(key)}
              />
              {collapsed ? null : block.rows.length ? (
                <div className="grid min-w-0 gap-0.5">{renderSectionRows(orderedRows, `${key}:`, botOrder)}</div>
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
          <div className="grid min-w-0 gap-0.5">
            {renderUserSections(section.rows, `${section.id}:`, { connectionId: section.id })}
          </div>
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
          tip={b.group.globalCount(sortedGroupRows.length)}
        />
        {collapsed ? null : (
          <div className="grid min-w-0 gap-0.5">
            {renderUserSections(sortedGroupRows, 'groups:', { reorderBots: false })}
          </div>
        )}
      </div>
    )
  }

  const renderHiddenGatewaySection = (section: ResolvedRosterGatewaySection) => (
    <div className="min-w-0" key={`hidden-gateway:${section.id}`}>
      <div className="flex min-w-0 items-center gap-1.5 px-2 py-1 text-[0.625rem] font-semibold uppercase tracking-wider text-(--ui-text-quaternary)">
        <GatewayKindGlyph kind={section.option?.kind} />
        <span className="min-w-0 flex-1 truncate">
          {section.option?.label || section.option?.connectionId || b.roster.currentGateway}
        </span>
        <span className="shrink-0 font-normal tabular-nums">{section.rows.length}</span>
      </div>
      {section.rows.map(row => renderBotRow(row.bot, `hidden:${section.id}:`))}
    </div>
  )

  return { renderUserSections, renderGatewaySection, renderGroupChatSection, renderHiddenGatewaySection }
}

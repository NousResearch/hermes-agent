import { host } from '@hermes/plugin-sdk'
import type { ReactNode } from 'react'

import { botRosterKey } from './data'
import type { useBots } from './i18n'
import type { RosterGroupRow } from './roster-pane-derivation'
import { GatewayKindGlyph, GatewaySectionHeading, RosterSectionHeader } from './roster-sections'
import type { ResolvedRosterGatewaySection } from './roster-sections'
import type { BotMeta, GroupChat, RosterRow } from './types'
import type { $botSections, SectionBlock, SectionDialogState } from './user-sections'
import {
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

/** A row user sections file: a bot, or a group chat. */
type UserSectionRow = { bot: RosterRow; kind?: 'bot' } | RosterGroupRow

interface RosterSectionRenderersProps {
  b: ReturnType<typeof useBots>
  userSections: ReturnType<typeof $botSections.get>
  roster: RosterRow[]
  allMeta: Record<string, BotMeta>
  groupRooms: Record<string, GroupChat>
  dragging: string | null
  /** `dragging`, from the tick after its `dragstart` (useBotDragUnderway). */
  dragUnderway: boolean
  rosterSectionCollapsed: (id: string) => boolean
  toggleRosterSection: (id: string) => void
  setSectionDialog: (value: SectionDialogState) => void
  renderBotRow: (bot: RosterRow, keyPrefix?: string) => ReactNode
  renderGroupRow: (row: RosterGroupRow) => ReactNode
  sortedGroupRows: RosterGroupRow[]
  /** Every row the roster lists: what the gateway buckets hold between them. */
  rosterRows: UserSectionRow[]
}

export function rosterSectionRenderers({
  b,
  userSections,
  roster,
  allMeta,
  groupRooms,
  dragging,
  dragUnderway,
  rosterSectionCollapsed,
  toggleRosterSection,
  setSectionDialog,
  renderBotRow,
  renderGroupRow,
  sortedGroupRows,
  rosterRows
}: RosterSectionRenderersProps) {
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

  // Where each named section stands across the whole roster. One with a row
  // is seated: drawn inside every bucket that holds one of its rows. One
  // without is drawn by no bucket (renderEmptyUserSections).
  const namedBlocks = groupRowsBySection(rosterRows, userSections, allMeta, groupRooms).filter(block => block.id)
  const seated = new Set(namedBlocks.filter(block => block.rows.length).map(block => block.id))

  // One section block: its heading, then its rows or the empty slot, as one
  // drop target. `keyPrefix` names the bucket it is drawn in; '' is the top
  // level.
  const renderSectionBlock = (block: SectionBlock<UserSectionRow>, keyPrefix = '') => {
    const key = `${keyPrefix}${block.id ? `user-section:${block.id}` : UNASSIGNED_SECTION_KEY}`
    const collapsed = rosterSectionCollapsed(key)
    const order = userSections.findIndex(section => section.id === block.id)

    return (
      <SectionDropZone
        isSource={
          Boolean(dragging) &&
          block.rows.some(row =>
            row.kind === 'group' ? groupDragKey(row.name) === dragging : botRosterKey(row.bot) === dragging
          )
        }
        key={key}
        nested={Boolean(keyPrefix)}
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
          id={block.id}
          name={block.name}
          onDelete={() => block.id && removeSection(block.id)}
          onMove={delta => block.id && moveBotSection(block.id, delta)}
          onRename={() => block.id && setSectionDialog({ id: block.id, mode: 'rename', name: block.name })}
          onToggle={() => toggleRosterSection(key)}
        />
        {collapsed ? null : block.rows.length ? (
          <div className="grid min-w-0 gap-0.5">
            {block.rows.map(row => (row.kind === 'group' ? renderGroupRow(row) : renderBotRow(row.bot, `${key}:`)))}
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
  }

  const renderUserSections = (rows: UserSectionRow[], keyPrefix = '') => {
    // No sections made: the plain list, exactly as before this feature.
    if (!userSections.length) {
      return rows.map(row => (row.kind === 'group' ? renderGroupRow(row) : renderBotRow(row.bot, keyPrefix)))
    }

    const nested = Boolean(keyPrefix)

    return (
      groupRowsBySection(rows, userSections, allMeta, groupRooms)
        // An empty Unassigned is not worth a heading; an empty NAMED section
        // is, because it is somewhere the user made and is about to drop into.
        // Inside a gateway bucket the same empty section would repeat under
        // every connection, so at rest a bucket draws only the sections it has
        // rows for. While a drag is under way it also draws, empty, the ones
        // seated in another bucket: that is how one section gathers bots of
        // several gateways. Without the copy its only target is the block
        // under another gateway's heading, which may be folded and is not
        // where the dropped bot will show. A section with rows in no bucket is
        // not repeated here: it already has its one block below them all.
        .filter(block => block.rows.length || (block.id && (!nested || (dragUnderway && seated.has(block.id)))))
        .map(block => renderSectionBlock(block, keyPrefix))
    )
  }

  // The named sections no gateway bucket draws: those without a row anywhere
  // in the roster. Each gets its one heading at the top level, so a section is
  // on screen from the moment it is made (#155196) and takes a row dragged
  // from any bucket; filing one seats the section inside that row's bucket,
  // and it leaves this list.
  //
  // It is the only copy, drag or not. Repeated inside every bucket during a
  // drag, as every empty section used to be, it would show the same empty
  // section several times; and hidden mid-drag to make room for those copies,
  // it would pull the slot that reads "Drag bots here" from under the pointer.
  const renderEmptyUserSections = () =>
    namedBlocks.filter(block => !block.rows.length).map(block => renderSectionBlock(block))

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
          tip={b.group.globalCount(sortedGroupRows.length)}
        />
        {collapsed ? null : (
          <div className="grid min-w-0 gap-0.5">{renderUserSections(sortedGroupRows, 'groups:')}</div>
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

  return {
    renderUserSections,
    renderEmptyUserSections,
    renderGatewaySection,
    renderGroupChatSection,
    renderHiddenGatewaySection
  }
}

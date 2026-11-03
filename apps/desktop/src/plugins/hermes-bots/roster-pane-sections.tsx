import { host } from '@hermes/plugin-sdk'
import type { ReactNode } from 'react'

import { botRosterKey } from './data'
import type { useBots } from './i18n'
import type { RosterGroupRow } from './roster-pane-derivation'
import { GatewayKindGlyph, GatewaySectionHeading, RosterSectionHeader } from './roster-sections'
import type { ResolvedRosterGatewaySection } from './roster-sections'
import type { BotMeta, GroupChat, GroupMember, RosterRow } from './types'
import type { $botSections, BotSection, SectionBlock, SectionDialogState } from './user-sections'
import {
  botSectionParentOptions,
  botSectionPath,
  deleteBotSection,
  GROUP_DRAG_PREFIX,
  groupDragKey,
  groupRowsBySection,
  moveBotSection,
  moveBotSectionToParent,
  moveBotsToSection,
  moveGroupChatsToSection,
  placeBotSectionBefore,
  SECTION_DRAG_PREFIX,
  UNASSIGNED_SECTION_KEY,
  visibleSectionBlocks
} from './user-sections'
import { SectionDropZone, UserSectionHeader } from './user-sections-ui'

interface RosterSectionRenderersProps {
  b: ReturnType<typeof useBots>
  userSections: ReturnType<typeof $botSections.get>
  roster: RosterRow[]
  allMeta: Record<string, BotMeta>
  groupRooms: Record<string, GroupChat>
  dragging: string | null
  rosterSectionCollapsed: (id: string) => boolean
  toggleRosterSection: (id: string) => void
  setSectionDialog: (value: SectionDialogState) => void
  renderBotRow: (bot: RosterRow, keyPrefix?: string) => ReactNode
  renderGroupRow: (row: { members: GroupMember[]; name: string }) => ReactNode
  sortedGroupRows: RosterGroupRow[]
}

/** A roster row a user section can hold: a bot, or a group chat. */
type UserSectionRow = { bot: RosterRow; kind?: 'bot' } | RosterGroupRow

interface SectionBlockView {
  block: SectionBlock<UserSectionRow>
  blockKey: (id: null | string) => string
  nested: boolean
  visible: SectionBlock<UserSectionRow>[]
}

export function rosterSectionRenderers({
  b,
  userSections,
  roster,
  allMeta,
  groupRooms,
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
    const { moved, parent, undo } = deleteBotSection(id, roster)

    // No confirmation: nothing is lost (bots, group chats and subsections move
    // up a level) and the toast's Undo puts the section and its members back.
    host.notify({
      action: { label: b.sections.undo, onClick: undo },
      durationMs: 8_000,
      kind: 'info',
      message: parent ? b.sections.deletedInto(name, moved, parent.name) : b.sections.deleted(name, moved)
    })
  }

  // "Move into" targets per section: top level first, then every folder that
  // is not the section itself or below it, labelled with its full path.
  const parentOptions = (id: string) => [
    { id: null, label: b.sections.topLevel },
    ...botSectionParentOptions(id, userSections).map(section => ({
      id: section.id,
      label: botSectionPath(section.id, userSections)
    }))
  ]

  // USER SECTIONS — composed with the gateway sections, not instead of them.
  // The gateway headings own the top level whenever the roster shows more
  // than one connection (that axis answers "where does this run", which no
  // folder name can, and a bot's membership lives in its profile on THAT
  // gateway); user sections group the rows INSIDE each connection bucket,
  // indented under it, and group the flat list when there is only one.
  // `keyPrefix` keeps row keys unique across the gateway buckets.
  const renderUserSections = (rows: UserSectionRow[], keyPrefix = '') => {
    // No sections made: the plain list, exactly as before this feature.
    if (!userSections.length) {
      return rows.map(row => (row.kind === 'group' ? renderGroupRow(row) : renderBotRow(row.bot, keyPrefix)))
    }

    const nested = Boolean(keyPrefix)
    const blockKey = (id: null | string) => `${keyPrefix}${id ? `user-section:${id}` : UNASSIGNED_SECTION_KEY}`

    const visible = visibleSectionBlocks(groupRowsBySection(rows, userSections, allMeta, groupRooms), {
      collapsed: id => rosterSectionCollapsed(blockKey(id)),
      dragging,
      nested
    })

    return visible.map(block => renderSectionBlock({ block, blockKey, nested, visible }))
  }

  const renderSectionBlock = ({ block, blockKey, nested, visible }: SectionBlockView) => {
    const key = blockKey(block.id)
    const collapsed = rosterSectionCollapsed(key)
    // Order among the siblings drawn here: an empty sibling hidden in this
    // gateway bucket is not something "Move up" can visibly pass.
    const siblings = visible.filter(other => other.id && other.parentId === block.parentId)
    const order = siblings.findIndex(other => other.id === block.id)
    const hasChildren = visible.some(other => other.parentId === block.id)

    // Collapse state is per block key; moving or creating into a collapsed
    // folder opens it, so the result is never filed somewhere out of sight.
    const expand = (id: string) => {
      if (rosterSectionCollapsed(blockKey(id))) {
        toggleRosterSection(blockKey(id))
      }
    }

    return (
      <SectionDropZone
        indent={block.depth + (nested ? 1 : 0)}
        isSource={isDropSource(block, dragging, userSections)}
        key={key}
        onDropBot={dragKey => dropOnSection(dragKey, block.id, roster)}
      >
        <UserSectionHeader
          canMoveDown={order >= 0 && order < siblings.length - 1}
          canMoveUp={order > 0}
          collapsed={collapsed}
          count={block.total}
          id={block.id}
          name={block.name}
          onDelete={() => block.id && removeSection(block.id)}
          onMove={delta => moveAmongVisible(userSections, siblings[order]?.id, siblings[order + delta]?.id)}
          onMoveInto={parentId => {
            if (block.id && moveBotSectionToParent(block.id, parentId) && parentId) {
              expand(parentId)
            }
          }}
          onNewSubsection={() => {
            if (block.id) {
              expand(block.id)
              setSectionDialog({ mode: 'create', parentId: block.id })
            }
          }}
          onRename={() => block.id && setSectionDialog({ id: block.id, mode: 'rename', name: block.name })}
          onToggle={() => toggleRosterSection(key)}
          parentId={block.parentId}
          parentOptions={block.id ? parentOptions(block.id) : []}
        />
        {collapsed ? null : block.rows.length ? (
          <div className="grid min-w-0 gap-0.5">
            {block.rows.map(row => (row.kind === 'group' ? renderGroupRow(row) : renderBotRow(row.bot, `${key}:`)))}
          </div>
        ) : block.total || hasChildren ? null : (
          // Empty leaf section: a quiet dashed slot that says what it is for,
          // and doubles as a roomy drop target.
          <div className="mx-1 mb-1 rounded-md border border-dashed border-(--ui-stroke-secondary) px-2 py-2 text-center text-[0.6875rem] text-(--ui-text-quaternary)">
            {b.sections.emptyHint}
          </div>
        )}
      </SectionDropZone>
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
          {section.option?.label || section.option?.connectionId || 'Current gateway'}
        </span>
        <span className="shrink-0 font-normal tabular-nums">{section.rows.length}</span>
      </div>
      {section.rows.map(row => renderBotRow(row.bot, `hidden:${section.id}:`))}
    </div>
  )

  return { renderUserSections, renderGatewaySection, renderGroupChatSection, renderHiddenGatewaySection }
}

/** Whether a drag in flight may NOT drop on this block: a bot or group already
 *  filed here, or — for a folder in flight — the folder itself or anything
 *  inside it (that would make a loop). */
function isDropSource(block: SectionBlock<UserSectionRow>, dragging: null | string, sections: BotSection[]): boolean {
  if (!dragging) {
    return false
  }

  if (dragging.startsWith(SECTION_DRAG_PREFIX)) {
    const id = dragging.slice(SECTION_DRAG_PREFIX.length)

    return Boolean(block.id) && !botSectionParentOptions(id, sections).some(section => section.id === block.id)
  }

  return block.rows.some(row =>
    row.kind === 'group' ? groupDragKey(row.name) === dragging : botRosterKey(row.bot) === dragging
  )
}

/** Whatever was dragged, filed onto this block (`null` = Unassigned). A folder
 *  dropped on a folder lands just before it, at its level; on Unassigned it
 *  goes to the end of the top level. Bots and groups move into the block —
 *  `null` is exactly what both movers want for "clear the assignment". */
function dropOnSection(dragKey: string, sectionId: null | string, roster: RosterRow[]): void {
  if (dragKey.startsWith(SECTION_DRAG_PREFIX)) {
    const id = dragKey.slice(SECTION_DRAG_PREFIX.length)

    if (sectionId) {
      placeBotSectionBefore(id, sectionId)
    } else {
      moveBotSectionToParent(id, null)
    }

    return
  }

  if (dragKey.startsWith(GROUP_DRAG_PREFIX)) {
    moveGroupChatsToSection([dragKey.slice(GROUP_DRAG_PREFIX.length)], sectionId)

    return
  }

  const bot = roster.find(row => botRosterKey(row) === dragKey)

  if (bot) {
    void moveBotsToSection([bot], sectionId)
  }
}

/** Swap `self` with the neighbour drawn next to it, which may not be its
 *  adjacent sibling in the stored list when an empty one is hidden between. */
function moveAmongVisible(sections: BotSection[], self?: null | string, neighbour?: null | string): void {
  if (!self || !neighbour) {
    return
  }

  const own = sections.find(section => section.id === self)?.parentId ?? null
  const peers = sections.filter(section => (section.parentId ?? null) === own).map(section => section.id)

  moveBotSection(self, peers.indexOf(neighbour) - peers.indexOf(self))
}

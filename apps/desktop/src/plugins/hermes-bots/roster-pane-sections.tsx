import { host } from '@hermes/plugin-sdk'
import type { ReactNode } from 'react'

import { botRosterKey } from './data'
import type { useBots } from './i18n'
import type { RosterGroupRow } from './roster-pane-derivation'
import { GatewayKindGlyph, GatewaySectionHeading, RosterSectionHeader } from './roster-sections'
import type { ResolvedRosterGatewaySection } from './roster-sections'
import type { BotMeta, GroupMember, RosterRow } from './types'
import type { $botSections } from './user-sections'
import {
  deleteBotSection,
  groupRowsBySection,
  moveBotSection,
  moveBotsToSection,
  UNASSIGNED_SECTION_KEY
} from './user-sections'
import { SectionDropZone, UserSectionHeader } from './user-sections-ui'

interface RosterSectionRenderersProps {
  b: ReturnType<typeof useBots>
  userSections: ReturnType<typeof $botSections.get>
  roster: RosterRow[]
  allMeta: Record<string, BotMeta>
  dragging: string | null
  rosterSectionCollapsed: (id: string) => boolean
  toggleRosterSection: (id: string) => void
  setSectionDialog: (
    value: null | { bot?: RosterRow; mode: 'create' } | { id: string; mode: 'rename'; name: string }
  ) => void
  renderBotRow: (bot: RosterRow, keyPrefix?: string) => ReactNode
  renderGroupRow: (row: { members: GroupMember[]; name: string }) => ReactNode
  sortedGroupRows: RosterGroupRow[]
}

export function rosterSectionRenderers({
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

  return { renderUserSections, renderGatewaySection, renderGroupChatSection, renderHiddenGatewaySection }
}

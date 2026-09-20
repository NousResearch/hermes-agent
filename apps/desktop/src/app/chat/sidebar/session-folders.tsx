/**
 * Session folders — the sidebar's own grouping of sessions, rendered as a strip
 * above the flat list (folders first, then the sessions that are in none).
 *
 * Folders are not sessions and not projects: a project groups sessions by WHERE
 * they run (its folders are filesystem paths, and moving a session into one
 * re-homes its cwd), while a session folder is the user's own label and touches
 * nothing about the session. So this rides the desktop's local folder store
 * (`@/store/session-folders`) and the sidebar's existing dnd-kit list: a row is
 * dropped onto a folder header to file it, onto "None" to take it back out.
 */

import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useCallback, useMemo, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { RowButton } from '@/components/ui/row-button'
import { Tip } from '@/components/ui/tooltip'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import {
  $sessionFoldersAvailable,
  $sessionFolderScope,
  createSessionFolder,
  deleteSessionFolder,
  fileSessionInFolder,
  groupSessionsByFolder,
  renameSessionFolder,
  type SessionFolder,
  toggleSessionFolderCollapsed
} from '@/store/session-folders'

import { SidebarRowLead } from './chrome'
import { type ReorderDropTarget, useDropTargetBindings } from './reorderable-list'
import { SIDEBAR_LEAD_ICON_SIZE, SIDEBAR_ROW_LABEL } from './row-geometry'

// dnd-kit ids of the strip's drop targets. Prefixed so they can never collide
// with a session id (the ids the reorder list carries).
const FOLDER_DROP_PREFIX = 'session-folder:'

/** Drop target that takes a session back out of every folder. */
export const UNFILED_DROP_TARGET = 'session-folder:none'

export const folderDropTarget = (folderId: string) => `${FOLDER_DROP_PREFIX}${folderId}`

// A folder that a drag is hovering over has to read as "it will land HERE" —
// the same weight the row's own lifted state uses.
const DROP_HIGHLIGHT = 'rounded-md bg-(--ui-row-active-background) ring-1 ring-inset ring-(--ui-border-strong)'

const HOVER_ACTION =
  'grid size-4 shrink-0 place-items-center rounded-sm bg-transparent text-(--ui-text-quaternary) opacity-0 transition-opacity hover:bg-(--ui-control-hover-background) hover:text-foreground group-hover/folder:opacity-100 focus-visible:opacity-100'

export interface SessionFolderSection {
  /** Session ids the flat list must leave out of its rows (empty wherever the
   *  folders are not on screen). */
  filedIds: ReadonlySet<string>
  /** Extra dnd-kit drop targets for the list the strip renders in. */
  dropTargets: ReorderDropTarget[] | undefined
  /** The strip itself, or null when this list doesn't group by folder. */
  strip: React.ReactNode
}

const NO_FILED_IDS: ReadonlySet<string> = new Set()

/**
 * Everything the flat Sessions list needs to group by folder: the ids that leave
 * the list, the drop targets their rows can land on, and the strip that renders
 * the folders. One call, so the section component doesn't grow a second,
 * folder-shaped body.
 */
export function useSessionFolderSection({
  enabled,
  renderMembers,
  sessions
}: {
  /** Whether THIS list groups by folder at all. */
  enabled: boolean
  /** The folder's member rows, rendered by the owner (which owns the row
   *  renderer and its per-row props). */
  renderMembers: (members: SessionInfo[]) => React.ReactNode
  sessions: SessionInfo[]
}): SessionFolderSection {
  const { t } = useI18n()
  const scope = useStore($sessionFolderScope)
  const available = useStore($sessionFoldersAvailable)
  const active = enabled && available
  // Folders first, then the sessions in none — which is what the ordinary flat
  // rendering already is when there are no folders.
  const split = useMemo(() => groupSessionsByFolder(sessions, scope), [scope, sessions])
  // Empty wherever folders are NOT on screen (archived, all-profiles, grouped
  // views): those lists still show every session, filed or not, or a filed chat
  // would go missing the moment the sidebar left the flat list.
  const filedIds = useMemo(() => (active ? new Set(Object.keys(scope.filed)) : NO_FILED_IDS), [active, scope.filed])

  // Dropping a row on a folder files it there; on "Not in a folder" it leaves
  // every folder. Both ride the ONE dnd-kit list the rows already belong to, so
  // a drag over the sidebar routes here and a drag over a chat pane still routes
  // to the pane drag session.
  const dropTargets = useMemo(
    () =>
      active && scope.folders.length
        ? [
            ...scope.folders.map(folder => ({
              id: folderDropTarget(folder.id),
              onDrop: (sessionId: string) => fileSessionInFolder(sessionId, folder.id)
            })),
            { id: UNFILED_DROP_TARGET, onDrop: (sessionId: string) => fileSessionInFolder(sessionId, null) }
          ]
        : undefined,
    [active, scope.folders]
  )

  const counts = useMemo(
    () => Object.fromEntries(split.groups.map(group => [group.folder.id, group.sessions.length])),
    [split.groups]
  )

  const renderSessions = useCallback(
    (folderId: string) => {
      const group = split.groups.find(candidate => candidate.folder.id === folderId)

      return group?.sessions.length ? renderMembers(group.sessions) : null
    },
    [renderMembers, split.groups]
  )

  const strip = active ? (
    <SessionFolderStrip
      counts={counts}
      folders={scope.folders}
      onCreate={() => createSessionFolder(t.sidebar.folders.newFolder)}
      onDelete={deleteSessionFolder}
      onRename={renameSessionFolder}
      onToggleCollapsed={toggleSessionFolderCollapsed}
      renderSessions={renderSessions}
    />
  ) : null

  return { dropTargets, filedIds, strip }
}

export interface SessionFolderStripProps {
  counts: Record<string, number>
  folders: SessionFolder[]
  onCreate: () => void
  onDelete: (folderId: string) => void
  onRename: (folderId: string, name: string) => void
  onToggleCollapsed: (folderId: string) => void
  /** The folder's member rows, in list order (rendered by the owner, which owns
   *  the row renderer). */
  renderSessions: (folderId: string) => React.ReactNode
}

export function SessionFolderStrip({
  counts,
  folders,
  onCreate,
  onDelete,
  onRename,
  onToggleCollapsed,
  renderSessions
}: SessionFolderStripProps) {
  const { t } = useI18n()
  const f = t.sidebar.folders
  const { dropHighlight, dropRef } = useDropTargetBindings(UNFILED_DROP_TARGET)

  return (
    <div className="flex flex-col gap-px pb-1" data-session-folders="">
      <div className="group/folders flex min-h-6 items-center gap-1.5 px-2 text-[0.6875rem] font-medium text-(--ui-text-tertiary)">
        <SidebarRowLead>
          <Codicon name="new-folder" size={SIDEBAR_LEAD_ICON_SIZE} />
        </SidebarRowLead>
        <span className="min-w-0 flex-1 truncate">{f.label}</span>
        <Tip label={f.newFolder}>
          <button
            aria-label={f.newFolder}
            className={cn(HOVER_ACTION, 'group-hover/folders:opacity-100')}
            onClick={onCreate}
            type="button"
          >
            <Codicon name="add" size="0.75rem" />
          </button>
        </Tip>
      </div>

      {folders.map(folder => (
        <SessionFolderRow
          count={counts[folder.id] ?? 0}
          folder={folder}
          key={folder.id}
          onDelete={onDelete}
          onRename={onRename}
          onToggleCollapsed={onToggleCollapsed}
          renderSessions={renderSessions}
        />
      ))}

      {folders.length > 0 && (
        <Tip label={f.unfiledHint}>
          <div
            className={cn(
              'flex min-h-6 items-center gap-1.5 px-2 text-[0.6875rem] text-(--ui-text-quaternary)',
              dropHighlight && DROP_HIGHLIGHT
            )}
            data-folder-drop="none"
            ref={dropRef}
          >
            {f.unfiled}
          </div>
        </Tip>
      )}
    </div>
  )
}

interface SessionFolderRowProps {
  count: number
  folder: SessionFolder
  onDelete: (folderId: string) => void
  onRename: (folderId: string, name: string) => void
  onToggleCollapsed: (folderId: string) => void
  renderSessions: (folderId: string) => React.ReactNode
}

function SessionFolderRow({
  count,
  folder,
  onDelete,
  onRename,
  onToggleCollapsed,
  renderSessions
}: SessionFolderRowProps) {
  const { t } = useI18n()
  const f = t.sidebar.folders
  const [draft, setDraft] = useState(folder.name)
  const [renaming, setRenaming] = useState(false)
  const { dropHighlight, dropRef } = useDropTargetBindings(folderDropTarget(folder.id))
  const open = !folder.collapsed

  const startRenaming = () => {
    setDraft(folder.name)
    setRenaming(true)
  }

  const commitRename = () => {
    setRenaming(false)

    if (draft.trim() && draft.trim() !== folder.name) {
      onRename(folder.id, draft)
    }
  }

  return (
    <div className="flex flex-col gap-px" data-session-folder-id={folder.id}>
      <div
        className={cn('group/folder flex min-h-6 items-center gap-1 px-2', dropHighlight && DROP_HIGHLIGHT)}
        data-folder-drop={folder.id}
        ref={dropRef}
      >
        {renaming ? (
          <input
            aria-label={f.rename}
            autoFocus
            className="min-w-0 flex-1 rounded-sm bg-transparent px-1 text-[0.8125rem] text-foreground outline-none ring-1 ring-inset ring-(--ui-border-strong)"
            onBlur={commitRename}
            onChange={event => setDraft(event.target.value)}
            onKeyDown={event => {
              if (event.key === 'Enter') {
                commitRename()
              } else if (event.key === 'Escape') {
                setRenaming(false)
              }
            }}
            value={draft}
          />
        ) : (
          <>
            <RowButton
              aria-expanded={open}
              className="flex min-w-0 flex-1 items-center gap-1.5 bg-transparent text-left"
              onClick={() => onToggleCollapsed(folder.id)}
            >
              <SidebarRowLead>
                <Codicon name={open ? 'chevron-down' : 'chevron-right'} size={SIDEBAR_LEAD_ICON_SIZE} />
              </SidebarRowLead>
              <span className={cn(SIDEBAR_ROW_LABEL, 'flex-1')}>{folder.name}</span>
              <span className="shrink-0 text-[0.6875rem] text-(--ui-text-quaternary)">{count}</span>
            </RowButton>
            <Tip label={f.rename}>
              <button
                aria-label={`${f.rename}: ${folder.name}`}
                className={HOVER_ACTION}
                onClick={startRenaming}
                type="button"
              >
                <Codicon name="edit" size="0.75rem" />
              </button>
            </Tip>
            <Tip label={f.deleteHint}>
              <button
                aria-label={`${f.delete}: ${folder.name}`}
                className={HOVER_ACTION}
                onClick={() => onDelete(folder.id)}
                type="button"
              >
                <Codicon name="trash" size="0.75rem" />
              </button>
            </Tip>
          </>
        )}
      </div>
      {open && renderSessions(folder.id)}
    </div>
  )
}

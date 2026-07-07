import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from '@/components/ui/context-menu'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { SidebarGroup, SidebarGroupContent } from '@/components/ui/sidebar'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import type { SessionFolder } from '@/hermes'
import { triggerHaptic } from '@/lib/haptics'
import { $folders, $foldersLoading, createFolder, deleteAndRemoveFolder, refreshFolders } from '@/store/session-folders'
import { $selectedStoredSessionId, $sessions, $workingSessionIds, sessionPinId } from '@/store/session'

import { SidebarPanelLabel } from '../../shell/sidebar-label'
import { SidebarSessionRow } from './session-row'

interface SidebarFoldersSectionProps {
  label: string
  open: boolean
  onToggle: () => void
  onResumeSession: (sessionId: string) => void
  onDeleteSession: (sessionId: string) => void
  onArchiveSession: (sessionId: string) => void
  onTogglePin: (sessionId: string) => void
}

export function SidebarFoldersSection({
  label,
  open,
  onToggle,
  onResumeSession,
  onDeleteSession,
  onArchiveSession,
  onTogglePin,
}: SidebarFoldersSectionProps) {
  const { t } = useI18n()
  const r = t.sidebar.row
  const [creatingFolder, setCreatingFolder] = useState(false)
  const [newFolderName, setNewFolderName] = useState('')
  const [creating, setCreating] = useState(false)
  const [expandedFolderId, setExpandedFolderId] = useState<null | string>(null)
  const folders = useStore($folders)
  const foldersLoading = useStore($foldersLoading)
  const sessions = useStore($sessions)
  const activeSessionId = useStore($selectedStoredSessionId)
  const workingSessionIds = useStore($workingSessionIds)
  const workingSessionIdSet = new Set(workingSessionIds)

  // Fetch folders on mount
  useEffect(() => {
    void refreshFolders()
  }, [])

  if (foldersLoading) {
    return null
  }

  if (folders.length === 0 && !open) {
    return null
  }

  const handleCreate = async () => {
    if (!newFolderName.trim() || creating) return
    setCreating(true)
    const result = await createFolder(newFolderName.trim())
    setCreating(false)
    if (result) {
      setCreatingFolder(false)
      setNewFolderName('')
    }
  }

  return (
    <SidebarGroup className="shrink-0 p-0 pb-1">
      <div className="group/section flex shrink-0 items-center justify-between pb-1 pt-1.5">
        <button
          className="group/section-label flex w-fit items-center gap-1 bg-transparent text-left leading-none"
          onClick={onToggle}
          type="button"
        >
          <SidebarPanelLabel>{label}</SidebarPanelLabel>
          <span className="text-[0.6875rem] font-medium text-(--ui-text-quaternary)">{folders.length}</span>
          <DisclosureCaret
            className="text-(--ui-text-tertiary) opacity-0 transition group-hover/section-label:opacity-100"
            open={open}
          />
        </button>
        {open && (
          <button
            aria-label={r.createFolder ?? 'New folder'}
            className="grid size-5 place-items-center rounded-sm text-(--ui-text-tertiary) opacity-0 hover:bg-(--ui-control-hover-background) hover:text-foreground group-hover/section:opacity-100"
            onClick={() => setCreatingFolder(true)}
            type="button"
          >
            <Codicon name="new-folder" size="0.75rem" />
          </button>
        )}
      </div>
      {open && (
        <SidebarGroupContent className="flex max-h-72 flex-col gap-px overflow-x-hidden overflow-y-auto overscroll-contain pb-1.75 compact:max-h-none compact:overflow-visible">
          {creatingFolder && (
            <div className="flex items-center gap-1 px-2 py-1">
              <input
                autoFocus
                className="min-w-0 flex-1 rounded-md border border-(--ui-stroke-secondary) bg-transparent px-1.5 py-0.5 text-[0.8125rem] outline-none focus:border-(--ui-stroke-focus)"
                onChange={e => setNewFolderName(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter') {
                    void handleCreate()
                  }
                  if (e.key === 'Escape') {
                    setCreatingFolder(false)
                    setNewFolderName('')
                  }
                }}
                placeholder={r.createFolder ?? 'Name…'}
                value={newFolderName}
              />
            </div>
          )}
          {folders.length === 0 && !creatingFolder ? (
            <div className="py-1 pl-2 text-[0.6875rem] text-(--ui-text-tertiary)">
              {r.noFolders ?? 'No folders yet'}
            </div>
          ) : (
            folders.map(folder => (
              <div key={folder.id}>
                <ContextMenu>
                  <ContextMenuTrigger asChild>
                    <button
                      aria-expanded={expandedFolderId === folder.id}
                      className="flex min-w-0 items-center gap-1.5 bg-transparent py-0.5 pl-2 pr-1 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40 hover:bg-(--chrome-action-hover)"
                      onClick={() => setExpandedFolderId(prev => (prev === folder.id ? null : folder.id))}
                      type="button"
                    >
                      <span className="grid w-3.5 shrink-0 place-items-center">
                        <Codicon name="folder" size="0.8125rem" />
                      </span>
                      <span className="min-w-0 truncate text-[0.8125rem] text-(--ui-text-secondary)">
                        {folder.name}
                      </span>
                      <DisclosureCaret
                        className="shrink-0 text-(--ui-text-tertiary) transition"
                        open={expandedFolderId === folder.id}
                      />
                      <span className="ml-auto text-[0.6875rem] text-(--ui-text-quaternary)">
                        {folder.session_count}
                      </span>
                    </button>
                  </ContextMenuTrigger>
                  <ContextMenuContent className="w-40">
                    <ContextMenuItem
                      onSelect={() => {
                        triggerHaptic('selection')
                        navigator.clipboard.writeText(folder.id)
                      }}
                    >
                      <Codicon name="copy" size="0.875rem" />
                      <span>{r.copyId ?? 'Copy ID'}</span>
                    </ContextMenuItem>
                    <ContextMenuItem
                      variant="destructive"
                      onSelect={() => {
                        triggerHaptic('warning')
                        void deleteAndRemoveFolder(folder.id)
                      }}
                    >
                      <Codicon name="trash" size="0.875rem" />
                      <span>{r.deleteFolder ?? 'Delete folder'}</span>
                    </ContextMenuItem>
                  </ContextMenuContent>
                </ContextMenu>
                {expandedFolderId === folder.id && (
                  <FolderMemberList
                    folder={folder}
                    sessions={sessions}
                    activeSessionId={activeSessionId}
                    workingSessionIdSet={workingSessionIdSet}
                    onResumeSession={onResumeSession}
                    onDeleteSession={onDeleteSession}
                    onArchiveSession={onArchiveSession}
                    onTogglePin={onTogglePin}
                  />
                )}
              </div>
            ))
          )}
        </SidebarGroupContent>
      )}
    </SidebarGroup>
  )
}

function FolderMemberList({
  folder,
  sessions,
  activeSessionId,
  workingSessionIdSet,
  onResumeSession,
  onDeleteSession,
  onArchiveSession,
  onTogglePin,
}: {
  folder: SessionFolder
  sessions: import('@/hermes').SessionInfo[]
  activeSessionId: string | null
  workingSessionIdSet: Set<string>
  onResumeSession: (id: string) => void
  onDeleteSession: (id: string) => void
  onArchiveSession: (id: string) => void
  onTogglePin: (id: string) => void
}) {
  const sessionIds = folder.session_ids ?? []
  const memberSessions = sessionIds
    .map(fsid => sessions.find(s => sessionPinId(s) === fsid || s.id === fsid))
    .filter((s): s is import('@/hermes').SessionInfo => s !== undefined)

  if (memberSessions.length === 0) {
    return (
      <div className="ml-[1.125rem] py-1 pl-1 text-[0.6875rem] text-(--ui-text-tertiary)">
        No sessions
      </div>
    )
  }

  return (
    <div className="ml-[1.125rem] flex flex-col gap-px">
      {memberSessions.map(session => (
        <SidebarSessionRow
          active={session.id === activeSessionId}
          key={session.id}
          onArchive={() => onArchiveSession(session.id)}
          onDelete={() => onDeleteSession(session.id)}
          onPin={() => onTogglePin(session.id)}
          onResume={() => onResumeSession(session.id)}
          session={session}
          working={workingSessionIdSet.has(session.id)}
        />
      ))}
    </div>
  )
}

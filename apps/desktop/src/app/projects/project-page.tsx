import { useStore } from '@nanostores/react'
import { Suspense, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { ChatRuntimeBoundary } from '@/app/chat'
import { ChatBar, ChatBarFallback } from '@/app/chat/composer'
import type { DroppedFile } from '@/app/chat/hooks/use-composer-actions'
import { SidebarSessionRow } from '@/app/chat/sidebar/session-row'
import { CreateProjectDialog } from '@/app/projects/create-project-dialog'
import { EditInstructionsDialog } from '@/app/projects/edit-instructions-dialog'
import { sessionRoute } from '@/app/routes'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import type { SessionInfo } from '@/hermes'
import { deleteSession, setSessionArchived } from '@/hermes'
import { projectAgentsMdPath, readDesktopFileText, writeDesktopFileText } from '@/lib/desktop-fs'
import { cn } from '@/lib/utils'
import type { ComposerAttachment } from '@/store/composer'
import { $gateway } from '@/store/gateway'
import { $pinnedSessionIds, pinSession, unpinSession } from '@/store/layout'
import { $projects, getProject } from '@/store/projects'
import { $busy, $currentModel, $currentProvider, $gatewayState, $sessions, sessionPinId, setSessions } from '@/store/session'

interface ProjectPageProps {
  projectId: string
  onStartSession?: (projectPath: string) => void
  onSubmit?: (value: string, options?: { attachments?: ComposerAttachment[]; fromQueue?: boolean }) => Promise<boolean> | boolean
  onCancel?: () => Promise<void> | void
  onPickFiles?: () => void
  onPickFolders?: () => void
  onPickImages?: () => void
  onPasteClipboardImage?: () => void
  onAttachImageBlob?: (blob: Blob) => Promise<boolean | void> | boolean | void
  onAttachDroppedItems?: (candidates: DroppedFile[]) => Promise<boolean | void> | boolean | void
  onRemoveAttachment?: (id: string) => void
}

export function ProjectPageView({
  projectId,
  onStartSession,
  onSubmit,
  onCancel,
  onPickFiles,
  onPickFolders,
  onPickImages,
  onPasteClipboardImage,
  onAttachImageBlob,
  onAttachDroppedItems,
  onRemoveAttachment
}: ProjectPageProps) {
  useStore($projects)
  const project = getProject(projectId)

  if (!project) {
    return (
      <div className="flex h-full w-full items-center justify-center text-sm text-muted-foreground">
        Project not found
      </div>
    )
  }

  return (
    <div className="flex h-full w-full min-w-0 overflow-hidden">
      <ProjectMainPanel
        onAttachDroppedItems={onAttachDroppedItems}
        onAttachImageBlob={onAttachImageBlob}
        onCancel={onCancel}
        onPasteClipboardImage={onPasteClipboardImage}
        onPickFiles={onPickFiles}
        onPickFolders={onPickFolders}
        onPickImages={onPickImages}
        onRemoveAttachment={onRemoveAttachment}
        onSubmit={onSubmit}
        projectId={projectId}
      />
      <ProjectSettingsPanel
        onStartSession={onStartSession}
        projectId={projectId}
      />
    </div>
  )
}

function ProjectMainPanel({
  onAttachDroppedItems,
  onAttachImageBlob,
  onCancel,
  onPasteClipboardImage,
  onPickFiles,
  onPickFolders,
  onPickImages,
  onRemoveAttachment,
  onSubmit,
  projectId
}: {
  onAttachDroppedItems?: (candidates: DroppedFile[]) => Promise<boolean | void> | boolean | void
  onAttachImageBlob?: (blob: Blob) => Promise<boolean | void> | boolean | void
  onCancel?: () => Promise<void> | void
  onPasteClipboardImage?: () => void
  onPickFiles?: () => void
  onPickFolders?: () => void
  onPickImages?: () => void
  onRemoveAttachment?: (id: string) => void
  onSubmit?: (value: string, options?: { attachments?: ComposerAttachment[]; fromQueue?: boolean }) => Promise<boolean> | boolean
  projectId: string
}) {
  useStore($projects)
  const project = getProject(projectId)
  const allSessions = useStore($sessions)
  const busy = useStore($busy)
  const gatewayState = useStore($gatewayState)
  const currentModel = useStore($currentModel)
  const currentProvider = useStore($currentProvider)
  const gateway = useStore($gateway)
  const pinnedSessionIds = useStore($pinnedSessionIds)
  const navigate = useNavigate()

  const projectSessions = useMemo<SessionInfo[]>(
    () => (project ? allSessions.filter(s => s.cwd?.startsWith(project.path)) : []),
    [allSessions, project]
  )

  const gatewayOpen = gatewayState === 'open'

  const chatBarState = useMemo(
    () => ({
      model: {
        model: currentModel,
        provider: currentProvider,
        canSwitch: gatewayOpen,
        loading: !gatewayOpen || (!currentModel && !currentProvider)
      },
      tools: { enabled: true, label: 'Add context' },
      voice: { enabled: true, active: false }
    }),
    [currentModel, currentProvider, gatewayOpen]
  )

  const handleSubmit = useMemo(
    () =>
      (value: string, options?: { attachments?: ComposerAttachment[]; fromQueue?: boolean }) =>
        onSubmit?.(value, options) ?? false,
    [onSubmit]
  )

  const [editOpen, setEditOpen] = useState(false)

  if (!project) {
    return null
  }

  return (
    <div className="flex min-w-0 flex-1 flex-col overflow-hidden pt-(--titlebar-height)">
      {/* Header: folder icon + title + description */}
      <div className="flex shrink-0 items-center gap-3 px-6 pb-4 pt-6">
        <Codicon
          className="shrink-0 text-(--ui-text-tertiary)"
          name="folder-opened"
          size="1.75rem"
        />
        <div className="min-w-0 flex-1">
          <p className="truncate text-base font-semibold text-foreground" title={project.title}>
            {project.title}
          </p>
          {project.description && (
            <p className="truncate text-xs text-muted-foreground" title={project.description}>
              {project.description}
            </p>
          )}
        </div>
        <Button
          className="size-6 shrink-0"
          onClick={() => setEditOpen(true)}
          size="icon"
          variant="ghost"
        >
          <Codicon name="edit" size="1rem" />
        </Button>
      </div>

      <CreateProjectDialog
        onClose={() => setEditOpen(false)}
        onCreated={() => undefined}
        open={editOpen}
        project={project}
      />

      {/* Sessions section */}
      <div className="shrink-0 px-6 py-2">
        <span className="text-[0.6875rem] font-medium uppercase tracking-wide text-(--ui-text-tertiary)">
          Sessions
        </span>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-3">
        {projectSessions.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <p className="text-sm text-muted-foreground">
              Start a session in <span className="font-medium text-foreground">{project.title}</span>
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-px">
            {projectSessions.map(session => {
              const pinId = sessionPinId(session)
              const isPinned = pinnedSessionIds.includes(pinId)

              return (
                <SidebarSessionRow
                  isPinned={isPinned}
                  isSelected={false}
                  isWorking={false}
                  key={session.id}
                  onArchive={() => {
                    void setSessionArchived(session.id, true, session.profile).then(() =>
                      setSessions(prev => prev.filter(s => s.id !== session.id))
                    )
                  }}
                  onDelete={() => {
                    void deleteSession(session.id, session.profile).then(() =>
                      setSessions(prev => prev.filter(s => s.id !== session.id))
                    )
                  }}
                  onPin={() => (isPinned ? unpinSession(pinId) : pinSession(pinId))}
                  onResume={() => navigate(sessionRoute(session.id))}
                  session={session}
                />
              )
            })}
          </div>
        )}
      </div>

      {/* ChatBar — relative container so absolute-positioned ChatBar anchors here.
           No overflow-hidden: the ChatBar's text input sits near the container's
           top edge; clipping it would make it unclickable. Width is self-constrained
           by the ChatBar's own w-[min(--composer-width, calc(100%-2rem))]. */}
      <div
        className="relative shrink-0"
        style={{ height: 'calc(var(--composer-measured-height) + 1.25rem)' }}
      >
        <ChatRuntimeBoundary
          busy={busy}
          onCancel={onCancel ?? (() => undefined)}
          onEdit={async () => undefined}
          onReload={async () => undefined}
          onThreadMessagesChange={() => undefined}
          suppressMessages={false}
        >
          <Suspense fallback={<ChatBarFallback />}>
            <ChatBar
              busy={busy}
              cwd={project.path}
              disabled={!gatewayOpen}
              gateway={gateway}
              onAttachDroppedItems={onAttachDroppedItems}
              onAttachImageBlob={onAttachImageBlob}
              onCancel={onCancel ?? (() => undefined)}
              onPasteClipboardImage={onPasteClipboardImage}
              onPickFiles={onPickFiles}
              onPickFolders={onPickFolders}
              onPickImages={onPickImages}
              onRemoveAttachment={onRemoveAttachment}
              onSubmit={handleSubmit}
              sessionId={null}
              state={chatBarState}
            />
          </Suspense>
        </ChatRuntimeBoundary>
      </div>
    </div>
  )
}

function ProjectSettingsPanel({
  onStartSession,
  projectId
}: {
  onStartSession?: (projectPath: string) => void
  projectId: string
}) {
  useStore($projects)
  const project = getProject(projectId)
  const [instructions, setInstructions] = useState('')
  const [instructionsLoaded, setInstructionsLoaded] = useState(false)
  const [instructionsSaving, setInstructionsSaving] = useState(false)
  const [instructionsDialogOpen, setInstructionsDialogOpen] = useState(false)

  const projectPath = project?.path ?? ''

  useEffect(() => {
    if (!project || !projectPath) { return }

    setInstructionsLoaded(false)
    readDesktopFileText(projectAgentsMdPath(projectPath))
      .then(result => {
        setInstructions(result.text)
        setInstructionsLoaded(true)
      })
      .catch(() => {
        setInstructions('')
        setInstructionsLoaded(true)
      })
  }, [projectPath, project])

  if (!project) {
    return null
  }

  const saveInstructions = async () => {
    if (!project || instructionsSaving) { return }

    setInstructionsSaving(true)

    try {
      await writeDesktopFileText(projectAgentsMdPath(projectPath), instructions)
    } catch {
      // silent
    } finally {
      setInstructionsSaving(false)
    }
  }

  const saveInstructionsText = async (text: string) => {
    if (!project) { return }

    setInstructionsSaving(true)

    try {
      await writeDesktopFileText(projectAgentsMdPath(projectPath), text)
    } catch {
      // silent
    } finally {
      setInstructionsSaving(false)
    }
  }

  const openFolder = () => {
    // Use hermesDesktop.openExternal with a file:// URL — works cross-platform
    // (Windows, macOS, Linux) via Electron's shell.openPath under the hood.
    const fileUrl = `file://${project.path.replaceAll('\\', '/')}`
    void window.hermesDesktop?.openExternal(fileUrl).catch(() => {})
  }

  const folderName =
    project.path
      .split(/[\\/]+/)
      .filter(Boolean)
      .pop() ?? project.path

  return (
    <aside
      className={cn(
        'flex h-full flex-col overflow-hidden border-l border-(--ui-stroke-secondary) bg-(--ui-sidebar-surface-background) text-(--ui-text-tertiary)',
        'pt-(--titlebar-height)'
      )}
      style={{ maxWidth: 360, minWidth: 200, width: 280 }}
    >
      <div className="flex shrink-0 flex-col gap-0.5 border-b border-(--ui-stroke-secondary) px-3 py-3">
        <p className="truncate text-[0.68rem] text-muted-foreground" title={project.path}>
          {project.path.split('/').slice(-2).join('/')}
        </p>
      </div>

      <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-3 py-3">
        <div className="grid gap-1.5">
          <div
            className="relative rounded-xl border border-(--ui-stroke-secondary) bg-(--ui-bg-quinary) p-3"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-foreground">Instructions</span>
              <Button
                className="size-5"
                disabled={!instructionsLoaded}
                onClick={() => setInstructionsDialogOpen(true)}
                size="icon"
                variant="ghost"
              >
                <Codicon name="edit" size="0.75rem" />
              </Button>
            </div>
            <p className="mt-1.5 text-[0.6875rem] leading-4 text-(--ui-text-tertiary)">
              {instructions
                ? instructions.length > 150 ? `${instructions.slice(0, 150)}...` : instructions
                : 'No instructions set'}
            </p>
          </div>
          <EditInstructionsDialog
            onClose={() => setInstructionsDialogOpen(false)}
            onSave={text => {
              setInstructions(text)
              setInstructionsDialogOpen(false)
              void saveInstructionsText(text)
            }}
            open={instructionsDialogOpen}
            projectTitle={project.title}
            value={instructions}
          />
        </div>

      </div>

      <div className="shrink-0 border-t border-(--ui-stroke-secondary) px-3 py-3">
        <Button className="w-full justify-start gap-1.5 text-xs" onClick={openFolder} size="sm" variant="outline">
          <Codicon name="folder-opened" size="0.875rem" />
          Open in Explorer
        </Button>
      </div>
    </aside>
  )
}

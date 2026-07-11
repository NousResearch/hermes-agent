import { useStore } from '@nanostores/react'
import { type ComponentProps, useEffect, useRef, useState } from 'react'

import { TreeSkeleton } from '@/components/chat/skeletons'
import { ErrorBoundary } from '@/components/error-boundary'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { useDelayedTrue } from '@/hooks/use-delayed-true'
import { useI18n } from '@/i18n'
import { createDesktopFile, createDesktopFolder, readDesktopDir, selectDesktopPaths } from '@/lib/desktop-fs'
import { normalizeOrLocalPreviewTarget } from '@/lib/local-preview'
import { cn } from '@/lib/utils'
import { $panesFlipped } from '@/store/layout'
import { notify, notifyError } from '@/store/notifications'
import { setCurrentSessionPreviewTarget } from '@/store/preview'
import { $activeSessionId, $connection, $currentCwd, $selectedStoredSessionId } from '@/store/session'
import { notifyWorkspaceChanged } from '@/store/workspace-events'

import { SidebarPanelLabel } from '../shell/sidebar-label'

import {
  browserBack,
  browserForward,
  browserNavigate,
  browserParentPath,
  browserSessionRoot,
  browserUp,
  useBrowserWorkspace
} from './files/browser-workspace'
import { ProjectTree } from './files/tree'
import { useProjectTree } from './files/use-project-tree'

interface RightSidebarPaneProps {
  onActivateFile: (path: string) => void
  onActivateFolder: (path: string) => void
}

export function RightSidebarPane({ onActivateFile, onActivateFolder }: RightSidebarPaneProps) {
  const { t } = useI18n()
  const r = t.rightSidebar
  const panesFlipped = useStore($panesFlipped)
  const currentCwd = useStore($currentCwd).trim()
  const activeSessionId = useStore($activeSessionId)
  const selectedStoredSessionId = useStore($selectedStoredSessionId)
  const connection = useStore($connection)
  const connectionKey = `${connection?.mode || 'local'}:${connection?.profile || ''}:${connection?.baseUrl || ''}`
  const sessionKey = selectedStoredSessionId || activeSessionId || 'detached'
  const workspace = useBrowserWorkspace(currentCwd, connectionKey, sessionKey)
  const browserCwd = workspace.location
  const hasWorkspace = Boolean(browserCwd || currentCwd)

  const {
    collapseAll,
    collapseNonce,
    data,
    effectiveCwd,
    loadChildren,
    openState,
    refreshRoot,
    rootError,
    rootLoading,
    setNodeOpen
  } = useProjectTree(browserCwd, sessionKey)

  const canCollapse = Object.values(openState).some(Boolean)

  const previewFile = async (path: string) => {
    try {
      const preview = await normalizeOrLocalPreviewTarget(path, effectiveCwd || undefined)

      if (!preview) {
        throw new Error(r.couldNotPreview(path))
      }

      setCurrentSessionPreviewTarget(preview, 'file-browser', path)
    } catch (error) {
      notifyError(error, r.previewUnavailable)
    }
  }

  const chooseFolder = async () => {
    try {
      const [picked] = await selectDesktopPaths({ directories: true, multiple: false })

      if (picked) {
        browserNavigate(picked)
      }
    } catch (error) {
      notifyError(error, r.couldNotOpenLocation)
    }
  }

  return (
    <aside
      aria-label={r.aria}
      className={cn(
        'before:pointer-events-none relative flex h-full w-full min-w-0 flex-col overflow-hidden border-(--ui-stroke-secondary) bg-(--ui-sidebar-surface-background) pt-(--titlebar-height) text-(--ui-text-tertiary)',
        panesFlipped
          ? 'border-r shadow-[inset_-0.0625rem_0_0_color-mix(in_srgb,white_18%,transparent)]'
          : 'border-l shadow-[inset_0.0625rem_0_0_color-mix(in_srgb,white_18%,transparent)]'
      )}
    >
      <FilesystemTab
        browserState={workspace}
        canCollapse={canCollapse}
        collapseNonce={collapseNonce}
        cwd={effectiveCwd}
        data={data}
        error={rootError}
        hasWorkspace={hasWorkspace}
        loading={rootLoading}
        onActivateFile={onActivateFile}
        onActivateFolder={onActivateFolder}
        onChooseFolder={() => void chooseFolder()}
        onCollapseAll={collapseAll}
        onLoadChildren={loadChildren}
        onNodeOpenChange={setNodeOpen}
        onPreviewFile={previewFile}
        onRefresh={() => void refreshRoot()}
        openState={openState}
      />
    </aside>
  )
}

interface FilesystemTabProps extends FileTreeBodyProps {
  browserState: ReturnType<typeof useBrowserWorkspace>
  canCollapse: boolean
  hasWorkspace: boolean
  onChooseFolder: () => void
  onCollapseAll: () => void
  onRefresh: () => void
}

const HEADER_ACTION_CLASS =
  'text-sidebar-foreground/70 hover:bg-sidebar-accent! hover:text-sidebar-accent-foreground! focus-visible:ring-sidebar-ring'

function FilesystemTab({
  browserState,
  canCollapse,
  collapseNonce,
  cwd,
  data,
  error,
  hasWorkspace,
  loading,
  onActivateFile,
  onActivateFolder,
  onChooseFolder,
  onCollapseAll,
  onLoadChildren,
  onNodeOpenChange,
  onPreviewFile,
  onRefresh,
  openState
}: FilesystemTabProps) {
  const { t } = useI18n()
  const r = t.rightSidebar
  const [createKind, setCreateKind] = useState<'file' | 'folder' | null>(null)

  if (!hasWorkspace) {
    return (
      <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-3 px-4">
        <PaneEmptyState label={r.noProjectOpen} />
        <Button onClick={onChooseFolder} size="sm" variant="outline">
          {r.chooseFolder}
        </Button>
      </div>
    )
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <RightSidebarSectionHeader className="gap-0.5 px-1.5">
        <WorkspaceLocationControls browserState={browserState} cwd={cwd || browserState.location} />
        <Button
          aria-label={r.newFile}
          className={HEADER_ACTION_CLASS}
          onClick={() => setCreateKind('file')}
          size="icon-xs"
          title={r.newFile}
          variant="ghost"
        >
          <Codicon name="new-file" size="0.8125rem" />
        </Button>
        <Button
          aria-label={r.newFolder}
          className={HEADER_ACTION_CLASS}
          onClick={() => setCreateKind('folder')}
          size="icon-xs"
          title={r.newFolder}
          variant="ghost"
        >
          <Codicon name="new-folder" size="0.8125rem" />
        </Button>
        <Button
          aria-label={r.refreshTree}
          className={HEADER_ACTION_CLASS}
          disabled={loading}
          onClick={onRefresh}
          size="icon-xs"
          title={r.refreshTree}
          variant="ghost"
        >
          <Codicon name="refresh" size="0.8125rem" spinning={loading} />
        </Button>
        <Button
          aria-label={r.collapseAll}
          className={cn(HEADER_ACTION_CLASS, !canCollapse && 'pointer-events-none opacity-0')}
          disabled={!canCollapse}
          onClick={onCollapseAll}
          size="icon-xs"
          title={r.collapseAll}
          variant="ghost"
        >
          <Codicon name="collapse-all" size="0.8125rem" />
        </Button>
      </RightSidebarSectionHeader>
      <FileTreeBody
        collapseNonce={collapseNonce}
        cwd={cwd}
        data={data}
        error={error}
        loading={loading}
        onActivateFile={onActivateFile}
        onActivateFolder={onActivateFolder}
        onChooseFolder={onChooseFolder}
        onLoadChildren={onLoadChildren}
        onNodeOpenChange={onNodeOpenChange}
        onPreviewFile={onPreviewFile}
        onRetry={onRefresh}
        openState={openState}
      />
      <CreateEntryDialog cwd={cwd || browserState.location} kind={createKind} onClose={() => setCreateKind(null)} />
    </div>
  )
}

function compactLocation(path: string): string {
  if (path === '/') {
    return '/'
  }

  const parts = path.split(/[\\/]+/).filter(Boolean)

  return parts.slice(-2).join(' / ') || path
}

function WorkspaceLocationControls({
  browserState,
  cwd
}: {
  browserState: ReturnType<typeof useBrowserWorkspace>
  cwd: string
}) {
  const { t } = useI18n()
  const r = t.rightSidebar
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(cwd)
  const [validationError, setValidationError] = useState<string | null>(null)
  const [validating, setValidating] = useState(false)
  const inputRef = useRef<HTMLInputElement | null>(null)
  const parent = browserParentPath(browserState.location)

  const beginEditing = () => {
    setDraft(cwd)
    setValidationError(null)
    setEditing(true)
  }

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'l') {
        event.preventDefault()
        beginEditing()
      }
    }

    window.addEventListener('keydown', onKeyDown)

    return () => window.removeEventListener('keydown', onKeyDown)
  })

  useEffect(() => {
    if (editing) {
      inputRef.current?.focus()
    }
  }, [editing])

  const commit = async () => {
    const candidate = draft.trim()

    if (!candidate) {
      return
    }

    if (candidate === browserState.location) {
      setEditing(false)

      return
    }

    setValidating(true)
    setValidationError(null)

    try {
      const result = await readDesktopDir(candidate)

      if (result.error) {
        throw new Error(result.error)
      }

      browserNavigate(candidate)
      setEditing(false)
    } catch {
      setValidationError(r.couldNotOpenLocation)
    } finally {
      setValidating(false)
    }
  }

  return (
    <>
      <Button
        aria-label={r.back}
        className={HEADER_ACTION_CLASS}
        disabled={browserState.back.length === 0}
        onClick={browserBack}
        size="icon-xs"
        title={r.back}
        variant="ghost"
      >
        <Codicon name="arrow-left" size="0.75rem" />
      </Button>
      <Button
        aria-label={r.forward}
        className={HEADER_ACTION_CLASS}
        disabled={browserState.forward.length === 0}
        onClick={browserForward}
        size="icon-xs"
        title={r.forward}
        variant="ghost"
      >
        <Codicon name="arrow-right" size="0.75rem" />
      </Button>
      <Button
        aria-label={r.up}
        className={HEADER_ACTION_CLASS}
        disabled={!browserState.location || parent === browserState.location}
        onClick={browserUp}
        size="icon-xs"
        title={r.up}
        variant="ghost"
      >
        <Codicon name="arrow-up" size="0.75rem" />
      </Button>
      <Button
        aria-label={r.sessionRoot}
        className={HEADER_ACTION_CLASS}
        disabled={!browserState.sessionRoot || browserState.sessionRoot === browserState.location}
        onClick={browserSessionRoot}
        size="icon-xs"
        title={r.sessionRoot}
        variant="ghost"
      >
        <Codicon name="home" size="0.75rem" />
      </Button>
      <div className="relative min-w-0 flex-1">
        {editing ? (
          <input
            aria-invalid={Boolean(validationError)}
            aria-label={r.location}
            autoCapitalize="off"
            autoComplete="off"
            autoCorrect="off"
            className="h-5 w-full rounded-sm border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) px-1.5 text-[0.66rem] text-foreground outline-none focus:border-primary"
            disabled={validating}
            onChange={event => setDraft(event.target.value)}
            onKeyDown={event => {
              if (event.key === 'Enter') {
                event.preventDefault()
                void commit()
              } else if (event.key === 'Escape') {
                event.preventDefault()
                setEditing(false)
                setValidationError(null)
              }
            }}
            ref={inputRef}
            spellCheck={false}
            value={draft}
          />
        ) : (
          <button
            aria-label={r.currentLocation}
            className="h-5 w-full truncate rounded-sm px-1 text-left text-[0.66rem] font-medium text-(--ui-text-secondary) hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
            onClick={beginEditing}
            title={cwd}
            type="button"
          >
            {compactLocation(cwd)}
          </button>
        )}
        {validationError && editing ? (
          <div
            className="absolute left-0 top-full z-20 mt-1 rounded border border-destructive/30 bg-(--ui-bg-elevated) px-2 py-1 text-[0.65rem] text-destructive shadow-nous"
            role="alert"
          >
            {validationError}
          </div>
        ) : null}
      </div>
    </>
  )
}

function CreateEntryDialog({
  cwd,
  kind,
  onClose
}: {
  cwd: string
  kind: 'file' | 'folder' | null
  onClose: () => void
}) {
  const { t } = useI18n()
  const r = t.rightSidebar
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (kind) {
      setName('')
      setError(null)
      setBusy(false)
    }
  }, [kind])

  const submit = async () => {
    const nextName = name.trim()

    if (!kind || !nextName || busy) {
      return
    }

    setBusy(true)
    setError(null)

    try {
      if (kind === 'file') {
        await createDesktopFile(cwd, nextName)
      } else {
        await createDesktopFolder(cwd, nextName)
      }

      notifyWorkspaceChanged()
      notify({ kind: 'success', message: kind === 'file' ? r.createdFile(nextName) : r.createdFolder(nextName) })
      onClose()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : t.errors.genericFailure)
      setBusy(false)
    }
  }

  const label = kind === 'file' ? r.fileName : r.folderName
  const action = kind === 'file' ? r.createFile : r.createFolder

  return (
    <Dialog onOpenChange={open => !open && !busy && onClose()} open={Boolean(kind)}>
      <DialogContent className="max-w-sm">
        <DialogHeader>
          <DialogTitle>{action}</DialogTitle>
          <p className="truncate font-mono text-[0.7rem] text-(--ui-text-secondary)" title={cwd}>
            {cwd}
          </p>
        </DialogHeader>
        <form
          className="grid gap-3"
          onSubmit={event => {
            event.preventDefault()
            void submit()
          }}
        >
          <input
            aria-label={label}
            autoFocus
            className="h-8 rounded-md border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) px-2 text-sm outline-none focus:border-primary"
            disabled={busy}
            onChange={event => setName(event.target.value)}
            value={name}
          />
          {error ? (
            <div className="text-xs text-destructive" role="alert">
              {error}
            </div>
          ) : null}
          <DialogFooter>
            <Button disabled={busy} onClick={onClose} type="button" variant="ghost">
              {t.common.cancel}
            </Button>
            <Button disabled={busy || !name.trim()} type="submit">
              {action}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

export function RightSidebarSectionHeader({ children, className, ...props }: ComponentProps<'div'>) {
  return (
    <div className={cn('group/project-header flex h-7 shrink-0 items-center px-2.5', className)} {...props}>
      {children}
    </div>
  )
}

interface FileTreeBodyProps {
  collapseNonce: number
  cwd: string
  data: ReturnType<typeof useProjectTree>['data']
  error: string | null
  loading: boolean
  onActivateFile: (path: string) => void
  onActivateFolder: (path: string) => void
  onChooseFolder?: () => void
  onLoadChildren: (id: string) => void | Promise<void>
  onNodeOpenChange: (id: string, open: boolean) => void
  onPreviewFile?: (path: string) => void
  /** Force-reload the root. The hook also auto-retries while errored, so this
   *  is the impatient-user path. */
  onRetry?: () => void
  openState: ReturnType<typeof useProjectTree>['openState']
}

function FileTreeBody({
  collapseNonce,
  cwd,
  data,
  error,
  loading,
  onActivateFile,
  onActivateFolder,
  onChooseFolder,
  onLoadChildren,
  onNodeOpenChange,
  onPreviewFile,
  onRetry,
  openState
}: FileTreeBodyProps) {
  const { t } = useI18n()
  const r = t.rightSidebar
  // Stay blank for a beat, then skeleton — so a fast project switch doesn't
  // flash a jarring loading state.
  const showSkeleton = useDelayedTrue(loading && data.length === 0)

  if (!cwd) {
    return <EmptyState body={r.noProjectBody} title={r.noProjectTitle} />
  }

  if (error) {
    return (
      <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-2 px-4 text-center">
        <EmptyState body={r.unreadableBody(error)} title={r.unreadableTitle} />
        {onRetry && (
          <button
            className="text-[0.68rem] font-medium text-muted-foreground transition hover:text-foreground"
            onClick={onRetry}
            type="button"
          >
            {r.tryAgain}
          </button>
        )}
        {onChooseFolder && /EACCES|EPERM|permission/i.test(error) ? (
          <Button onClick={onChooseFolder} size="sm" variant="outline">
            {r.chooseFolder}
          </Button>
        ) : null}
      </div>
    )
  }

  if (loading && data.length === 0) {
    return showSkeleton ? <FileTreeLoadingState /> : <div className="min-h-0 flex-1" />
  }

  if (data.length === 0) {
    return <EmptyState body={r.emptyBody} title={r.emptyTitle} />
  }

  return (
    <ErrorBoundary
      fallback={({ reset }) => (
        <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-2 px-4 text-center">
          <EmptyState body={r.treeErrorBody} title={r.treeErrorTitle} />
          <button
            className="text-[0.68rem] font-medium text-muted-foreground transition hover:text-foreground"
            onClick={reset}
            type="button"
          >
            {r.tryAgain}
          </button>
        </div>
      )}
      key={cwd}
      label="file-tree"
    >
      <ProjectTree
        collapseNonce={collapseNonce}
        cwd={cwd}
        data={data}
        onActivateFile={onActivateFile}
        onActivateFolder={onActivateFolder}
        onLoadChildren={onLoadChildren}
        onNodeOpenChange={onNodeOpenChange}
        onPreviewFile={onPreviewFile}
        openState={openState}
      />
    </ErrorBoundary>
  )
}

function FileTreeLoadingState() {
  const { t } = useI18n()

  return (
    <div aria-label={t.rightSidebar.loadingTree} className="min-h-0 flex-1" role="status">
      <TreeSkeleton />
    </div>
  )
}

// Terse pane empty state ("No files" / "No diffs"): the panel label itself —
// same uppercase/tracking + dither dot — just muted instead of theme-primary,
// centered. Shared by the file tree and review panes so both read identically.
export function PaneEmptyState({ label }: { label: string }) {
  return (
    <div className="flex min-h-0 flex-1 items-center justify-center px-4">
      <SidebarPanelLabel className="pl-0 text-(--ui-text-quaternary)">{label}</SidebarPanelLabel>
    </div>
  )
}

// Richer empty/error state (title + body) for the file tree's read failures.
export function EmptyState({ body, title }: { body: string; title?: string }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-1 px-4 text-center">
      {title && (
        <div className="text-[0.7rem] font-semibold uppercase tracking-[0.07em] text-muted-foreground/75">{title}</div>
      )}
      <div className="text-[0.68rem] leading-relaxed text-muted-foreground/65">{body}</div>
    </div>
  )
}

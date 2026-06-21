import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import { ErrorBoundary } from '@/components/error-boundary'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Loader } from '@/components/ui/loader'
import { useI18n } from '@/i18n'
import { selectDesktopPaths } from '@/lib/desktop-fs'
import { normalizeOrLocalPreviewTarget } from '@/lib/local-preview'
import { cn } from '@/lib/utils'
import { $panesFlipped } from '@/store/layout'
import { notifyError } from '@/store/notifications'
import { setCurrentSessionPreviewTarget } from '@/store/preview'
import { $currentCwd } from '@/store/session'

import { SidebarPanelLabel } from '../shell/sidebar-label'

import { RemoteFolderPicker } from './files/remote-picker'
import { ProjectTree } from './files/tree'
import { useProjectTree } from './files/use-project-tree'

interface RightSidebarPaneProps {
  onActivateFile: (path: string) => void
  onActivateFolder: (path: string) => void
  onChangeCwd: (path: string) => Promise<void> | void
}

export function RightSidebarPane({ onActivateFile, onActivateFolder, onChangeCwd }: RightSidebarPaneProps) {
  const { t } = useI18n()
  const r = t.rightSidebar
  const panesFlipped = useStore($panesFlipped)
  const currentCwd = useStore($currentCwd).trim()
  const hasCwd = currentCwd.length > 0

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
  } = useProjectTree(currentCwd)

  const cwdName = hasCwd
    ? (effectiveCwd
        .split(/[\\/]+/)
        .filter(Boolean)
        .pop() ?? effectiveCwd)
    : r.noFolderSelected

  const canCollapse = Object.values(openState).some(Boolean)

  const chooseFolder = async () => {
    const selected = await selectDesktopPaths({
      defaultPath: hasCwd ? effectiveCwd : undefined,
      directories: true,
      multiple: false,
      title: r.changeCwdTitle
    })

    if (selected?.[0]) {
      await onChangeCwd(selected[0])
    }
  }

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

  return (
    <aside
      aria-label={r.aria}
      className={cn(
        'before:pointer-events-none before:absolute before:bottom-0 before:top-(--titlebar-height) before:w-px before:bg-(--ui-stroke-secondary) relative flex h-full w-full min-w-0 flex-col overflow-hidden bg-(--ui-sidebar-surface-background) pt-(--titlebar-height) text-(--ui-text-tertiary)',
        panesFlipped ? 'before:right-0' : 'before:left-0'
      )}
    >
      <RemoteFolderPicker />

      <FilesystemTab
        canCollapse={canCollapse}
        collapseNonce={collapseNonce}
        cwd={effectiveCwd}
        cwdName={cwdName}
        data={data}
        error={rootError}
        hasCwd={hasCwd}
        loading={rootLoading}
        onActivateFile={onActivateFile}
        onActivateFolder={onActivateFolder}
        onChangeFolder={chooseFolder}
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
  canCollapse: boolean
  cwdName: string
  hasCwd: boolean
  onChangeFolder: () => Promise<void> | void
  onCollapseAll: () => void
  onRefresh: () => void
}

// Sidebar palette + hover-reveal: header actions stay reachable while moving
// from the project label to the action buttons.
const HEADER_ACTION_CLASS =
  'text-sidebar-foreground/70 hover:bg-sidebar-accent! hover:text-sidebar-accent-foreground! focus-visible:ring-sidebar-ring'

const HEADER_ACTION_LABEL_REVEAL = `${HEADER_ACTION_CLASS} pointer-events-none opacity-0 transition-opacity focus-visible:pointer-events-auto focus-visible:opacity-100 group-focus-within/project-header:pointer-events-auto group-focus-within/project-header:opacity-100 group-hover/project-header:pointer-events-auto group-hover/project-header:opacity-100`

function FilesystemTab({
  canCollapse,
  collapseNonce,
  cwd,
  cwdName,
  data,
  error,
  hasCwd,
  loading,
  onActivateFile,
  onActivateFolder,
  onChangeFolder,
  onCollapseAll,
  onLoadChildren,
  onNodeOpenChange,
  onPreviewFile,
  onRefresh,
  openState
}: FilesystemTabProps) {
  const { t } = useI18n()
  const r = t.rightSidebar

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <RightSidebarSectionHeader>
        <div className="flex min-w-0 flex-1">
          <button
            className="flex w-full min-w-0 items-center rounded-md text-left hover:text-(--ui-text-secondary)"
            onClick={() => void onChangeFolder()}
            type="button"
          >
            <SidebarPanelLabel>{cwdName}</SidebarPanelLabel>
          </button>
        </div>
        <Button
          aria-label={r.refreshTree}
          className={HEADER_ACTION_LABEL_REVEAL}
          disabled={!hasCwd || loading}
          onClick={onRefresh}
          size="icon-xs"
          title={r.refreshTree}
          variant="ghost"
        >
          <Codicon name="refresh" size="0.8125rem" spinning={loading} />
        </Button>
        <Button
          aria-label={r.openFolder}
          className={HEADER_ACTION_CLASS}
          onClick={() => void onChangeFolder()}
          size="icon-xs"
          title={r.openFolder}
          variant="ghost"
        >
          <Codicon name="folder-opened" size="0.8125rem" />
        </Button>
        <Button
          aria-label={r.collapseAll}
          className={cn(HEADER_ACTION_CLASS, !canCollapse && 'pointer-events-none opacity-0')}
          disabled={!hasCwd || !canCollapse}
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
        onLoadChildren={onLoadChildren}
        onNodeOpenChange={onNodeOpenChange}
        onPreviewFile={onPreviewFile}
        onRetry={onRefresh}
        openState={openState}
      />
    </div>
  )
}

export function RightSidebarSectionHeader({ children }: { children: ReactNode }) {
  return <div className="group/project-header flex h-7 shrink-0 items-center px-2.5">{children}</div>
}

interface FileTreeBodyProps {
  collapseNonce: number
  cwd: string
  data: ReturnType<typeof useProjectTree>['data']
  error: string | null
  loading: boolean
  onActivateFile: (path: string) => void
  onActivateFolder: (path: string) => void
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
  onLoadChildren,
  onNodeOpenChange,
  onPreviewFile,
  onRetry,
  openState
}: FileTreeBodyProps) {
  const { t } = useI18n()
  const r = t.rightSidebar

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
      </div>
    )
  }

  if (loading && data.length === 0) {
    return <FileTreeLoadingState />
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
    <div aria-label={t.rightSidebar.loadingTree} className="grid min-h-0 flex-1 place-items-center px-3" role="status">
      <Loader
        aria-hidden="true"
        className="size-8 text-(--ui-text-tertiary)"
        pathSteps={180}
        role="presentation"
        strokeScale={0.68}
        type="spiral-search"
      />
    </div>
  )
}

function EmptyState({ body, title }: { body: string; title: string }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-1 px-4 text-center">
      <div className="text-[0.7rem] font-semibold uppercase tracking-[0.07em] text-muted-foreground/75">{title}</div>
      <div className="text-[0.68rem] leading-relaxed text-muted-foreground/65">{body}</div>
    </div>
  )
}

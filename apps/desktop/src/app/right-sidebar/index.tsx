import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import { ErrorBoundary } from '@/components/error-boundary'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Loader } from '@/components/ui/loader'
import { normalizeOrLocalPreviewTarget } from '@/lib/local-preview'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'
import { setCurrentSessionPreviewTarget } from '@/store/preview'
import { $currentBranch, $currentCwd } from '@/store/session'

import { SidebarPanelLabel } from '../shell/sidebar-label'

import { ProjectTree } from './files/tree'
import { useProjectTree } from './files/use-project-tree'
import { $rightSidebarTab, $terminalTakeover, type RightSidebarTabId, setRightSidebarTab } from './store'
import { TerminalSlot } from './terminal/persistent'

interface RightSidebarPaneProps {
  onActivateFile: (path: string) => void
  onActivateFolder: (path: string) => void
  onChangeCwd: (path: string) => Promise<void> | void
}

interface RightSidebarTab {
  icon: string
  id: RightSidebarTabId
  label: string
}

const RIGHT_SIDEBAR_TABS: readonly RightSidebarTab[] = [
  { id: 'files', label: 'File system', icon: 'files' },
  { id: 'terminal', label: 'Terminal', icon: 'terminal' }
]

export function RightSidebarPane({ onActivateFile, onActivateFolder, onChangeCwd }: RightSidebarPaneProps) {
  const activeTab = useStore($rightSidebarTab)
  const terminalTakeover = useStore($terminalTakeover)
  const currentBranch = useStore($currentBranch).trim()
  const currentCwd = useStore($currentCwd).trim()
  const hasCwd = currentCwd.length > 0

  const cwdName = hasCwd
    ? (currentCwd
        .split(/[\\/]+/)
        .filter(Boolean)
        .pop() ?? currentCwd)
    : 'No folder selected'

  const {
    collapseAll,
    collapseNonce,
    data,
    loadChildren,
    openState,
    refreshRoot,
    rootError,
    rootLoading,
    setNodeOpen
  } = useProjectTree(currentCwd)

  const canCollapse = Object.values(openState).some(Boolean)
  const effectiveTab: RightSidebarTabId = terminalTakeover ? 'files' : activeTab

  const chooseFolder = async () => {
    const selected = await window.hermesDesktop?.selectPaths({
      defaultPath: hasCwd ? currentCwd : undefined,
      directories: true,
      multiple: false,
      title: 'Change working directory'
    })

    if (selected?.[0]) {
      await onChangeCwd(selected[0])
    }
  }

  const previewFile = async (path: string) => {
    try {
      const preview = await normalizeOrLocalPreviewTarget(path, currentCwd || undefined)

      if (!preview) {
        throw new Error(`Could not preview ${path}`)
      }

      setCurrentSessionPreviewTarget(preview, 'file-browser', path)
    } catch (error) {
      notifyError(error, 'Preview unavailable')
    }
  }

  const tabs = terminalTakeover ? RIGHT_SIDEBAR_TABS.filter(tab => tab.id !== 'terminal') : RIGHT_SIDEBAR_TABS

  return (
    <aside
      aria-label="Right sidebar"
      className="before:pointer-events-none relative flex h-full w-full min-w-0 flex-col overflow-hidden border-l border-(--ui-stroke-secondary) bg-(--ui-sidebar-surface-background) pt-(--titlebar-height) text-(--ui-text-tertiary) shadow-[inset_0.0625rem_0_0_color-mix(in_srgb,white_18%,transparent)]"
    >
      <RightSidebarChrome activeTab={effectiveTab} branch={currentBranch} tabs={tabs} />

      {effectiveTab === 'terminal' ? (
        <TerminalSlot />
      ) : (
        <FilesystemTab
          canCollapse={canCollapse}
          collapseNonce={collapseNonce}
          cwd={currentCwd}
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
      )}
    </aside>
  )
}

function RightSidebarChrome({
  activeTab,
  branch,
  tabs
}: {
  activeTab: RightSidebarTabId
  branch: string
  tabs: readonly RightSidebarTab[]
}) {
  return (
    <header className="shrink-0 bg-transparent text-[0.75rem]">
      <div className="flex items-center gap-2 px-2.5 py-1">
        <nav aria-label="Right sidebar panels" className="flex min-w-0 items-center gap-1">
          {tabs.map(tab => (
            <button
              aria-label={tab.label}
              aria-pressed={tab.id === activeTab}
              className={cn(
                'grid size-6 shrink-0 place-items-center rounded-lg text-(--ui-text-tertiary) transition-colors hover:bg-(--ui-control-hover-background) hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring active:bg-(--ui-control-active-background) active:text-foreground',
                'data-[active=true]:bg-(--ui-control-active-background) data-[active=true]:text-foreground'
              )}
              data-active={tab.id === activeTab}
              key={tab.id}
              onClick={() => setRightSidebarTab(tab.id)}
              title={tab.label}
              type="button"
            >
              <Codicon name={tab.icon} size="0.875rem" />
            </button>
          ))}
        </nav>
        {branch && (
          <span className="ml-auto flex min-w-0 items-center gap-1 text-[0.6875rem] text-(--ui-text-tertiary)">
            <Codicon className="shrink-0" name="git-branch" size="0.75rem" />
            <span className="truncate">{branch}</span>
          </span>
        )}
      </div>
    </header>
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

const HEADER_ACTION_CLASS =
  'size-6 shrink-0 rounded-md text-sidebar-foreground/70 transition-colors hover:bg-sidebar-accent! hover:text-sidebar-accent-foreground! focus-visible:ring-2 focus-visible:ring-sidebar-ring'

const HEADER_ACTION_REVEAL_CLASS = `${HEADER_ACTION_CLASS} pointer-events-none opacity-0 transition-opacity focus-visible:opacity-100 group-focus-within/project-header:pointer-events-auto group-focus-within/project-header:opacity-100 group-hover/project-header:pointer-events-auto group-hover/project-header:opacity-100`

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
  return (
    <div className="group/project-header flex min-h-0 flex-1 flex-col">
      <RightSidebarSectionHeader>
        <button
          className="flex min-w-0 flex-1 items-center rounded-md text-left hover:text-(--ui-text-secondary)"
          onClick={() => void onChangeFolder()}
          title={hasCwd ? `${cwd} — click to change folder` : 'Open a folder'}
          type="button"
        >
          <SidebarPanelLabel>{cwdName}</SidebarPanelLabel>
        </button>
        <Button
          aria-label="Open folder"
          className={HEADER_ACTION_CLASS}
          onClick={() => void onChangeFolder()}
          size="icon"
          title={hasCwd ? 'Open a different folder' : 'Open a folder'}
          variant="ghost"
        >
          <Codicon name="folder-opened" size="0.8125rem" />
        </Button>
        <Button
          aria-label="Collapse all folders"
          className={HEADER_ACTION_REVEAL_CLASS}
          disabled={!hasCwd || !canCollapse}
          onClick={onCollapseAll}
          size="icon"
          title="Collapse all folders"
          variant="ghost"
        >
          <Codicon name="collapse-all" size="0.8125rem" />
        </Button>
        <Button
          aria-label="Refresh tree"
          className={HEADER_ACTION_REVEAL_CLASS}
          disabled={!hasCwd || loading}
          onClick={onRefresh}
          size="icon"
          title="Refresh tree"
          variant="ghost"
        >
          <Codicon name="refresh" size="0.8125rem" spinning={loading} />
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
        openState={openState}
      />
    </div>
  )
}

export function RightSidebarSectionHeader({ children }: { children: ReactNode }) {
  return <div className="flex h-7 shrink-0 items-center px-2">{children}</div>
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
  openState
}: FileTreeBodyProps) {
  if (!cwd) {
    return <EmptyState body="Set a working directory from the status bar to browse files." title="No project" />
  }

  if (error) {
    return <EmptyState body={`Could not read this folder (${error}).`} title="Unreadable" />
  }

  if (loading && data.length === 0) {
    return <FileTreeLoadingState />
  }

  if (data.length === 0) {
    return <EmptyState body="This folder is empty." title="Empty" />
  }

  return (
    <ErrorBoundary
      fallback={({ reset }) => (
        <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-2 px-4 text-center">
          <EmptyState body="The file tree hit an error rendering this folder." title="Tree error" />
          <button
            className="text-[0.68rem] font-medium text-muted-foreground transition hover:text-foreground"
            onClick={reset}
            type="button"
          >
            Try again
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
  return (
    <div aria-label="Loading file tree" className="grid min-h-0 flex-1 place-items-center px-3" role="status">
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

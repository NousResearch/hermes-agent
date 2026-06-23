import {
  closestCenter,
  DndContext,
  type DragEndEvent,
  DragOverlay,
  type DragStartEvent,
  KeyboardSensor,
  PointerSensor,
  useDroppable,
  useSensor,
  useSensors
} from '@dnd-kit/core'
import {
  horizontalListSortingStrategy,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { useStore } from '@nanostores/react'
import { type KeyboardEvent, type PointerEvent, type ReactNode, useEffect, useRef, useState } from 'react'

import { ErrorBoundary } from '@/components/error-boundary'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Loader } from '@/components/ui/loader'
import { Tip } from '@/components/ui/tooltip'
import type { HermesGitStatusEntry } from '@/global'
import { useI18n } from '@/i18n'
import { selectDesktopPaths } from '@/lib/desktop-fs'
import { normalizeOrLocalPreviewTarget } from '@/lib/local-preview'
import { cn } from '@/lib/utils'
import { workspaceKey } from '@/lib/workspace-key'
import { $panesFlipped } from '@/store/layout'
import { notifyError } from '@/store/notifications'
import { setCurrentSessionPreviewTarget } from '@/store/preview'
import { $currentBranch, $currentCwd } from '@/store/session'

import { SidebarPanelLabel } from '../shell/sidebar-label'

import { RemoteFolderPicker } from './files/remote-picker'
import { ProjectTree } from './files/tree'
import { useProjectTree } from './files/use-project-tree'
import { GitPanel } from './git/git-panel'
import { RightSidebarSectionHeader } from './section-header'
import {
  $rightSidebarWorkspaceLayouts,
  $terminalTakeover,
  moveRightSidebarPanel,
  type RightSidebarIsland,
  type RightSidebarIslandId,
  rightSidebarLayoutFor,
  type RightSidebarTabId,
  type RightSidebarWorkspaceLayout,
  selectRightSidebarPanel,
  setRightSidebarTab,
  setSidebarSplitRatio
} from './store'
import { TerminalSlot } from './terminal/persistent'

interface RightSidebarPaneProps {
  onActivateFile: (path: string) => void
  onActivateFolder: (path: string) => void
  onChangeCwd: (path: string) => Promise<void> | void
}

interface RightSidebarTab {
  icon: string
  id: RightSidebarTabId
  labelKey: 'files' | 'sourceControl' | 'terminal'
}

const RIGHT_SIDEBAR_TABS: readonly RightSidebarTab[] = [
  { id: 'files', labelKey: 'files', icon: 'list-tree' },
  { id: 'git', labelKey: 'sourceControl', icon: 'source-control' },
  { id: 'terminal', labelKey: 'terminal', icon: 'terminal' }
]

export function RightSidebarPane({ onActivateFile, onActivateFolder, onChangeCwd }: RightSidebarPaneProps) {
  const { t } = useI18n()
  const r = t.rightSidebar
  const layouts = useStore($rightSidebarWorkspaceLayouts)
  const terminalTakeover = useStore($terminalTakeover)
  const panesFlipped = useStore($panesFlipped)
  const currentBranch = useStore($currentBranch).trim()
  const currentCwd = useStore($currentCwd).trim()
  const hasCwd = currentCwd.length > 0
  const workspaceId = workspaceKey(currentCwd)
  const storedLayout = layouts[workspaceId] ?? rightSidebarLayoutFor(workspaceId)

  const layout = terminalTakeover ? layoutWithoutTerminal(storedLayout) : storedLayout

  useEffect(() => {
    setRightSidebarTab(
      layout.primary.active === 'terminal' || layout.secondary.active === 'terminal'
        ? 'terminal'
        : layout.primary.active ?? layout.secondary.active ?? 'files'
    )
  }, [layout.primary.active, layout.secondary.active])

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

  const openGitChange = async (entry: HermesGitStatusEntry, gitRoot: string) => {
    try {
      const preview = await normalizeOrLocalPreviewTarget(entry.absolutePath, currentCwd || undefined)

      if (!preview) {
        throw new Error(r.couldNotPreview(entry.absolutePath))
      }

      setCurrentSessionPreviewTarget(
        {
          ...preview,
          gitOriginalPath: entry.originalPath ? `${gitRoot}/${entry.originalPath}` : undefined,
          renderMode: 'diff'
        },
        'file-browser',
        entry.absolutePath
      )
    } catch (error) {
      notifyError(error, r.previewUnavailable)
    }
  }

  const revealFolder = async () => {
    const result = await window.hermesDesktop?.openPath?.(currentCwd)

    if (result && !result.ok) {
      notifyError(result.error || r.couldNotOpenFolder, r.couldNotOpenFolder)
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
      <RemoteFolderPicker />

      <SidebarWorkspace
        branch={currentBranch}
        layout={layout}
        renderPanel={panel => {
          if (panel === 'terminal') {
            return terminalTakeover ? null : <TerminalSlot />
          }

          if (panel === 'git') {
            return (
              <GitPanel
                active
                cwd={currentCwd}
                onOpenChange={(entry, gitRoot) => void openGitChange(entry, gitRoot)}
              />
            )
          }

          return (
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
              onRevealFolder={revealFolder}
              openState={openState}
            />
          )
        }}
        terminalTakeover={terminalTakeover}
        workspaceId={workspaceId}
      />
    </aside>
  )
}

function layoutWithoutTerminal(layout: RightSidebarWorkspaceLayout): RightSidebarWorkspaceLayout {
  const clean = (island: RightSidebarIsland): RightSidebarIsland => {
    const tabs: RightSidebarTabId[] = island.tabs.filter(panel => panel !== 'terminal')

    return { active: island.active && tabs.includes(island.active) ? island.active : tabs[0] ?? null, tabs }
  }

  const primary = clean(layout.primary)
  const secondary = clean(layout.secondary)

  if (primary.tabs.length === 0) {
    return { ...layout, primary: secondary, secondary: { active: null, tabs: [] } }
  }

  return { ...layout, primary, secondary }
}

function SidebarWorkspace({
  branch,
  layout,
  renderPanel,
  terminalTakeover,
  workspaceId
}: {
  branch: string
  layout: RightSidebarWorkspaceLayout
  renderPanel: (panel: RightSidebarTabId) => ReactNode
  terminalTakeover: boolean
  workspaceId: string
}) {
  const { t } = useI18n()
  const bodyRef = useRef<HTMLDivElement>(null)
  const [draggingPanel, setDraggingPanel] = useState<RightSidebarTabId | null>(null)

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  )

  const hasSecondary = layout.secondary.tabs.length > 0
  const previewSecondary = Boolean(draggingPanel && !hasSecondary && !terminalTakeover)
  const showSecondary = hasSecondary || previewSecondary
  const rows = showSecondary ? `${layout.splitRatio}fr 4px ${1 - layout.splitRatio}fr` : 'minmax(0, 1fr) 0px 0fr'

  const resizeFromClientY = (clientY: number) => {
    const rect = bodyRef.current?.getBoundingClientRect()

    if (rect && rect.height > 0) {
      setSidebarSplitRatio(workspaceId, (clientY - rect.top) / rect.height)
    }
  }

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId)
    resizeFromClientY(event.clientY)
  }

  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      resizeFromClientY(event.clientY)
    }
  }

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') {
      return
    }

    event.preventDefault()
    setSidebarSplitRatio(workspaceId, layout.splitRatio + (event.key === 'ArrowDown' ? 0.05 : -0.05))
  }

  const islandForPanel = (panel: RightSidebarTabId): RightSidebarIslandId =>
    layout.secondary.tabs.includes(panel) ? 'secondary' : 'primary'

  const onDragStart = (event: DragStartEvent) => {
    setDraggingPanel(String(event.active.id) as RightSidebarTabId)
  }

  const onDragEnd = (event: DragEndEvent) => {
    setDraggingPanel(null)

    if (!event.over) {
      return
    }

    const panel = String(event.active.id) as RightSidebarTabId
    const overId = String(event.over.id)

    const targetIsland: RightSidebarIslandId = overId.startsWith('island:')
      ? (overId.slice('island:'.length) as RightSidebarIslandId)
      : islandForPanel(overId as RightSidebarTabId)

    const targetTabs = layout[targetIsland].tabs
    const targetIndex = overId.startsWith('island:') ? targetTabs.length : targetTabs.indexOf(overId as RightSidebarTabId)

    moveRightSidebarPanel(workspaceId, panel, targetIsland, targetIndex < 0 ? targetTabs.length : targetIndex)
  }

  return (
    <DndContext
      collisionDetection={closestCenter}
      onDragCancel={() => setDraggingPanel(null)}
      onDragEnd={onDragEnd}
      onDragStart={onDragStart}
      sensors={sensors}
    >
      <div
        className="grid min-h-0 flex-1 transition-[grid-template-rows] duration-300 ease-out"
        ref={bodyRef}
        style={{ gridTemplateRows: rows }}
      >
        <PanelIsland
          branch={branch}
          draggingPanel={draggingPanel}
          island={layout.primary}
          islandId="primary"
          renderPanel={renderPanel}
          workspaceId={workspaceId}
        />
        <div
          aria-label={t.rightSidebar.resizePanelSplit}
          aria-orientation="horizontal"
          aria-valuemax={75}
          aria-valuemin={25}
          aria-valuenow={Math.round(layout.splitRatio * 100)}
          className={cn(
            'relative z-10 cursor-row-resize bg-(--ui-stroke-secondary) opacity-0 transition-opacity hover:opacity-100 focus:opacity-100 focus:outline-none',
            showSecondary && 'opacity-60'
          )}
          onKeyDown={onKeyDown}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          role="separator"
          tabIndex={hasSecondary ? 0 : -1}
        />
        <PanelIsland
          branch=""
          draggingPanel={draggingPanel}
          island={layout.secondary}
          islandId="secondary"
          preview={previewSecondary}
          renderPanel={renderPanel}
          workspaceId={workspaceId}
        />
      </div>
      <DragOverlay dropAnimation={{ duration: 220, easing: 'cubic-bezier(0.34, 1.56, 0.64, 1)' }}>
        {draggingPanel ? <PanelIconGhost panel={draggingPanel} /> : null}
      </DragOverlay>
    </DndContext>
  )
}

function PanelIsland({
  branch,
  draggingPanel,
  island,
  islandId,
  preview = false,
  renderPanel,
  workspaceId
}: {
  branch: string
  draggingPanel: RightSidebarTabId | null
  island: RightSidebarIsland
  islandId: RightSidebarIslandId
  preview?: boolean
  renderPanel: (panel: RightSidebarTabId) => ReactNode
  workspaceId: string
}) {
  const { t } = useI18n()
  const { isOver, setNodeRef } = useDroppable({ id: `island:${islandId}` })

  return (
    <section
      className={cn(
        'flex min-h-0 flex-col overflow-hidden transition-[background-color,opacity] duration-200',
        preview && 'opacity-70',
        isOver && draggingPanel && 'bg-(--ui-control-hover-background)'
      )}
      ref={setNodeRef}
    >
      <header
        className={cn(
          'flex h-8 shrink-0 items-center gap-2 border-(--ui-stroke-secondary) px-2.5 text-[0.75rem]',
          islandId === 'secondary' && 'border-b'
        )}
      >
        <SortableContext items={island.tabs} strategy={horizontalListSortingStrategy}>
          <nav aria-label={t.rightSidebar.panelsAria} className="flex min-w-0 items-center gap-1">
            {island.tabs.map(panel => (
              <PanelIcon
                active={island.active === panel}
                key={panel}
                onSelect={() => selectRightSidebarPanel(workspaceId, panel)}
                panel={panel}
              />
            ))}
          </nav>
        </SortableContext>
        {preview && island.tabs.length === 0 && (
          <span className="text-[0.625rem] font-medium uppercase tracking-[0.06em] text-(--ui-text-quaternary)">
            {t.rightSidebar.dropPanelHere}
          </span>
        )}
        {branch && (
          <span className="ml-auto flex min-w-0 items-center gap-1 text-[0.6875rem] text-(--ui-text-tertiary)">
            <Codicon className="shrink-0" name="git-branch" size="0.75rem" />
            <span className="truncate">{branch}</span>
          </span>
        )}
      </header>
      <div className="flex min-h-0 flex-1 flex-col">{island.active ? renderPanel(island.active) : null}</div>
    </section>
  )
}

function PanelIcon({
  active,
  onSelect,
  panel
}: {
  active: boolean
  onSelect: () => void
  panel: RightSidebarTabId
}) {
  const { t } = useI18n()
  const tab = RIGHT_SIDEBAR_TABS.find(item => item.id === panel)!
  const label = t.rightSidebar[tab.labelKey]

  const { attributes, isDragging, listeners, setNodeRef, transform, transition } = useSortable({
    id: panel,
    transition: { duration: 280, easing: 'cubic-bezier(0.34, 1.56, 0.64, 1)' }
  })

  return (
    <Tip label={label}>
      <Button
        {...attributes}
        {...listeners}
        aria-label={label}
        aria-pressed={active}
        className={cn(
          'cursor-grab touch-none text-(--ui-text-tertiary) transition-[background-color,color,box-shadow,opacity] active:cursor-grabbing hover:bg-(--ui-control-hover-background) hover:text-foreground',
          active && 'bg-(--ui-control-active-background) text-foreground',
          isDragging && 'z-20 opacity-45 shadow-lg'
        )}
        onClick={onSelect}
        ref={setNodeRef}
        size="icon-xs"
        style={{ transform: CSS.Transform.toString(transform), transition }}
        variant="ghost"
      >
        <Codicon name={tab.icon} size="0.875rem" />
      </Button>
    </Tip>
  )
}

function PanelIconGhost({ panel }: { panel: RightSidebarTabId }) {
  const tab = RIGHT_SIDEBAR_TABS.find(item => item.id === panel)!

  return (
    <div className="grid size-7 place-items-center rounded-md border border-(--ui-stroke-secondary) bg-(--ui-sidebar-surface-background) text-foreground shadow-xl">
      <Codicon name={tab.icon} size="0.875rem" />
    </div>
  )
}

interface FilesystemTabProps extends FileTreeBodyProps {
  canCollapse: boolean
  cwdName: string
  hasCwd: boolean
  onChangeFolder: () => Promise<void> | void
  onCollapseAll: () => void
  onRefresh: () => void
  onRevealFolder: () => Promise<void> | void
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
  onRevealFolder,
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
          aria-label={r.openInFileManager}
          className={HEADER_ACTION_CLASS}
          disabled={!hasCwd}
          onClick={() => void onRevealFolder()}
          size="icon-xs"
          variant="ghost"
        >
          <Codicon name="go-to-file" size="0.8125rem" />
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

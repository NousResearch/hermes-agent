import { useStore } from '@nanostores/react'
import { useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from 'react'

import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger
} from '@/components/ui/context-menu'
import { Codicon } from '@/components/ui/codicon'
import { DiffCount } from '@/components/ui/diff-count'
import type { HermesGitLogEntry, HermesReviewFile } from '@/global'
import { useI18n } from '@/i18n'
import { isDesktopFsRemoteMode } from '@/lib/desktop-fs'
import { normalizeOrLocalPreviewTarget } from '@/lib/local-preview'
import { cn } from '@/lib/utils'
import { copyFilePath, revealFile, toRelativePath } from '@/store/file-actions'
import { revealFileInTree } from '@/store/layout'
import { notifyError } from '@/store/notifications'
import { setCurrentSessionPreviewTarget } from '@/store/preview'
import {
  $reviewFiles,
  requestRevert,
  selectReviewFile,
  showCommitFileDiff,
  stageReviewFile,
  unstageReviewFile
} from '@/store/review'
import {
  $graphCommitFiles,
  $graphDivergence,
  $graphExpanded,
  $graphHeight,
  $graphLoading,
  $graphLog,
  resetGraphHeight,
  setGraphHeight,
  toggleCommit
} from '@/store/git-graph'
import { $currentCwd } from '@/store/session'
import { showCommitDiff } from '@/store/review'
import { pickRevealLabel } from '../file-actions'

const STATUS_LETTER: Record<string, { letter: string; tone: string }> = {
  A: { letter: 'A', tone: 'text-(--ui-green)' },
  C: { letter: 'C', tone: 'text-(--ui-green)' },
  D: { letter: 'D', tone: 'text-(--ui-red)' },
  M: { letter: 'M', tone: 'text-amber-500/85' },
  R: { letter: 'R', tone: 'text-sky-500/85' },
  T: { letter: 'T', tone: 'text-amber-500/85' },
  U: { letter: 'U', tone: 'text-(--ui-red)' }
}

function pathName(p: string): string {
  return p.split(/[\\/]+/).filter(Boolean).pop() ?? p
}

function pathDir(p: string): string {
  const parts = p.split(/[\\\\/]+/).filter(Boolean)
  return parts.length > 1 ? parts.slice(0, -1).join('/') : ''
}

// Single-click → inline diff in the review panel; double-click → open the file
// in the main editor preview. Uses a 200ms timer so a double-click can cancel
// the single-click (matching the original review pane's file-tree pattern).
function useClickPreview(onSingle: () => void, abs: string) {
  const timer = useRef<null | ReturnType<typeof setTimeout>>(null)

  const single = () => {
    if (timer.current != null) clearTimeout(timer.current)
    timer.current = setTimeout(() => {
      timer.current = null
      onSingle()
    }, 200)
  }

  const double = () => {
    if (timer.current != null) {
      clearTimeout(timer.current)
      timer.current = null
    }
    void (async () => {
      try {
        const target = await normalizeOrLocalPreviewTarget(abs)
        if (target) setCurrentSessionPreviewTarget(target, 'file-browser', abs)
      } catch (err) {
        notifyError(err, '')
      }
    })()
  }

  return { single, double }
}

// Collapsible commit graph section for the review pane. Shows a working-tree
// hollow circle at top (if there are uncommitted changes), then the commit
// history with a visual graph rail. Click a commit to expand its file list.
export function ReviewGraph() {
  const { t } = useI18n()
  const c = t.statusStack.coding
  const log = useStore($graphLog)
  const expanded = useStore($graphExpanded)
  const commitFiles = useStore($graphCommitFiles)
  const loading = useStore($graphLoading)
  const reviewFiles = useStore($reviewFiles)
  const div = useStore($graphDivergence)
  const height = useStore($graphHeight)
  const [collapsed, setCollapsed] = useState(true)

  const hasUncommitted = reviewFiles.length > 0
  const hasDivergence = div && (div.ahead > 0 || div.behind > 0)

  const startResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault()
    const handle = event.currentTarget
    const pointerId = event.pointerId
    const startY = event.clientY
    const startHeight = $graphHeight.get()
    const prevCursor = document.body.style.cursor
    const prevSelect = document.body.style.userSelect
    let active = true

    handle.setPointerCapture?.(pointerId)
    document.body.style.cursor = 'row-resize'
    document.body.style.userSelect = 'none'

    const onMove = (e: PointerEvent) => {
      if (!active) return
      setGraphHeight(startHeight + startY - e.clientY)
    }

    const cleanup = () => {
      if (!active) return
      active = false
      document.body.style.cursor = prevCursor
      document.body.style.userSelect = prevSelect
      handle.releasePointerCapture?.(pointerId)
      window.removeEventListener('pointermove', onMove, true)
      window.removeEventListener('pointerup', cleanup, true)
      window.removeEventListener('pointercancel', cleanup, true)
    }

    window.addEventListener('pointermove', onMove, true)
    window.addEventListener('pointerup', cleanup, true)
    window.addEventListener('pointercancel', cleanup, true)
  }

  return (
    <div className="shrink-0 border-b border-(--ui-stroke-secondary)">
      {!collapsed && (
        <div
          className="group relative -mt-px h-1.5 cursor-row-resize"
          onDoubleClick={resetGraphHeight}
          onPointerDown={startResize}
          role="separator"
        >
          <span className="absolute left-1/2 top-1/2 h-0.5 w-12 -translate-x-1/2 -translate-y-1/2 rounded-full bg-muted-foreground/40 opacity-0 transition-opacity group-hover:opacity-100" />
        </div>
      )}
      <button
        className="flex w-full items-center gap-1.5 px-2.5 py-1.5 text-left text-[0.62rem] font-semibold uppercase tracking-[0.16em] text-(--ui-text-tertiary) transition-colors hover:bg-(--ui-row-hover-background)"
        onClick={() => setCollapsed(prev => !prev)}
        type="button"
      >
        <Codicon
          className="shrink-0 transition-transform"
          name={collapsed ? 'chevron-right' : 'chevron-down'}
          size="0.72rem"
        />
        <span className="flex-1">{c.graph}</span>
        {hasDivergence && (
          <span className="flex shrink-0 items-center gap-1 text-[0.55rem] font-normal normal-case tracking-normal">
            {div.behind > 0 && (
              <span className="flex items-center gap-0.5 text-(--ui-red)" title={`${div.behind} behind upstream`}>
                <Codicon name="arrow-down" size="0.6rem" />
                {div.behind}
              </span>
            )}
            {div.ahead > 0 && (
              <span className="flex items-center gap-0.5 text-(--ui-green)" title={`${div.ahead} ahead of upstream`}>
                <Codicon name="arrow-up" size="0.6rem" />
                {div.ahead}
              </span>
            )}
          </span>
        )}
        {log.length > 0 && (
          <span className="shrink-0 rounded-full bg-(--ui-surface-background-subtle) px-1.5 text-[0.58rem] tabular-nums">{log.length}</span>
        )}
      </button>

      {!collapsed && (
        <div className="overflow-y-auto overflow-x-hidden pb-1" style={{ height }}>
          {log.length === 0 ? (
            <div className="px-2.5 py-2 text-[0.66rem] text-muted-foreground/60">{c.noChanges}</div>
          ) : (
            <div>
              {hasUncommitted && (
                <WorkingTreeRow
                  files={reviewFiles}
                />
              )}
              {log.map((entry, i) => {
                const prevUpstream = i > 0 && log[i - 1].side === 'upstream'
                const currLocal = !entry.side
                const showSeparator = prevUpstream && currLocal

                return (
                  <div key={entry.hash}>
                    {showSeparator && (
                      <div className="flex items-center gap-2 px-2.5 py-0.5">
                        <div className="h-px flex-1 bg-(--ui-stroke-secondary)" />
                        <span className="text-[0.5rem] font-semibold uppercase tracking-wider text-(--ui-text-tertiary)">HEAD</span>
                        <div className="h-px flex-1 bg-(--ui-stroke-secondary)" />
                      </div>
                    )}
                    <GraphRow
                      commitFiles={commitFiles[entry.hash] ?? []}
                      commitLoading={loading.has(entry.hash)}
                      entry={entry}
                      expanded={expanded.has(entry.hash)}
                      isFirst={i === 0 && !hasUncommitted && !entry.side}
                      isUpstream={entry.side === 'upstream'}
                      onToggle={() => void toggleCommit(entry.hash)}
                    />
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Hollow circle at the top representing uncommitted working-tree changes.
// Click to expand and see the list of changed files (from $reviewFiles).
function WorkingTreeRow({ files }: { files: HermesReviewFile[] }) {
  const { t } = useI18n()
  const c = t.statusStack.coding
  const [expanded, setExpanded] = useState(false)

  const added = files.reduce((sum, f) => sum + f.added, 0)
  const removed = files.reduce((sum, f) => sum + f.removed, 0)

  return (
    <div className="group/graph-row relative">
      <div
        className="absolute bg-(--ui-text-tertiary)"
        style={{ left: 11, top: 22, bottom: 0, width: 1, transform: 'translateX(-50%)' }}
      />
      <button
        className="flex w-full cursor-pointer items-center pr-2 text-xs text-(--ui-text-secondary) relative"
        onClick={() => setExpanded(prev => !prev)}
        style={{ height: 22 }}
        title={c.scopeUncommitted}
        type="button"
      >
        <div className="flex w-[22px] shrink-0 justify-center relative z-10">
          <div
            className="rounded-full border-2 border-(--ui-accent) transition-all group-hover/graph-row:scale-110"
            style={{ height: 12, width: 12 }}
          />
        </div>
        <span className="truncate pl-1" title={c.scopeUncommitted}>
          {c.scopeUncommitted}
        </span>
        <div className="flex-1" />
        <DiffCount added={added} className="text-[0.58rem] leading-4" removed={removed} />
      </button>
      {expanded && (
        <div className="relative z-10 pl-[22px]">
          {files.map(f => {
            const s = STATUS_LETTER[f.status] ?? STATUS_LETTER.M
            return (
              <GraphFileContextMenu key={f.path} path={f.path} file={f}>
                <GraphFileRow
                  letter={s.letter}
                  tone={s.tone}
                  name={pathName(f.path)}
                  dir={pathDir(f.path)}
                  title={f.path}
                  onSingle={() => void selectReviewFile(f)}
                  abs={absolutePath(f.path)}
                />
              </GraphFileContextMenu>
            )
          })}
        </div>
      )}
    </div>
  )
}

function GraphRow({ commitFiles, commitLoading, entry, expanded, isFirst, isUpstream, onToggle }: {
  commitFiles: { path: string; status: string }[]
  commitLoading: boolean
  entry: HermesGitLogEntry
  expanded: boolean
  isFirst: boolean
  isUpstream: boolean
  onToggle: () => void
}) {
  return (
    <div className="group/graph-row relative">
      {!isFirst && (
        <div
          className={cn('absolute bg-(--ui-text-tertiary) transition-all', expanded ? 'w-0.5' : 'w-px')}
          style={{ left: 11, top: 0, height: 7, transform: 'translateX(-50%)' }}
        />
      )}
      <div
        className={cn('absolute transition-all', expanded ? 'w-0.5' : 'w-px', isUpstream ? 'bg-sky-500/40' : 'bg-(--ui-text-tertiary)')}
        style={{ left: 11, top: 15, bottom: 0, transform: 'translateX(-50%)' }}
      />
      <GraphCommitContextMenu entry={entry}>
        <button
          className={cn(
            'flex w-full cursor-pointer items-center rounded-md pr-2 text-left text-xs transition-colors hover:bg-(--ui-row-hover-background) relative',
            isUpstream && 'opacity-70'
          )}
          onClick={onToggle}
          style={{ height: 22 }}
          title={entry.hash}
          type="button"
        >
          <div className="flex w-[22px] shrink-0 justify-center relative z-10">
            {isFirst ? (
              <div
                className="rounded-full border-2 border-(--ui-accent) transition-all group-hover/graph-row:scale-110"
                style={{ height: 12, width: 12 }}
              />
            ) : isUpstream ? (
              <div
                className="rounded-full border-2 border-sky-500/60 transition-all group-hover/graph-row:scale-110"
                style={{ height: 10, width: 10 }}
              />
            ) : entry.parents.length > 1 ? (
              <div
                className="relative flex items-center justify-center rounded-full border-2 border-(--ui-text-tertiary) transition-all group-hover/graph-row:scale-110"
                style={{ height: 14, width: 14 }}
              >
                <div className="rounded-full border-2 border-(--ui-text-tertiary)" style={{ height: 6, width: 6 }} />
              </div>
            ) : (
              <div
                className="rounded-full bg-(--ui-text-tertiary) transition-all group-hover/graph-row:scale-150"
                style={{ height: 8, width: 8 }}
              />
            )}
          </div>
          <span className={cn('min-w-0 flex-1 truncate pl-1', isUpstream ? 'text-sky-600 dark:text-sky-400/80' : 'text-(--ui-text-secondary)')} title={entry.message}>
            {entry.message}
          </span>
          {isFirst && (
            <span className="shrink-0 rounded bg-(--ui-accent)/15 px-1 text-[0.52rem] font-semibold uppercase tracking-wider text-(--ui-accent)">
              HEAD
            </span>
          )}
          <span className="shrink-0 font-mono text-[0.58rem] text-(--ui-text-tertiary)">{entry.hash.slice(0, 7)}</span>
          <span className="shrink-0 pl-1 text-[0.58rem] text-(--ui-text-tertiary)">{entry.date}</span>
        </button>
      </GraphCommitContextMenu>
      {expanded && (
        <div className="relative z-10 pl-[22px]">
          {commitLoading ? (
            <div className="px-2 py-1 text-[0.6rem] text-muted-foreground/60">...</div>
          ) : commitFiles.length === 0 ? (
            <div className="px-2 py-1 text-[0.6rem] text-muted-foreground/60">No files</div>
          ) : (
            commitFiles.map(f => {
              const s = STATUS_LETTER[f.status] ?? STATUS_LETTER.M
              return (
                <GraphFileContextMenu key={f.path} path={f.path} hash={entry.hash}>
                  <GraphFileRow
                    letter={s.letter}
                    tone={s.tone}
                    name={pathName(f.path)}
                    dir={pathDir(f.path)}
                    title={f.path}
                    onSingle={() => void showCommitFileDiff(f.path, entry.hash)}
                    abs={absolutePath(f.path)}
                  />
                </GraphFileContextMenu>
              )
            })
          )}
        </div>
      )}
    </div>
  )
}

function GraphFileRow({ letter, tone, name, dir, title, onSingle, abs }: {
  letter: string
  tone: string
  name: string
  dir: string
  title: string
  onSingle: () => void
  abs: string
}) {
  const { single, double } = useClickPreview(onSingle, abs)

  return (
    <div
      className="flex h-5 cursor-pointer items-center gap-1.5 rounded-md px-2 text-[0.66rem] text-(--ui-text-secondary) transition-colors hover:bg-(--ui-row-hover-background) hover:text-foreground"
      onClick={single}
      onDoubleClick={double}
      title={title}
    >
      <span className={cn('w-3 shrink-0 text-center font-bold', tone)}>{letter}</span>
      <span className="min-w-0 flex-1 truncate">{name}</span>
      {dir && <span className="min-w-0 shrink-[9999] truncate text-[0.62rem] text-(--ui-text-tertiary)">{dir}</span>}
    </div>
  )
}

function GraphCommitContextMenu({ children, entry }: { children: ReactNode; entry: HermesGitLogEntry }) {
  const { t } = useI18n()
  const c = t.statusStack.coding

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>{children}</ContextMenuTrigger>
      <ContextMenuContent>
        <ContextMenuItem onSelect={() => void showCommitDiff(entry.hash, entry.message)}>
          {c.openChanges}
        </ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem onSelect={() => void navigator.clipboard.writeText(entry.hash)}>
          {c.copyHash}
        </ContextMenuItem>
        <ContextMenuItem onSelect={() => void navigator.clipboard.writeText(entry.message)}>
          {c.copyMessage}
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  )
}

function absolutePath(relative: string): string {
  if (/^([a-zA-Z]:[\\/]|\/)/.test(relative)) return relative
  const cwd = $currentCwd.get()?.trim().replace(/[\\/]+$/, '')
  return cwd ? `${cwd}/${relative}` : relative
}

function GraphFileContextMenu({ children, path, file, hash }: {
  children: ReactNode
  path: string
  file?: HermesReviewFile
  hash?: string
}) {
  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        <div className="contents">{children}</div>
      </ContextMenuTrigger>
      <GraphFileMenuItems path={path} file={file} hash={hash} />
    </ContextMenu>
  )
}

function GraphFileMenuItems({ path, file, hash }: { path: string; file?: HermesReviewFile; hash?: string }) {
  const { t } = useI18n()
  const c = t.statusStack.coding
  const m = t.fileMenu
  const localFs = !isDesktopFsRemoteMode()
  const abs = absolutePath(path)
  const cwd = $currentCwd.get()?.trim() || undefined

  const openInPreview = () => {
    void (async () => {
      try {
        const target = await normalizeOrLocalPreviewTarget(abs)
        if (target) setCurrentSessionPreviewTarget(target, 'file-browser', abs)
      } catch (err) {
        notifyError(err, t.rightSidebar.previewUnavailable)
      }
    })()
  }

  const openChanges = () => {
    if (file) {
      void selectReviewFile(file)
    } else if (hash) {
      void showCommitFileDiff(path, hash)
    }
  }

  const showOpenChanges = !!file || !!hash

  return (
    <ContextMenuContent>
      {showOpenChanges && <ContextMenuItem onSelect={openChanges}>{c.openChanges}</ContextMenuItem>}
      <ContextMenuItem onSelect={openInPreview}>{c.openFile}</ContextMenuItem>
      {file && (
        <>
          <ContextMenuSeparator />
          <ContextMenuItem
            onSelect={() =>
              void (file.staged ? unstageReviewFile(file.path) : stageReviewFile(file.path)).catch(err =>
                notifyError(err, file.staged ? c.unstage : c.stage)
              )
            }
          >
            {file.staged ? c.unstage : c.stage}
          </ContextMenuItem>
          <ContextMenuItem onSelect={() => requestRevert(file.path)} variant="destructive">
            {c.revert}
          </ContextMenuItem>
        </>
      )}
      <ContextMenuSeparator />
      <ContextMenuItem onSelect={() => revealFileInTree(abs)}>{m.revealInSidebar}</ContextMenuItem>
      {localFs && (
        <ContextMenuItem onSelect={() => void revealFile(abs)}>
          {pickRevealLabel(m.revealFinder, m.revealExplorer, m.revealFileManager)}
        </ContextMenuItem>
      )}
      <ContextMenuSeparator />
      <ContextMenuItem onSelect={() => void copyFilePath(abs)}>{m.copyPath}</ContextMenuItem>
      {cwd && (
        <ContextMenuItem onSelect={() => void copyFilePath(toRelativePath(abs, cwd))}>
          {m.copyRelativePath}
        </ContextMenuItem>
      )}
    </ContextMenuContent>
  )
}

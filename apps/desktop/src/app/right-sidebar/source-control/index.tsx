import { useStore } from '@nanostores/react'
import { type ReactNode, useCallback, useEffect, useImperativeHandle, useMemo, useState, forwardRef } from 'react'

import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from '@/components/ui/context-menu'
import { Codicon } from '@/components/ui/codicon'
import { DiffCount } from '@/components/ui/diff-count'
import type { HermesRepoStatus, HermesReviewFile } from '@/global'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $currentCwd } from '@/store/session'
import {
  $scmCommitFiles,
  $scmCommitFilesLoading,
  $scmExpandedCommits,
  $scmFiles,
  $scmIsRepo,
  $scmLoaded,
  $scmLog,
  $scmStatus,
  clearGraphState,
  refreshAll,
  toggleCommit
} from '@/store/git'

import { SidebarPanelLabel } from '../../shell/sidebar-label'
import { FileEntryContextMenu } from '../file-actions'

const STATUS_LETTER: Record<string, { letter: string; tone: string }> = {
  A: { letter: 'A', tone: 'text-(--ui-green)' },
  C: { letter: 'C', tone: 'text-(--ui-green)' },
  D: { letter: 'D', tone: 'text-(--ui-red)' },
  M: { letter: 'M', tone: 'text-amber-500/85' },
  R: { letter: 'R', tone: 'text-sky-500/85' },
  U: { letter: 'U', tone: 'text-(--ui-red)' },
  '?': { letter: 'U', tone: 'text-muted-foreground/60' }
}

function pathName(p: string): string {
  return p.split(/[\\/]+/).filter(Boolean).pop() ?? p
}

function pathDir(p: string): string {
  const parts = p.split(/[\\/]+/).filter(Boolean)
  return parts.length > 1 ? parts.slice(0, -1).join('/') : ''
}

function absolutePath(cwd: string, file: string): string {
  if (/^([a-zA-Z]:[\\/]|\/)/.test(file)) return file
  return `${cwd.replace(/[\\/]+$/, '')}/${file}`
}

interface SourceControlTabProps {
  cwd: string
  onOpenFile: (path: string) => void
}

export const SourceControlTab = forwardRef<{ refresh: () => void; collapseAll: () => void }, SourceControlTabProps>(function SourceControlTab({ cwd: treeCwd, onOpenFile }: SourceControlTabProps, ref) {
  const { t } = useI18n()
  const c = t.statusStack.coding
  const r = t.rightSidebar
  const sessionCwd = useStore($currentCwd).trim()
  const cwd = (sessionCwd || treeCwd).trim()
  const status = useStore($scmStatus)
  const files = useStore($scmFiles)
  const log = useStore($scmLog)
  const isRepo = useStore($scmIsRepo)
  const loaded = useStore($scmLoaded)
  const expandedCommits = useStore($scmExpandedCommits)
  const commitFiles = useStore($scmCommitFiles)
  const commitFilesLoading = useStore($scmCommitFilesLoading)
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({})
  const [viewMode, setViewMode] = useState<'list' | 'tree'>('list')

  // Initial load + clear graph state on cwd change.
  useEffect(() => {
    if (cwd) refreshAll()
  }, [cwd])

  useEffect(() => {
    clearGraphState()
  }, [cwd])

  const openFile = useCallback((file: HermesReviewFile) => {
    onOpenFile(absolutePath(cwd, file.path))
  }, [cwd, onOpenFile])

  const groups = useMemo(() => {
    const byPath = (a: HermesReviewFile, b: HermesReviewFile) => a.path.localeCompare(b.path)
    const toTree = (list: HermesReviewFile[]) => {
      if (viewMode !== 'tree') return null
      const tree = new Map<string, HermesReviewFile[]>()
      for (const f of list.sort(byPath)) {
        const dir = pathDir(f.path) || '.'
        if (!tree.has(dir)) tree.set(dir, [])
        tree.get(dir)!.push(f)
      }
      return tree
    }

    return [
      { id: 'staged', label: c.staged, files: files.filter(f => f.staged).sort(byPath), tree: toTree(files.filter(f => f.staged)) },
      { id: 'changes', label: c.scopeUncommitted, files: files.filter(f => !f.staged && f.status !== '?').sort(byPath), tree: toTree(files.filter(f => !f.staged && f.status !== '?')) },
      { id: 'untracked', label: c.untracked, files: files.filter(f => !f.staged && f.status === '?').sort(byPath), tree: toTree(files.filter(f => !f.staged && f.status === '?')) },
    ].filter(g => g.files.length > 0)
  }, [c.scopeUncommitted, c.staged, c.untracked, files, viewMode])

  const toggle = useCallback((id: string) => {
    setCollapsed(prev => ({ ...prev, [id]: !prev[id] }))
  }, [])

  useImperativeHandle(ref, () => ({
    refresh: () => refreshAll(),
    collapseAll: () => {
      clearGraphState()
      setCollapsed({ repo: true, changes: true, graph: true })
    }
  }))

  if (!cwd) return <PaneEmptyState label={r.noProjectOpen} />

  const chevron = (id: string) => (
    <Codicon
      className="shrink-0 text-(--ui-text-tertiary) transition-transform"
      name={collapsed[id] ? 'chevron-right' : 'chevron-down'}
      size="0.72rem"
    />
  )

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* REPOSITORY */}
      <SectionHeader collapsed={collapsed.repo} label={r.repository} onToggle={() => toggle('repo')} chevron={chevron('repo')} />
      {!collapsed.repo && (
        <div className="border-b border-(--ui-stroke-secondary) px-2.5 py-2">
          <div className="flex min-w-0 items-center gap-1.5 text-[0.68rem] text-(--ui-text-secondary)">
            <span className="min-w-0 flex-1 truncate font-medium" title={cwd}>
              {cwd.split(/[\\\/]+/).filter(Boolean).pop() ?? cwd}
            </span>
            <span className="flex items-center gap-1 shrink-0 text-(--ui-text-tertiary)">
              <Codicon className="shrink-0" name="git-branch" size="0.72rem" />
              <span className="truncate" title={status?.branch ?? undefined}>
                {status?.detached ? c.detached : status?.branch || c.noBranch}
              </span>
            </span>
          </div>
          {status && (
            <div className="mt-1 flex min-w-0 items-center gap-1.5 text-[0.62rem] text-(--ui-text-tertiary)">
              <RepoSummary status={status} />
            </div>
          )}
        </div>
      )}

      {/* CHANGES */}
      <ContextMenu>
        <ContextMenuTrigger asChild>
          <div>
            <SectionHeader collapsed={collapsed.changes} label={c.scopeUncommitted} onToggle={() => toggle('changes')} chevron={chevron('changes')} />
          </div>
        </ContextMenuTrigger>
        <ContextMenuContent>
          <ContextMenuItem onSelect={() => setViewMode('list')}>
            <Codicon name="list-flat" size="0.875rem" />
            <span>{c.viewAsList}</span>
          </ContextMenuItem>
          <ContextMenuItem onSelect={() => setViewMode('tree')}>
            <Codicon name="list-tree" size="0.875rem" />
            <span>{c.viewAsTree}</span>
          </ContextMenuItem>
        </ContextMenuContent>
      </ContextMenu>
      {!collapsed.changes && (
        <>
          {!isRepo ? (
            <PaneEmptyState label={c.notRepo} />
          ) : files.length === 0 ? (
            <PaneEmptyState label={c.noChanges} />
          ) : (
            <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden px-1 py-1">
              {groups.map(group => (
                <section className="py-1" key={group.id}>
                  <button
                    className="flex w-full items-center px-2 py-1 text-[0.62rem] font-semibold uppercase tracking-[0.16em] text-(--ui-text-tertiary) transition-colors hover:bg-(--ui-row-hover-background)"
                    onClick={() => toggle(`group-${group.id}`)}
                    type="button"
                  >
                    <Codicon
                      className="shrink-0 text-(--ui-text-tertiary) transition-transform"
                      name={collapsed[`group-${group.id}`] ? 'chevron-right' : 'chevron-down'}
                      size="0.72rem"
                    />
                    <span className="flex-1 pl-1">{group.label}</span>
                    <span className="rounded-full bg-(--ui-surface-background-subtle) px-1.5 text-[0.58rem] tabular-nums">{group.files.length}</span>
                  </button>
                  {!collapsed[`group-${group.id}`] && (
                    viewMode === 'tree' && group.tree ? (
                      Array.from(group.tree.entries()).map(([dir, dirFiles]) => (
                        <div key={`${group.id}-${dir}`}>
                          <FileEntryContextMenu
                            isDirectory
                            name={dir}
                            path={absolutePath(cwd, dir)}
                            relativeTo={cwd}
                          >
                            <div className="flex items-center gap-1.5 px-2 py-1 text-[0.66rem] text-(--ui-text-tertiary)">
                              <Codicon name="folder" size="0.75rem" />
                              <span className="truncate">{dir}</span>
                            </div>
                          </FileEntryContextMenu>
                          {dirFiles.map(file => (
                            <FileEntryContextMenu
                              isDirectory={false}
                              key={`${file.staged ? 's' : 'w'}:${file.path}`}
                              name={pathName(file.path)}
                              path={absolutePath(cwd, file.path)}
                              relativeTo={cwd}
                            >
                              <div className="pl-4">
                                <SourceControlRow file={file} onOpenFile={openFile} />
                              </div>
                            </FileEntryContextMenu>
                          ))}
                        </div>
                      ))
                    ) : (
                      group.files.map(file => {
                        const sl = STATUS_LETTER[file.status] ?? STATUS_LETTER.M
                        return (
                          <FileEntryContextMenu
                            isDirectory={false}
                            key={`${file.staged ? 's' : 'w'}:${file.path}`}
                            name={pathName(file.path)}
                            path={absolutePath(cwd, file.path)}
                            relativeTo={cwd}
                          >
                            <button
                              className="group/source-row flex h-6 w-full cursor-pointer select-none items-center gap-1.5 rounded-md px-2 text-left text-xs text-(--ui-text-secondary) transition-colors duration-100 ease-out hover:bg-(--ui-row-hover-background) hover:text-foreground hover:transition-none"
                              onClick={() => openFile(file)}
                              title={file.path}
                              type="button"
                            >
                              <span className="flex min-w-0 flex-1 items-baseline gap-1.5">
                                <span className="min-w-0 shrink truncate" title={pathName(file.path)}>{pathName(file.path)}</span>
                                {pathDir(file.path) && <span className="min-w-0 shrink-[9999] truncate text-[0.68rem] text-(--ui-text-tertiary)" title={pathDir(file.path)}>{pathDir(file.path)}</span>}
                              </span>
                              <DiffCount added={file.added} className="text-[0.62rem] leading-4" removed={file.removed} />
                              <span className={cn('shrink-0 font-mono text-[0.64rem] font-semibold', sl.tone)}>{sl.letter}</span>
                            </button>
                          </FileEntryContextMenu>
                        )
                      })
                    )
                  )}
                </section>
              ))}
            </div>
          )}
        </>
      )}

      {/* GRAPH */}
      <SectionHeader collapsed={collapsed.graph} count={log.length} label={r.graph} onToggle={() => toggle('graph')} chevron={chevron('graph')} />
      {!collapsed.graph && (
        <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden">
          {log.length === 0 ? (
            <PaneEmptyState label={c.noChanges} />
          ) : (
            <div>
              {log.map((entry, i) => (
                <GraphRow
                  commitFiles={commitFiles[entry.hash] ?? []}
                  commitFilesLoading={commitFilesLoading.has(entry.hash)}
                  cwd={cwd}
                  entry={entry}
                  expanded={expandedCommits.has(entry.hash)}
                  isFirst={i === 0}
                  key={entry.hash}
                  onOpenFile={onOpenFile}
                  onToggle={() => void toggleCommit(entry.hash)}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
})

function SectionHeader({ collapsed, count, label, onToggle, chevron }: {
  collapsed: boolean
  count?: number
  label: string
  onToggle: () => void
  chevron: ReactNode
}) {
  return (
    <button
      className="flex w-full items-center gap-1.5 border-b border-(--ui-stroke-secondary) px-2.5 py-1.5 text-left text-[0.62rem] font-semibold uppercase tracking-[0.16em] text-(--ui-text-tertiary) transition-colors hover:bg-(--ui-row-hover-background)"
      onClick={onToggle}
      type="button"
    >
      {chevron}
      <span className="flex-1">{label}</span>
      {count !== undefined && count > 0 && (
        <span className="rounded-full bg-(--ui-surface-background-subtle) px-1.5 text-[0.58rem] tabular-nums">{count}</span>
      )}
    </button>
  )
}

function RepoSummary({ status }: { status: HermesRepoStatus }) {
  const c = useI18n().t.statusStack.coding
  const bits: ReactNode[] = []
  if (status.changed > 0) bits.push(c.changed(status.changed))
  if (status.ahead > 0) bits.push(c.ahead(status.ahead))
  if (status.behind > 0) bits.push(c.behind(status.behind))
  if (bits.length === 0) bits.push(c.clean)
  return <span className="shrink-0 text-[0.62rem] text-(--ui-text-tertiary)">{bits.map((bit, i) => <span key={String(bit)}>{i > 0 ? ' · ' : ''}{bit}</span>)}</span>
}

function PaneEmptyState({ label }: { label: string }) {
  return (
    <div className="flex min-h-0 flex-1 items-center justify-center px-4">
      <SidebarPanelLabel className="pl-0 text-(--ui-text-quaternary)">{label}</SidebarPanelLabel>
    </div>
  )
}

function SourceControlRow({ file, onOpenFile }: {
  file: HermesReviewFile
  onOpenFile: (file: HermesReviewFile) => void
}) {
  const sl = STATUS_LETTER[file.status] ?? STATUS_LETTER.M
  const dir = pathDir(file.path)

  return (
    <button
      className="group/source-row flex h-6 w-full cursor-pointer select-none items-center gap-1.5 rounded-md px-2 text-left text-xs text-(--ui-text-secondary) transition-colors duration-100 ease-out hover:bg-(--ui-row-hover-background) hover:text-foreground hover:transition-none"
      onClick={() => onOpenFile(file)}
      title={file.path}
      type="button"
    >
      <span className="flex min-w-0 flex-1 items-baseline gap-1.5">
        <span className="min-w-0 shrink truncate" title={pathName(file.path)}>{pathName(file.path)}</span>
        {dir && <span className="min-w-0 shrink-[9999] truncate text-[0.68rem] text-(--ui-text-tertiary)" title={dir}>{dir}</span>}
      </span>
      <DiffCount added={file.added} className="text-[0.62rem] leading-4" removed={file.removed} />
      <span className={cn('shrink-0 font-mono text-[0.64rem] font-semibold', sl.tone)}>{sl.letter}</span>
    </button>
  )
}

function GraphRow({ commitFiles, commitFilesLoading, cwd, entry, expanded, isFirst, onOpenFile, onToggle }: {
  commitFiles: { path: string; status: string }[]
  commitFilesLoading: boolean
  cwd: string
  entry: { hash: string; message: string; date: string; parents: string[] }
  expanded: boolean
  isFirst: boolean
  onOpenFile: (path: string) => void
  onToggle: () => void
}) {
  return (
    <div className="group/graph-row relative">
      {/* Two-segment line: top (into circle) + bottom (out of circle).
          Gap in the middle where the circle sits — no masking needed. */}
      {!isFirst && (
        <div
          className={cn('absolute bg-(--ui-text-tertiary) transition-all', expanded ? 'w-0.5' : 'w-px')}
          style={{ left: 11, top: 0, height: 7, transform: 'translateX(-50%)' }}
        />
      )}
      <div
        className={cn('absolute bg-(--ui-text-tertiary) transition-all', expanded ? 'w-0.5' : 'w-px')}
        style={{ left: 11, top: 15, bottom: 0, transform: 'translateX(-50%)' }}
      />
      <button
        className="flex w-full cursor-pointer items-center rounded-md pr-2 text-left text-xs transition-colors hover:bg-(--ui-row-hover-background) relative"
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
        <span className="min-w-0 flex-1 truncate pl-1 text-(--ui-text-secondary)" title={entry.message}>
          {entry.message}
        </span>
        <span className="shrink-0 font-mono text-[0.58rem] text-(--ui-text-tertiary)">{entry.hash.slice(0, 7)}</span>
        <span className="shrink-0 pl-1 text-[0.58rem] text-(--ui-text-tertiary)">{entry.date}</span>
      </button>
      {expanded && (
        <div className="relative z-10 pl-[22px]">
          {commitFilesLoading ? (
            <div className="px-2 py-1 text-[0.6rem] text-muted-foreground/60">...</div>
          ) : commitFiles.length === 0 ? (
            <div className="px-2 py-1 text-[0.6rem] text-muted-foreground/60">No files</div>
          ) : (
            commitFiles.map(f => {
              const sl = STATUS_LETTER[f.status] ?? STATUS_LETTER.M
              return (
                <FileEntryContextMenu
                  isDirectory={false}
                  key={f.path}
                  name={pathName(f.path)}
                  path={absolutePath(cwd, f.path)}
                  relativeTo={cwd}
                >
                  <button
                    className="flex h-5 w-full items-center gap-1.5 rounded-md px-2 text-left text-[0.66rem] text-(--ui-text-secondary) transition-colors hover:bg-(--ui-row-hover-background) hover:text-foreground"
                    onClick={() => onOpenFile(absolutePath(cwd, f.path))}
                    title={f.path}
                    type="button"
                  >
                    <span className="min-w-0 flex-1 truncate">{pathName(f.path)}</span>
                    {pathDir(f.path) && (
                      <span className="min-w-0 shrink-[9999] truncate text-[0.62rem] text-(--ui-text-tertiary)">{pathDir(f.path)}</span>
                    )}
                    <span className={cn('shrink-0 font-mono text-[0.62rem] font-semibold', sl.tone)}>{sl.letter}</span>
                  </button>
                </FileEntryContextMenu>
              )
            })
          )}
        </div>
      )}
    </div>
  )
}

import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import {
  $gitAvailable,
  $gitBranches,
  $gitBusy,
  $gitError,
  $gitLoaded,
  $gitLog,
  $gitStashes,
  $gitStatus,
  commit,
  discardFiles,
  type GitFile,
  pull,
  push,
  refreshAll,
  refreshGitStatus,
  refreshLog,
  refreshStashes,
  stageAll,
  stageFiles,
  stashAction,
  stashPush,
  unstageAll,
  unstageFiles
} from '@/store/git'

import { BranchPicker } from './branch-picker'
import { DiffViewer } from './diff-viewer'

const STATUS_BADGE: Record<GitFile['status'], { letter: string; tone: string }> = {
  added: { letter: 'A', tone: 'text-emerald-500' },
  conflicted: { letter: '!', tone: 'text-amber-500' },
  copied: { letter: 'C', tone: 'text-emerald-500' },
  deleted: { letter: 'D', tone: 'text-red-500' },
  modified: { letter: 'M', tone: 'text-yellow-500' },
  renamed: { letter: 'R', tone: 'text-blue-400' },
  typechange: { letter: 'T', tone: 'text-yellow-500' },
  untracked: { letter: 'U', tone: 'text-emerald-400' }
}

function basename(p: string): string {
  return p.split(/[\\/]/).filter(Boolean).at(-1) ?? p
}

function dirname(p: string): string {
  const parts = p.split(/[\\/]/).filter(Boolean)
  parts.pop()

  return parts.join('/')
}

// What's open in the diff viewer overlay.
type DiffSelection =
  | { kind: 'file'; path: string; staged: boolean }
  | { kind: 'commit'; sha: string; subject: string }
  | null

interface FileRowProps {
  file: GitFile
  busy: boolean
  primaryIcon: string
  primaryLabel: string
  onPrimary: () => void
  onOpen: () => void
  onDiscard?: () => void
  discardLabel?: string
}

function FileRow({ file, busy, primaryIcon, primaryLabel, onPrimary, onOpen, onDiscard, discardLabel }: FileRowProps) {
  const badge = STATUS_BADGE[file.status]
  const dir = dirname(file.path)

  return (
    <div className="group/row flex h-6 items-center gap-1.5 rounded-sm px-1.5 hover:bg-(--ui-control-hover-background)">
      <button className="flex min-w-0 flex-1 items-center text-left" onClick={onOpen} type="button">
        <span className="truncate text-[0.72rem] leading-5 text-foreground">{basename(file.path)}</span>
        {dir && <span className="ml-1.5 truncate text-[0.62rem] text-muted-foreground/55">{dir}</span>}
      </button>

      <div className="flex shrink-0 items-center gap-0.5 opacity-0 transition-opacity group-hover/row:opacity-100 group-focus-within/row:opacity-100">
        {onDiscard && (
          <Tip label={discardLabel ?? 'Discard'}>
            <button
              aria-label={discardLabel ?? 'Discard'}
              className="grid size-5 place-items-center rounded-sm text-(--ui-text-tertiary) hover:bg-(--ui-bg-secondary) hover:text-foreground"
              disabled={busy}
              onClick={onDiscard}
              type="button"
            >
              <Codicon name="discard" size="0.75rem" />
            </button>
          </Tip>
        )}
        <Tip label={primaryLabel}>
          <button
            aria-label={primaryLabel}
            className="grid size-5 place-items-center rounded-sm text-(--ui-text-tertiary) hover:bg-(--ui-bg-secondary) hover:text-foreground"
            disabled={busy}
            onClick={onPrimary}
            type="button"
          >
            <Codicon name={primaryIcon} size="0.8rem" />
          </button>
        </Tip>
      </div>

      <span className={cn('w-3 shrink-0 text-center text-[0.7rem] font-semibold tabular-nums', badge.tone)}>
        {badge.letter}
      </span>
    </div>
  )
}

interface SectionProps {
  title: string
  count: number
  action?: { icon: string; label: string; onClick: () => void }
  collapsed?: boolean
  onToggle?: () => void
  children: React.ReactNode
}

function Section({ title, count, action, collapsed, onToggle, children }: SectionProps) {
  if (count === 0) {
    return null
  }

  return (
    <div className="mb-1">
      <div className="group/sec flex h-6 items-center gap-1 px-1.5">
        {onToggle && (
          <button aria-label={title} className="grid size-4 place-items-center text-muted-foreground/60" onClick={onToggle} type="button">
            <Codicon name={collapsed ? 'chevron-right' : 'chevron-down'} size="0.7rem" />
          </button>
        )}
        <span className="flex-1 text-[0.62rem] font-semibold uppercase tracking-[0.06em] text-muted-foreground/70">
          {title}
        </span>
        {action && (
          <Tip label={action.label}>
            <button
              aria-label={action.label}
              className="grid size-5 place-items-center rounded-sm text-(--ui-text-tertiary) opacity-0 transition-opacity hover:bg-(--ui-bg-secondary) hover:text-foreground group-hover/sec:opacity-100"
              onClick={action.onClick}
              type="button"
            >
              <Codicon name={action.icon} size="0.8rem" />
            </button>
          </Tip>
        )}
        <span className="grid h-4 min-w-4 place-items-center rounded-full bg-(--ui-bg-secondary) px-1 text-[0.6rem] font-semibold tabular-nums text-muted-foreground/70">
          {count}
        </span>
      </div>
      {!collapsed && <div className="px-0.5">{children}</div>}
    </div>
  )
}

/**
 * VS Code / Cursor-style Source Control panel. Branch picker + push/pull/fetch +
 * stash in the header, a commit box, staged/changes/untracked/conflict sections
 * with per-file and bulk stage/unstage/discard, plus collapsible History and
 * Stashes sections. Clicking a file opens an inline diff viewer with per-hunk
 * staging.
 */
export function SourceControlPanel() {
  const { t } = useI18n()
  const g = t.sourceControl
  const status = useStore($gitStatus)
  const available = useStore($gitAvailable)
  const busy = useStore($gitBusy)
  const error = useStore($gitError)
  const loaded = useStore($gitLoaded)
  const branches = useStore($gitBranches)
  const log = useStore($gitLog)
  const stashes = useStore($gitStashes)
  const [message, setMessage] = useState('')
  const [diff, setDiff] = useState<DiffSelection>(null)
  const [showHistory, setShowHistory] = useState(false)
  const [showStashes, setShowStashes] = useState(false)

  useEffect(() => {
    void refreshAll()
  }, [])

  const doCommit = useCallback(async () => {
    const text = message.trim()

    if (!text) {
      return
    }

    const ok = await commit(text)

    if (ok) {
      setMessage('')
    }
  }, [message])

  if (loaded && !available) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-1 px-6 text-center">
        <Codicon className="text-muted-foreground/40" name="source-control" size="1.5rem" />
        <div className="text-[0.72rem] font-semibold text-foreground">{g.notARepoTitle}</div>
        <div className="text-[0.66rem] leading-relaxed text-muted-foreground/65">{g.notARepoBody}</div>
      </div>
    )
  }

  // Diff viewer takes over the whole panel when a file/commit is open.
  if (diff) {
    return (
      <DiffViewer
        commitSha={diff.kind === 'commit' ? diff.sha : undefined}
        filePath={diff.kind === 'file' ? diff.path : undefined}
        onChanged={() => void refreshGitStatus()}
        onClose={() => setDiff(null)}
        staged={diff.kind === 'file' ? diff.staged : undefined}
        title={diff.kind === 'file' ? basename(diff.path) : diff.subject}
      />
    )
  }

  const hasStaged = status.staged.length > 0

  const totalChanges =
    status.staged.length + status.unstaged.length + status.untracked.length + status.conflicted.length

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Branch + sync header */}
      <div className="flex h-8 shrink-0 items-center gap-1 border-b border-(--ui-stroke-tertiary) px-1.5">
        <BranchPicker busy={busy} current={branches.local.find(b => b.current)?.name ?? status.branch} />
        <div className="flex-1" />
        {status.behind > 0 && (
          <span className="flex items-center gap-0.5 text-[0.62rem] text-muted-foreground/70">
            <Codicon name="arrow-down" size="0.62rem" />
            {status.behind}
          </span>
        )}
        {status.ahead > 0 && (
          <span className="flex items-center gap-0.5 text-[0.62rem] text-muted-foreground/70">
            <Codicon name="arrow-up" size="0.62rem" />
            {status.ahead}
          </span>
        )}
        <HeaderButton busy={busy} icon="arrow-down" label={g.pull} onClick={() => void pull()} />
        <HeaderButton
          busy={busy}
          icon={status.upstream ? 'repo-push' : 'cloud-upload'}
          label={status.upstream ? g.push : g.publish}
          onClick={() => void push({ setUpstream: !status.upstream })}
        />
        <HeaderButton busy={busy} icon="archive" label={g.stash} onClick={() => void stashPush()} />
        <HeaderButton busy={busy} icon="refresh" label={g.refresh} onClick={() => void refreshAll()} spinning={busy} />
      </div>

      {/* Commit box */}
      <div className="shrink-0 border-b border-(--ui-stroke-tertiary) p-2">
        <textarea
          aria-label={g.commitMessage}
          className="h-14 w-full resize-none rounded-md border border-(--ui-stroke-quaternary) bg-(--ui-bg-secondary) px-2 py-1.5 text-[0.72rem] text-foreground outline-none transition-colors focus:border-(--ui-stroke-primary) focus-visible:ring-1 focus-visible:ring-sidebar-ring"
          onChange={event => setMessage(event.target.value)}
          onKeyDown={event => {
            if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
              event.preventDefault()
              void doCommit()
            }
          }}
          placeholder={status.branch ? g.commitPlaceholder(status.branch) : g.commitMessage}
          value={message}
        />
        <Button
          className="mt-1.5 h-7 w-full text-[0.72rem]"
          disabled={busy || !message.trim() || !hasStaged}
          onClick={() => void doCommit()}
          size="sm"
        >
          <Codicon name="check" size="0.8rem" />
          {g.commit}
        </Button>
      </div>

      {/* Sections */}
      <div className="min-h-0 flex-1 overflow-y-auto py-1">
        {error && (
          <div className="mx-1.5 mb-1.5 rounded-md border border-destructive/40 bg-destructive/10 px-2 py-1.5 text-[0.66rem] leading-relaxed text-destructive">
            {error}
          </div>
        )}

        {totalChanges === 0 && !error && (
          <div className="grid place-items-center px-6 py-6 text-center text-[0.68rem] text-muted-foreground/60">
            {g.noChanges}
          </div>
        )}

        <Section
          action={{ icon: 'remove', label: g.unstageAll, onClick: () => void unstageAll() }}
          count={status.staged.length}
          title={g.stagedChanges}
        >
          {status.staged.map(file => (
            <FileRow
              busy={busy}
              file={file}
              key={`staged:${file.path}`}
              onOpen={() => setDiff({ kind: 'file', path: file.path, staged: true })}
              onPrimary={() => void unstageFiles([file.path])}
              primaryIcon="remove"
              primaryLabel={g.unstage}
            />
          ))}
        </Section>

        <Section
          action={{ icon: 'add', label: g.stageAll, onClick: () => void stageAll() }}
          count={status.conflicted.length}
          title={g.mergeChanges}
        >
          {status.conflicted.map(file => (
            <FileRow
              busy={busy}
              file={file}
              key={`conflict:${file.path}`}
              onOpen={() => setDiff({ kind: 'file', path: file.path, staged: false })}
              onPrimary={() => void stageFiles([file.path])}
              primaryIcon="add"
              primaryLabel={g.stage}
            />
          ))}
        </Section>

        <Section
          action={{ icon: 'add', label: g.stageAll, onClick: () => void stageAll() }}
          count={status.unstaged.length}
          title={g.changes}
        >
          {status.unstaged.map(file => (
            <FileRow
              busy={busy}
              discardLabel={g.discard}
              file={file}
              key={`unstaged:${file.path}`}
              onDiscard={() => void discardFiles([file.path])}
              onOpen={() => setDiff({ kind: 'file', path: file.path, staged: false })}
              onPrimary={() => void stageFiles([file.path])}
              primaryIcon="add"
              primaryLabel={g.stage}
            />
          ))}
        </Section>

        <Section
          action={{ icon: 'add', label: g.stageAll, onClick: () => void stageAll() }}
          count={status.untracked.length}
          title={g.untracked}
        >
          {status.untracked.map(file => (
            <FileRow
              busy={busy}
              discardLabel={g.delete}
              file={file}
              key={`untracked:${file.path}`}
              onDiscard={() => void discardFiles([file.path])}
              onOpen={() => setDiff({ kind: 'file', path: file.path, staged: false })}
              onPrimary={() => void stageFiles([file.path])}
              primaryIcon="add"
              primaryLabel={g.stage}
            />
          ))}
        </Section>

        {/* Stashes */}
        <Section
          collapsed={!showStashes}
          count={stashes.length}
          onToggle={() => {
            setShowStashes(s => !s)
            void refreshStashes()
          }}
          title={g.stashes}
        >
          {stashes.map(s => (
            <div
              className="group/st flex h-6 items-center gap-1.5 rounded-sm px-1.5 hover:bg-(--ui-control-hover-background)"
              key={s.ref}
            >
              <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="archive" size="0.7rem" />
              <span className="min-w-0 flex-1 truncate text-[0.7rem] text-foreground">{s.subject}</span>
              <div className="flex shrink-0 items-center gap-0.5 opacity-0 group-hover/st:opacity-100">
                <Tip label={g.stashPop}>
                  <button
                    aria-label={g.stashPop}
                    className="grid size-5 place-items-center rounded-sm text-(--ui-text-tertiary) hover:bg-(--ui-bg-secondary) hover:text-foreground"
                    onClick={() => void stashAction('pop', s.ref)}
                    type="button"
                  >
                    <Codicon name="arrow-up" size="0.72rem" />
                  </button>
                </Tip>
                <Tip label={g.stashDrop}>
                  <button
                    aria-label={g.stashDrop}
                    className="grid size-5 place-items-center rounded-sm text-(--ui-text-tertiary) hover:bg-(--ui-bg-secondary) hover:text-destructive"
                    onClick={() => void stashAction('drop', s.ref)}
                    type="button"
                  >
                    <Codicon name="trash" size="0.72rem" />
                  </button>
                </Tip>
              </div>
            </div>
          ))}
        </Section>

        {/* History */}
        <div className="mt-1">
          <button
            className="flex h-6 w-full items-center gap-1 px-1.5 text-[0.62rem] font-semibold uppercase tracking-[0.06em] text-muted-foreground/70 hover:text-foreground"
            onClick={() => {
              setShowHistory(s => !s)
              void refreshLog()
            }}
            type="button"
          >
            <Codicon name={showHistory ? 'chevron-down' : 'chevron-right'} size="0.7rem" />
            <span className="flex-1 text-left">{g.history}</span>
          </button>

          {showHistory && (
            <div className="px-0.5">
              {log.length === 0 && (
                <div className="px-2 py-2 text-[0.66rem] text-muted-foreground/50">{g.noChanges}</div>
              )}
              {log.map(commitItem => (
                <button
                  className="flex w-full flex-col items-start gap-0.5 rounded-sm px-2 py-1 text-left hover:bg-(--ui-control-hover-background)"
                  key={commitItem.sha}
                  onClick={() => setDiff({ kind: 'commit', sha: commitItem.sha, subject: commitItem.subject })}
                  type="button"
                >
                  <span className="w-full truncate text-[0.7rem] text-foreground">{commitItem.subject}</span>
                  <span className="flex items-center gap-1.5 text-[0.6rem] text-muted-foreground/55">
                    <span className="font-mono">{commitItem.shortSha}</span>
                    <span>·</span>
                    <span className="truncate">{commitItem.author}</span>
                    <span>·</span>
                    <span>{commitItem.relativeDate}</span>
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

interface HeaderButtonProps {
  busy: boolean
  icon: string
  label: string
  onClick: () => void
  spinning?: boolean
}

function HeaderButton({ busy, icon, label, onClick, spinning }: HeaderButtonProps) {
  return (
    <Tip label={label}>
      <button
        aria-label={label}
        className="grid size-5 place-items-center rounded-sm text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground disabled:opacity-40"
        disabled={busy}
        onClick={onClick}
        type="button"
      >
        <Codicon name={icon} size="0.8rem" spinning={spinning} />
      </button>
    </Tip>
  )
}

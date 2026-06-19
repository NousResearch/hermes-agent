import { useEffect, useMemo, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { buildHunkPatch, type DiffHunk, parseUnifiedDiff } from '@/lib/git-diff'
import { cn } from '@/lib/utils'
import { applyHunk, fetchCommitDiff, fetchDiff } from '@/store/git'

interface DiffViewerProps {
  // Either a working-tree/index file diff…
  filePath?: string
  staged?: boolean
  // …or a commit diff (read-only, no hunk staging).
  commitSha?: string
  title: string
  onClose: () => void
  onChanged?: () => void
}

/**
 * Inline unified-diff viewer. Renders hunks with old/new line gutters and, for
 * working-tree/index file diffs, a per-hunk Stage/Unstage button that applies
 * just that hunk via `git apply --cached` (VS Code's killer feature).
 */
export function DiffViewer({ filePath, staged, commitSha, title, onClose, onChanged }: DiffViewerProps) {
  const { t } = useI18n()
  const g = t.sourceControl
  const [raw, setRaw] = useState<string | null>(null)
  const [busyHunk, setBusyHunk] = useState<number | null>(null)

  useEffect(() => {
    let alive = true
    setRaw(null)

    const load = async () => {
      const diff = commitSha ? await fetchCommitDiff(commitSha) : await fetchDiff(filePath ?? '', Boolean(staged))

      if (alive) {
        setRaw(diff)
      }
    }

    void load()

    return () => {
      alive = false
    }
  }, [filePath, staged, commitSha])

  const parsed = useMemo(() => (raw == null ? null : parseUnifiedDiff(raw)), [raw])
  const canStageHunks = !commitSha && Boolean(filePath)

  const onHunk = async (hunk: DiffHunk, index: number) => {
    if (!parsed) {
      return
    }

    setBusyHunk(index)
    const patch = buildHunkPatch(parsed, hunk)
    // staged diff → unstage that hunk (reverse); unstaged diff → stage it.
    await applyHunk(patch, Boolean(staged))
    setBusyHunk(null)
    onChanged?.()
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex h-8 shrink-0 items-center gap-1.5 border-b border-(--ui-stroke-tertiary) px-2">
        <button
          aria-label={g.backToChanges}
          className="grid size-5 place-items-center rounded-sm text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground"
          onClick={onClose}
          type="button"
        >
          <Codicon name="arrow-left" size="0.85rem" />
        </button>
        <span className="min-w-0 flex-1 truncate text-[0.72rem] font-medium text-foreground">{title}</span>
      </div>

      <div className="min-h-0 flex-1 overflow-auto font-mono text-[0.68rem] leading-[1.5]">
        {parsed == null && <div className="px-3 py-4 text-muted-foreground/60">{g.loadingDiff}</div>}

        {parsed != null && parsed.binary && (
          <div className="px-3 py-4 text-muted-foreground/60">{g.binaryFile}</div>
        )}

        {parsed != null && !parsed.binary && parsed.hunks.length === 0 && (
          <div className="px-3 py-4 text-muted-foreground/60">{g.noChanges}</div>
        )}

        {parsed != null &&
          parsed.hunks.map((hunk, index) => (
            <div className="border-b border-(--ui-stroke-quaternary)/40" key={index}>
              <div className="flex items-center gap-2 bg-(--ui-bg-secondary)/40 px-2 py-0.5 text-(--ui-text-tertiary)">
                <span className="min-w-0 flex-1 truncate text-[0.64rem]">{hunk.header}</span>
                {canStageHunks && (
                  <Tip label={staged ? g.unstageHunk : g.stageHunk}>
                    <button
                      aria-label={staged ? g.unstageHunk : g.stageHunk}
                      className="grid size-4 place-items-center rounded-sm text-(--ui-text-tertiary) hover:bg-(--ui-bg-secondary) hover:text-foreground disabled:opacity-40"
                      disabled={busyHunk === index}
                      onClick={() => void onHunk(hunk, index)}
                      type="button"
                    >
                      <Codicon name={staged ? 'remove' : 'add'} size="0.7rem" />
                    </button>
                  </Tip>
                )}
              </div>

              {hunk.lines.map((line, li) => (
                <div
                  className={cn(
                    'flex whitespace-pre',
                    line.kind === 'add' && 'bg-emerald-500/12',
                    line.kind === 'del' && 'bg-red-500/12',
                    line.kind === 'meta' && 'text-muted-foreground/50'
                  )}
                  key={li}
                >
                  <span className="w-9 shrink-0 select-none px-1 text-right text-muted-foreground/35">
                    {line.oldLine ?? ''}
                  </span>
                  <span className="w-9 shrink-0 select-none px-1 text-right text-muted-foreground/35">
                    {line.newLine ?? ''}
                  </span>
                  <span
                    className={cn(
                      'w-3 shrink-0 select-none text-center',
                      line.kind === 'add' && 'text-emerald-500',
                      line.kind === 'del' && 'text-red-500'
                    )}
                  >
                    {line.kind === 'add' ? '+' : line.kind === 'del' ? '-' : ''}
                  </span>
                  <span className="min-w-0 flex-1 pr-2 text-foreground/90">{line.text}</span>
                </div>
              ))}
            </div>
          ))}
      </div>
    </div>
  )
}

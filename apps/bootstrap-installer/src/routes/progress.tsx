import { useEffect, useRef, useState } from 'react'
import { useStore } from '@nanostores/react'
import { Button } from '../components/button'
import {
  cancelInstall,
  $progress,
  type BootstrapStateModel,
  type StageState
} from '../store'
import { Check, X, ChevronRight, FileText, Loader2 } from 'lucide-react'
import clsx from 'clsx'

interface ProgressProps {
  bootstrap: BootstrapStateModel
}

/*
 * Progress screen — drives a stage list + collapsible log panel. Uses
 * the DS <Progress> for the top bar so its motion + ring match the rest
 * of the product.
 */
export default function ProgressScreen({ bootstrap }: ProgressProps) {
  const progress = useStore($progress)
  const [showLogs, setShowLogs] = useState(false)
  const logEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (showLogs && logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [bootstrap.logs.length, showLogs])

  const currentStage =
    bootstrap.currentStage != null
      ? bootstrap.stages[bootstrap.currentStage]
      : null

  return (
    <div className="hermes-fade-in flex h-full flex-col">
      <div className="border-b border-border px-6 py-4">
        <div className="mb-3 flex items-center justify-between text-xs">
          <div className="flex items-center gap-2 text-foreground">
            {bootstrap.status === 'running' && (
              <Loader2 size={12} className="animate-spin text-primary" />
            )}
            <span>
              {bootstrap.status === 'running'
                ? currentStage
                  ? currentStage.info.title
                  : 'Preparing\u2026'
                : bootstrap.status === 'completed'
                  ? 'Done'
                  : 'Installing'}
            </span>
          </div>
          <div className="text-muted-foreground">
            {progress.done} of {progress.total} steps
          </div>
        </div>
        {/* Top progress bar — plain HTML, derived from --primary so it
            tracks the theme accent. */}
        <div className="h-1 w-full overflow-hidden rounded-full bg-muted">
          <div
            className="h-full bg-primary transition-all duration-300 ease-out"
            style={{ width: `${Math.max(2, progress.fraction * 100)}%` }}
          />
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        <div className="flex-1 overflow-y-auto px-6 py-4">
          <ol className="space-y-1">
            {bootstrap.stageOrder.map((name) => {
              const rec = bootstrap.stages[name]
              if (!rec) return null
              return (
                <li
                  key={name}
                  className={clsx(
                    'flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors',
                    rec.state === 'running' && 'bg-card text-foreground',
                    rec.state === 'succeeded' && 'text-foreground/80',
                    rec.state === 'skipped' && 'text-muted-foreground',
                    rec.state === 'failed' &&
                      'bg-destructive/10 text-destructive',
                    !rec.state && 'text-muted-foreground/60'
                  )}
                >
                  <StateIcon state={rec.state ?? null} />
                  <span className="flex-1 truncate">{rec.info.title}</span>
                  {rec.durationMs != null && (
                    <span className="text-xs text-muted-foreground">
                      {formatDuration(rec.durationMs)}
                    </span>
                  )}
                </li>
              )
            })}
          </ol>
        </div>

        {showLogs && (
          <div className="flex w-1/2 flex-col border-l border-border bg-card/40">
            <div className="flex shrink-0 items-center justify-between border-b border-border px-3 py-2">
              <div className="text-xs font-medium text-foreground/80">
                Live output
              </div>
              <div className="text-xs text-muted-foreground">
                {bootstrap.logs.length} lines
              </div>
            </div>
            <div className="flex-1 overflow-y-auto px-3 py-2 font-mono text-[11px] leading-relaxed">
              {bootstrap.logs.map((entry, idx) => (
                <div
                  key={idx}
                  className={clsx(
                    'whitespace-pre-wrap',
                    entry.line.startsWith('stderr:')
                      ? 'text-destructive'
                      : 'text-foreground/70'
                  )}
                >
                  {entry.line}
                </div>
              ))}
              <div ref={logEndRef} />
            </div>
          </div>
        )}
      </div>

      <div className="flex shrink-0 items-center justify-between border-t border-border px-6 py-3">
        <button
          type="button"
          onClick={() => setShowLogs((v) => !v)}
          className="inline-flex items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
        >
          <FileText size={14} />
          {showLogs ? 'Hide details' : 'Show details'}
          <ChevronRight
            size={12}
            className={clsx(
              'transition-transform',
              showLogs && 'rotate-90'
            )}
          />
        </button>

        {bootstrap.status === 'running' && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => void cancelInstall()}
          >
            Cancel
          </Button>
        )}
      </div>
    </div>
  )
}

function StateIcon({ state }: { state: StageState | null }) {
  if (state === 'running') {
    return <Loader2 size={14} className="animate-spin text-primary" />
  }
  if (state === 'succeeded') {
    return <Check size={14} className="text-emerald-400" />
  }
  if (state === 'skipped') {
    return <ChevronRight size={14} className="text-muted-foreground/70" />
  }
  if (state === 'failed') {
    return <X size={14} className="text-destructive" />
  }
  return (
    <div
      className="h-[6px] w-[6px] rounded-full bg-muted-foreground/40"
      aria-hidden
    />
  )
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  const m = Math.floor(ms / 60000)
  const s = Math.round((ms % 60000) / 1000)
  return `${m}m ${s}s`
}

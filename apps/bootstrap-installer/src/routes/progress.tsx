import { useStore } from '@nanostores/react'
import clsx from 'clsx'
import { Check, ChevronRight, FileText, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import { BrandMark } from '../components/brand-mark'
import { Button } from '../components/button'
import { Loader } from '../components/loader'
import {
  $mode,
  $progress,
  type BootstrapStateModel,
  cancelInstall,
  type StageState
} from '../store'

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
  const mode = useStore($mode)
  const [showLogs, setShowLogs] = useState(false)
  const [now, setNow] = useState(() => Date.now())
  const logEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (showLogs && logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [bootstrap.logs.length, showLogs])

  // Tick once a second while the run is in flight so the active step shows a
  // live elapsed timer — a long single step (e.g. the dependency download)
  // reads as working, not frozen. Stops when nothing is running.
  useEffect(() => {
    if (bootstrap.status !== 'running') {
      return
    }

    const id = window.setInterval(() => setNow(Date.now()), 1000)

    return () => window.clearInterval(id)
  }, [bootstrap.status])

  const isUpdate = mode === 'update'
  const title = bootstrap.status === 'completed' ? 'Done' : isUpdate ? 'Updating Hermes' : 'Setting up Hermes Agent'

  const description = isUpdate
    ? 'Hermes is updating to the latest version — this only takes a moment.'
    : 'This is a one-time setup. The Hermes installer is downloading dependencies and configuring your machine. Subsequent launches will skip this step.'

  const pct = Math.round(progress.fraction * 100)

  return (
    <div className="hermes-fade-in flex h-full flex-col">
      {/* Header: brand + title + description, matching the desktop install overlay. */}
      <div className="flex shrink-0 items-start gap-4 px-6 pt-6 pb-4">
        <BrandMark className="size-11" />
        <div className="min-w-0">
          <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
          <p className="mt-1.5 text-sm text-muted-foreground">{description}</p>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        <div className="flex-1 overflow-y-auto px-6 pt-2 pb-4">
          {/* Progress line + bar; the count shimmers while the install runs.
              pt-2 matches the log header's py-2 so the "steps complete" line and
              the "Live output" header share a baseline. */}
          <div className="mb-4">
            <div className="mb-1 flex items-center justify-between text-xs text-muted-foreground">
              <span className={clsx(bootstrap.status === 'running' && 'shimmer')}>
                {progress.done} of {progress.total} steps complete
              </span>
              <span className="tabular-nums">{pct}%</span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-(--ui-bg-tertiary)">
              <div
                className="h-full bg-primary transition-all duration-300 ease-out"
                style={{ width: `${Math.max(2, progress.fraction * 100)}%` }}
              />
            </div>
          </div>

          {/* Flat stage list: only the running step is opaque; the rest read as
              muted. Running loader overhangs left so labels stay aligned; the
              terminal check/cross sits right of the label. */}
          <ol className="space-y-0.5">
            {bootstrap.stageOrder.map((name) => {
              const rec = bootstrap.stages[name]

              if (!rec) {return null}

              const meta =
                rec.state === 'running' && rec.startedAt != null
                  ? formatElapsed(now - rec.startedAt)
                  : rec.durationMs != null && rec.state !== 'failed'
                    ? formatDuration(rec.durationMs)
                    : null

              return (
                <li
                  className={clsx(
                    'flex items-center gap-2.5 px-3 py-1.5 text-sm',
                    rec.state === 'running'
                      ? 'font-medium text-foreground'
                      : 'text-muted-foreground'
                  )}
                  key={name}
                >
                  {rec.state === 'running' && <Loader className="-ml-2 size-6 shrink-0" />}
                  <span className="flex-1 truncate">{rec.info.title}</span>
                  {meta && <span className="text-xs tabular-nums text-muted-foreground/70">{meta}</span>}
                  <StateIcon state={rec.state ?? null} />
                </li>
              )
            })}
          </ol>
        </div>

        {showLogs && (
          <div className="flex w-1/2 flex-col border-l border-(--stroke-nous)">
            <div className="flex shrink-0 items-center justify-between border-b border-(--stroke-nous) px-3 py-2 text-xs">
              <span className="font-medium text-foreground/80">Live output</span>
              <span className="tabular-nums text-muted-foreground">{bootstrap.logs.length} lines</span>
            </div>
            <div className="flex-1 overflow-y-auto px-3 py-2 font-mono text-[10.5px] leading-relaxed">
              {bootstrap.logs.map((entry, idx) => (
                <div
                  className={clsx(
                    'whitespace-pre-wrap',
                    entry.stream === 'stderr' ? 'text-foreground/45' : 'text-foreground/70'
                  )}
                  key={idx}
                >
                  {entry.line}
                </div>
              ))}
              <div ref={logEndRef} />
            </div>
          </div>
        )}
      </div>

      <div className="flex shrink-0 items-center justify-between border-t border-(--stroke-nous) px-6 py-3">
        <button
          className="inline-flex cursor-pointer items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
          onClick={() => setShowLogs((v) => !v)}
          type="button"
        >
          <FileText size={14} />
          {showLogs ? 'Hide details' : 'Show details'}
          <ChevronRight className={clsx('transition-transform', showLogs && 'rotate-90')} size={12} />
        </button>

        {bootstrap.status === 'running' && (
          <Button onClick={() => void cancelInstall()} size="sm" variant="outline">
            Cancel
          </Button>
        )}
      </div>
    </div>
  )
}

// Terminal-state markers, neutral by design: a muted check for done/skipped
// (no celebratory green), a destructive cross for failure. Running renders its
// spinner on the left; pending stays icon-less.
function StateIcon({ state }: { state: StageState | null }) {
  if (state === 'succeeded') {
    return <Check className="shrink-0 text-muted-foreground" size={13} />
  }

  if (state === 'skipped') {
    return <Check className="shrink-0 text-muted-foreground/50" size={13} />
  }

  if (state === 'failed') {
    return <X className="shrink-0 text-destructive" size={13} />
  }

  return null
}

function formatDuration(ms: number): string {
  if (ms < 1000) {return `${ms}ms`}

  if (ms < 60000) {return `${(ms / 1000).toFixed(1)}s`}
  const m = Math.floor(ms / 60000)
  const s = Math.round((ms % 60000) / 1000)

  return `${m}m ${s}s`
}

// Live elapsed for a running stage: bare seconds under a minute, then m:ss.
function formatElapsed(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000))

  if (s < 60) {return `${s}s`}
  const m = Math.floor(s / 60)

  return `${m}:${String(s - m * 60).padStart(2, '0')}`
}

import clsx from 'clsx'
import { ChevronRight, FileText } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import { BrandMark } from '../components/brand-mark'
import { Button } from '../components/button'
import { Loader } from '../components/loader'
import { formatElapsed } from '../lib/format'
import { type BootstrapStateModel, cancelInstall } from '../store'

interface ProgressProps {
  bootstrap: BootstrapStateModel
}

/*
 * Progress screen.
 *
 * bootstrap-north-forge.ps1 is one monolithic run with no stage protocol,
 * so this is an INDETERMINATE view: a moving bar, a live elapsed timer, and
 * the collapsible log panel (the real detail). No per-step checklist.
 */
export default function ProgressScreen({ bootstrap }: ProgressProps) {
  const [showLogs, setShowLogs] = useState(true)
  const [startedAt] = useState(() => Date.now())
  const [now, setNow] = useState(() => Date.now())
  const logEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (showLogs && logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [bootstrap.logs.length, showLogs])

  useEffect(() => {
    if (bootstrap.status !== 'running') {
      return
    }

    const id = window.setInterval(() => setNow(Date.now()), 1000)

    return () => window.clearInterval(id)
  }, [bootstrap.status])

  const running = bootstrap.status === 'running'
  const lastLine = bootstrap.logs.length > 0 ? bootstrap.logs[bootstrap.logs.length - 1]!.line : null

  return (
    <div className="nf-fade-in flex h-full flex-col">
      <div className="flex shrink-0 items-start gap-4 px-6 pt-6 pb-4">
        <BrandMark className="size-11" />
        <div className="min-w-0">
          <h2 className="text-xl font-semibold tracking-tight">
            {running ? 'Setting up North Forge' : 'Done'}
          </h2>
          <p className="mt-1.5 text-sm text-muted-foreground">
            Running <code className="font-mono text-xs">bootstrap-north-forge.ps1</code> — creating the
            Python environment and data folder next to your checkout. This is a one-time step.
          </p>
        </div>
      </div>

      <div className="flex flex-1 flex-col overflow-hidden px-6">
        {/* Indeterminate bar + elapsed */}
        <div className="mb-3 shrink-0">
          <div className="mb-1 flex items-center justify-between text-xs text-muted-foreground">
            <span className={clsx('flex items-center gap-2', running && 'shimmer')}>
              {running && <Loader className="-ml-1 size-5 shrink-0" />}
              {running ? 'Working…' : 'Complete'}
            </span>
            <span className="tabular-nums">{formatElapsed(now - startedAt)}</span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-(--ui-bg-tertiary)">
            <div
              className={clsx(
                'h-full bg-primary',
                running ? 'progress-slide w-1/3' : 'w-full transition-all duration-300'
              )}
            />
          </div>
          {lastLine && running && (
            <p className="mt-1.5 truncate font-mono text-[11px] text-muted-foreground/70">{lastLine}</p>
          )}
        </div>

        {/* Collapsible log panel — the real detail */}
        <div className="flex min-h-0 flex-1 flex-col rounded-md border border-(--stroke-nous)">
          <button
            className="flex shrink-0 items-center justify-between border-b border-(--stroke-nous) px-3 py-2 text-xs"
            onClick={() => setShowLogs((v) => !v)}
            type="button"
          >
            <span className="inline-flex items-center gap-1.5 font-medium text-foreground/80">
              <FileText size={13} />
              Live output
              <ChevronRight className={clsx('transition-transform', showLogs && 'rotate-90')} size={12} />
            </span>
            <span className="tabular-nums text-muted-foreground">{bootstrap.logs.length} lines</span>
          </button>
          {showLogs && (
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
          )}
        </div>
      </div>

      <div className="flex shrink-0 items-center justify-end px-6 py-3">
        {running && (
          <Button onClick={() => void cancelInstall()} size="sm" variant="outline">
            Cancel
          </Button>
        )}
      </div>
    </div>
  )
}

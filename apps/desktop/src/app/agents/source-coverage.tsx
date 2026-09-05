import { useState } from 'react'

import { Button } from '@/components/ui/button'
import type { OverviewSource } from '@/global'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { overviewCache } from '@/store/agent-overview'

import { overviewActions } from './actions'

// One quiet footer entry per source: a status dot, the name, and only the
// facts that need acting on. "Ready · 357/357" is bookkeeping, not a status —
// the count only appears while coverage is still incomplete.
export function SourceCoverage({ source }: { source: OverviewSource }) {
  const { t } = useI18n()
  const a = t.agents
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const connect = async () => {
    setPending(true)
    setError(null)

    try {
      await overviewActions.connect(source.connectionId)
      await overviewCache.refresh({ force: true })
    } catch (error) {
      setError(String(error instanceof Error ? error.message : error).slice(0, 240))
    } finally {
      setPending(false)
    }
  }

  // Known error tokens get product copy; anything else is a sanitized backend message.
  const errorCopy: Record<string, string> = {
    'read-only-inventory-unavailable': a.inventoryUnavailable,
    'history-shifted': a.historyShifted
  }

  const ready = source.state === 'ready' && source.complete

  const labels = {
    ready: a.ready,
    'on-demand': a.onDemand,
    offline: a.offline,
    unsupported: a.unsupported,
    partial: a.partial
  }

  const dot =
    source.state === 'ready'
      ? 'bg-(--ui-success)'
      : source.state === 'on-demand'
        ? 'border border-(--ui-text-tertiary)'
        : source.state === 'partial'
          ? 'bg-amber-500'
          : 'bg-(--ui-text-quaternary)'

  return (
    <div className="flex flex-wrap items-baseline gap-x-1.5 gap-y-0.5">
      <span aria-hidden="true" className={cn('size-1.5 self-center rounded-full', dot)} />
      <span className="text-foreground/80">{source.label}</span>
      {!ready ? (
        <span>
          {labels[source.state]}
          {source.state === 'ready' || source.state === 'partial'
            ? ` · ${a.history} ${source.offset}/${source.total}`
            : ''}
        </span>
      ) : null}
      {source.liveCoverage === 'unavailable' && source.state !== 'unsupported' ? (
        <span>· {a.liveUnavailable}</span>
      ) : null}
      {source.state === 'on-demand' || source.state === 'offline' ? (
        <Button
          aria-label={`${a.connect} ${source.label}`}
          disabled={pending}
          onClick={() => void connect()}
          size="inline"
          variant="textStrong"
        >
          {a.connect}
        </Button>
      ) : null}
      {source.state !== 'ready' ? (
        <Button
          aria-label={`${a.retry} ${source.label}`}
          disabled={pending}
          onClick={() => void overviewCache.refresh({ force: true })}
          size="inline"
          variant="text"
        >
          {a.retry}
        </Button>
      ) : null}
      {error || source.error ? (
        <span className="text-destructive" role="alert">
          {error || errorCopy[source.error ?? ''] || source.error}
        </span>
      ) : null}
      {source.errors.map((item, index) => (
        <span className="text-destructive" key={index}>
          {item.profile} · {item.error}
        </span>
      ))}
    </div>
  )
}

import { StatusDot } from '@/components/status-dot'
import type { FallbackStatus as FallbackStatusSnapshot } from '@/types/hermes'

interface FallbackStatusProps {
  status?: FallbackStatusSnapshot | null
}

/** Compact, secret-free operational view of the active fallback route. */
export function FallbackStatus({ status }: FallbackStatusProps) {
  if (!status) {
    return null
  }

  const active = status.active
  const chain = status.chain ?? []
  const cooldown = status.cooldown_until
  const retryAt = cooldown ? new Date(cooldown * 1000).toLocaleTimeString() : null

  return (
    <section aria-label="Fallback chain status" className="border-t border-border/50 px-3 py-2 text-xs">
      <div className="flex items-center gap-1.5 font-medium">
        <StatusDot tone={active ? 'warn' : 'good'} />
        <span>{active ? `Using ${active.provider}/${active.model}` : 'Primary route active'}</span>
      </div>
      {chain.length > 0 && (
        <ol aria-label="Fallback priority" className="mt-1.5 space-y-0.5 text-muted-foreground">
          {chain.map((entry, index) => (
            <li className="flex gap-1.5" key={`${entry.provider}/${entry.model}-${index}`}>
              <span className="w-3 shrink-0 text-right">{index + 1}.</span>
              <span className="truncate">{entry.provider}/{entry.model}</span>
            </li>
          ))}
        </ol>
      )}
      {(status.reason || retryAt) && (
        <div className="mt-1.5 text-muted-foreground">
          {status.reason && <div className="line-clamp-2">{status.reason}</div>}
          {retryAt && <div>Next retry: {retryAt}</div>}
        </div>
      )}
    </section>
  )
}

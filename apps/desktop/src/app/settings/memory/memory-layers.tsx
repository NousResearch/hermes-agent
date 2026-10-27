import { useEffect, useState } from 'react'

import { getMemoryStatus } from '@/hermes'
import { useI18n } from '@/i18n'
import type { MemoryProviderStatus, MemoryStatusResponse } from '@/types/hermes'

import { isExternalMemoryProvider } from '../helpers'
import { ListRow, Pill } from '../primitives'

const STATUS_TONE: Record<MemoryProviderStatus, 'destructive' | 'success' | 'warn'> = {
  missing: 'destructive',
  needs_config: 'warn',
  ready: 'success',
  unavailable: 'destructive'
}

/** The external provider `active` names, with its readiness when the status
 *  lists it. Built-in memory is not a provider: `''`/`builtin` read as none. */
export function activeMemoryProvider(
  status: MemoryStatusResponse | null,
  provider?: string
): { name: string; status?: MemoryProviderStatus } | null {
  const name = provider ?? status?.active

  if (!isExternalMemoryProvider(name)) {
    return null
  }

  return { name, status: status?.providers.find(row => row.name === name)?.status }
}

/** Which memory provider is active, and that MEMORY.md / USER.md are the
 *  built-in layer that stays on under any provider (#145175). */
export function MemoryLayersSummary({ provider, status }: { provider?: string; status: MemoryStatusResponse | null }) {
  const { t } = useI18n()
  const m = t.settings.memoryLayers
  const active = activeMemoryProvider(status, provider)

  return (
    <div className="grid gap-1" data-testid="memory-layers">
      <ListRow
        action={
          active ? (
            <span className="flex min-w-0 items-center gap-2">
              <span className="truncate font-mono text-[length:var(--conversation-text-font-size)]">{active.name}</span>
              {active.status ? <Pill tone={STATUS_TONE[active.status]}>{m.status[active.status]}</Pill> : null}
            </span>
          ) : (
            <span className="text-[length:var(--conversation-text-font-size)] text-(--ui-text-secondary)">
              {m.noProvider}
            </span>
          )
        }
        description={active ? m.providerDesc(active.name) : m.noProviderDesc}
        title={m.activeProvider}
      />
      <ListRow action={<Pill tone="success">{m.alwaysOn}</Pill>} description={m.builtinDesc} title={m.builtinLayer} />
    </div>
  )
}

/** Settings > Memory & Context: `provider` is the page's live `memory.provider`
 *  draft, so the summary follows the select; readiness comes from GET /api/memory. */
export function MemoryLayersPanel({ profile, provider }: { profile?: string; provider: string }) {
  const [status, setStatus] = useState<MemoryStatusResponse | null>(null)

  useEffect(() => {
    let cancelled = false

    getMemoryStatus(profile)
      .then(next => !cancelled && setStatus(next))
      .catch(() => !cancelled && setStatus(null))

    return () => {
      cancelled = true
    }
  }, [profile, provider])

  return <MemoryLayersSummary provider={provider} status={status} />
}

import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { getMemoryStatus } from '@/hermes'
import { useI18n } from '@/i18n'
import type { MemoryLayersTranslations } from '@/i18n/types_memory_layers'
import { $memoryStatusRevision } from '@/store/memory-status'
import type { MemoryProviderStatus, MemoryStatusResponse } from '@/types/hermes'

import { isExternalMemoryProvider } from '../helpers'
import { ListRow, Pill } from '../primitives'

const STATUS_TONE: Record<MemoryProviderStatus, 'destructive' | 'success' | 'warn'> = {
  missing: 'destructive',
  needs_config: 'warn',
  ready: 'success',
  unavailable: 'destructive'
}

interface ProviderRow {
  name: string
  status?: MemoryProviderStatus
}

/** Whether MEMORY.md / USER.md enter sessions (`memory.memory_enabled` /
 *  `memory.user_profile_enabled`); omitted where the config isn't loaded. */
export interface BuiltinToggles {
  memory: boolean
  user: boolean
}

function providerRow(status: MemoryStatusResponse | null, name: unknown): null | ProviderRow {
  if (!isExternalMemoryProvider(name)) {
    return null
  }

  const trimmed = name.trim()

  return { name: trimmed, status: status?.providers.find(row => row.name === trimmed)?.status }
}

/** `active` is what GET /api/memory reports (the saved provider); `selected`
 *  is the Settings draft, set only while it differs from `active` or the
 *  status is unknown. Built-in aliases (`''`, `builtin`, `default`, ...) read
 *  as no external provider. */
export function memoryProviderRows(
  status: MemoryStatusResponse | null,
  provider?: string
): { active: null | ProviderRow; selected?: null | ProviderRow } {
  const active = status ? providerRow(status, status.active) : null

  if (provider === undefined) {
    return { active }
  }

  const selected = providerRow(status, provider)

  return status && selected?.name === active?.name ? { active } : { active, selected }
}

function ProviderName({ m, row }: { m: MemoryLayersTranslations; row: null | ProviderRow }) {
  if (!row) {
    return (
      <span className="text-[length:var(--conversation-text-font-size)] text-(--ui-text-secondary)">
        {m.noProvider}
      </span>
    )
  }

  return (
    <span className="flex min-w-0 items-center gap-2">
      <span className="truncate font-mono text-[length:var(--conversation-text-font-size)]">{row.name}</span>
      {row.status ? <Pill tone={STATUS_TONE[row.status]}>{m.status[row.status]}</Pill> : null}
    </span>
  )
}

// Recall is claimed only for a saved provider the status calls ready.
function activeDescription(m: MemoryLayersTranslations, row: null | ProviderRow): string {
  if (!row) {
    return m.noProviderDesc
  }

  if (!row.status) {
    return m.providerUnknownDesc(row.name)
  }

  return row.status === 'ready' ? m.providerDesc(row.name) : m.providerNotReadyDesc(row.name)
}

function builtinPill(m: MemoryLayersTranslations, builtin?: BuiltinToggles) {
  if (!builtin) {
    return <Pill tone="success">{m.onByDefault}</Pill>
  }

  if (builtin.memory && builtin.user) {
    return <Pill tone="success">{m.on}</Pill>
  }

  if (!builtin.memory && !builtin.user) {
    return <Pill>{m.off}</Pill>
  }

  return <Pill tone="warn">{m.onlyFile(builtin.memory ? 'MEMORY.md' : 'USER.md')}</Pill>
}

interface MemoryLayersSummaryProps {
  builtin?: BuiltinToggles
  onRetry?: () => void
  provider?: string
  status: MemoryStatusResponse | null
  statusFailed?: boolean
}

/** Which memory provider is active (and, in Settings, which one is selected
 *  but not saved yet), plus the built-in MEMORY.md / USER.md layer that no
 *  provider replaces (#145175). */
export function MemoryLayersSummary({ builtin, onRetry, provider, status, statusFailed }: MemoryLayersSummaryProps) {
  const { t } = useI18n()
  const m = t.settings.memoryLayers
  const { active, selected } = memoryProviderRows(status, provider)

  return (
    <div className="grid gap-1" data-testid="memory-layers">
      {selected !== undefined ? (
        <ListRow
          action={<ProviderName m={m} row={selected} />}
          description={status ? m.selectedDesc(active?.name ?? m.builtinMemory) : undefined}
          title={m.selectedProvider}
        />
      ) : null}
      {status ? (
        <ListRow
          action={<ProviderName m={m} row={active} />}
          description={activeDescription(m, active)}
          title={m.activeProvider}
        />
      ) : null}
      {statusFailed ? (
        <ListRow
          action={
            onRetry ? (
              <Button onClick={onRetry} size="xs" type="button" variant="textStrong">
                {m.retry}
              </Button>
            ) : null
          }
          description={m.statusUnavailableDesc}
          title={m.statusUnavailable}
        />
      ) : null}
      <ListRow action={builtinPill(m, builtin)} description={m.builtinDesc} title={m.builtinLayer} />
    </div>
  )
}

interface MemoryLayersPanelProps {
  builtin?: BuiltinToggles
  profile?: string
  provider: string
}

interface MemoryLayersSlotProps extends MemoryLayersPanelProps {
  sectionId: string
  subpage?: string
}

/** Settings > Memory & Context > Persistent memory (or the whole section). */
export function MemoryLayersSlot({ sectionId, subpage, ...panel }: MemoryLayersSlotProps) {
  if (sectionId !== 'memory' || (subpage !== undefined && subpage !== 'persistent')) {
    return null
  }

  return <MemoryLayersPanel {...panel} />
}

/** `provider` is the page's live `memory.provider` draft; the active provider
 *  and readiness come from GET /api/memory, refetched whenever
 *  `$memoryStatusRevision` moves (config autosave, provider setup, OAuth). */
export function MemoryLayersPanel({ builtin, profile, provider }: MemoryLayersPanelProps) {
  const revision = useStore($memoryStatusRevision)
  const [status, setStatus] = useState<MemoryStatusResponse | null>(null)
  const [failed, setFailed] = useState(false)
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    let cancelled = false

    getMemoryStatus(profile)
      .then(next => {
        if (!cancelled) {
          setStatus(next)
          setFailed(false)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setStatus(null)
          setFailed(true)
        }
      })

    return () => {
      cancelled = true
    }
  }, [attempt, profile, revision])

  return (
    <MemoryLayersSummary
      builtin={builtin}
      onRetry={() => setAttempt(n => n + 1)}
      provider={provider}
      status={status}
      statusFailed={failed}
    />
  )
}

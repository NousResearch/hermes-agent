import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router'

import { type OwnerScope, profileScopeKey } from '@/api/client'
import { MEMORY_PLUGINS_ROUTE } from '@/app/routes'
import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { ErrorState } from '@/components/ui/error-state'
import { getMemoryStatus, setMemoryProvider } from '@/hermes'
import { useI18n } from '@/i18n'
import { openPluginInstallRequest } from '@/store/plugin-install-request'
import type { MemoryProviderStatus, MemoryStatusResponse } from '@/types/hermes'

import { ListRow, Pill } from '../primitives'

import { MemoryConnect } from './connect'
import { ProviderConfigPanel } from './provider-config'

interface MemoryProviderSettingsProps {
  profile: OwnerScope
}

export function MemoryProviderSettings({ profile }: MemoryProviderSettingsProps) {
  return <MemoryProviderSettingsOwner key={profileScopeKey(profile)} profile={profile} />
}

// The backend omits builtin and may name an active provider it no longer finds.
function providerRows(status: MemoryStatusResponse, builtinDescription: string): MemoryProviderStatus[] {
  const rows = [...status.providers]
  const active = status.active || 'builtin'

  if (!rows.some(row => row.name === 'builtin')) {
    rows.unshift({ name: 'builtin', description: builtinDescription, configured: true, status: 'ready' })
  }

  if (!rows.some(row => row.name === active)) {
    rows.push({ name: active, description: '', configured: false, status: 'missing' })
  }

  return rows
}

function MemoryProviderSettingsOwner({ profile }: MemoryProviderSettingsProps) {
  const { t } = useI18n()
  const c = t.memoryProviders

  const status = useQuery({
    queryKey: ['memory-status', profileScopeKey(profile)],
    queryFn: () => getMemoryStatus(profile),
    staleTime: 0
  })

  const [searchParams] = useSearchParams()
  const linkedProvider = searchParams.get('provider')
  const [inspected, setInspected] = useState<string | null>(linkedProvider)

  useEffect(() => {
    setInspected(linkedProvider)
  }, [linkedProvider])

  const active = status.data?.active || 'builtin'
  const rows = status.data ? providerRows(status.data, c.builtinDescription) : []
  const selected = rows.find(row => row.name === inspected)
  const statusLabels = { ready: c.ready, needs_config: c.needsConfig, missing: c.missing, unavailable: c.unavailable }

  const loadError = (
    <ErrorState title={c.loadFailed}>
      <Button onClick={() => void status.refetch()} size="sm">
        {c.retry}
      </Button>
    </ErrorState>
  )

  return (
    <section aria-label={c.title} className="grid gap-5">
      <div>
        <h2 className="font-medium">{c.title}</h2>
        <p className="mt-1 text-sm text-muted-foreground">{c.newSessions}</p>
      </div>
      {!status.data ? (
        status.isError ? (
          loadError
        ) : (
          <PageLoader className="min-h-24" label={c.loading} />
        )
      ) : (
        <>
          <div>
            {rows.map(row => (
              <ListRow
                action={
                  <Button
                    aria-label={`${c.inspect} ${row.name}`}
                    aria-pressed={inspected === row.name}
                    onClick={() => setInspected(row.name)}
                    size="sm"
                    variant="secondary"
                  >
                    {c.inspect}
                  </Button>
                }
                description={row.description}
                key={row.name}
                title={
                  <span className="flex items-center gap-2">
                    {row.name === 'builtin' ? c.builtin : row.name}
                    {row.name === active && <Pill>{c.active}</Pill>}
                    <Pill>{row.status ? statusLabels[row.status] : c.unknown}</Pill>
                  </span>
                }
              />
            ))}
          </div>
          {status.isError && loadError}
          {selected && (
            <ProviderDetail
              active={active}
              key={selected.name}
              profile={profile}
              refetch={status.refetch}
              row={selected}
            />
          )}
        </>
      )}
      <Button asChild className="justify-self-start" size="sm" variant="link">
        <Link state={{ capabilityScope: profile }} to={MEMORY_PLUGINS_ROUTE}>
          {c.explore}
        </Link>
      </Button>
    </section>
  )
}

interface ProviderDetailProps {
  active: string
  profile: OwnerScope
  refetch: () => Promise<{ data?: MemoryStatusResponse; error: unknown }>
  row: MemoryProviderStatus
}

function ProviderDetail({ active, profile, refetch, row }: ProviderDetailProps) {
  const { t } = useI18n()
  const c = t.memoryProviders
  const queryClient = useQueryClient()
  const [selecting, setSelecting] = useState(false)
  const [selectionFailed, setSelectionFailed] = useState(false)
  const isActive = row.name === active

  // Selection is verified by reading the owner back, never assumed from the PUT.
  async function selectProvider() {
    setSelecting(true)
    setSelectionFailed(false)

    try {
      const result = await setMemoryProvider(row.name, profile)
      const verified = await refetch()

      if (!result.ok || verified.error || (verified.data?.active || 'builtin') !== row.name) {
        throw new Error('Selection not verified')
      }
    } catch {
      setSelectionFailed(true)
    } finally {
      setSelecting(false)
    }
  }

  // A connected provider may now be ready, and its declared fields may have changed.
  const connected = () => {
    void refetch()
    void queryClient.invalidateQueries({ queryKey: ['memory-provider-config', profileScopeKey(profile), row.name] })
  }

  const install = () =>
    openPluginInstallRequest({
      repo: '',
      profile,
      legacyHint: 'agent',
      origin: { kind: 'memory', providerId: row.name }
    })

  return (
    <div>
      {(row.status === 'missing' || row.status === 'unavailable') && (
        <p className="text-sm text-muted-foreground">{c.repair}</p>
      )}
      {row.status === 'missing' ? (
        <Button onClick={install} size="sm" variant="secondary">
          {t.settings.plugins.installModal.installFromGit}
        </Button>
      ) : (
        <>
          {row.name !== 'builtin' && <MemoryConnect onConnected={connected} profile={profile} provider={row.name} />}
          <ProviderConfigPanel active={isActive} onSaved={() => void refetch()} profile={profile} provider={row.name} />
        </>
      )}
      {selectionFailed && (
        <div role="alert">
          <ErrorState title={c.selectionFailed} />
        </div>
      )}
      <div className="flex gap-2">
        <Button
          disabled={selecting || row.status !== 'ready' || isActive}
          onClick={() => void selectProvider()}
          size="sm"
        >
          {isActive ? c.active : c.use}
        </Button>
        <Button onClick={() => void refetch()} size="sm" variant="ghost">
          {c.retry}
        </Button>
      </div>
    </div>
  )
}

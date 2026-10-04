import { useStore } from '@nanostores/react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useRef, useState } from 'react'

import { $apiRequestScope, hermesApiAs, type ResolvedOwner } from '@/api/client'
import { getMemoryStatus, memoryDiscoveryKey } from '@/api/system'
import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { ErrorState } from '@/components/ui/error-state'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectSeparator,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import { useI18n } from '@/i18n'
import { openLink } from '@/lib/external-link'
import { CATALOG_ORIGIN } from '@/lib/plugin-catalog'
import { openPluginInstallRequest } from '@/store/plugin-install-request'

import { hermesConfigKey } from '../../hooks/use-config-record'
import { ListRow } from '../primitives'

import { MemoryConnect } from './connect'
import { ProviderConfigPanel } from './provider-config-panel'

export function MemoryProviderSelection({ profile }: { profile?: string }) {
  const ambient = useStore($apiRequestScope)

  const owner = useMemo(
    () => ({ connectionId: ambient.connectionId, profile: profile ?? ambient.profile }),
    [ambient.connectionId, ambient.profile, profile]
  )

  return <Selection key={JSON.stringify(owner)} owner={owner} />
}

function Selection({ owner }: { owner: ResolvedOwner }) {
  const { t } = useI18n()
  const c = t.memoryDiscovery
  const client = useQueryClient()
  const [inspected, setInspected] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(false)
  const pending = useRef(false)

  const { data, isError, refetch } = useQuery({
    queryKey: memoryDiscoveryKey(owner),
    queryFn: () => getMemoryStatus(owner)
  })

  const catalog =
    data?.catalog_providers?.filter(
      entry =>
        entry.featured &&
        !data.providers.some(provider => provider.name === entry.name && provider.status !== 'missing')
    ) ?? []

  const inspect = (value: string) => {
    if (value === 'explore:') {
      openLink(`${CATALOG_ORIGIN}/docs/plugins?kind=memory`)

      return
    }

    setInspected(value)
  }

  const selected = inspected ?? (data?.active || 'builtin')
  const entry = catalog.find(provider => provider.name === selected)
  const isInstalled = data?.providers.some(provider => provider.name === selected && provider.status !== 'missing')
  const isActive = selected === (data?.active || 'builtin')

  const ready =
    selected === 'builtin' ||
    data?.providers.some(provider => provider.name === selected && provider.status === 'ready')

  const refresh = async () => {
    const next = await getMemoryStatus(owner)
    client.setQueryData(memoryDiscoveryKey(owner), next)

    return next
  }

  const activateProvider = async () => {
    if (pending.current || !ready) {
      return
    }

    pending.current = true
    setBusy(true)
    setError(false)

    try {
      const result = await hermesApiAs<{ ok: boolean }>(owner, {
        path: '/api/memory/provider',
        method: 'PUT',
        body: { provider: selected === 'builtin' ? '' : selected }
      })

      if (!result.ok) {
        setError(true)

        return
      }

      const next = await refresh()

      if ((next.active || 'builtin') !== selected) {
        setError(true)

        return
      }

      // The generic settings cache is not selection authority. Never enqueue an
      // autosave with an old provider value when this independent mutation lands.
      // Retire only this owner's cached config; no background refetch through
      // an ambient query function after a connection switch.
      void client.invalidateQueries({
        queryKey: hermesConfigKey(owner.profile ?? undefined, owner.connectionId),
        exact: true,
        refetchType: 'none'
      })
    } catch {
      setError(true)
    } finally {
      pending.current = false
      setBusy(false)
    }
  }

  return (
    <>
      {isError ? (
        <ErrorState title={c.loadFailed}>
          <Button onClick={() => void refetch()} size="sm" variant="secondary">
            {t.common.retry}
          </Button>
        </ErrorState>
      ) : !data ? (
        <PageLoader className="min-h-16" label={c.providerSettings} />
      ) : null}
      {data && (
        <>
          <ListRow
            action={
              <div className="flex flex-wrap items-center justify-end gap-2">
                <Select onValueChange={inspect} value={selected}>
                  <SelectTrigger aria-label={c.providerSettings}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectGroup>
                      <SelectLabel className="py-1 text-[10px] leading-4 font-semibold tracking-wider uppercase">
                        {c.installed}
                      </SelectLabel>
                      <SelectItem value="builtin">{c.builtin}</SelectItem>
                      {data.providers
                        .filter(provider => provider.name !== 'builtin' && provider.status !== 'missing')
                        .map(provider => (
                          <SelectItem key={provider.name} value={provider.name}>
                            {provider.name}
                          </SelectItem>
                        ))}
                    </SelectGroup>
                    {data.providers
                      .filter(provider => provider.status === 'missing')
                      .map(provider => (
                        <SelectItem
                          key={provider.name}
                          value={provider.name}
                        >{`${provider.name} (${c.missing})`}</SelectItem>
                      ))}
                    {catalog.some(entry => !data.providers.some(provider => provider.name === entry.name)) && (
                      <SelectGroup>
                        <SelectSeparator />
                        <SelectLabel className="py-1 text-[10px] leading-4 font-semibold tracking-wider uppercase">
                          {c.availableToInstall}
                        </SelectLabel>
                        {catalog
                          .filter(entry => !data.providers.some(provider => provider.name === entry.name))
                          .map(entry => (
                            <SelectItem key={`catalog:${entry.name}`} value={entry.name}>
                              {entry.title}
                            </SelectItem>
                          ))}
                      </SelectGroup>
                    )}
                    <SelectSeparator />
                    <SelectItem value="explore:">{c.exploreAll}</SelectItem>
                  </SelectContent>
                </Select>
                {entry && (
                  <Button
                    onClick={() =>
                      openPluginInstallRequest({
                        catalogName: entry.name,
                        repo: entry.subdir ? `${entry.repo}#${entry.subdir}` : entry.repo,
                        sha: entry.sha,
                        profile: owner.profile,
                        legacyHint: 'agent',
                        enable: true,
                        memory: { name: entry.name, owner }
                      })
                    }
                    size="sm"
                    variant="secondary"
                  >
                    {c.reviewInstall}
                  </Button>
                )}
              </div>
            }
            description={
              <>
                <span>{`${c.active}: ${data.active || 'builtin'}`}</span>
                {entry && <span className="block">{c.installationRequired}</span>}
              </>
            }
            title={c.providerSettings}
          />
          {selected !== 'builtin' && isInstalled && (
            <MemoryConnect
              key={`connect:${selected}`}
              onConnected={refresh}
              owner={owner}
              profile={owner.profile ?? undefined}
              provider={selected}
            />
          )}
          {selected !== 'builtin' && isInstalled && (
            <ProviderConfigPanel
              isActive={isActive}
              key={`config:${selected}`}
              onSaved={async () => {
                await refresh()
              }}
              owner={owner}
              provider={selected}
            />
          )}
          {selected && !isActive && !entry && (
            <div className="flex items-center gap-3 py-3">
              <Button
                disabled={busy || !ready || isActive}
                onClick={() => void activateProvider()}
                size="sm"
                variant="secondary"
              >
                {c.useProvider}
              </Button>
              {!ready && (
                <>
                  <span className="text-sm text-muted-foreground">{c.notReady}</span>
                  <Button onClick={() => void refresh().catch(() => setError(true))} size="sm" variant="ghost">
                    {t.common.retry}
                  </Button>
                </>
              )}
            </div>
          )}
          {error && <span role="alert">{c.useFailed}</span>}
        </>
      )}
    </>
  )
}

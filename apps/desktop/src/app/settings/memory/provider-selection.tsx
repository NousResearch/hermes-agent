import { useStore } from '@nanostores/react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useRef, useState } from 'react'

import { $apiRequestScope, type ResolvedOwner } from '@/api/client'
import { getMemoryStatus, memoryDiscoveryKey, setMemoryProvider } from '@/api/system'
import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { ErrorState } from '@/components/ui/error-state'
import { useI18n } from '@/i18n'
import { openLink } from '@/lib/external-link'
import { CATALOG_ORIGIN } from '@/lib/plugin-catalog'
import { prettyName } from '@/lib/text'
import { openPluginInstallRequest } from '@/store/plugin-install-request'

import { hermesConfigKey } from '../../hooks/use-config-record'
import { ListRow, RowFootnoteAction } from '../primitives'
import { SETTING_IDS, settingElementId } from '../settings-manifest'

import { MemoryConnect } from './connect'
import { ProviderConfigPanel } from './provider-config-panel'
import { memoryProviderLabel, memoryRowState, type MemoryRowStep } from './provider-row-state'
import { EXPLORE_MEMORY_PLUGINS, MemoryProviderSelect } from './provider-select'

/** The provider row belongs to Memory › Persistent (and the unfiltered Memory page). */
export function MemoryProviderSection({
  profile,
  sectionId,
  subpage
}: {
  profile?: string
  sectionId: string
  subpage?: string
}) {
  return sectionId === 'memory' && (!subpage || subpage === 'persistent') ? (
    <MemoryProviderSelection profile={profile} />
  ) : null
}

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

  const row = memoryRowState(data, inspected)
  const label = (name: string) => memoryProviderLabel(name, data, row, c.builtin, prettyName)

  const inspect = (value: string) => {
    if (value === EXPLORE_MEMORY_PLUGINS) {
      openLink(`${CATALOG_ORIGIN}/docs/plugins?kind=memory`)

      return
    }

    setError(false)
    setInspected(value)
  }

  const refresh = async () => {
    const next = await getMemoryStatus(owner)
    client.setQueryData(memoryDiscoveryKey(owner), next)

    return next
  }

  const activateProvider = async () => {
    if (pending.current) {
      return
    }

    pending.current = true
    setBusy(true)
    setError(false)

    try {
      const result = await setMemoryProvider(owner, row.selected === 'builtin' ? '' : row.selected)
      const next = result.ok ? await refresh() : null

      if (!next || (next.active || 'builtin') !== row.selected) {
        setError(true)

        return
      }

      // The settings record caches memory.provider too: retire only this
      // owner's copy, without a refetch through the ambient connection.
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

  const openInstall = () => {
    const { entry } = row

    if (entry) {
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
  }

  if (isError) {
    return (
      <ErrorState title={c.loadFailed}>
        <Button onClick={() => void refetch()} size="sm" variant="secondary">
          {t.common.retry}
        </Button>
      </ErrorState>
    )
  }

  if (!data) {
    return <PageLoader className="min-h-16" label={t.settings.fieldLabels['memory.provider']} />
  }

  // One quiet follow-up in the settings footnote slot, never a second control beside the select.
  const steps: Record<Exclude<MemoryRowStep, null>, { label: string; run: () => void }> = {
    install: { label: c.reviewInstall, run: openInstall },
    use: { label: c.useProvider, run: () => void activateProvider() },
    retry: { label: t.common.retry, run: () => void refresh().catch(() => setError(true)) }
  }

  const footnote = row.step && !busy ? steps[row.step] : null
  const status = { install: c.installationRequired, retry: c.notReady, use: null }[row.step ?? 'use']
  const showProvider = row.selected !== 'builtin' && row.isInstalled

  return (
    <>
      <ListRow
        action={<MemoryProviderSelect label={label} onValueChange={inspect} row={row} />}
        below={
          <>
            {footnote && <RowFootnoteAction onClick={footnote.run}>{footnote.label}</RowFootnoteAction>}
            {error && (
              <p className="mt-1 text-[length:var(--conversation-caption-font-size)] text-destructive" role="alert">
                {c.useFailed}
              </p>
            )}
            {showProvider && (
              <div className="mt-1.5">
                <MemoryConnect
                  key={`connect:${row.selected}`}
                  onConnected={refresh}
                  owner={owner}
                  profile={owner.profile ?? undefined}
                  provider={row.selected}
                />
              </div>
            )}
          </>
        }
        description={
          row.isActive ? undefined : (
            <>
              <span className="block">{c.activeProvider(label(row.active))}</span>
              {status && <span className="block">{status}</span>}
            </>
          )
        }
        id={settingElementId(SETTING_IDS.memory.provider)}
        title={t.settings.fieldLabels['memory.provider']}
      />
      {showProvider && (
        <ProviderConfigPanel
          isActive={row.isActive}
          key={`config:${row.selected}`}
          onSaved={async () => {
            await refresh()
          }}
          owner={owner}
          provider={row.selected}
        />
      )}
    </>
  )
}

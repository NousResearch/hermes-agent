import { useCallback, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import type { HermesGateway } from '@/hermes'
import { Archive, Loader2, RefreshCw, Search } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'

import { ListRow, SectionHeading, SettingsContent } from './primitives'

interface ContinuitySettingsPayload {
  obsidian_allowlisted_folders?: string[]
  obsidian_last_indexed_at?: string
  obsidian_mirror_enabled?: boolean
  obsidian_read_enabled?: boolean
  obsidian_vault_path?: string
}

interface ContinuityStatus {
  enabled: boolean
  ledger_path?: string
  record_count?: number
  settings?: ContinuitySettingsPayload
}

interface ContinuityHit {
  id: string
  kind: string
  source_path?: string
  snippet?: string
}

export function ContinuitySettings({ gateway }: { gateway?: HermesGateway | null }) {
  const [status, setStatus] = useState<ContinuityStatus | null>(null)
  const [settings, setSettings] = useState<ContinuitySettingsPayload>({})
  const [query, setQuery] = useState('')
  const [hits, setHits] = useState<ContinuityHit[]>([])
  const [busy, setBusy] = useState(false)

  const load = useCallback(async () => {
    if (!gateway) {
      return
    }

    setBusy(true)

    try {
      const next = await gateway.request<ContinuityStatus>('continuity.status')
      setStatus(next)
      setSettings(next.settings ?? {})
    } catch (err) {
      notifyError(err, 'Could not load Context Continuity status')
    } finally {
      setBusy(false)
    }
  }, [gateway])

  useEffect(() => {
    void load()
  }, [load])

  const save = useCallback(
    async (patch: ContinuitySettingsPayload) => {
      if (!gateway) {
        return
      }

      const next = { ...settings, ...patch }
      setSettings(next)

      try {
        const saved = await gateway.request<ContinuitySettingsPayload>('continuity.settings.set', { settings: next })
        setSettings(saved)
        notify({ durationMs: 2_000, kind: 'success', message: 'Context Continuity settings saved' })
      } catch (err) {
        notifyError(err, 'Could not save Context Continuity settings')
        void load()
      }
    },
    [gateway, load, settings]
  )

  const runSearch = useCallback(async () => {
    if (!gateway || !query.trim()) {
      return
    }

    setBusy(true)

    try {
      const result = await gateway.request<{ hits: ContinuityHit[] }>('continuity.search', {
        limit: 5,
        query,
        timeout_s: 0.5
      })

      setHits(result.hits ?? [])
    } catch (err) {
      notifyError(err, 'Continuity search failed')
    } finally {
      setBusy(false)
    }
  }, [gateway, query])

  const indexObsidian = useCallback(async () => {
    if (!gateway) {
      return
    }

    setBusy(true)

    try {
      const result = await gateway.request<{ indexed?: number; skipped?: number }>('continuity.obsidian.index', {
        timeout_s: 2
      })

      notify({
        durationMs: 4_000,
        kind: 'success',
        message: `Indexed ${result.indexed ?? 0} note blocks; skipped ${result.skipped ?? 0}.`
      })
      void load()
    } catch (err) {
      notifyError(err, 'Obsidian indexing failed')
    } finally {
      setBusy(false)
    }
  }, [gateway, load])

  return (
    <SettingsContent>
      <SectionHeading icon={Archive} meta={String(status?.record_count ?? 0)} title="Context Continuity" />

      <ListRow
        action={
          <Button disabled={busy || !gateway} onClick={() => void load()} size="sm" type="button" variant="textStrong">
            {busy ? <Loader2 className="size-3.5 animate-spin" /> : <RefreshCw className="size-3.5" />}
            <span>Refresh</span>
          </Button>
        }
        description="Oversized Desktop chats continue in a child session, with a local dropoff ledger for recall."
        hint={status?.ledger_path}
        title={status?.enabled ? 'Enabled' : 'Waiting for gateway'}
      />

      <SectionHeading icon={Search} title="Local Recall" />
      <ListRow
        action={
          <Button disabled={busy || !query.trim()} onClick={() => void runSearch()} size="sm" type="button">
            <Search className="size-3.5" />
            <span>Search</span>
          </Button>
        }
        below={
          <input
            className="mt-2 h-8 w-full rounded-md border border-border bg-transparent px-2 text-sm outline-none focus:border-ring"
            onChange={event => setQuery(event.target.value)}
            placeholder="Search by project, file, error, task, or session"
            value={query}
          />
        }
        description="FTS search runs locally and is bounded so it never blocks chat continuation."
        title="Continuity records"
        wide
      />

      {hits.length > 0 && (
        <div className="grid gap-1">
          {hits.map(hit => (
            <ListRow
              description={hit.snippet}
              hint={hit.source_path || hit.id}
              key={hit.id}
              title={`${hit.kind}: ${hit.id}`}
              wide
            />
          ))}
        </div>
      )}

      <SectionHeading icon={Archive} title="Obsidian Recall" />
      <ListRow
        below={
          <div className="mt-2 grid gap-2">
            <input
              className="h-8 w-full rounded-md border border-border bg-transparent px-2 text-sm outline-none focus:border-ring"
              onBlur={() => void save({ obsidian_vault_path: settings.obsidian_vault_path || '' })}
              onChange={event => setSettings(prev => ({ ...prev, obsidian_vault_path: event.target.value }))}
              placeholder="/path/to/vault"
              value={settings.obsidian_vault_path ?? ''}
            />
            <input
              className="h-8 w-full rounded-md border border-border bg-transparent px-2 text-sm outline-none focus:border-ring"
              onBlur={() => void save({ obsidian_allowlisted_folders: settings.obsidian_allowlisted_folders ?? [] })}
              onChange={event =>
                setSettings(prev => ({
                  ...prev,
                  obsidian_allowlisted_folders: event.target.value
                    .split(',')
                    .map(v => v.trim())
                    .filter(Boolean)
                }))
              }
              placeholder="Projects, Notes/Hermes"
              value={(settings.obsidian_allowlisted_folders ?? []).join(', ')}
            />
          </div>
        }
        description="Read-only Markdown indexing. .obsidian, secrets, binaries, and oversized files are ignored."
        hint={settings.obsidian_last_indexed_at ? `Last indexed: ${settings.obsidian_last_indexed_at}` : undefined}
        title="Vault path and allowlisted folders"
        wide
      />
      <ListRow
        action={
          <div className="flex flex-wrap justify-end gap-2">
            <Button
              onClick={() => void save({ obsidian_read_enabled: !settings.obsidian_read_enabled })}
              size="sm"
              type="button"
              variant={settings.obsidian_read_enabled ? 'default' : 'outline'}
            >
              {settings.obsidian_read_enabled ? 'Read on' : 'Read off'}
            </Button>
            <Button
              onClick={() => void save({ obsidian_mirror_enabled: !settings.obsidian_mirror_enabled })}
              size="sm"
              type="button"
              variant={settings.obsidian_mirror_enabled ? 'default' : 'outline'}
            >
              {settings.obsidian_mirror_enabled ? 'Mirror on' : 'Mirror off'}
            </Button>
            <Button disabled={busy || !settings.obsidian_read_enabled} onClick={() => void indexObsidian()} size="sm">
              Index now
            </Button>
          </div>
        }
        description="Obsidian is optional; continuation and ledger search work when it is disabled."
        title="Optional note recall"
      />
    </SettingsContent>
  )
}

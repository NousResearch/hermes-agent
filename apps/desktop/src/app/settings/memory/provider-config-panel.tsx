import { useCallback, useEffect, useRef, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { getMemoryProviderConfig, runMemoryProviderAction, saveMemoryProviderConfig } from '@/hermes'
import { SlidersHorizontal } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'
import type { MemoryProviderConfig, MemoryProviderField, MemoryProviderSummary } from '@/types/hermes'

import { ListRow, Pill } from '../primitives'

import { FieldControl, FieldTitle } from './field-control'
import { ProviderConfigModal } from './provider-config-modal'

// Inline fields only: the compact panel must never re-write modal-owned keys.
function seedValues(config: MemoryProviderConfig): Record<string, string> {
  return Object.fromEntries(
    config.fields.filter(field => field.inline).map(field => [field.key, field.kind === 'secret' ? '' : field.value])
  )
}

export function ProviderConfigPanel({ profile, provider }: { profile?: string; provider: string }) {
  const [config, setConfig] = useState<MemoryProviderConfig | null>(null)
  const [loadError, setLoadError] = useState<null | string>(null)
  const [values, setValues] = useState<Record<string, string>>({})
  const [saved, setSaved] = useState<Record<string, string>>({})
  const [expanded, setExpanded] = useState(true)
  const [showModal, setShowModal] = useState(false)
  const [statusRevision, setStatusRevision] = useState(0)
  const refreshGeneration = useRef(0)

  const refresh = useCallback(async () => {
    const generation = ++refreshGeneration.current

    try {
      const next = await getMemoryProviderConfig(provider, profile)

      if (generation !== refreshGeneration.current) {
        return
      }

      const seed = seedValues(next)
      setConfig(next)
      setValues(seed)
      setSaved(seed)
      setLoadError(null)
      setStatusRevision(current => current + 1)
    } catch (err) {
      if (generation !== refreshGeneration.current) {
        return
      }

      setConfig(null)
      setLoadError(err instanceof Error ? err.message : 'Memory provider settings failed to load')
    }
  }, [profile, provider])

  useEffect(() => {
    setConfig(null)
    void refresh()

    return () => {
      refreshGeneration.current += 1
    }
  }, [refresh])

  useEffect(() => {
    const action = config?.status_action

    if (!action) {
      return
    }

    let cancelled = false

    void runMemoryProviderAction<NonNullable<MemoryProviderSummary['status']>>(provider, action, {}, profile)
      .then(status => {
        if (cancelled) {
          return
        }

        setConfig(current =>
          current
            ? {
                ...current,
                summary: { items: current.summary?.items ?? [], status }
              }
            : current
        )
      })
      .catch(() => {
        if (cancelled) {
          return
        }

        setConfig(current =>
          current
            ? {
                ...current,
                summary: {
                  items: current.summary?.items ?? [],
                  status: {
                    label: 'Unavailable',
                    message: `${current.label} status could not be checked.`,
                    state: 'unreachable'
                  }
                }
              }
            : current
        )
      })

    return () => {
      cancelled = true
    }
  }, [config?.status_action, profile, provider, statusRevision])

  // Autosave, matching the settings page around the panel: one-key partial PUT
  // on commit, silent on success, no full refresh (it would reset sibling drafts).
  const commitField = useCallback(
    async (field: MemoryProviderField, value: string) => {
      if (value === (saved[field.key] ?? '') || (field.kind === 'secret' && !value.trim())) {
        return
      }

      try {
        await saveMemoryProviderConfig(provider, { [field.key]: value }, profile)

        if (field.kind === 'secret') {
          setValues(current => ({ ...current, [field.key]: '' }))
          setConfig(
            current =>
              current && {
                ...current,
                fields: current.fields.map(f => (f.key === field.key ? { ...f, is_set: true } : f))
              }
          )
        } else {
          setSaved(current => ({ ...current, [field.key]: value }))
        }
      } catch (err) {
        notifyError(err, `Failed to save ${field.label}`)
      }
    },
    [profile, provider, saved]
  )

  // Providers without a declared config surface (e.g. builtin) render nothing.
  if (config && config.fields.length === 0) {
    return null
  }

  if (!config) {
    if (loadError) {
      return (
        <div className="flex items-center justify-between gap-3 py-2">
          <span className="text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
            Memory provider settings failed to load: {loadError}
          </span>
          <Button onClick={() => void refresh()} size="sm" type="button" variant="secondary">
            Retry
          </Button>
        </div>
      )
    }

    return <PageLoader className="min-h-24" label="Loading memory provider settings..." />
  }

  const inlineFields = config.fields.filter(field => field.inline)
  const secretFields = config.fields.filter(field => field.kind === 'secret')
  const hasFullConfig = config.fields.some(field => !field.inline)

  if (config.submit_action) {
    const status = config.summary?.status

    const statusTone =
      status?.state === 'healthy'
        ? 'text-emerald-600 dark:text-emerald-400'
        : status?.state === 'unhealthy'
          ? 'text-amber-600 dark:text-amber-400'
          : status?.state === 'checking'
            ? 'text-muted-foreground'
            : 'text-destructive'

    return (
      <section className="py-1">
        <div className="flex items-center justify-between gap-3 py-2">
          <div className="min-w-0">
            <div className="text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
              {config.label} settings
            </div>
            {config.description ? (
              <div className="mt-1 text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
                {config.description}
              </div>
            ) : null}
          </div>
          <Button onClick={() => setShowModal(true)} size="sm" type="button" variant="secondary">
            <SlidersHorizontal className="size-3.5" />
            Configure
          </Button>
        </div>

        {config.summary ? (
          <div className="@container">
            <div
              aria-label={`${config.label} connection summary`}
              className="grid gap-x-6 gap-y-2 rounded-md border border-(--ui-accent-secondary)/30 px-3 py-2.5 @2xl:grid-cols-[repeat(2,minmax(0,1fr))_auto] @2xl:items-center"
              role="group"
            >
              {config.summary.items.map(item => (
                <div className="min-w-0" key={item.label}>
                  <div className="text-[0.65rem] font-medium uppercase text-muted-foreground">{item.label}</div>
                  <div className="mt-0.5 truncate font-mono text-xs text-foreground" title={item.value}>
                    {item.value}
                  </div>
                </div>
              ))}
              {status ? (
                <button
                  className="min-w-0 text-left @2xl:text-right"
                  disabled={!status.message || status.state === 'healthy' || status.state === 'checking'}
                  onClick={() =>
                    notify({
                      kind: status.state === 'unhealthy' ? 'warning' : 'error',
                      title: status.label,
                      message: status.message
                    })
                  }
                  title={status.message || undefined}
                  type="button"
                >
                  <div className="text-[0.65rem] font-medium uppercase text-muted-foreground">Status</div>
                  <div className={`mt-0.5 inline-flex items-center gap-1.5 text-xs font-medium ${statusTone}`}>
                    <span className="size-1.5 rounded-full bg-current" />
                    {status.label}
                  </div>
                </button>
              ) : null}
            </div>
          </div>
        ) : null}

        <ProviderConfigModal
          config={config}
          onOpenChange={setShowModal}
          onSaved={refresh}
          open={showModal}
          profile={profile}
          provider={provider}
        />
      </section>
    )
  }

  return (
    <section className="py-1">
      <div className="flex items-center gap-2 py-2">
        <button
          aria-expanded={expanded}
          className="flex min-w-0 flex-1 items-center gap-2 text-left"
          onClick={() => setExpanded(open => !open)}
          type="button"
        >
          <DisclosureCaret open={expanded} />
          <span className="text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
            {config.label} settings
          </span>
          {secretFields.map(field => (
            <Pill key={field.key}>{field.is_set ? `${field.label} set` : `${field.label} not set`}</Pill>
          ))}
        </button>
        {hasFullConfig && (
          <Button onClick={() => setShowModal(true)} size="sm" type="button" variant="secondary">
            <SlidersHorizontal className="size-3.5" />
            Full config…
          </Button>
        )}
      </div>

      {expanded && (
        <div className="ml-1.5 border-l-2 border-(--ui-accent-secondary)/25 pb-4 pl-4 pr-4">
          {inlineFields.map(field => (
            <div className="border-b border-border/40 last:border-b-0" key={field.key}>
              <ListRow
                action={
                  <FieldControl
                    field={field}
                    onChange={value => setValues(current => ({ ...current, [field.key]: value }))}
                    onCommit={value => void commitField(field, value)}
                    value={values[field.key] ?? ''}
                  />
                }
                description={field.description}
                title={<FieldTitle field={field} />}
              />
            </div>
          ))}
        </div>
      )}

      {hasFullConfig && (
        <ProviderConfigModal
          config={config}
          onOpenChange={setShowModal}
          onSaved={refresh}
          open={showModal}
          profile={profile}
          provider={provider}
        />
      )}
    </section>
  )
}

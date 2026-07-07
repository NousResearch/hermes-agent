import { useCallback, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { getMemoryProviderConfig, runMemoryProviderAction, saveMemoryProviderConfig } from '@/hermes'
import { Loader2, Save, SlidersHorizontal } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'
import type { MemoryProviderConfig } from '@/types/hermes'

import { FieldControl, FieldTitle } from './field-control'
import { ListRow, LoadingState, Pill } from '../primitives'
import { ProviderConfigModal } from './provider-config-modal'

// Inline fields only: the compact panel must never re-write modal-owned keys.
function seedValues(config: MemoryProviderConfig): Record<string, string> {
  return Object.fromEntries(
    config.fields.filter(field => field.inline).map(field => [field.key, field.kind === 'secret' ? '' : field.value])
  )
}

export function ProviderConfigPanel({ provider }: { provider: string }) {
  const [config, setConfig] = useState<MemoryProviderConfig | null>(null)
  const [loadError, setLoadError] = useState<null | string>(null)
  const [values, setValues] = useState<Record<string, string>>({})
  const [expanded, setExpanded] = useState(true)
  const [saving, setSaving] = useState(false)
  const [runningAction, setRunningAction] = useState<null | string>(null)
  const [showModal, setShowModal] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const next = await getMemoryProviderConfig(provider)
      setConfig(next)
      setValues(seedValues(next))
      setLoadError(null)
    } catch (err) {
      setConfig(null)
      setLoadError(err instanceof Error ? err.message : 'Memory provider settings failed to load')
    }
  }, [provider])

  useEffect(() => {
    setConfig(null)
    void refresh()
  }, [refresh])

  const save = useCallback(async () => {
    if (!config) {
      return
    }

    setSaving(true)

    try {
      await saveMemoryProviderConfig(provider, values)
      notify({ kind: 'success', title: `${config.label} saved`, message: 'Memory provider configuration updated.' })
      await refresh()
    } catch (err) {
      notifyError(err, `Failed to save ${config.label} settings`)
    } finally {
      setSaving(false)
    }
  }, [config, provider, refresh, values])

  const runAction = useCallback(
    async (key: string, label: string) => {
      setRunningAction(key)
      try {
        const { result } = await runMemoryProviderAction(provider, key, values)
        notify({
          kind: 'success',
          title: label,
          message: typeof result.message === 'string' ? result.message : 'Action completed.'
        })
        await refresh()
      } catch (err) {
        notifyError(err, `${label} failed`)
      } finally {
        setRunningAction(null)
      }
    },
    [provider, refresh, values]
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
    return <LoadingState label="Loading memory provider settings..." />
  }

  const inlineFields = config.fields.filter(field => field.inline)
  const secretFields = config.fields.filter(field => field.kind === 'secret')
  const hasFullConfig = config.fields.some(field => !field.inline)

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
                    value={values[field.key] ?? ''}
                  />
                }
                description={field.description}
                title={<FieldTitle field={field} />}
              />
            </div>
          ))}

          <div className="flex items-center justify-end gap-2 pt-3">
            {config.actions.map(action => (
              <Button
                disabled={runningAction !== null || saving}
                key={action.key}
                onClick={() => void runAction(action.key, action.label)}
                size="sm"
                title={action.description || undefined}
                type="button"
                variant="secondary"
              >
                {runningAction === action.key && <Loader2 className="size-3.5 animate-spin" />}
                {action.label}
              </Button>
            ))}
            <Button disabled={saving || runningAction !== null} onClick={() => void save()} size="sm">
              {saving ? <Loader2 className="size-3.5 animate-spin" /> : <Save />}
              Save
            </Button>
          </div>
        </div>
      )}

      {hasFullConfig && (
        <ProviderConfigModal
          config={config}
          onOpenChange={setShowModal}
          onSaved={refresh}
          open={showModal}
          provider={provider}
        />
      )}
    </section>
  )
}

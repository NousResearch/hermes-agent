import { useCallback, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { getMemoryProviderConfig, saveMemoryProviderConfig } from '@/hermes'
import { TRANSLATIONS, useI18n } from '@/i18n'
import { Check, Loader2, Save } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'
import type { MemoryProviderConfig, MemoryProviderField } from '@/types/hermes'

import { CONTROL_TEXT } from './constants'
import { LoadingState, Pill } from './primitives'

/** Seed editable values from the schema: non-secret fields keep their current
 *  value, secret fields start blank (their value is never returned). */
function seedValues(config: MemoryProviderConfig): Record<string, string> {
  return Object.fromEntries(config.fields.map(field => [field.key, field.kind === 'secret' ? '' : field.value]))
}

function FieldControl({
  copy,
  field,
  value,
  onChange
}: {
  copy: NonNullable<(typeof TRANSLATIONS.en.settings)['memoryProvider']>
  field: MemoryProviderField
  value: string
  onChange: (value: string) => void
}) {
  if (field.kind === 'select') {
    const selected = field.options.find(option => option.value === value)
    const fieldDescription = copy.fieldDescriptions[field.key] || field.description
    const selectedDescription = selected
      ? copy.optionDescriptions[selected.value] || selected.description
      : fieldDescription

    return (
      <>
        <Select onValueChange={onChange} value={value}>
          <SelectTrigger className={CONTROL_TEXT}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {field.options.map(option => (
              <SelectItem key={option.value} value={option.value}>
                {copy.optionLabels[option.value] || option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {selectedDescription && <span className="text-xs text-muted-foreground">{selectedDescription}</span>}
      </>
    )
  }

  if (field.kind === 'secret') {
    return (
      <div className="flex flex-wrap items-center gap-2">
        <Input
          className="min-w-64 flex-1 font-mono"
          onChange={event => onChange(event.target.value)}
          placeholder={field.is_set ? copy.leaveBlankToKeep : copy.fieldPlaceholders[field.key] || field.placeholder}
          type="password"
          value={value}
        />
        {field.is_set && (
          <Pill tone="primary">
            <Check className="size-3" />
            {copy.set}
          </Pill>
        )}
      </div>
    )
  }

  return (
    <Input
      className="font-mono"
      onChange={event => onChange(event.target.value)}
      placeholder={copy.fieldPlaceholders[field.key] || field.placeholder}
      value={value}
    />
  )
}

export function ProviderConfigPanel({ provider }: { provider: string }) {
  const { t } = useI18n()
  const copy = t.settings.memoryProvider ?? TRANSLATIONS.en.settings.memoryProvider!
  const [config, setConfig] = useState<MemoryProviderConfig | null>(null)
  const [values, setValues] = useState<Record<string, string>>({})
  const [expanded, setExpanded] = useState(true)
  const [saving, setSaving] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const next = await getMemoryProviderConfig(provider)
      setConfig(next)
      setValues(seedValues(next))
    } catch (err) {
      notifyError(err, copy.failedLoad)
      setConfig(null)
    }
  }, [copy.failedLoad, provider])

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
      const providerLabel = copy.providerNames[config.name] || config.label
      notify({ kind: 'success', title: copy.savedTitle(providerLabel), message: copy.savedMessage })
      await refresh()
    } catch (err) {
      notifyError(err, copy.failedSave(copy.providerNames[config.name] || config.label))
    } finally {
      setSaving(false)
    }
  }, [config, copy, provider, refresh, values])

  // Providers without a declared config surface (e.g. builtin) render nothing.
  if (config && config.fields.length === 0) {
    return null
  }

  if (!config) {
    return <LoadingState label={copy.loading} />
  }

  const secretFields = config.fields.filter(field => field.kind === 'secret')
  const providerLabel = copy.providerNames[config.name] || config.label

  return (
    <section className="py-3">
      <button
        aria-expanded={expanded}
        className="flex w-full items-center justify-between gap-3 rounded-lg bg-background/60 px-3 py-2 text-left hover:bg-accent/50"
        onClick={() => setExpanded(open => !open)}
        type="button"
      >
        <span className="flex min-w-0 items-center gap-2">
          <DisclosureCaret open={expanded} />
          <span className="text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
            {copy.settings(providerLabel)}
          </span>
          {secretFields.map(field => (
            <Pill key={field.key}>
              {field.is_set
                ? copy.fieldSet(copy.fieldLabels[field.key] || field.label)
                : copy.fieldNotSet(copy.fieldLabels[field.key] || field.label)}
            </Pill>
          ))}
        </span>
      </button>

      {expanded && (
        <div className="mt-3 grid gap-4 rounded-xl bg-background/60 p-4">
          {config.fields.map(field => (
            <label className="grid gap-1.5" key={field.key}>
              <span className="text-xs font-medium text-muted-foreground">
                {copy.fieldLabels[field.key] || field.label}
              </span>
              <FieldControl
                copy={copy}
                field={field}
                onChange={value => setValues(current => ({ ...current, [field.key]: value }))}
                value={values[field.key] ?? ''}
              />
              {field.kind !== 'select' && (copy.fieldDescriptions[field.key] || field.description) && (
                <span className="text-xs text-muted-foreground">
                  {copy.fieldDescriptions[field.key] || field.description}
                </span>
              )}
            </label>
          ))}

          <div className="flex justify-end">
            <Button disabled={saving} onClick={() => void save()} size="sm">
              {saving ? <Loader2 className="size-3.5 animate-spin" /> : <Save />}
              {copy.save}
            </Button>
          </div>
        </div>
      )}
    </section>
  )
}

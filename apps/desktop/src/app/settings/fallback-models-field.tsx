import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { getGlobalModelOptions } from '@/hermes'
import { useI18n } from '@/i18n'
import { Plus, X } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $customModels, withCustomModels } from '@/store/custom-models'

import { CONTROL_TEXT } from './constants'
import { ModelSelect } from './model-select'

// An entry is `{provider, model}` plus whatever routing the user hand-wrote
// (`base_url`, `api_key`, `key_env`, `api_mode`, ...). The editor only edits
// the two selects; every other key rides along untouched, or an autosave
// would rewrite a local-gateway chain as bare provider/model pairs and route
// fallbacks to the public provider (#89184).
interface FallbackEntry extends Record<string, unknown> {
  provider: string
  model: string
}

// Normalize the raw config value (`fallback_providers`: a list of
// `{provider, model, ...}` dicts) into editor rows. Defensive against legacy
// string entries ("provider/model") so the editor never crashes on odd data.
function normalizeEntries(value: unknown): FallbackEntry[] {
  if (!Array.isArray(value)) {
    return []
  }

  return value.map(item => {
    if (item && typeof item === 'object') {
      const record = item as Record<string, unknown>

      return { ...record, provider: String(record.provider ?? ''), model: String(record.model ?? '') }
    }

    if (typeof item === 'string') {
      const slash = item.indexOf('/')

      return slash > 0
        ? { provider: item.slice(0, slash), model: item.slice(slash + 1) }
        : { provider: '', model: item }
    }

    return { provider: '', model: '' }
  })
}

function completeEntries(rows: FallbackEntry[]): FallbackEntry[] {
  return rows.filter(entry => entry.provider && entry.model)
}

function entriesEqual(a: FallbackEntry[], b: FallbackEntry[]): boolean {
  return a.length === b.length && a.every((entry, index) => JSON.stringify(entry) === JSON.stringify(b[index]))
}

/**
 * Structured editor for the top-level `fallback_providers` config list — a
 * chain of `{provider, model}` pairs tried in order when the default model
 * fails. Replaces the generic comma-string `list` input, which stringified the
 * objects to "[object Object], [object Object]".
 *
 * Mirrors the Auxiliary Models picker in `model-settings.tsx`: provider + model
 * selects sourced from `getGlobalModelOptions()`. Half-filled rows are kept in
 * local state and only complete pairs are emitted upward, so the config
 * autosave never persists a partial `{provider, model: ''}`.
 */
export function FallbackModelsField({
  value,
  onChange
}: {
  value: unknown
  onChange: (next: FallbackEntry[]) => void
}) {
  const { t } = useI18n()
  const m = t.settings.model

  const modelOptions = useQuery({
    queryKey: ['model-options', 'global'],
    queryFn: () => getGlobalModelOptions()
  })

  const customModels = useStore($customModels)

  const providers = withCustomModels(
    (modelOptions.data?.providers ?? []).filter(provider => provider.slug),
    customModels
  )

  const [rows, setRows] = useState<FallbackEntry[]>(() => normalizeEntries(value))
  // Last complete chain we emitted (or seeded). Autosave echoes the same
  // filtered list back through `value`; ignore that echo so draft rows stay.
  const lastEmittedRef = useRef(normalizeEntries(value))

  // Resync on real external changes (profile switch / config reload). Skip
  // when `value` is just our own commit echoing through the parent.
  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    const persisted = normalizeEntries(value)

    if (entriesEqual(persisted, lastEmittedRef.current)) {
      return
    }

    lastEmittedRef.current = persisted
    setRows(persisted)
  }, [value])

  const commit = (next: FallbackEntry[]) => {
    const complete = completeEntries(next)

    setRows(next)
    lastEmittedRef.current = complete
    onChange(complete)
  }

  const updateRow = (index: number, patch: Partial<FallbackEntry>) =>
    commit(rows.map((entry, i) => (i === index ? { ...entry, ...patch } : entry)))

  return (
    <div className="grid w-full gap-1.5">
      {rows.length === 0 && <p className="text-xs text-muted-foreground">{m.fallbackEmpty}</p>}
      {rows.map((entry, index) => {
        const providerRow = providers.find(provider => provider.slug === entry.provider)

        return (
          <div className="flex flex-wrap items-center gap-2" key={index}>
            <span className="w-4 shrink-0 text-center font-mono text-[0.7rem] text-muted-foreground">{index + 1}</span>
            <Select onValueChange={provider => updateRow(index, { provider, model: '' })} value={entry.provider}>
              <SelectTrigger className={cn('min-w-36', CONTROL_TEXT)}>
                <SelectValue placeholder={m.provider} />
              </SelectTrigger>
              <SelectContent>
                {providers.map(provider => (
                  <SelectItem key={provider.slug} value={provider.slug}>
                    {provider.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <ModelSelect
              className="min-w-52 flex-1"
              models={providerRow?.models ?? []}
              onValueChange={model => updateRow(index, { model })}
              provider={providerRow}
              providerSlug={entry.provider}
              value={entry.model}
            />
            <Button
              aria-label={t.common.remove}
              onClick={() => commit(rows.filter((_, i) => i !== index))}
              size="icon-xs"
              variant="ghost"
            >
              <X className="size-3.5" />
            </Button>
          </div>
        )
      })}
      <div>
        <Button onClick={() => commit([...rows, { provider: '', model: '' }])} size="sm" variant="textStrong">
          <Plus className="size-3.5" />
          {m.fallbackAdd}
        </Button>
      </div>
    </div>
  )
}

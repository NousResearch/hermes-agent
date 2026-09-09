import { useQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { getGlobalModelOptions } from '@/hermes'
import { useI18n } from '@/i18n'
import { Plus, X } from '@/lib/icons'
import { modelSearchText } from '@/lib/model-search-text'

import { SearchableSelect } from './searchable-select'

interface FallbackEntry {
  provider: string
  model: string
}

// Normalize the raw config value (`fallback_providers`: a list of
// `{provider, model}` dicts) into editor rows. Defensive against legacy string
// entries ("provider/model") so the editor never crashes on odd data.
function normalizeEntries(value: unknown): FallbackEntry[] {
  if (!Array.isArray(value)) {
    return []
  }

  return value.map(item => {
    if (item && typeof item === 'object') {
      const record = item as Record<string, unknown>

      return { provider: String(record.provider ?? ''), model: String(record.model ?? '') }
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
  return (
    a.length === b.length &&
    a.every((entry, index) => entry.provider === b[index]?.provider && entry.model === b[index]?.model)
  )
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

  const providers = (modelOptions.data?.providers ?? []).filter(provider => provider.slug)

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
        const catalog = providerRow?.models ?? []
        // Keep an out-of-catalog model selectable so an existing custom
        // provider/model renders instead of showing a blank box.
        const modelItems = entry.model && !catalog.includes(entry.model) ? [entry.model, ...catalog] : catalog

        return (
          <div className="flex flex-wrap items-center gap-2" key={index}>
            <span className="w-4 shrink-0 text-center font-mono text-[0.7rem] text-muted-foreground">{index + 1}</span>
            <SearchableSelect
              className="min-w-36"
              emptyMessage={m.noResults}
              onChange={provider => updateRow(index, { provider, model: '' })}
              options={providers.map(provider => ({
                value: provider.slug,
                label: provider.name,
                keywords: [provider.name, provider.slug]
              }))}
              placeholder={m.searchProvider}
              value={entry.provider}
            />
            <SearchableSelect
              className="min-w-52 flex-1"
              emptyMessage={m.noResults}
              onChange={model => updateRow(index, { model })}
              options={modelItems.map(model => ({
                value: model,
                keywords: [modelSearchText(model)]
              }))}
              placeholder={m.searchModel}
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

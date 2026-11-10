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

/** Figma empty-state badge icon (351:8605) — stacked layers. */
function FallbackStackIcon({ className }: { className?: string }) {
  return (
    <svg aria-hidden className={className} fill="none" height={24} viewBox="0 0 24 24" width={24}>
      <path
        d="M12.83 2.18C12.5694 2.06115 12.2864 1.99964 12 1.99964C11.7136 1.99964 11.4306 2.06115 11.17 2.18L2.6 6.08C2.42255 6.15824 2.27168 6.2864 2.16576 6.44886C2.05985 6.61131 2.00345 6.80106 2.00345 6.995C2.00345 7.18894 2.05985 7.37869 2.16576 7.54114C2.27168 7.7036 2.42255 7.83176 2.6 7.91L11.18 11.82C11.4406 11.9389 11.7236 12.0004 12.01 12.0004C12.2964 12.0004 12.5794 11.9389 12.84 11.82L21.42 7.92C21.5975 7.84176 21.7483 7.7136 21.8542 7.55114C21.9602 7.38869 22.0165 7.19894 22.0165 7.005C22.0165 6.81106 21.9602 6.62131 21.8542 6.45886C21.7483 6.2964 21.5975 6.16824 21.42 6.09L12.83 2.18Z"
        stroke="#2F5BEA"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
      />
      <path
        d="M22 12.5L13.42 16.41C13.1594 16.5289 12.8764 16.5904 12.59 16.5904C12.3036 16.5904 12.0206 16.5289 11.76 16.41L2.6 12.5"
        stroke="#2F5BEA"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
      />
      <path
        d="M22 17.5L13.42 21.41C13.1594 21.5289 12.8764 21.5904 12.59 21.5904C12.3036 21.5904 12.0206 21.5289 11.76 21.41L2.6 17.5"
        stroke="#2F5BEA"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
      />
    </svg>
  )
}

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

const ADD_FALLBACK_CLASS =
  'h-auto gap-2 rounded-full border-0 bg-gradient-to-r from-[#4361ee] to-[#c04de2] px-[18px] py-[11px] text-[14px] font-bold text-white shadow-none hover:brightness-105 hover:text-white'

/**
 * Structured editor for the top-level `fallback_providers` config list — a
 * chain of `{provider, model}` pairs tried in order when the default model
 * fails. Replaces the generic comma-string `list` input, which stringified the
 * objects to "[object Object], [object Object]".
 *
 * `page` layout matches Figma Settings → Model → Fallback models (351:7932):
 * title, description, empty-state card or editable chain card.
 */
export function FallbackModelsField({
  layout = 'embedded',
  onChange,
  value
}: {
  layout?: 'page' | 'embedded'
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

  const addRow = () => commit([...rows, { provider: '', model: '' }])

  const addButton = (
    <Button className={ADD_FALLBACK_CLASS} onClick={addRow} size="sm" type="button" variant="ghost">
      <Plus className="size-4" />
      {m.fallbackAdd}
    </Button>
  )

  const rowEditor = (
    <div className="grid w-full gap-2.5">
      {rows.map((entry, index) => {
        const providerRow = providers.find(provider => provider.slug === entry.provider)

        return (
          <div className="flex flex-wrap items-center gap-2" key={index}>
            <span className="w-5 shrink-0 text-center font-mono text-[0.7rem] text-[#94a3b8]">{index + 1}</span>
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
    </div>
  )

  if (layout === 'embedded') {
    return (
      <div className="grid w-full gap-1.5">
        {rows.length === 0 && <p className="text-xs text-muted-foreground">{m.fallbackEmpty}</p>}
        {rowEditor}
        <div>
          <Button onClick={addRow} size="sm" variant="textStrong">
            <Plus className="size-3.5" />
            {m.fallbackAdd}
          </Button>
        </div>
      </div>
    )
  }

  return (
    <section className="grid gap-5" data-model-fallbacks>
      <header className="space-y-2">
        <h1 className="text-[32px] font-extrabold leading-none tracking-tight text-[#0f172a] dark:text-foreground">
          {m.fallbackTitle}
        </h1>
        <p className="text-[14px] text-[#475569]">{m.fallbackDesc}</p>
      </header>

      {rows.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-4 rounded-[20px] border border-[#e2e8f0] bg-white px-8 py-12 shadow-[0_4px_16px_rgba(38,64,140,0.06)] dark:border-(--ui-stroke-tertiary) dark:bg-(--ui-chat-bubble-background) dark:shadow-none">
          <div className="flex size-14 items-center justify-center rounded-2xl bg-[#e3ecfd] dark:bg-[color-mix(in_srgb,var(--ui-blue)_16%,transparent)]">
            <FallbackStackIcon className="size-6" />
          </div>
          <div className="max-w-[520px] space-y-2 text-center">
            <p className="text-[18px] font-bold text-[#0f172a] dark:text-foreground">{m.fallbackEmptyTitle}</p>
            <p className="text-[14px] text-[#475569]">{m.fallbackEmptyDesc}</p>
          </div>
          {addButton}
        </div>
      ) : (
        <div className="flex flex-col gap-4 rounded-[20px] border border-[#e2e8f0] bg-white p-5 shadow-[0_4px_16px_rgba(38,64,140,0.06)] dark:border-(--ui-stroke-tertiary) dark:bg-(--ui-chat-bubble-background) dark:shadow-none">
          {rowEditor}
          <div className="flex justify-start">{addButton}</div>
        </div>
      )}
    </section>
  )
}

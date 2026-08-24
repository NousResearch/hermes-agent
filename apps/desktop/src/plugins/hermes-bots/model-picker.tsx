/**
 * Provider + model dropdowns backed by the gateway's `model.options`
 * inventory, plus the bounded fetch that keeps a wedged bot socket from
 * spinning the picker forever.
 *
 * Shared by the advanced profile editor and the create dialog.
 */

import {
  Button,
  GlyphSpinner,
  Input,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  useQuery
} from '@hermes/plugin-sdk'
import { useState } from 'react'

import { labeled } from './dialog-parts'
import { botRouteKey, requestForBot, resolveBotConnectionRoute } from './routing'
import { ID } from './shared'
import type { RosterRow } from './types'

// ── model picker (provider/model dropdowns via model.options) ───────────────

// #95279: the picker's catalog read rides the BOT's own socket — a lazily
// dialed second backend that can wedge (cold pool spawn, dropped remote hop)
// without the primary socket ever noticing. An unbounded RPC there left the
// query pending forever and the picker spinning ("never settles"). Bound every
// attempt: past the budget the query rejects and ModelPicker falls back to its
// free-text inputs instead of an eternal GlyphSpinner.
const MODEL_OPTIONS_SETTLE_MS = 20000

function boundedModelOptionsFetch<T>(fetch: Promise<T>, settleMs = MODEL_OPTIONS_SETTLE_MS): Promise<T> {
  // window.setTimeout (not bare setTimeout): bare vm test harnesses expose
  // timers only through the window shim. With no scheduler at all, degrade to
  // the old unbounded behavior rather than not fetching.
  const scope = typeof window === 'undefined' ? null : window

  if (!scope || typeof scope.setTimeout !== 'function') {
    return fetch
  }

  // `any`: the timer id is assigned synchronously by the executor below, but
  // it is still `null` on the declaration TypeScript sees from the closure,
  // and clearTimeout's signature takes `number | undefined`.
  let timerId: any = null

  const deadline = new Promise<never>((_, reject) => {
    timerId = scope.setTimeout(() => {
      reject(new Error(`model.options did not answer within ${Math.round(settleMs / 1000)}s (#95279 settle guard)`))
    }, settleMs)
  })

  return Promise.race([fetch, deadline]).finally(() => scope.clearTimeout(timerId))
}

/** One provider row of the gateway's `model.options` inventory. Entries in
 *  `models` are bare slugs on current gateways and objects on older ones. */
interface ModelProviderOption {
  models?: Array<string | { id?: string; name?: string }>
  name?: string
  slug: string
}
interface ModelOptionsResponse {
  providers?: ModelProviderOption[]
}

function useModelOptions(bot: null | RosterRow = null) {
  // Hook body runs during render: an orphaned row must paint the picker
  // disabled/erroring, not throw into the pane's error boundary.
  const resolved = bot ? resolveBotConnectionRoute(bot) : null
  const route = resolved?.status === 'resolved' ? resolved.route : null
  const orphaned = resolved?.status === 'owner_removed'

  return useQuery<ModelOptionsResponse>({
    queryKey: [ID, 'model-options', route ? botRouteKey(route) : 'active'],
    // No forced `refresh`: forcing a network read on EVERY mount bypassed the
    // staleTime cache, so each Bots view remount (tab re-front, dialog reopen,
    // pane visibility flip) knocked the picker back into its loading state and
    // discarded the user's staged selection mid-edit (#95279). The cached read
    // still refreshes per staleTime like every other surface's catalog.
    queryFn: () =>
      boundedModelOptionsFetch(
        requestForBot(bot, 'model.options', {
          include_unconfigured: true,
          explicit_only: false
        }) as Promise<ModelOptionsResponse>
      ),
    enabled: !orphaned,
    staleTime: 120000,
    retry: false
  })
}

/**
 * Provider + model dropdowns from the gateway's configured inventory — the
 * same data the core model picker shows. `value = {provider, model}`;
 * onChange receives the merged patch.
 */
/** The two fields a profile pins for its model. */
interface ModelSelection {
  model: string
  provider: string
}
/** ModelPicker only ever emits the field(s) it just changed, so consumers can
 *  test membership with `in` and leave the rest of their state untouched. */
type ModelSelectionPatch = { model: string } | { model: string; provider: string } | { provider: string }
interface ModelPickerProps {
  bot?: null | RosterRow
  onChange: (patch: ModelSelectionPatch) => void
  placeholderModel?: string
  value: ModelSelection
}

export function ModelPicker({ bot = null, value, onChange, placeholderModel = 'gateway default' }: ModelPickerProps) {
  const { data, isLoading, error } = useModelOptions(bot)

  // Hooks are ALWAYS declared up front, before any conditional return.
  // Declaring them after a return trips React error #310.
  const NONE = '__default__'
  const CUSTOM = '__custom__'
  const providers = (data?.providers || []).filter(p => p && p.slug)
  const isKnown = !value.provider || value.provider === NONE || providers.some(p => p.slug === value.provider)
  const [useFreeText, setUseFreeText] = useState(!isKnown)

  if (isLoading) {
    return (
      <div className="flex justify-center py-2">
        <GlyphSpinner className="text-(--ui-text-tertiary)" spinner="breathe" />
      </div>
    )
  }

  if (error || !providers.length) {
    // Fallback: free text (older gateway or empty inventory).
    return (
      <div className="grid grid-cols-2 gap-2.5">
        {labeled(
          'Provider',
          <Input
            onChange={event =>
              onChange({
                provider: event.target.value
              })
            }
            placeholder="omnirouter / 9router / nous …"
            value={value.provider}
          />
        )}
        {labeled(
          'Model',
          <Input
            onChange={event =>
              onChange({
                model: event.target.value
              })
            }
            placeholder="antigravity/gemini-3.6-flash-high"
            value={value.model}
          />
        )}
      </div>
    )
  }

  if (useFreeText) {
    return (
      <div className="flex flex-col gap-2">
        <div className="grid grid-cols-2 gap-2.5">
          {labeled(
            'Provider (Custom)',
            <Input
              onChange={event =>
                onChange({
                  provider: event.target.value
                })
              }
              placeholder="e.g. omnirouter, inferx, 9router"
              value={value.provider}
            />
          )}
          {labeled(
            'Model (Custom)',
            <Input
              onChange={event =>
                onChange({
                  model: event.target.value
                })
              }
              placeholder="e.g. antigravity/gemini-3.6-flash-high"
              value={value.model}
            />
          )}
        </div>
        <Button
          className="h-6 self-start text-xs text-(--ui-text-tertiary)"
          onClick={() => setUseFreeText(false)}
          size="sm"
          variant="ghost"
        >
          ← Back to dropdowns
        </Button>
      </div>
    )
  }

  const activeProvider = providers.find(p => p.slug === value.provider) || null

  const models = activeProvider
    ? (activeProvider.models || []).map(m => (typeof m === 'string' ? m : m.id || m.name || ''))
    : []

  return (
    <div className="grid grid-cols-[1fr_1.4fr] gap-2.5">
      {labeled(
        'Provider',
        <Select
          onValueChange={v => {
            if (v === NONE) {
              onChange({
                provider: '',
                model: ''
              })
            } else if (v === CUSTOM) {
              setUseFreeText(true)
            } else {
              const prov = providers.find(p => p.slug === v)
              const provModels = (prov?.models || []).map(m => (typeof m === 'string' ? m : m.id || m.name || ''))
              const first = provModels[0] || ''
              onChange({
                provider: v,
                model: prov && provModels.includes(value.model) ? value.model : first
              })
            }
          }}
          value={value.provider || NONE}
        >
          <SelectTrigger className="h-8 rounded-md">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={NONE}>Inherit (launch profile)</SelectItem>
            {providers.map(p => (
              <SelectItem key={p.slug} value={p.slug}>
                {p.name ? `${p.name} (${p.slug})` : p.slug}
              </SelectItem>
            ))}
            <SelectItem value={CUSTOM}>✏️ Enter manually…</SelectItem>
          </SelectContent>
        </Select>
      )}
      {labeled(
        'Model',
        activeProvider && models.length > 0 ? (
          <Select
            onValueChange={v =>
              onChange({
                model: v
              })
            }
            value={value.model || (models[0] ?? '')}
          >
            <SelectTrigger className="h-8 rounded-md">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {models.map(m => (
                <SelectItem key={m} value={m}>
                  {m}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        ) : (
          <Input
            onChange={event =>
              onChange({
                model: event.target.value
              })
            }
            placeholder={placeholderModel || 'e.g. model name'}
            value={value.model}
          />
        )
      )}
    </div>
  )
}

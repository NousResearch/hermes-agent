/**
 * Provider + model dropdowns backed by the gateway's `model.options`
 * inventory, plus the bounded fetch that keeps a wedged bot socket from
 * spinning the picker forever.
 *
 * Shared by the advanced profile editor and the create dialog.
 */

import {
  GlyphSpinner,
  Input,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  useQuery
} from '@hermes/plugin-sdk'

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
interface ModelOptionsResult {
  providers?: ModelProviderOption[]
}

function useModelOptions(bot: null | RosterRow = null) {
  // Hook body runs during render: an orphaned row must paint the picker
  // disabled/erroring, not throw into the pane's error boundary.
  const resolved = bot ? resolveBotConnectionRoute(bot) : null
  const route = resolved?.status === 'resolved' ? resolved.route : null
  const orphaned = resolved?.status === 'owner_removed'

  return useQuery<ModelOptionsResult>({
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
        }) as Promise<ModelOptionsResult>
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

  const providers = (data?.providers || []).filter(p => p && p.slug)

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

  const activeProvider = providers.find(p => p.slug === value.provider) || null

  const models = activeProvider
    ? (activeProvider.models || []).map(m => (typeof m === 'string' ? m : m.id || m.name || ''))
    : []

  return (
    <div className="grid grid-cols-[1fr_1.4fr] gap-2.5">
      {labeled(
        'Provider',
        <>
          <Input
            aria-label="Provider"
            list="hermes-bot-provider-options"
            onChange={event => {
              const provider = event.target.value
              const selected = providers.find(p => p.slug.toLowerCase() === provider.trim().toLowerCase())
              const canonicalProvider = selected?.slug || provider

              const providerModels = (selected?.models || []).map(m =>
                typeof m === 'string' ? m : m.id || m.name || ''
              )

              const nextModel = selected
                ? providerModels.includes(value.model)
                  ? value.model
                  : providerModels[0] || ''
                : provider
                  ? value.model
                  : ''

              onChange({
                provider: canonicalProvider,
                model: nextModel
              })
            }}
            placeholder="type or choose a provider"
            value={value.provider}
          />
          <datalist id="hermes-bot-provider-options">
            <option label="Inherit (launch profile)" value="" />
            {providers.map(p => (
              <option key={p.slug} label={p.name ? `${p.name} (${p.slug})` : p.slug} value={p.slug} />
            ))}
          </datalist>
        </>
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

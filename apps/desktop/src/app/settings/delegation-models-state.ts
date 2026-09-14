/** Config-only state for the #67523 picker + #80479 fallback-matrix integration. */
export interface DelegationModelProviderValue {
  provider: string
  model: string
}

export interface DelegationFallbackEntry extends DelegationModelProviderValue {
  [key: string]: unknown
}

export type DelegationFallbackMode = 'auto' | 'inherit' | 'none' | 'custom' | 'invalid'

export interface DelegationModelsDraft extends DelegationModelProviderValue {
  mode: DelegationFallbackMode
  rows: DelegationFallbackEntry[]
  clearEndpoint: boolean
}

export const asRecord = (value: unknown): Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {}

const text = (value: unknown) => (typeof value === 'string' ? value : '')

export const pairComplete = (pair: DelegationModelProviderValue): boolean =>
  !!pair.provider.trim() && !!pair.model.trim()

export function readDelegationModels(value: unknown): DelegationModelsDraft {
  const config = asRecord(value)
  const raw = config.fallback_providers ?? config.fallback_chain ?? config.fallback_model
  const entries = Array.isArray(raw) ? raw : raw !== null && typeof raw === 'object' ? [raw] : []
  let mode: DelegationFallbackMode = 'invalid'

  if (raw == null) {
    mode = 'auto'
  } else if (typeof raw === 'string' && ['inherit', 'parent'].includes(raw.trim().toLowerCase())) {
    mode = 'inherit'
  } else if (Array.isArray(raw) && raw.length === 0) {
    mode = 'none'
  } else if (entries.length && entries.every(entry => {
    const row = asRecord(entry)

    return pairComplete({ provider: text(row.provider), model: text(row.model) })
  })) {
    mode = 'custom'
  }

  return {
    provider: text(config.provider),
    model: text(config.model),
    mode,
    rows: mode === 'custom' ? entries.map(entry => ({ ...asRecord(entry) } as DelegationFallbackEntry)) : [],
    clearEndpoint: false
  }
}

export function delegationDraftValid(draft: DelegationModelsDraft, baseline: unknown): boolean {
  const original = readDelegationModels(baseline)
  const routeChanged = draft.clearEndpoint || draft.provider !== original.provider || draft.model !== original.model

  // Preserve an existing provider-default selection, but never persist the
  // temporary provider/new-empty-model state during a new selection (#67523).
  if (routeChanged && draft.provider.trim() && !draft.model.trim()) {
    return false
  }

  return draft.mode !== 'invalid' && (draft.mode !== 'custom' || (draft.rows.length > 0 && draft.rows.every(pairComplete)))
}

export function delegationModelsPatch(baseline: unknown, draft: DelegationModelsDraft): Record<string, unknown> {
  if (!delegationDraftValid(draft, baseline)) {
    throw new Error('Incomplete delegation selection')
  }

  const original = readDelegationModels(baseline)
  const patch: Record<string, unknown> = {}

  if (draft.clearEndpoint || draft.provider !== original.provider || draft.model !== original.model) {
    patch.provider = draft.provider.trim()
    patch.model = draft.model.trim()
  }

  if (draft.clearEndpoint) {
    // Provider selection means its configured route, not a stale direct URL
    // and credential left by a previous override. Model-only edits retain it.
    patch.base_url = ''
    patch.api_key = ''
    patch.api_mode = ''
    patch.request_overrides = null
  }

  if (draft.mode !== original.mode || JSON.stringify(draft.rows) !== JSON.stringify(original.rows)) {
    patch.fallback_providers = draft.mode === 'auto' ? null
      : draft.mode === 'inherit' ? 'inherit'
        : draft.mode === 'none' ? []
          : draft.rows.map(entry => ({ ...entry, provider: entry.provider.trim(), model: entry.model.trim() }))
    // A deliberate reset must not reanimate a legacy alias (#101017).
    patch.fallback_chain = null
    patch.fallback_model = null
  }

  return patch
}

export function delegationDraftChanged(draft: DelegationModelsDraft, baseline: unknown): boolean {
  return JSON.stringify(draft) !== JSON.stringify(readDelegationModels(baseline))
}

/** Provider changes drop only that row's old endpoint/auth; model edits and moves preserve it. */
export function updateDelegationFallback(
  row: DelegationFallbackEntry, pair: DelegationModelProviderValue, providerChanged: boolean
): DelegationFallbackEntry {
  return providerChanged ? { ...pair } : { ...row, ...pair }
}

export function moveDelegationFallback(rows: DelegationFallbackEntry[], index: number, delta: number): DelegationFallbackEntry[] {
  const target = index + delta

  if (index < 0 || index >= rows.length || target < 0 || target >= rows.length) {
    return rows
  }

  const next = [...rows]

  ;[next[index], next[target]] = [next[target], next[index]]

  return next
}

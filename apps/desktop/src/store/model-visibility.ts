import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'
import type { ModelOptionProvider } from '@/types/hermes'
import { $gateway } from './gateway'

const STORAGE_KEY = 'hermes.desktop.visible-models'

/** Models shown per provider in the status-bar dropdown before the user has
 *  customized the list. Backend `models` are already relevance-ordered. */
export const DEFAULT_VISIBLE_PER_PROVIDER = 50

/** Stable key for a provider/model pair (`::` avoids colliding with model ids
 *  that contain a single colon, e.g. `model:tag`). */
export const modelVisibilityKey = (provider: string, model: string): string => `${provider}::${model}`

/** Sentinel key suffix stored when the user explicitly hides ALL models for a
 *  provider.  Distinguishes "user hid everything" from "never customized" so
 *  `effectiveVisibleKeys` does not re-add defaults for that provider. */
export const EMPTY_PROVIDER_SENTINEL = ''

/** Build the sentinel key for a provider whose last model was toggled off. */
export const emptyProviderSentinelKey = (provider: string): string =>
  modelVisibilityKey(provider, EMPTY_PROVIDER_SENTINEL)

/** Check whether a stored key is a provider-hidden sentinel. */
export const isProviderSentinel = (key: string): boolean => key.endsWith('::')

/** A model and its optional `…-fast` sibling, collapsed into one logical row.
 *  `id` is the canonical (base) model; `fastId` is the fast variant if present. */
export interface ModelFamily {
  fastId: string | null
  id: string
}

/** Collapse a provider's model list so a base model and its `…-fast` variant
 *  become a single family (one row, one toggle). Order is preserved by the
 *  base model's position. A `…-fast` model with no base stands on its own. */
export function collapseModelFamilies(models: readonly string[]): ModelFamily[] {
  const present = new Set(models)
  const families: ModelFamily[] = []
  const consumed = new Set<string>()

  for (const model of models) {
    if (consumed.has(model)) {
      continue
    }

    if (/-fast$/i.test(model) && present.has(model.replace(/-fast$/i, ''))) {
      // Represented by its base entry — the base attaches it as `fastId`.
      continue
    }

    if (/-\d{8}$/.test(model) && present.has(model.replace(/-\d{8}$/, ''))) {
      // A date-pinned snapshot superseded by its rolling alias — drop the dupe.
      continue
    }

    const fastId = `${model}-fast`
    const hasFast = present.has(fastId)
    families.push({ fastId: hasFast ? fastId : null, id: model })
    consumed.add(model)

    if (hasFast) {
      consumed.add(fastId)
    }
  }

  return families
}

// ── Backend persistence ──────────────────────────────────────────────
// Model visibility preferences are persisted on the backend so they survive
// cache clears, origin changes, and Electron userData resets.

/** Try to load visibility keys from the backend. Returns null on failure. */
async function loadVisibleFromBackend(): Promise<Set<string> | null> {
  const gateway = $gateway.get()
  if (!gateway) return null
  try {
    const result = await gateway.request<{ keys: string[] }>('model_visibility.get')
    if (result && Array.isArray(result.keys)) {
      return new Set(result.keys.filter((x): x is string => typeof x === 'string'))
    }
    return null
  } catch {
    return null
  }
}

/** Persist visibility keys to the backend. Best-effort; localStorage fallback still works. */
async function saveVisibleToBackend(keys: Set<string>): Promise<void> {
  const gateway = $gateway.get()
  if (!gateway) return
  try {
    await gateway.request('model_visibility.set', { keys: [...keys] })
  } catch {
    // Best-effort
  }
}

/** Migrate existing localStorage data to the backend. Returns the migrated
 *  set, or null if there was nothing to migrate. */
function migrateFromLocalStorage(): Set<string> | null {
  const raw = storedString(STORAGE_KEY)
  if (!raw) return null
  try {
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? new Set(parsed.filter((x): x is string => typeof x === 'string')) : null
  } catch {
    return null
  }
}

/** Call once after gateway boot to sync visibility preferences: load from
 *  backend, migrate localStorage if needed, store into the atom. */
export async function syncVisibleModels(): Promise<void> {
  const backend = await loadVisibleFromBackend()
  if (backend !== null) {
    $visibleModels.set(backend)
    // Keep localStorage in sync for offline resilience
    persistString(STORAGE_KEY, JSON.stringify([...backend]))
    return
  }

  const local = loadVisible()
  if (local !== null) {
    // Migrate localStorage data to backend (fire-and-forget)
    saveVisibleToBackend(local)
    return
  }

  // Neither has data — keep the null sentinel (curated defaults apply)
}

function loadVisible(): Set<string> | null {
  const raw = storedString(STORAGE_KEY)

  if (!raw) {
    return null
  }

  try {
    const parsed = JSON.parse(raw)

    return Array.isArray(parsed) ? new Set(parsed.filter((x): x is string => typeof x === 'string')) : null
  } catch {
    return null
  }
}

/** Explicit set of visible `provider::model` keys, or null when the user
 *  hasn't customized — in which case the curated default applies. */
export const $visibleModels = atom<Set<string> | null>(loadVisible())

export const $modelVisibilityOpen = atom(false)

export function setVisibleModels(keys: Set<string>): void {
  $visibleModels.set(new Set(keys))
  persistString(STORAGE_KEY, JSON.stringify([...keys]))
  // Fire-and-forget backend persist for durability across storage contexts
  saveVisibleToBackend(keys)
}

export function setModelVisibilityOpen(open: boolean): void {
  $modelVisibilityOpen.set(open)
}

/** The default-visible key set: the curated top-N per provider. Used both as
 *  the dropdown fallback and to seed the Edit Models dialog. */
export function defaultVisibleKeys(providers: readonly ModelOptionProvider[]): Set<string> {
  const keys = new Set<string>()

  for (const provider of providers) {
    expandProviderDefaults(provider, keys)
  }

  return keys
}

/** Add a provider's curated default model keys (top-N collapsed families) to
 *  `target`. Shared by `defaultVisibleKeys` and `resolveVisibleKeys` so the
 *  expansion rule lives in exactly one place. */
function expandProviderDefaults(provider: ModelOptionProvider, target: Set<string>): void {
  const families = collapseModelFamilies(provider.models ?? [])

  for (const family of families.slice(0, DEFAULT_VISIBLE_PER_PROVIDER)) {
    target.add(modelVisibilityKey(provider.slug, family.id))
  }
}

/** Resolve the canonical working set: the user's stored keys plus the curated
 *  default expansion for any provider they haven't customized. Hide-all
 *  sentinels are PRESERVED here — this is the set the toggle handler mutates and
 *  persists, so dropping a sentinel would silently re-enable a provider the user
 *  emptied. Use `effectiveVisibleKeys` for display (sentinels stripped). */
export function resolveVisibleKeys(stored: Set<string> | null, providers: readonly ModelOptionProvider[]): Set<string> {
  if (!stored) {
    return defaultVisibleKeys(providers)
  }

  if (stored.size === 0) {
    return new Set()
  }

  const next = new Set(stored)

  for (const provider of providers) {
    const providerPrefix = `${provider.slug}::`

    const hasStoredProvider = [...stored].some(key => key.startsWith(providerPrefix) && !isProviderSentinel(key))

    const hasSentinel = stored.has(emptyProviderSentinelKey(provider.slug))

    if (hasStoredProvider || hasSentinel) {
      continue
    }

    expandProviderDefaults(provider, next)
  }

  return next
}

/** Resolve which keys are currently visible for DISPLAY: the resolved working
 *  set with bookkeeping sentinels stripped (they are not real models). */
export function effectiveVisibleKeys(
  stored: Set<string> | null,
  providers: readonly ModelOptionProvider[]
): Set<string> {
  const next = resolveVisibleKeys(stored, providers)

  // Strip sentinel keys — they are bookkeeping, not real visibility entries.
  for (const key of [...next]) {
    if (isProviderSentinel(key)) {
      next.delete(key)
    }
  }

  return next
}

/** Compute the next persisted visibility set when one model row is toggled.
 *  Seeds from `resolveVisibleKeys` (NOT `effectiveVisibleKeys`) so other
 *  providers' hide-all sentinels survive the persist. When the last visible
 *  model of a provider is toggled off, a sentinel records the explicit
 *  hide-all; re-enabling a model clears THAT provider's sentinel (only). */
export function toggleModelVisibility(
  stored: Set<string> | null,
  providers: readonly ModelOptionProvider[],
  providerSlug: string,
  model: string
): Set<string> {
  // `resolveVisibleKeys` always returns a fresh Set, so we can mutate it directly.
  const next = resolveVisibleKeys(stored, providers)
  const key = modelVisibilityKey(providerSlug, model)
  const sentinel = emptyProviderSentinelKey(providerSlug)

  if (next.has(key)) {
    next.delete(key)

    // Check if this was the last real model for this provider.
    const remainingForProvider = [...next].some(k => k.startsWith(`${providerSlug}::`) && !isProviderSentinel(k))

    if (!remainingForProvider) {
      next.add(sentinel)
    }
  } else {
    // Re-enabling promotes a previously hidden-all provider to an explicit
    // set of exactly the one re-enabled model — the curated defaults are NOT
    // restored. Intentional: "you hid everything, you get back only what you
    // re-enable." (Locked in by the sentinel-clear-on-re-enable test.)
    next.delete(sentinel)
    next.add(key)
  }

  return next
}

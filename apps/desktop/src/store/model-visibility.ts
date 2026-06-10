import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'
import type { ModelOptionProvider } from '@/types/hermes'

const STORAGE_KEY = 'hermes.desktop.visible-models'

/** Models shown per provider in the status-bar dropdown before the user has
 *  customized the list. Backend `models` are already relevance-ordered. */
export const DEFAULT_VISIBLE_PER_PROVIDER = 50

/** Stable key for a provider/model pair (`::` avoids colliding with model ids
 *  that contain a single colon, e.g. `model:tag`). */
export const modelVisibilityKey = (provider: string, model: string): string => `${provider}::${model}`

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
}

export function setModelVisibilityOpen(open: boolean): void {
  $modelVisibilityOpen.set(open)
}

/** The default-visible key set: the curated top-N per provider. Used both as
 *  the dropdown fallback and to seed the Edit Models dialog. */
export function defaultVisibleKeys(providers: readonly ModelOptionProvider[]): Set<string> {
  const keys = new Set<string>()

  for (const provider of providers) {
    const families = collapseModelFamilies(provider.models ?? [])

    for (const family of families.slice(0, DEFAULT_VISIBLE_PER_PROVIDER)) {
      keys.add(modelVisibilityKey(provider.slug, family.id))
    }
  }

  return keys
}

/** Collect all family base-ids that belong to a provider (from current
 *  backend response). */
export function providerFamilyIds(
  provider: { slug: string; models?: readonly string[] },
): string[] {
  return collapseModelFamilies(provider.models ?? []).map(family => family.id)
}

/** Ensure every family of a provider is present in the visible set. */
export function enableProvider(provider: { slug: string; models?: readonly string[] }, keys: Set<string>): void {
  // Remove tombstone if present — this provider is now active again.
  keys.delete(`${provider.slug}::`)

  for (const id of providerFamilyIds(provider)) {
    keys.add(modelVisibilityKey(provider.slug, id))
  }
}

/** Remove every key belonging to a provider from the set, leaving a tombstone
 *  (`slug::`) so `effectiveVisibleKeys` knows this was an intentional choice,
 *  not "never customized". */
export function disableProvider(provider: { slug: string; models?: readonly string[] }, keys: Set<string>): void {
  const prefix = `${provider.slug}::`
  for (const key of [...keys]) {
    if (key.startsWith(prefix) && key.length > prefix.length) {
      keys.delete(key)
    }
  }
  // Tombstone — prevents defaults from being auto-added back.
  keys.add(prefix)
}

/** Toggle an entire provider's visibility in one shot. If any family of the
 *  provider is currently visible, disable all; otherwise enable all. */
export function toggleProviderVisibility(
  provider: { slug: string; models?: readonly string[] },
  keys: Set<string>,
): void {
  const prefix = `${provider.slug}::`
  const anyVisible = [...keys].some(key => key.startsWith(prefix) && key.length > prefix.length)

  if (anyVisible) {
    disableProvider(provider, keys)
  } else {
    enableProvider(provider, keys)
  }
}

/** Resolve which keys are currently visible: the user's explicit set when
 *  configured, otherwise the curated default for the given providers. */
export function effectiveVisibleKeys(
  stored: Set<string> | null,
  providers: readonly ModelOptionProvider[]
): Set<string> {
  if (!stored) {
    return defaultVisibleKeys(providers)
  }

  if (stored.size === 0) {
    return new Set()
  }

  const next = new Set(stored)

  for (const provider of providers) {
    const providerPrefix = `${provider.slug}::`
    const hasStoredProvider = [...stored].some(key => key.startsWith(providerPrefix))

    if (hasStoredProvider) {
      continue
    }

    const families = collapseModelFamilies(provider.models ?? [])

    for (const family of families.slice(0, DEFAULT_VISIBLE_PER_PROVIDER)) {
      next.add(modelVisibilityKey(provider.slug, family.id))
    }
  }

  return next
}

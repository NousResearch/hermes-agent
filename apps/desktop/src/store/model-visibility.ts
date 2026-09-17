import type { ModelOptionProvider } from '@hermes/shared'
import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

const STORAGE_KEY = 'hermes.desktop.visible-models'

// Companion key: the `provider::family` keys the catalog contained the last
// time visibility was committed. It is what lets a model the catalog added
// AFTER the user last curated be distinguished from one they hid on purpose —
// without it both collapse to "absent from the visible set" and new models are
// silently frozen out of the dropdown forever (#107391 root cause B, #114369).
const SEEN_STORAGE_KEY = 'hermes.desktop.visible-models.seen'

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

function loadSeenFamilies(): Set<string> | null {
  const raw = storedString(SEEN_STORAGE_KEY)

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

/** Every `provider::family` key the catalog contained the last time visibility
 *  state was committed (or backfilled on first read after the key landed).
 *  `resolveVisibleKeys` treats a catalog family ABSENT from this baseline as
 *  "added after the user last curated" and defaults it visible — the visible
 *  set alone cannot express that, because hides are expressed as absence too,
 *  and that collision froze the dropdown on a point-in-time snapshot
 *  (#107391 root cause B, #114369). Keys present in the baseline and absent
 *  from the visible set are deliberate hides and stay hidden. */
export const $seenFamilies = atom<Set<string> | null>(loadSeenFamilies())

/** Every collapsed-family key the current catalog exposes, across providers. */
export function allCatalogKeys(providers: readonly ModelOptionProvider[]): Set<string> {
  const keys = new Set<string>()

  for (const provider of providers) {
    for (const family of collapseModelFamilies(provider.models ?? [])) {
      keys.add(modelVisibilityKey(provider.slug, family.id))
    }
  }

  return keys
}

/** Adopt a seen baseline for an install that curated visibility before this key
 *  existed: adopt the CURRENT catalog wholesale, so an upgrade changes nothing
 *  the user can see — deliberate hides (indistinguishable from frozen-out
 *  additions in legacy state) stay hidden — and every addition from here on
 *  defaults visible. Persist-on-read; a null visible set means "never
 *  customized" and needs no baseline. Returns the baseline in force after the
 *  call (null when none applies yet). Callers: the two curation surfaces. */
export function ensureSeenBaseline(
  stored: Set<string> | null,
  providers: readonly ModelOptionProvider[],
  seen: Set<string> | null = $seenFamilies.get()
): Set<string> | null {
  if (seen !== null) {
    return seen
  }

  if (!stored || providers.length === 0) {
    return null
  }

  const keys = allCatalogKeys(providers)

  $seenFamilies.set(keys)
  persistString(SEEN_STORAGE_KEY, JSON.stringify([...keys]))

  return keys
}

export const $modelVisibilityOpen = atom(false)

export function setVisibleModels(keys: Set<string>, providers?: readonly ModelOptionProvider[]): void {
  $visibleModels.set(new Set(keys))
  persistString(STORAGE_KEY, JSON.stringify([...keys]))

  // Committing visibility re-baselines "seen": every family in the catalog the
  // user just looked at is now something they've had the chance to hide. The
  // next catalog addition is what defaults back to visible.
  if (providers) {
    const seen = allCatalogKeys(providers)

    $seenFamilies.set(seen)
    persistString(SEEN_STORAGE_KEY, JSON.stringify([...seen]))
  }
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

/** Add a provider's curated default model keys to `target`. Prefers the
 *  backend's `featured_models` shortlist (one flagship per lab) for aggregator
 *  providers that would otherwise flood the default view with dozens of models;
 *  falls back to the top-N collapsed families when a provider ships no featured
 *  list. Shared by `defaultVisibleKeys` and `resolveVisibleKeys` so the
 *  expansion rule lives in exactly one place. */
function expandProviderDefaults(provider: ModelOptionProvider, target: Set<string>): void {
  const families = collapseModelFamilies(provider.models ?? [])

  const featured = provider.featured_models ?? []

  const defaults = featured.length
    ? families.filter(family => featured.includes(family.id))
    : families.slice(0, DEFAULT_VISIBLE_PER_PROVIDER)

  for (const family of defaults) {
    target.add(modelVisibilityKey(provider.slug, family.id))
  }
}

/** Resolve the canonical working set: the user's stored keys plus the curated
 *  default expansion for any provider they haven't customized, plus every
 *  catalog family ABSENT from `seen` (the additions since the user last
 *  curated — visible by default, #114369). Hide-all sentinels are PRESERVED
 *  here — this is the set the toggle handler mutates and persists, so dropping
 *  a sentinel would silently re-enable a provider the user emptied. Use
 *  `effectiveVisibleKeys` for display (sentinels stripped). `seen` null (legacy
 *  state predating the baseline key) keeps the old exact-honor behavior. */
export function resolveVisibleKeys(
  stored: Set<string> | null,
  providers: readonly ModelOptionProvider[],
  seen?: Set<string> | null
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

    const hasStoredProvider = [...stored].some(key => key.startsWith(providerPrefix) && !isProviderSentinel(key))

    const hasSentinel = stored.has(emptyProviderSentinelKey(provider.slug))

    if (hasSentinel) {
      // Hide-all is intent about the WHOLE provider — new families stay hidden.
      continue
    }

    if (hasStoredProvider) {
      // Model-level curation is intent about SPECIFIC models only. Families the
      // catalog added after the seen baseline carry no user intent at all, so
      // they default visible. (Providers without any stored keys get the full
      // default expansion below, which already covers them.)
      if (seen) {
        for (const family of collapseModelFamilies(provider.models ?? [])) {
          const key = modelVisibilityKey(provider.slug, family.id)

          if (!seen.has(key)) {
            next.add(key)
          }
        }
      }

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
  providers: readonly ModelOptionProvider[],
  seen?: Set<string> | null
): Set<string> {
  const next = resolveVisibleKeys(stored, providers, seen)

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
  model: string,
  seen?: Set<string> | null
): Set<string> {
  // `resolveVisibleKeys` always returns a fresh Set, so we can mutate it directly.
  const next = resolveVisibleKeys(stored, providers, seen)
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

/** Compute the next persisted visibility set when a provider's master switch is
 *  flipped. `visible=true` enables every one of the provider's collapsed model
 *  families (and clears its hide-all sentinel); `visible=false` removes them all
 *  and records the sentinel so the defaults are not silently re-expanded.
 *  Seeds from `resolveVisibleKeys` so other providers' state (including their
 *  sentinels) survives the persist, mirroring `toggleModelVisibility`. */
export function setProviderVisibility(
  stored: Set<string> | null,
  providers: readonly ModelOptionProvider[],
  providerSlug: string,
  visible: boolean,
  seen?: Set<string> | null
): Set<string> {
  const next = resolveVisibleKeys(stored, providers, seen)
  const sentinel = emptyProviderSentinelKey(providerSlug)
  const provider = providers.find(p => p.slug === providerSlug)
  const families = collapseModelFamilies(provider?.models ?? [])

  // Drop every existing entry for this provider (real keys + sentinel); we
  // rebuild its state from scratch below.
  for (const key of [...next]) {
    if (key.startsWith(`${providerSlug}::`)) {
      next.delete(key)
    }
  }

  if (visible) {
    for (const family of families) {
      next.add(modelVisibilityKey(providerSlug, family.id))
    }

    // A provider with zero models can't be "all on" — leave it empty rather
    // than stranding a sentinel that reads as an explicit hide-all.
    if (families.length === 0) {
      next.delete(sentinel)
    }
  } else {
    next.add(sentinel)
  }

  return next
}

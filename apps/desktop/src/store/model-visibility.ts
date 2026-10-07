import type { ModelOptionProvider } from '@hermes/shared'
import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'
import { $activeGatewayRoute, $gateway, activeGatewayProfileKey, requestGatewayForProfile } from '@/store/gateway'

const STORAGE_KEY = 'hermes.desktop.visible-models'

/** Every `provider::model` key the user has had a chance to judge — snapshotted
 *  each time the visible set is persisted. A model absent from here appeared
 *  AFTER the user last curated (plugin update, catalog refresh, new release), so
 *  it falls through to the curated default rule instead of defaulting to hidden. */
const KNOWN_STORAGE_KEY = 'hermes.desktop.known-models'

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

function loadKeySet(storageKey: string): Set<string> | null {
  const raw = storedString(storageKey)

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
export const $visibleModels = atom<Set<string> | null>(loadKeySet(STORAGE_KEY))

/** Keys the user has seen, or null when nothing has been recorded yet (a fresh
 *  install, or a store written before the snapshot existed). */
export const $knownModels = atom<Set<string> | null>(loadKeySet(KNOWN_STORAGE_KEY))

export const $modelVisibilityOpen = atom(false)

/** Every collapsed-family key across `providers`. */
function allFamilyKeys(providers: readonly ModelOptionProvider[]): Set<string> {
  const keys = new Set<string>()

  for (const provider of providers) {
    for (const family of collapseModelFamilies(provider.models ?? [])) {
      keys.add(modelVisibilityKey(provider.slug, family.id))
    }
  }

  return keys
}

/** Push the roster to the gateway so OTHER surfaces (phone, a second client,
 *  a TUI on the same backend) show the models this operator actually chose.
 *  Kept local-first: localStorage stays the instant read so the picker never
 *  waits on a round-trip, and a disconnected gateway is not an error — the next
 *  edit and the next connect both retry. */
function pushVisibleModels(keys: null | Set<string> | readonly string[]): void {
  if (!$gateway.get()) {
    return
  }

  void requestGatewayForProfile(activeGatewayProfileKey(), 'config.set', {
    key: 'visible_models',
    value: keys === null ? null : [...keys]
  }).catch(() => {
    // Not connected, or a gateway too old to know the key.
  })
}

/** Adopt the roster the backend holds. Called on connect: a gateway that already
 *  has an answer wins over this renderer's cache, otherwise two surfaces on one
 *  backend drift apart and the phone keeps showing models hidden on the Mac. */
export function adoptVisibleModels(keys: null | readonly string[]): void {
  if (keys === null) {
    $visibleModels.set(null)
    persistString(STORAGE_KEY, null)

    return
  }

  $visibleModels.set(new Set(keys))
  persistString(STORAGE_KEY, JSON.stringify([...keys]))
}

/** Listen for the gateway pushing an updated roster (from another surface
 *  editing its settings or the daemon adopting a config change) and apply it
 *  without a reload. Unsubscribe on disconnect. */
export function initVisibleModelsGatewaySync(): () => void {
  let offEvent: (() => void) | undefined
  let generation = 0
  const bind = () => {
    offEvent?.()
    offEvent = undefined
    const currentGeneration = ++generation
    const gateway = $gateway.get()
    const profile = activeGatewayProfileKey()
    const isCurrent = () =>
      generation === currentGeneration && $gateway.get() === gateway && activeGatewayProfileKey() === profile
    let receivedUpdate = false

    if (!gateway) {
      return
    }

    // Live updates win over an in-flight initial read.
    offEvent = gateway.onEvent(event => {
      if (!isCurrent() || (event.profile && event.profile !== profile)) {
        return
      }
      if ((event.type as string) === 'visible_models.changed') {
        const payload = event.payload as { value?: null | readonly string[] } | undefined
        if (payload && 'value' in payload) {
          receivedUpdate = true
          adoptVisibleModels(payload.value ?? null)
        }
      }
    })

    void requestGatewayForProfile<{ value?: null | readonly string[] }>(profile, 'config.get', {
      key: 'visible_models'
    })
      .then(result => {
        if (!isCurrent() || receivedUpdate) {
          return
        }
        const value = result?.value

        if (value === undefined) {
          return
        }

        if (value === null) {
          // Seed an uncustomised backend from the renderer's existing roster.
          const local = $visibleModels.get()
          if (local !== null) {
            pushVisibleModels(local)
          }
          return
        }

        adoptVisibleModels(value)
      })
      .catch(() => {
        // Older gateway: keep the renderer-local list.
      })
  }

  // nanostores ignores cleanup returned from subscribe callbacks. Own both
  // subscriptions and the socket listener explicitly; profiles can share a socket.
  const offGateway = $gateway.listen(bind)
  const offProfile = $activeGatewayRoute.listen(bind)
  bind()
  return () => {
    ++generation
    offGateway()
    offProfile()
    offEvent?.()
    offEvent = undefined
  }
}

if (typeof window !== 'undefined') {
  initVisibleModelsGatewaySync()
}

/** Persist the visible set and, when the current catalog is supplied, mark every
 *  model in it as judged so only models that appear later count as new. */
export function setVisibleModels(keys: Set<string>, providers: readonly ModelOptionProvider[] = []): void {
  $visibleModels.set(new Set(keys))
  persistString(STORAGE_KEY, JSON.stringify([...keys]))
  pushVisibleModels(keys)

  if (providers.length === 0) {
    return
  }

  persistKnownModels(new Set([...($knownModels.get() ?? []), ...allFamilyKeys(providers)]))
}

function persistKnownModels(known: Set<string>): void {
  $knownModels.set(known)
  persistString(KNOWN_STORAGE_KEY, JSON.stringify([...known]))
}

/** One-time adoption for a visible set persisted before the known snapshot
 *  existed. The old store predates the snapshot machinery, so "absent from the
 *  allowlist" is ambiguous: a deliberate hide and a model that arrived after
 *  the user last curated look identical. Recording everything in the catalog
 *  as judged therefore strands catalog-present defaults behind a stale
 *  allowlist forever (https://github.com/NousResearch/hermes-agent/issues/122053)
 *  — so the curated defaults the old allowlist does NOT contain stay unknown,
 *  and the default rule re-admits them on the next resolve. The one-time cost
 *  is that a deliberately hidden default comes back once; the user's next save
 *  records the re-hide properly and it locks. Non-default models keep the
 *  verbatim-hide semantics — the default rule never showed them anyway — and
 *  a provider hidden outright (sentinel) is skipped entirely. Never a running
 *  union — that would mark a newcomer judged on the very render that first
 *  shows it. Call when the catalog has loaded. */
export function seedKnownModels(providers: readonly ModelOptionProvider[]): void {
  const stored = $visibleModels.get()

  if ($knownModels.get() !== null || stored === null || providers.length === 0) {
    return
  }

  const known = allFamilyKeys(providers)

  for (const provider of providers) {
    if (stored.has(emptyProviderSentinelKey(provider.slug))) {
      continue
    }

    const defaults = new Set<string>()
    expandProviderDefaults(provider, defaults)

    for (const key of defaults) {
      if (!stored.has(key)) {
        known.delete(key)
      }
    }
  }

  persistKnownModels(known)
}

/** Back to "never customized": the curated defaults apply again and the known
 *  snapshot starts over. The snapshot records what was LISTED at each persist,
 *  not what the user chose, so a model hidden when it was recorded (e.g. by the
 *  one-time adoption above) stays hidden through every later catalog change.
 *  This is the user's way out of that without a global storage-key bump; Edit
 *  Models reaches it through `resetModelVisibilityKeepingCustoms`. */
export function resetModelVisibility(): void {
  $visibleModels.set(null)
  persistString(STORAGE_KEY, null)
  $knownModels.set(null)
  persistString(KNOWN_STORAGE_KEY, null)
  pushVisibleModels(null)
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
function expandProviderDefaults(
  provider: ModelOptionProvider,
  target: Set<string>,
  admit: (key: string) => boolean = () => true
): void {
  const families = collapseModelFamilies(provider.models ?? [])

  const featured = provider.featured_models ?? []

  const defaults = featured.length
    ? families.filter(family => featured.includes(family.id))
    : families.slice(0, DEFAULT_VISIBLE_PER_PROVIDER)

  for (const family of defaults) {
    const key = modelVisibilityKey(provider.slug, family.id)

    if (admit(key)) {
      target.add(key)
    }
  }
}

/** Resolve the canonical working set: the user's stored keys plus the curated
 *  default expansion for any provider they haven't customized. Hide-all
 *  sentinels are PRESERVED here — this is the set the toggle handler mutates and
 *  persists, so dropping a sentinel would silently re-enable a provider the user
 *  emptied. Use `effectiveVisibleKeys` for display (sentinels stripped).
 *
 *  A provider the user has curated still admits models absent from `known`
 *  (they arrived after the last curation) through the same default rule, so a
 *  plugin or catalog update never lands a model silently switched off. A
 *  provider the user hid outright (sentinel) stays hidden, new models included.
 *  With no snapshot yet (`known` null) nothing counts as new. */
export function resolveVisibleKeys(
  stored: Set<string> | null,
  providers: readonly ModelOptionProvider[],
  known: Set<string> | null = $knownModels.get()
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

    if (stored.has(emptyProviderSentinelKey(provider.slug))) {
      continue
    }

    const hasStoredProvider = [...stored].some(key => key.startsWith(providerPrefix) && !isProviderSentinel(key))

    if (!hasStoredProvider) {
      expandProviderDefaults(provider, next)
    } else if (known) {
      expandProviderDefaults(provider, next, key => !known.has(key))
    }
  }

  return next
}

/** Resolve which keys are currently visible for DISPLAY: the resolved working
 *  set with bookkeeping sentinels stripped (they are not real models). */
export function effectiveVisibleKeys(
  stored: Set<string> | null,
  providers: readonly ModelOptionProvider[],
  known: Set<string> | null = $knownModels.get()
): Set<string> {
  const next = resolveVisibleKeys(stored, providers, known)

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
  known: Set<string> | null = $knownModels.get()
): Set<string> {
  // `resolveVisibleKeys` always returns a fresh Set, so we can mutate it directly.
  const next = resolveVisibleKeys(stored, providers, known)
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
  known: Set<string> | null = $knownModels.get()
): Set<string> {
  const next = resolveVisibleKeys(stored, providers, known)
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

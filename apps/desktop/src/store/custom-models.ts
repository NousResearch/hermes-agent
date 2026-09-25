import type { ModelOptionProvider } from '@hermes/shared'
import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'
import { normalize } from '@/lib/text'

import {
  $visibleModels,
  defaultVisibleKeys,
  emptyProviderSentinelKey,
  modelVisibilityKey,
  resetModelVisibility,
  resolveVisibleKeys,
  setVisibleModels
} from './model-visibility'

/** A model id the user typed in rather than picked from a provider's catalog.
 *  The catalog is a hint list; a slug the row lacks is still a real model to
 *  the backend (a newer release, a custom endpoint), so remembering it makes
 *  it a normal row on every picker instead of something to retype. */
export interface CustomModel {
  model: string
  provider: string
}

const STORAGE_KEY = 'hermes.desktop.custom-models'

function loadCustomModels(): CustomModel[] {
  const raw = storedString(STORAGE_KEY)

  if (!raw) {
    return []
  }

  try {
    const parsed: unknown = JSON.parse(raw)

    if (!Array.isArray(parsed)) {
      return []
    }

    return parsed.filter(
      (entry): entry is CustomModel =>
        !!entry &&
        typeof entry === 'object' &&
        typeof (entry as CustomModel).model === 'string' &&
        typeof (entry as CustomModel).provider === 'string'
    )
  } catch {
    return []
  }
}

/** Global like model visibility: a slug is scoped by its provider slug, so it
 *  is harmless under any profile that lacks the provider and correct under any
 *  profile that has it. */
export const $customModels = atom<readonly CustomModel[]>(loadCustomModels())

function persist(next: readonly CustomModel[]): void {
  $customModels.set(next)
  persistString(STORAGE_KEY, JSON.stringify(next))
}

const sameEntry = (entry: CustomModel, provider: string, model: string): boolean =>
  entry.provider === provider && entry.model === model

export function isCustomModel(customs: readonly CustomModel[], provider: string, model: string): boolean {
  return customs.some(entry => sameEntry(entry, provider, model))
}

/** A typed model id: something with no whitespace inside it. Nothing stricter —
 *  ids differ per provider (`org/model:tag`, `model@date`) and the gateway's
 *  switch result is the only authority on validity. */
export function customModelSlug(text: string): string | null {
  const slug = text.trim()

  return slug && !/\s/.test(slug) ? slug : null
}

/** The slug a search query would add, or null when some provider already lists
 *  it exactly (then the normal row is the thing to pick). */
export function customModelCandidate(search: string, providers: readonly ModelOptionProvider[]): string | null {
  const slug = customModelSlug(search)

  if (!slug) {
    return null
  }

  const q = normalize(slug)

  return providers.some(provider => (provider.models ?? []).some(model => normalize(model) === q)) ? null : slug
}

/** Catalog rows with the user's custom models appended to their providers.
 *  Returns `providers` itself when nothing applies so memoized consumers keep
 *  their reference. */
export function withCustomModels(
  providers: readonly ModelOptionProvider[],
  customs: readonly CustomModel[]
): readonly ModelOptionProvider[] {
  if (customs.length === 0) {
    return providers
  }

  let changed = false

  const next = providers.map(provider => {
    const models = provider.models ?? []
    const extra = customs.filter(entry => entry.provider === provider.slug && !models.includes(entry.model))

    if (extra.length === 0) {
      return provider
    }

    changed = true

    return { ...provider, models: [...models, ...extra.map(entry => entry.model)] }
  })

  return changed ? next : providers
}

/** Remember `model` under `provider` and make sure it shows in the composer
 *  menu. Pass the provider's catalog row when you have it: the visible set is
 *  re-resolved against that row before the key is added, so a provider the
 *  user never curated keeps its default shortlist instead of collapsing to
 *  the one new row. */
export function addCustomModel(provider: string, model: string, row?: ModelOptionProvider): void {
  const slug = customModelSlug(model)

  if (!slug) {
    return
  }

  const customs = $customModels.get()

  if (!isCustomModel(customs, provider, slug)) {
    persist([...customs, { model: slug, provider }])
  }

  if (!row) {
    return
  }

  const merged = withCustomModels([row], $customModels.get())
  const next = resolveVisibleKeys($visibleModels.get(), merged)
  next.delete(emptyProviderSentinelKey(provider))
  next.add(modelVisibilityKey(provider, slug))
  setVisibleModels(next, merged)
}

/** Edit Models' "Reset to defaults". A custom id sits after the catalog, so on
 *  a featured-shortlist row or past the top-N the bare default rule would bring
 *  it back hidden. Only a provider holding such an id is curated again, the way
 *  adding the id curated it; every other provider returns to the live defaults. */
export function resetModelVisibilityKeepingCustoms(providers: readonly ModelOptionProvider[]): void {
  resetModelVisibility()

  const defaults = defaultVisibleKeys(providers)

  for (const { model, provider } of $customModels.get()) {
    const row = providers.find(candidate => candidate.slug === provider)

    if (row && !defaults.has(modelVisibilityKey(provider, model))) {
      addCustomModel(provider, model, row)
    }
  }
}

export function removeCustomModel(provider: string, model: string): void {
  const customs = $customModels.get()

  if (isCustomModel(customs, provider, model)) {
    persist(customs.filter(entry => !sameEntry(entry, provider, model)))
  }
}

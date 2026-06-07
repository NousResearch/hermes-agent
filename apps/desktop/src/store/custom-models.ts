import { atom } from 'nanostores'

import { persistStringRecord, storedStringRecord } from '@/lib/storage'

const STORAGE_KEY = 'hermes.desktop.custom-models'

/** Custom model IDs per provider, keyed by provider slug.
 *  Stores models the user manually added that aren't in the backend discovery list.
 *  Format: { "openrouter": ["claude-opus-4.6", "gpt-4-turbo"], ... } */
export const $customModels = atom<Record<string, string[]>>(loadCustomModels())

function loadCustomModels(): Record<string, string[]> {
  return storedStringRecord(STORAGE_KEY)
}

/** Add a custom model to a provider's list (if not already present). */
export function addCustomModel(provider: string, model: string): void {
  const current = $customModels.get()
  const models = current[provider] ?? []

  if (!models.includes(model)) {
    models.push(model)
  }

  const updated = { ...current, [provider]: models }
  $customModels.set(updated)
  persistCustomModels(updated)
}

/** Remove a custom model from a provider's list. */
export function removeCustomModel(provider: string, model: string): void {
  const current = $customModels.get()
  const models = current[provider] ?? []
  const filtered = models.filter(m => m !== model)

  if (filtered.length === 0) {
    const updated = { ...current }
    delete updated[provider]
    $customModels.set(updated)
    persistCustomModels(updated)
  } else {
    const updated = { ...current, [provider]: filtered }
    $customModels.set(updated)
    persistCustomModels(updated)
  }
}

/** Get all custom models for a provider. */
export function getCustomModelsForProvider(provider: string): string[] {
  const current = $customModels.get()
  return current[provider] ?? []
}

function persistCustomModels(models: Record<string, string[]>): void {
  persistStringRecord(STORAGE_KEY, models)
}

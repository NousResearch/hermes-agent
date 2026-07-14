import { atom } from 'nanostores'

import type { HermesConnection } from '@/global'
import { persistStringArrayRecord, storedStringArrayRecord } from '@/lib/storage'

import { $connection } from './session'

const STORAGE_KEY = 'hermes.desktop.custom-models'

/** Custom model IDs the user typed for a provider that aren't in the backend
 *  discovery list (gated/beta models, non-discoverable providers, etc.).
 *  Format: `{ "openrouter": ["claude-opus-4.8", "gpt-4-turbo"], ... }`.
 *
 *  Scope is connection-aware, mirroring `workspaceCwdKey`: a local backend uses
 *  the bare key; a remote connection keys by base URL + profile so a model
 *  typed against one backend/profile never bleeds into another (a different
 *  deployment may not expose the same provider). The atom re-homes when the
 *  active connection changes (a workspace switch, not a reboot). */
function customModelsKey(connection: HermesConnection | null = $connection.get()): string {
  if (connection?.mode !== 'remote') {
    return STORAGE_KEY
  }

  const base = encodeURIComponent(connection.baseUrl || 'remote')
  const profile = encodeURIComponent(connection.profile || 'default')

  return `${STORAGE_KEY}.remote.${base}.${profile}`
}

export const $customModels = atom<Record<string, string[]>>(storedStringArrayRecord(customModelsKey()))

// Re-home the store to the active connection's scope. `subscribe` fires
// immediately (re-seeding the same value at module load) and again on every
// connection change, so switching backend/profile swaps in that scope's list
// instead of leaking the previous one.
$connection.subscribe(() => {
  $customModels.set(storedStringArrayRecord(customModelsKey()))
})

function persist(models: Record<string, string[]>): void {
  persistStringArrayRecord(customModelsKey(), models)
}

/** Add a custom model to a provider's list (deduped, order-preserving). */
export function addCustomModel(provider: string, model: string): void {
  const slug = provider.trim()
  const id = model.trim()

  if (!slug || !id) {
    return
  }

  const current = $customModels.get()
  const existing = current[slug] ?? []

  if (existing.includes(id)) {
    return
  }

  const updated = { ...current, [slug]: [...existing, id] }
  $customModels.set(updated)
  persist(updated)
}

/** Remove a custom model from a provider's list. */
export function removeCustomModel(provider: string, model: string): void {
  const current = $customModels.get()
  const existing = current[provider] ?? []
  const filtered = existing.filter(entry => entry !== model)

  if (filtered.length === existing.length) {
    return
  }

  const updated = { ...current }

  if (filtered.length === 0) {
    delete updated[provider]
  } else {
    updated[provider] = filtered
  }

  $customModels.set(updated)
  persist(updated)
}

/** Get all custom models for a provider (empty array when none). */
export function getCustomModelsForProvider(provider: string): string[] {
  return $customModels.get()[provider] ?? []
}

/** Merge a provider's backend models with its custom models, backend first,
 *  deduped and order-preserving. Shared by every surface that lists models so
 *  they stay consistent. */
export function mergeCustomModels(
  provider: string,
  backendModels: readonly string[] | undefined,
  customModels: Record<string, string[]> = $customModels.get()
): string[] {
  const base = backendModels ? [...backendModels] : []
  const extra = customModels[provider] ?? []
  const seen = new Set(base)
  const merged = [...base]

  for (const model of extra) {
    if (!seen.has(model)) {
      seen.add(model)
      merged.push(model)
    }
  }

  return merged
}

/** Whether a given model id for a provider came from the user's custom list. */
export function isCustomModel(
  provider: string,
  model: string,
  customModels: Record<string, string[]> = $customModels.get()
): boolean {
  return (customModels[provider] ?? []).includes(model)
}

import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

const STORAGE_KEY = 'hermes.desktop.pinned-models'

/** Ordered list of pinned `provider::model` keys (same key format as
 *  `model-visibility.ts`'s `modelVisibilityKey`). Pinning is orthogonal to
 *  Edit Models visibility — a model can be pinned and later hidden from its
 *  provider group without losing its pinned slot for when it's shown again.
 *  Order is insertion order (oldest pin first, newest last), matching the
 *  sidebar's session-pin convention (`store/layout.ts`'s `pinSession`). */
function load(): string[] {
  const raw = storedString(STORAGE_KEY)

  if (!raw) {
    return []
  }

  try {
    const parsed = JSON.parse(raw)

    return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === 'string') : []
  } catch {
    return []
  }
}

export const $pinnedModelKeys = atom<string[]>(load())

function persist(next: string[]) {
  $pinnedModelKeys.set(next)
  persistString(STORAGE_KEY, next.length === 0 ? null : JSON.stringify(next))
}

export function isModelPinned(key: string): boolean {
  return $pinnedModelKeys.get().includes(key)
}

/** Pin a model key at the end of the pinned list. No-op if already pinned. */
export function pinModel(key: string) {
  const prev = $pinnedModelKeys.get()

  if (prev.includes(key)) {
    return
  }

  persist([...prev, key])
}

export function unpinModel(key: string) {
  const prev = $pinnedModelKeys.get()

  if (!prev.includes(key)) {
    return
  }

  persist(prev.filter(existing => existing !== key))
}

export function toggleModelPinned(key: string) {
  if (isModelPinned(key)) {
    unpinModel(key)
  } else {
    pinModel(key)
  }
}

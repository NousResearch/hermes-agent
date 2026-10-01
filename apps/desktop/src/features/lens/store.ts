import { atom } from 'nanostores'

import { readJson, readKey, writeKey } from '@/lib/storage'

import type { LensGuest } from './capture'
import { decodeCard, LENS_CARD_LIMIT, type LensCapture, type LensCard, refreshCard } from './model'

const guests = new Map<LensGuest, string>()

export function registerLensGuest(guest: LensGuest) {
  guests.set(guest, $lensScope.get())

  return () => {
    guests.delete(guest)
  }
}

export function findLensGuest(card: LensCard): LensGuest | undefined {
  return [...guests].find(([guest, scope]) => {
    try {
      return scope === card.scope && guest.getURL?.() === card.url
    } catch {
      return false
    }
  })?.[0]
}

const PREFIX = 'hermes.desktop.lens.card.v1.'
export const $lensScope = atom('conn:local::default')
export const $lensCards = atom<LensCard[]>([])
export const $unassignedLensCount = atom(0)

// A persisted epoch also invalidates work in other Desktop windows. Checking
// only the current name would miss delete/recreate and A → B → A races.
const epochKey = (scope: string) => 'hermes.desktop.lens.epoch.v1.' + encodeURIComponent(scope)
let activation = 0

export function lensOperationIsCurrent(scope: string, guest: LensGuest): () => boolean {
  const started = activation
  const epoch = readKey(epochKey(scope))

  return () =>
    started === activation &&
    $lensScope.get() === scope &&
    guests.get(guest) === scope &&
    readKey(epochKey(scope)) === epoch
}

function invalidateScope(scope: string) {
  writeKey(epochKey(scope), crypto.randomUUID())

  if ($lensScope.get() === scope) {
    activation += 1
  }
}

function loadCards(): LensCard[] {
  const cards: LensCard[] = []

  if (typeof window === 'undefined') {
    return cards
  }

  try {
    for (let i = 0; i < window.localStorage.length; i++) {
      const key = window.localStorage.key(i)

      if (!key?.startsWith(PREFIX)) {
        continue
      }

      const card = decodeCard(readJson(key))

      if (card && key === PREFIX + card.id) {
        cards.push(card)
      }
    }
  } catch {
    // Sandboxed or full storage must not break the browser rail at startup.
    return []
  }

  return cards.sort((a, b) => b.capturedAt.localeCompare(a.capturedAt))
}

/** Pre-release builds saved profile-only owners. Preserve those records for
 * explicit export; guessing a connection would expose one backend's evidence
 * in another backend with the same profile name. */
export function unassignedLensCards() {
  return loadCards().filter(card => !card.scope.startsWith('conn:'))
}

export function syncLensCards() {
  const cards = loadCards()
  $lensCards.set(cards.filter(card => card.scope === $lensScope.get()))
  $unassignedLensCount.set(cards.filter(card => !card.scope.startsWith('conn:')).length)
}

export function setLensScope(scope: string) {
  if ($lensScope.get() !== scope) {
    activation += 1
  }

  $lensScope.set(scope)
  syncLensCards()
}

function save(card: LensCard) {
  const key = PREFIX + card.id
  const value = JSON.stringify(card)
  writeKey(key, value)

  if (readKey(key) !== value) {
    throw new Error('saveFailed')
  }

  syncLensCards()
}

export function pinLensCapture(capture: LensCapture, scope: string): LensCard {
  const cards = loadCards().filter(card => card.scope === scope)
  const existing = cards.find(card => card.url === capture.url && card.selector === capture.selector)

  if (existing) {
    const updated = refreshCard(existing, capture, new Date().toISOString())
    save(updated)

    return updated
  }

  if (cards.length >= LENS_CARD_LIMIT) {
    throw new Error('boardFull')
  }

  const now = new Date().toISOString()
  const card = { ...capture, id: crypto.randomUUID(), scope, capturedAt: now, checkedAt: now, note: '' }
  save(card)

  return card
}

export function updateLensCapture(original: LensCard, capture: LensCapture) {
  const current = decodeCard(readJson(PREFIX + original.id))

  // A refresh can finish after a removal, edit, newer refresh, or profile rename.
  if (!current || current.scope !== original.scope || current.checkedAt !== original.checkedAt) {
    return
  }

  save(refreshCard(current, capture, new Date().toISOString()))
}

export function noteLensCard(id: string, note: string) {
  const current = decodeCard(readJson(PREFIX + id))

  if (current) {
    save({ ...current, note: note.slice(0, 2000) })
  }
}

export function removeLensCard(id: string) {
  writeKey(PREFIX + id, null)

  if (readKey(PREFIX + id) !== null) {
    throw new Error('saveFailed')
  }

  syncLensCards()
}

export function dropLensScope(scope: string) {
  invalidateScope(scope)

  for (const [guest, owner] of guests) {
    if (owner === scope) {
      guests.delete(guest)
    }
  }

  for (const card of loadCards().filter(card => card.scope === scope)) {
    removeLensCard(card.id)
  }
}

export function migrateLensScope(from: string, to: string) {
  if (from === to) {
    return
  }

  invalidateScope(from)
  invalidateScope(to)

  for (const [guest, owner] of guests) {
    if (owner === from) {
      guests.set(guest, to)
    }
  }

  for (const card of loadCards().filter(card => card.scope === from)) {
    save({ ...card, scope: to })
  }

  if ($lensScope.get() === from) {
    setLensScope(to)
  }
}

if (typeof window !== 'undefined') {
  window.addEventListener('storage', event => {
    if (!event.key || event.key.startsWith(PREFIX)) {
      syncLensCards()
    }
  })
}

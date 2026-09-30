import { atom } from 'nanostores'

import { readJson, readKey, writeKey } from '@/lib/storage'

import { decodeCard, LENS_CARD_LIMIT, type LensCapture, type LensCard, refreshCard } from './model'

const PREFIX = 'hermes.desktop.lens.card.v1.'
export const $lensScope = atom('default')
export const $lensCards = atom<LensCard[]>([])

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

export function syncLensCards() {
  $lensCards.set(loadCards().filter(card => card.scope === $lensScope.get()))
}

export function setLensScope(scope: string) {
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
  for (const card of loadCards().filter(card => card.scope === scope)) {
    removeLensCard(card.id)
  }
}

export function migrateLensScope(from: string, to: string) {
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

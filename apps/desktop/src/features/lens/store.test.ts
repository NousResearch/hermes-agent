import { beforeEach, describe, expect, it } from 'vitest'

import { type LensCapture, lensPrompt } from './model'
import {
  $lensCards,
  dropLensScope,
  findLensGuest,
  migrateLensScope,
  noteLensCard,
  pinLensCapture,
  registerLensGuest,
  removeLensCard,
  setLensScope,
  syncLensCards,
  updateLensCapture
} from './store'

const source: LensCapture = {
  url: 'https://example.com/item',
  title: 'Item',
  text: 'Price: $80',
  selector: '#offer',
  tag: 'ARTICLE',
  truncated: false
}

beforeEach(() => {
  localStorage.clear()
  setLensScope('device-a:research')
})

describe('Lens workspace lifecycle', () => {
  it('persists captures and notes across A → B → A, rename, reload, and deletion', () => {
    const card = pinLensCapture(source, 'device-a:research')
    const guest = { getURL: () => source.url, addEventListener() {}, removeEventListener() {} }
    const unregister = registerLensGuest(guest)
    noteLensCard(card.id, 'Check delivery')
    setLensScope('device-b:research')
    expect($lensCards.get()).toEqual([])
    pinLensCapture({ ...source, text: 'Different account' }, 'device-b:research')
    setLensScope('device-a:research')
    expect($lensCards.get()[0]).toMatchObject({ id: card.id, note: 'Check delivery', text: source.text })
    migrateLensScope('device-a:research', 'device-a:shopping')
    syncLensCards()
    expect($lensCards.get()[0].scope).toBe('device-a:shopping')
    expect(findLensGuest($lensCards.get()[0])).toBe(guest)
    unregister()
    dropLensScope('device-a:shopping')
    expect($lensCards.get()).toEqual([])
    setLensScope('device-b:research')
    expect($lensCards.get()[0].text).toBe('Different account')
  })

  it('preserves notes and prior evidence on refresh without reviving removed cards', () => {
    const card = pinLensCapture(source, 'device-a:research')
    noteLensCard(card.id, 'My criteria')
    updateLensCapture(card, { ...source, text: 'Price: $65' })
    const updated = $lensCards.get()[0]
    expect(updated).toMatchObject({ text: 'Price: $65', previousText: source.text, note: 'My criteria' })
    expect(lensPrompt([updated], 'Compare')).toContain(source.url)
    expect(lensPrompt([updated], 'Compare')).toContain('untrusted source evidence')
    expect(() => updateLensCapture(updated, { ...source, url: 'https://other.example/' })).toThrow('sourceChanged')
    removeLensCard(card.id)
    updateLensCapture(updated, source)
    expect($lensCards.get()).toEqual([])
  })
})

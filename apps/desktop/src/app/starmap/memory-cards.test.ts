import { expect, it } from 'vitest'

import { indexMemoryCards } from './memory-cards'
import type { MemoryCard } from './types'

it('keeps UUID cards addressable across edits and reorders alongside legacy graphs', () => {
  const id = 'd07ef158-32bc-497f-8407-bdbdfb6c0173'
  const first: MemoryCard = { source: 'memory', title: 'Preference', body: 'Original preference.', entry_id: id }
  const second: MemoryCard = {
    source: 'memory',
    title: 'Other',
    body: 'Other note.',
    entry_id: '8f2b128d-0f8c-4208-882b-f4cf63c6df71'
  }
  const legacy: MemoryCard = {
    source: 'profile',
    title: 'Legacy',
    body: 'Legacy preference.',
    fingerprint: 'legacy-digest'
  }
  const original = indexMemoryCards([first, second, legacy])
  const selected = `memory:memory:${id}`

  expect(original.get(selected)?.body).toBe('Original preference.')
  expect(original.get('memory:profile:2:legacy-digest')).toBe(legacy)
  expect(original.get('memory:profile:2')).toBe(legacy)
  const edited = { ...first, body: 'Edited preference.' }
  const reordered = indexMemoryCards([legacy, second, edited])
  expect(reordered.get(selected)).toBe(edited)
  expect(reordered.get('memory:profile:0:legacy-digest')).toBe(legacy)
  expect(reordered.get('memory:memory:0')).toBeUndefined()
  expect(reordered.get(`memory:profile:${id}`)).toBeUndefined()
})

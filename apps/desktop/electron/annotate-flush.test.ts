import assert from 'node:assert/strict'

import { test } from 'vitest'

import { ANNOTATE_FLUSH_MAX_ITEMS, isAnnotateFlushEnvelope } from './annotate-flush'

function envelope(partial: Record<string, unknown> = {}) {
  return {
    id: 'flush-1',
    items: [{ imageDataUrl: 'data:image/png;base64,AAAA', note: 'note', number: 1, prompt: 'Comment 1' }],
    pageUrl: 'http://127.0.0.1:4173/',
    ...partial
  }
}

test('isAnnotateFlushEnvelope accepts a well-shaped envelope', () => {
  assert.equal(isAnnotateFlushEnvelope(envelope()), true)
})

test('isAnnotateFlushEnvelope accepts pins without crops', () => {
  assert.equal(isAnnotateFlushEnvelope(envelope({ items: [{ number: 2, prompt: 'Comment 2' }] })), true)
})

test('isAnnotateFlushEnvelope rejects non-objects and envelopes without items', () => {
  for (const value of [null, undefined, 'flush', 42, [], { id: 'x' }, envelope({ items: [] })]) {
    assert.equal(isAnnotateFlushEnvelope(value), false)
  }
})

test('isAnnotateFlushEnvelope rejects malformed ids and items', () => {
  assert.equal(isAnnotateFlushEnvelope(envelope({ id: '' })), false)
  assert.equal(isAnnotateFlushEnvelope(envelope({ items: [{ number: '1', prompt: 'x' }] })), false)
  assert.equal(isAnnotateFlushEnvelope(envelope({ items: [{ number: 1 }] })), false)
  assert.equal(isAnnotateFlushEnvelope(envelope({ items: [{ number: 1, prompt: 'x', imageDataUrl: 7 }] })), false)
  assert.equal(isAnnotateFlushEnvelope(envelope({ pageUrl: 42 })), false)
})

test('isAnnotateFlushEnvelope caps the pin count', () => {
  const items = Array.from({ length: ANNOTATE_FLUSH_MAX_ITEMS + 1 }, (_, index) => ({
    number: index + 1,
    prompt: `Comment ${index + 1}`
  }))

  assert.equal(isAnnotateFlushEnvelope(envelope({ items })), false)
})

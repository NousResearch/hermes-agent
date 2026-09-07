import assert from 'node:assert/strict'

import { test } from 'vitest'

import { ensurePenEmbedUrl, isPenWebUrl, penEmbedDropped, restorePenEmbedUrl } from './embed-url'

const EDITOR = 'https://app.pen.dev/new?embed'

test('isPenWebUrl matches origin and ignores path or query', () => {
  assert.equal(isPenWebUrl('https://app.pen.dev/new?d=abc', EDITOR), true)
  assert.equal(isPenWebUrl('https://app.pen.dev/new?embed', EDITOR), true)
  assert.equal(isPenWebUrl('about:blank', EDITOR), false)
  assert.equal(isPenWebUrl('https://evil.example/new?embed', EDITOR), false)
})

test('ensurePenEmbedUrl adds embed when the override omitted it', () => {
  const url = new URL(ensurePenEmbedUrl('https://app.pen.dev/new'))

  assert.equal(url.searchParams.has('embed'), true)
})

test('ensurePenEmbedUrl leaves an existing embed flag alone', () => {
  assert.equal(new URL(ensurePenEmbedUrl(EDITOR)).searchParams.has('embed'), true)
  assert.equal(new URL(ensurePenEmbedUrl('https://app.pen.dev/new?embed=1')).searchParams.get('embed'), '1')
})

test('penEmbedDropped is true only for same-origin /new without embed', () => {
  assert.equal(penEmbedDropped('https://app.pen.dev/new?d=abc-uuid', EDITOR), true)
  assert.equal(penEmbedDropped('https://app.pen.dev/new', EDITOR), true)
  assert.equal(penEmbedDropped('https://app.pen.dev/new/', EDITOR), true)
  assert.equal(penEmbedDropped(EDITOR, EDITOR), false)
  assert.equal(penEmbedDropped('https://app.pen.dev/new?embed&d=abc', EDITOR), false)
  assert.equal(penEmbedDropped('https://app.pen.dev/pricing', EDITOR), false)
  assert.equal(penEmbedDropped('about:blank', EDITOR), false)
})

test('restorePenEmbedUrl keeps Pencil minted d and puts embed back', () => {
  const restored = new URL(restorePenEmbedUrl('https://app.pen.dev/new?d=minted-id', EDITOR))

  assert.equal(restored.origin, 'https://app.pen.dev')
  assert.equal(restored.pathname, '/new')
  assert.equal(restored.searchParams.has('embed'), true)
  assert.equal(restored.searchParams.get('d'), 'minted-id')
})

test('restorePenEmbedUrl does not invent a d when Pencil never set one', () => {
  const restored = new URL(restorePenEmbedUrl('https://app.pen.dev/new', EDITOR))

  assert.equal(restored.searchParams.has('embed'), true)
  assert.equal(restored.searchParams.get('d'), null)
})

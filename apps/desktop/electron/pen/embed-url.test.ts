import assert from 'node:assert/strict'

import { test } from 'vitest'

import { ensurePenEmbedUrl, isPenWebUrl } from './embed-url'

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

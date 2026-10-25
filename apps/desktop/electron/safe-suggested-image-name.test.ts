import assert from 'node:assert/strict'

import { test } from 'vitest'

import { safeSuggestedImageName } from './safe-suggested-image-name'

test('falls back to image plus extension when empty', () => {
  assert.equal(safeSuggestedImageName('', '.png'), 'image.png')
  assert.equal(safeSuggestedImageName(undefined, '.jpg'), 'image.jpg')
})

test('replaces Windows-invalid and control characters', () => {
  assert.equal(safeSuggestedImageName('foo:bar*.png', '.png'), 'foo_bar_.png')
  assert.equal(safeSuggestedImageName('hi\u0001there.png', '.png'), 'hi_there.png')
})

test('appends the mime extension when the name has none', () => {
  assert.equal(safeSuggestedImageName('slashstack-worktree-feedback-2026-08-18', '.png'), 'slashstack-worktree-feedback-2026-08-18.png')
})

test('keeps an existing extension', () => {
  assert.equal(safeSuggestedImageName('photo.webp', '.png'), 'photo.webp')
})

test('prefixes Windows reserved device names', () => {
  assert.equal(safeSuggestedImageName('CON', '.png'), '_CON.png')
  assert.equal(safeSuggestedImageName('NUL.txt', '.png'), '_NUL.txt')
  assert.equal(safeSuggestedImageName('prn.png', '.png'), '_prn.png')
  assert.equal(safeSuggestedImageName('COM1', '.jpg'), '_COM1.jpg')
})

test('caps the stem at 120 characters', () => {
  const name = safeSuggestedImageName(`${'a'.repeat(300)}.png`, '.png')
  assert.equal(name.length, 124)
  assert.equal(name, `${'a'.repeat(120)}.png`)
})

test('treats trailing-dot and all-dot names as empty', () => {
  assert.equal(safeSuggestedImageName('...', '.png'), 'image.png')
  assert.equal(safeSuggestedImageName('photo...', '.png'), 'photo.png')
})

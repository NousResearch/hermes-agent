import assert from 'node:assert/strict'

import { test } from 'vitest'

import { mergeActiveWork, normalizeActiveWork, quitPromptFor } from './quit-guard'

test('normalizeActiveWork drops junk and keeps the count at least the title count', () => {
  assert.deepEqual(normalizeActiveWork(null), { count: 0, titles: [] })
  assert.deepEqual(normalizeActiveWork({ count: 'many', titles: 'nope' }), { count: 0, titles: [] })
  assert.deepEqual(normalizeActiveWork({ count: -3, titles: ['  Fix login  ', '', 7] }), {
    count: 1,
    titles: ['Fix login']
  })
})

test('normalizeActiveWork keeps untitled sessions in the count', () => {
  assert.deepEqual(normalizeActiveWork({ count: 3, titles: ['Fix login'] }), { count: 3, titles: ['Fix login'] })
})

test('mergeActiveWork de-dupes a session two windows both report', () => {
  const merged = mergeActiveWork([
    { count: 2, titles: ['Fix login', 'Ship docs'] },
    { count: 1, titles: ['Fix login'] }
  ])

  assert.deepEqual(merged, { count: 2, titles: ['Fix login', 'Ship docs'] })
})

test('quitPromptFor stays out of the way when nothing is running', () => {
  assert.equal(quitPromptFor({ count: 0, titles: [] }, false), null)
})

test('quitPromptFor stays out of the way during an update handoff', () => {
  assert.equal(quitPromptFor({ count: 2, titles: ['Fix login'] }, true), null)
})

test('quitPromptFor names the running chats', () => {
  const prompt = quitPromptFor({ count: 2, titles: ['Fix login', 'Ship docs'] }, false)

  assert.ok(prompt)
  assert.equal(prompt.message, 'Hermes is still working on 2 chats.')
  assert.ok(prompt.detail.includes('• Fix login'))
  assert.ok(prompt.detail.includes('• Ship docs'))
})

test('quitPromptFor summarizes past the list cap and counts untitled work', () => {
  const prompt = quitPromptFor({ count: 9, titles: ['a', 'b', 'c', 'd', 'e', 'f'] }, false)

  assert.ok(prompt)
  assert.equal(prompt.message, 'Hermes is still working on 9 chats.')
  assert.ok(prompt.detail.includes('• d'))
  assert.ok(!prompt.detail.includes('• e'))
  assert.ok(prompt.detail.includes('• 5 more'))
})

test('quitPromptFor speaks singular for one chat', () => {
  const prompt = quitPromptFor({ count: 1, titles: [] }, false)

  assert.ok(prompt)
  assert.equal(prompt.message, 'Hermes is still working on 1 chat.')
  assert.ok(prompt.detail.includes('mid-turn'))
})

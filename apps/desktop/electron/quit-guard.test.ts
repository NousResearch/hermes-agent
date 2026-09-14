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

test('quitPromptFor only prompts for idle work when always is selected', () => {
  const idle = { count: 0, titles: [] }
  assert.equal(quitPromptFor(idle, false), null)
  assert.equal(quitPromptFor(idle, false, 'never'), null)
  assert.equal(quitPromptFor(idle, false, 'while-working'), null)
  const prompt = quitPromptFor(idle, false, 'always')
  assert.ok(prompt)
  assert.ok(prompt.detail.includes('Local models'))
})

test('quitPromptFor stays out of the way during an update handoff', () => {
  for (const mode of ['never', 'while-working', 'always'] as const) {
    assert.equal(quitPromptFor({ count: 2, titles: ['Fix login'] }, true, mode), null)
    assert.equal(quitPromptFor({ count: 0, titles: [] }, true, mode), null)
  }
})

test('quitPromptFor names the running chats', () => {
  const prompt = quitPromptFor({ count: 2, titles: ['Fix login', 'Ship docs'] }, false)

  assert.ok(prompt)
  assert.equal(prompt.message, 'Hermes is still working on 2 chats.')
  assert.ok(prompt.detail.includes('• Fix login'))
  assert.ok(prompt.detail.includes('• Ship docs'))
  assert.deepEqual(quitPromptFor({ count: 2, titles: ['Fix login', 'Ship docs'] }, false, 'always'), prompt)
  assert.equal(quitPromptFor({ count: 2, titles: ['Fix login', 'Ship docs'] }, false, 'never'), null)
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

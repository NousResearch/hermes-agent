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

test('quitPromptFor localizes to zh when locale is zh', () => {
  const prompt = quitPromptFor({ count: 1, titles: ['固化室审查'] }, false, 'zh')

  assert.ok(prompt)
  assert.equal(prompt.message, 'Hermes 正在处理 1 个对话。')
  assert.equal(prompt.buttons[0], '继续运行')
  assert.equal(prompt.buttons[1], '仍然退出')
  assert.ok(prompt.detail.includes('固化室审查'))
  assert.ok(prompt.detail.includes('未写入的工作将丢失'))
})

test('quitPromptFor falls back to English for an unknown locale', () => {
  const prompt = quitPromptFor({ count: 2, titles: ['Fix login'] }, false, 'xx')

  assert.ok(prompt)
  assert.equal(prompt.message, 'Hermes is still working on 2 chats.')
  assert.equal(prompt.buttons[0], 'Keep Running')
})

test('quitPromptFor localizes plural + "more" for ja', () => {
  const prompt = quitPromptFor({ count: 5, titles: ['a', 'b', 'c', 'd', 'e'] }, false, 'ja')

  assert.ok(prompt)
  assert.equal(prompt.message, 'Hermes は 5 件のチャットを処理中です。')
  assert.ok(prompt.detail.includes('さらに 1 件'))
  assert.equal(prompt.buttons[1], '強制終了')
})

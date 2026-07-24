import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createClickOnlyWindowCloseItem, shouldInterceptCloseTabShortcut } from './close-tab-shortcut'

const cases = [
  {
    expected: true,
    input: { key: 'w', meta: true },
    isMac: true,
    name: 'macOS Cmd+W'
  },
  {
    expected: false,
    input: { control: true, key: 'w' },
    isMac: false,
    name: 'Windows/Linux Ctrl+W'
  },
  {
    expected: false,
    input: { control: true, key: 'w' },
    isMac: true,
    name: 'macOS Ctrl+W'
  },
  {
    expected: false,
    input: { key: 'w', meta: true, shift: true },
    isMac: true,
    name: 'macOS Cmd+Shift+W'
  }
]

for (const { expected, input, isMac, name } of cases) {
  test(`close-tab main-process routing: ${name}`, () => {
    assert.equal(shouldInterceptCloseTabShortcut(input, isMac), expected)
  })
}

test('the non-macOS window close item has no Ctrl+W accelerator', () => {
  const item = createClickOnlyWindowCloseItem()
  let closeCalls = 0

  item.click(undefined, { close: () => closeCalls++ })

  assert.equal(closeCalls, 1)
  assert.equal('accelerator' in item, false)
  assert.equal('role' in item, false)
})

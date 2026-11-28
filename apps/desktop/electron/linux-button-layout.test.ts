import assert from 'node:assert/strict'

import { test } from 'vitest'

import { DEFAULT_LINUX_BUTTON_LAYOUT, parseLinuxButtonLayout } from './linux-button-layout'

test('parses a left-side layout', () => {
  assert.deepEqual(parseLinuxButtonLayout("'close,minimize,maximize:'\n"), { left: 3, right: 0 })
})

test('parses a right-side layout', () => {
  assert.deepEqual(parseLinuxButtonLayout("'appmenu:minimize,maximize,close'\n"), { left: 0, right: 3 })
})

test('parses a mixed layout on both sides of the first colon', () => {
  assert.deepEqual(parseLinuxButtonLayout("'close:minimize,maximize'"), { left: 1, right: 2 })
})

test('ignores appmenu, spacer, icon and unknown tokens', () => {
  assert.deepEqual(parseLinuxButtonLayout("'appmenu,close,spacer:icon,minimize,mystery'"), {
    left: 1,
    right: 1
  })
})

test('tolerates surrounding whitespace and double quotes', () => {
  assert.deepEqual(parseLinuxButtonLayout('  "close,minimize : maximize"  '), { left: 2, right: 1 })
})

test('falls back to all-right when the value is empty', () => {
  assert.deepEqual(parseLinuxButtonLayout(''), DEFAULT_LINUX_BUTTON_LAYOUT)
  assert.deepEqual(parseLinuxButtonLayout("''"), DEFAULT_LINUX_BUTTON_LAYOUT)
  assert.deepEqual(parseLinuxButtonLayout('   '), DEFAULT_LINUX_BUTTON_LAYOUT)
})

test('falls back to all-right for garbage and non-string values', () => {
  assert.deepEqual(parseLinuxButtonLayout('not a button layout'), DEFAULT_LINUX_BUTTON_LAYOUT)
  assert.deepEqual(parseLinuxButtonLayout(null), DEFAULT_LINUX_BUTTON_LAYOUT)
  assert.deepEqual(parseLinuxButtonLayout(undefined), DEFAULT_LINUX_BUTTON_LAYOUT)
  assert.deepEqual(parseLinuxButtonLayout({ layout: 'close' }), DEFAULT_LINUX_BUTTON_LAYOUT)
})

test('a colon-separated value with only ignored tokens counts zero buttons', () => {
  assert.deepEqual(parseLinuxButtonLayout("'appmenu:spacer,icon'"), { left: 0, right: 0 })
})

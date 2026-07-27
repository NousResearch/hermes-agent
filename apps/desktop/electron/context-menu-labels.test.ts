/**
 * Behavior tests for the context-menu label resolver: every renderer UI locale
 * resolves to a complete label set, and anything unknown falls back to English
 * instead of crashing the menu build.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { getContextMenuLabels } from './context-menu-labels'

const KEYS = [
  'openImage',
  'copyImage',
  'copyImageAddress',
  'saveImageAs',
  'openLink',
  'copyLink',
  'addToDictionary',
  'cut',
  'copy',
  'paste',
  'selectAll'
] as const

test('every supported renderer locale yields a complete, non-empty label set', () => {
  for (const locale of ['en', 'zh', 'zh-hant', 'ja']) {
    const labels = getContextMenuLabels(locale)

    for (const key of KEYS) {
      assert.equal(typeof labels[key], 'string', `${locale}.${key}`)
      assert.ok(labels[key].length > 0, `${locale}.${key} is empty`)
    }
  }
})

test('localized sets actually differ from English (not silently falling back)', () => {
  const en = getContextMenuLabels('en')

  for (const locale of ['zh', 'zh-hant', 'ja']) {
    assert.notEqual(getContextMenuLabels(locale).paste, en.paste, locale)
  }
})

test('unknown or malformed locales fall back to English', () => {
  const en = getContextMenuLabels('en')

  assert.deepEqual(getContextMenuLabels('ko'), en)
  assert.deepEqual(getContextMenuLabels(undefined), en)
  assert.deepEqual(getContextMenuLabels(42), en)
  assert.deepEqual(getContextMenuLabels(''), en)
})

test('locale lookup tolerates case and whitespace from the IPC boundary', () => {
  assert.equal(getContextMenuLabels(' ZH-Hant ').paste, getContextMenuLabels('zh-hant').paste)
  assert.equal(getContextMenuLabels('JA').paste, getContextMenuLabels('ja').paste)
})

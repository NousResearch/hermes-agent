import assert from 'node:assert/strict'

import { test } from 'vitest'

import { imageContextMenuItems } from './image-context-menu'

function createActions() {
  const calls = {
    copyImageAt: [],
    openImage: [],
    copyImageAddress: [],
    saveImage: []
  }

  return {
    calls,
    actions: {
      copyImageAt: (x, y) => calls.copyImageAt.push([x, y]),
      openImage: url => calls.openImage.push(url),
      copyImageAddress: url => calls.copyImageAddress.push(url),
      saveImage: url => calls.saveImage.push(url)
    }
  }
}

test('keeps Copy Image available when Chromium omits a large image srcURL', () => {
  const { actions, calls } = createActions()
  const items = imageContextMenuItems({ mediaType: 'image', srcURL: '', x: 100, y: 120 }, actions)

  assert.deepEqual(
    items.map(item => item.label),
    ['Copy Image']
  )

  items[0].click()
  assert.deepEqual(calls.copyImageAt, [[100, 120]])
})

test('keeps URL-dependent image actions when srcURL is available', () => {
  const { actions, calls } = createActions()
  const url = 'https://example.com/image.png'
  const items = imageContextMenuItems({ mediaType: 'image', srcURL: url, x: 5, y: 8 }, actions)

  assert.deepEqual(
    items.map(item => item.label),
    ['Open Image', 'Copy Image', 'Copy Image Address', 'Save Image As...']
  )

  items[0].click()
  items[1].click()
  items[2].click()
  items[3].click()

  assert.deepEqual(calls.openImage, [url])
  assert.deepEqual(calls.copyImageAt, [[5, 8]])
  assert.deepEqual(calls.copyImageAddress, [url])
  assert.deepEqual(calls.saveImage, [url])
})

test('does not add image actions for a non-image target', () => {
  const { actions } = createActions()

  assert.deepEqual(imageContextMenuItems({ mediaType: 'none', srcURL: '', x: 0, y: 0 }, actions), [])
})

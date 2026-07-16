import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import { type ImageContextMenuActions, imageContextMenuItems } from './image-context-menu'

type TestActions = Omit<ImageContextMenuActions, 'copyImageAt' | 'saveImageFromUrl' | 'writeText'> & {
  copyImageAt: ReturnType<typeof vi.fn<(x: number, y: number) => void>>
  saveImageFromUrl: ReturnType<typeof vi.fn<(url: string) => Promise<unknown>>>
  writeText: ReturnType<typeof vi.fn<(text: string) => void>>
}

function actions(): TestActions {
  return {
    copyImageAt: vi.fn(),
    openExternalUrl: vi.fn(),
    reportError: vi.fn(),
    saveImageFromUrl: vi.fn(async () => undefined),
    writeText: vi.fn()
  }
}

test('builds the existing image actions when Chromium provides a source URL', () => {
  const items = imageContextMenuItems(
    { hasImageContents: true, mediaType: 'image', srcURL: 'https://example.com/image.png', x: 12, y: 34 },
    actions()
  )

  assert.deepEqual(
    items.map(item => item.label),
    ['Open Image', 'Copy Image', 'Copy Image Address', 'Save Image As...']
  )
})

test('copies an above-limit image by coordinates when Chromium drops its source URL', () => {
  const imageActions = actions()

  const items = imageContextMenuItems(
    { hasImageContents: true, mediaType: 'image', srcURL: '', x: 100, y: 200 },
    imageActions
  )

  assert.deepEqual(
    items.map(item => item.label),
    ['Copy Image']
  )

  items[0]?.click?.({} as never, {} as never, {} as never)

  assert.equal(imageActions.copyImageAt.mock.calls.length, 1)
  assert.deepEqual(imageActions.copyImageAt.mock.calls[0], [100, 200])
  assert.equal(imageActions.saveImageFromUrl.mock.calls.length, 0)
  assert.equal(imageActions.writeText.mock.calls.length, 0)
})

test('does not add image actions for a non-image target', () => {
  assert.deepEqual(
    imageContextMenuItems(
      { hasImageContents: false, mediaType: 'none', srcURL: '', x: 0, y: 0 },
      actions()
    ),
    []
  )
})

test('does not offer Copy Image when the image has no decoded contents', () => {
  assert.deepEqual(
    imageContextMenuItems(
      { hasImageContents: false, mediaType: 'image', srcURL: '', x: 0, y: 0 },
      actions()
    ),
    []
  )
})

import { describe, expect, it } from 'vitest'

import { clipboardImageFiles } from './ui'

// The helper reads `clipboardData.items` structurally — a plain mock is enough
// to exercise extraction, filtering, and null-file handling without jsdom's
// (incomplete) clipboard implementation.
const textItem = () => ({ type: 'text/plain', getAsFile: () => null })
const imageItem = (file: File) => ({ type: 'image/png', getAsFile: () => file })

const event = (items: unknown) => ({ clipboardData: { items } }) as unknown as { clipboardData: DataTransfer }
const image = (name = 'shot.png') => new File(['png'], name, { type: 'image/png' })

describe('clipboardImageFiles', () => {
  it('returns an empty list when there is no clipboard data', () => {
    expect(clipboardImageFiles({ clipboardData: null })).toEqual([])
  })

  it('returns an empty list for a text-only paste', () => {
    expect(clipboardImageFiles(event([textItem()]))).toEqual([])
  })

  it('extracts a pasted image as a File', () => {
    const file = image()

    expect(clipboardImageFiles(event([imageItem(file)]))).toEqual([file])
  })

  it('ignores non-image items in a mixed paste', () => {
    const file = image()

    expect(clipboardImageFiles(event([textItem(), imageItem(file), textItem()]))).toEqual([file])
  })

  it('drops image items that fail to produce a file', () => {
    const file = image()
    const items = [imageItem(file), { type: 'image/png', getAsFile: () => null }]

    expect(clipboardImageFiles(event(items))).toEqual([file])
  })
})

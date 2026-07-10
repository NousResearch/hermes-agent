import { describe, expect, it } from 'vitest'

import { blobDedupeKey, detectTrigger, extractClipboardImageBlobs } from './text-utils'

describe('detectTrigger', () => {
  it('detects a bare slash trigger with an empty query', () => {
    expect(detectTrigger('/')).toEqual({ kind: '/', query: '', tokenLength: 1 })
  })

  it('detects a slash command query', () => {
    expect(detectTrigger('/skill')).toEqual({ kind: '/', query: 'skill', tokenLength: 6 })
  })

  it('detects a bare at-mention trigger with an empty query', () => {
    expect(detectTrigger('@')).toEqual({ kind: '@', query: '', tokenLength: 1 })
  })

  it('detects an at-mention query', () => {
    expect(detectTrigger('@file')).toEqual({ kind: '@', query: 'file', tokenLength: 5 })
  })

  it('returns null for plain text', () => {
    expect(detectTrigger('hello there')).toBeNull()
  })

  it('keeps the slash trigger live while typing args', () => {
    expect(detectTrigger('/personality ')).toEqual({
      kind: '/',
      query: 'personality ',
      tokenLength: 13
    })
    expect(detectTrigger('/personality alic')).toEqual({
      kind: '/',
      query: 'personality alic',
      tokenLength: 17
    })
    expect(detectTrigger('/tools enable foo')).toEqual({
      kind: '/',
      query: 'tools enable foo',
      tokenLength: 17
    })
  })

  it('does not treat file-style paths as slash triggers', () => {
    expect(detectTrigger('src/foo/bar')).toBeNull()
    expect(detectTrigger('/path/to/file')).toBeNull()
  })

  it('does not trigger slash popover mid-message', () => {
    expect(detectTrigger('hello /')).toBeNull()
    expect(detectTrigger('hello /skill')).toBeNull()
    expect(detectTrigger('hello there /personality alic')).toBeNull()
    expect(detectTrigger('text\n/skill')).toBeNull()
    expect(detectTrigger('multi word message /')).toBeNull()
  })

  it('still anchors at-mention triggers strictly at the token edge', () => {
    expect(detectTrigger('@file:path with space')).toBeNull()
  })
})

describe('extractClipboardImageBlobs', () => {
  it('dedupes the same image exposed on both items and files', () => {
    const image = new File([new Uint8Array([1, 2, 3])], 'paste.png', {
      type: 'image/png',
      lastModified: 1_700_000_000_000
    })

    const clipboard = {
      files: {
        length: 1,
        item: (index: number) => (index === 0 ? image : null)
      },
      getData: () => '',
      items: [
        {
          kind: 'file',
          type: 'image/png',
          getAsFile: () => image
        }
      ]
    } as unknown as DataTransfer

    expect(extractClipboardImageBlobs(clipboard)).toEqual([image])
  })

  it('falls back to files when items has no image', () => {
    const image = new File([new Uint8Array([4, 5])], 'shot.jpg', {
      type: 'image/jpeg',
      lastModified: 1_700_000_000_001
    })

    const clipboard = {
      files: {
        length: 1,
        item: (index: number) => (index === 0 ? image : null)
      },
      getData: () => '',
      items: []
    } as unknown as DataTransfer

    expect(extractClipboardImageBlobs(clipboard)).toEqual([image])
  })
})

describe('blobDedupeKey', () => {
  it('uses file metadata for File blobs', () => {
    const file = new File([], 'a.png', { type: 'image/png', lastModified: 42 })

    expect(blobDedupeKey(file)).toBe('file:a.png:0:image/png:42')
  })
})

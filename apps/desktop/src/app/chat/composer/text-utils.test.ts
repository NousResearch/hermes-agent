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
  it('extracts image blobs from macOS clipboard that also has text', () => {
    // macOS clipboard always includes a PDF/image flavor alongside text.
    // The caller (handlePaste) must check text FIRST — this test documents
    // why: extractClipboardImageBlobs WILL return the ghost image.
    // See https://github.com/NousResearch/hermes-agent/issues/42827
    const ghostImage = new File([new Uint8Array([0xff, 0xd8])], 'clip.png', {
      type: 'image/png',
      lastModified: 1_700_000_000_002
    })

    const clipboard = {
      files: { length: 0, item: () => null },
      getData: (type: string) => (type === 'text/plain' ? 'hello world' : ''),
      items: [
        {
          kind: 'file',
          type: 'image/png',
          getAsFile: () => ghostImage
        }
      ]
    } as unknown as DataTransfer

    // extractClipboardImageBlobs returns the ghost image — it doesn't know
    // about text priority.  The caller must prefer text when available.
    expect(extractClipboardImageBlobs(clipboard)).toEqual([ghostImage])
  })
})

describe('blobDedupeKey', () => {
  it('uses file metadata for File blobs', () => {
    const file = new File([], 'a.png', { type: 'image/png', lastModified: 42 })

    expect(blobDedupeKey(file)).toBe('file:a.png:0:image/png:42')
  })
})

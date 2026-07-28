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
    // Mid-message paths stay excluded too: a path keeps going past the command
    // token, so the trailing-anchored inline trigger never matches it.
    expect(detectTrigger('check src/foo/bar')).toBeNull()
    expect(detectTrigger('look at /usr/local/bin')).toBeNull()
    expect(detectTrigger('and/or')).toBeNull()
  })

  it('keeps the at-mention live while walking into subfolders', () => {
    // A `/` inside the query is path navigation, not the end of the token —
    // the popover has to stay open so the next directory level can load.
    expect(detectTrigger('@./')).toEqual({ kind: '@', query: './', tokenLength: 3 })
    expect(detectTrigger('@./src')).toEqual({ kind: '@', query: './src', tokenLength: 6 })
    expect(detectTrigger('@~/Desktop/')).toEqual({ kind: '@', query: '~/Desktop/', tokenLength: 11 })
    expect(detectTrigger('@/usr/local')).toEqual({ kind: '@', query: '/usr/local', tokenLength: 11 })
    expect(detectTrigger('@apps/desktop/src')).toEqual({
      kind: '@',
      query: 'apps/desktop/src',
      tokenLength: 17
    })
  })

  it('keeps the at-mention live for a typed ref kind with a path', () => {
    expect(detectTrigger('@file:src/main.tsx')).toEqual({
      kind: '@',
      query: 'file:src/main.tsx',
      tokenLength: 18
    })
    expect(detectTrigger('@folder:apps/')).toEqual({ kind: '@', query: 'folder:apps/', tokenLength: 13 })
  })

  it('still ends the at-mention token at whitespace', () => {
    // The token is whitespace-delimited; a path doesn't change that.
    expect(detectTrigger('@./src and more')).toBeNull()
    expect(detectTrigger('look at @apps/desktop')).toEqual({
      kind: '@',
      query: 'apps/desktop',
      tokenLength: 13
    })
  })

  it('treats a mid-message slash as an inline reference', () => {
    // Skills have to be reachable anywhere in a prompt, not just at position 0.
    expect(detectTrigger('hello /')).toEqual({ kind: '/', inline: true, query: '', tokenLength: 1 })
    expect(detectTrigger('hello /clean')).toEqual({ kind: '/', inline: true, query: 'clean', tokenLength: 6 })
    expect(detectTrigger('text\n/skill')).toEqual({ kind: '/', inline: true, query: 'skill', tokenLength: 6 })
  })

  it('does not carry arg completion into an inline slash reference', () => {
    // Only a position-0 slash is a real invocation, so `/personality alic`
    // mid-message is prose — the trigger ends at the command token.
    expect(detectTrigger('hello there /personality alic')).toBeNull()
    expect(detectTrigger('run /tools enable foo')).toBeNull()
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

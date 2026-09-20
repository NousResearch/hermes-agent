import { describe, expect, it } from 'vitest'

import { appendLink, attachmentName, base64FromBytes, imageMarkdown, storableImages } from './paste-image'

/** Minimal DataTransfer stand-in, shaped like the one the composer's extractor reads. */
function clipboard(files: File[]): DataTransfer {
  return {
    files: { item: (index: number) => files[index] ?? null, length: files.length },
    getData: () => '',
    items: files.map(file => ({ getAsFile: () => file, kind: 'file', type: file.type }))
  } as unknown as DataTransfer
}

describe('storableImages', () => {
  it('takes the pictures the page can keep and leaves everything else to the editor', () => {
    const png = new File([new Uint8Array([1])], 'shot.png', { type: 'image/png' })

    expect(storableImages(clipboard([png]))).toEqual([png])
    expect(storableImages(clipboard([new File(['x'], 'notes.txt', { type: 'text/plain' })]))).toEqual([])
    // A document that can carry script is not a picture; it pastes as text.
    expect(storableImages(clipboard([new File(['x'], 'art.svg', { type: 'image/svg+xml' })]))).toEqual([])
    expect(storableImages(null)).toEqual([])
  })

  it('keeps every picture of a several-image paste, in clipboard order', () => {
    const first = new File([new Uint8Array([1])], 'a.png', { type: 'image/png' })
    const second = new File([new Uint8Array([2, 2])], 'b.webp', { type: 'image/webp' })

    expect(storableImages(clipboard([first, second]))).toEqual([first, second])
  })
})

describe('attachmentName', () => {
  it('names a clipboard image by its type, which is all it has', () => {
    expect(attachmentName(new Blob([''], { type: 'image/png' }))).toBe('clipboard.png')
    expect(attachmentName(new File([''], 'diagram.webp', { type: 'image/webp' }))).toBe('clipboard.webp')
  })
})

describe('base64FromBytes', () => {
  it('encodes past the argument limit of a single spread call', () => {
    expect(base64FromBytes(new TextEncoder().encode('hello'))).toBe('aGVsbG8=')

    const big = new Uint8Array(200_000).fill(65)

    expect(atob(base64FromBytes(big))).toHaveLength(big.length)
  })
})

describe('imageMarkdown', () => {
  it('links the image relative to the page', () => {
    expect(imageMarkdown('assets/20260920_101010_abcdef.png')).toBe('![](assets/20260920_101010_abcdef.png)')
  })
})

describe('appendLink', () => {
  it('puts a picture that outlived the editor on a line of its own', () => {
    expect(appendLink('a note', '![](assets/a.png)')).toBe('a note\n![](assets/a.png)')
    expect(appendLink('a note\n', '![](assets/a.png)')).toBe('a note\n![](assets/a.png)')
    expect(appendLink('', '![](assets/a.png)')).toBe('![](assets/a.png)')
  })
})

import { describe, expect, it } from 'vitest'

import { extractClipboardDocumentCandidates } from './use-composer-actions'

/**
 * Copying a file in Finder (⌘C) puts the file AND a rendered icon of it on the
 * pasteboard. Chromium exposes the icon as `image/png`, which is the only kind
 * `extractClipboardImageBlobs` collects — and its `files` fallback is guarded by
 * `blobs.length === 0`, so the icon permanently shadowed the document. A pasted
 * PDF arrived as a 1024×1024 generic document glyph and the real bytes were
 * never read (observed: the same icon PNG attached 17 times over three weeks).
 *
 * These tests pin the contract: a real copied document wins over its icon, and
 * a genuine screenshot paste is untouched.
 */
function iconFile(): File {
  return new File([new Uint8Array([137, 80, 78, 71])], 'icon.png', { type: 'image/png' })
}

function pdfFile(name = 'contract.pdf', type = 'application/pdf'): File {
  return new File([new Uint8Array([0x25, 0x50, 0x44, 0x46])], name, { type })
}

/** Minimal DataTransfer stub — only the members `extractDroppedFiles` reads. */
function clipboardWith(files: File[]): DataTransfer {
  const list = {
    length: files.length,
    item: (i: number) => files[i] ?? null,
  }

  return {
    files: list,
    items: files.map((file) => ({
      kind: 'file',
      type: file.type,
      getAsFile: () => file,
      webkitGetAsEntry: () => null,
    })),
    getData: () => '',
  } as unknown as DataTransfer
}

describe('extractClipboardDocumentCandidates', () => {
  it('returns the copied PDF, not the icon macOS adds alongside it', () => {
    const candidates = extractClipboardDocumentCandidates(clipboardWith([iconFile(), pdfFile()]))

    expect(candidates).toHaveLength(1)
    expect(candidates[0]?.file?.name).toBe('contract.pdf')
    expect(candidates[0]?.file?.type).toBe('application/pdf')
  })

  it('returns the PDF even when the icon is listed first', () => {
    const candidates = extractClipboardDocumentCandidates(clipboardWith([iconFile(), pdfFile()]))

    expect(candidates.map((c) => c.file?.name)).toEqual(['contract.pdf'])
  })

  it('keeps a document whose MIME type is empty', () => {
    // macOS does not always report a concrete type for a copied file.
    const candidates = extractClipboardDocumentCandidates(clipboardWith([iconFile(), pdfFile('x.pdf', '')]))

    expect(candidates).toHaveLength(1)
    expect(candidates[0]?.file?.name).toBe('x.pdf')
  })

  it('ignores a plain screenshot paste', () => {
    expect(extractClipboardDocumentCandidates(clipboardWith([iconFile()]))).toEqual([])
  })

  it('ignores an empty clipboard', () => {
    expect(extractClipboardDocumentCandidates(clipboardWith([]))).toEqual([])
  })

  it('handles several copied documents', () => {
    const candidates = extractClipboardDocumentCandidates(
      clipboardWith([iconFile(), pdfFile('a.pdf'), pdfFile('b.docx', 'application/msword')]),
    )

    expect(candidates.map((c) => c.file?.name)).toEqual(['a.pdf', 'b.docx'])
  })

  it('never returns an image blob as a document', () => {
    const jpeg = new File([new Uint8Array([0xff, 0xd8])], 'photo.jpg', { type: 'image/jpeg' })
    const candidates = extractClipboardDocumentCandidates(clipboardWith([jpeg]))

    expect(candidates).toEqual([])
  })
})

import { describe, expect, it } from 'vitest'

import {
  type AttachmentToken,
  collectAttachCandidates,
  compileToWireText,
  type MessageDocument,
  importFromWireText,
  isAttachmentToken,
  isTextToken,
  normalizeDocument,
  updateTokenPath
} from './index'

function attachment(over: Partial<AttachmentToken> = {}): AttachmentToken {
  return {
    type: 'attachment',
    id: over.id ?? 'a1',
    kind: over.kind ?? 'file',
    path: over.path ?? '/tmp/x',
    displayName: over.displayName ?? 'x',
    ...over
  }
}

describe('compileToWireText', () => {
  it('preserves order: Chinese -> folder -> Chinese -> image (reporter scenario)', () => {
    const doc: MessageDocument = [
      { type: 'text', value: '你调用codex给我出封面图 ' },
      attachment({ id: 'f1', kind: 'folder', path: '/Users/k/Desktop/sage/xhs_covers', displayName: 'xhs_covers' }),
      { type: 'text', value: ' 封面图上面我的照片 ' },
      attachment({ id: 'i1', kind: 'image', path: '/Users/k/temp/7e321e0.jpg', displayName: '7e321e0.jpg' })
    ]

    const wire = compileToWireText(doc)

    expect(wire).toBe(
      '你调用codex给我出封面图 @folder:`/Users/k/Desktop/sage/xhs_covers` 封面图上面我的照片 @image:`/Users/k/temp/7e321e0.jpg`'
    )
    // Order is preserved: folder before image.
    expect(wire.indexOf('@folder:')).toBeLessThan(wire.indexOf('@image:'))
  })

  it('compiles folder kind to @folder: not @file:', () => {
    const doc: MessageDocument = [attachment({ kind: 'folder', path: '/a/b/dir', displayName: 'dir' })]
    expect(compileToWireText(doc)).toBe('@folder:`/a/b/dir`')
  })

  it('never emits a malformed space-after-colon form', () => {
    const doc: MessageDocument = [attachment({ kind: 'file', path: 'Desktop/sage/xhs covers' })]
    const wire = compileToWireText(doc)
    expect(wire).not.toMatch(/@file:\s/)
    expect(wire).toBe('@file:`Desktop/sage/xhs covers`')
  })

  it('returns empty string for an empty document', () => {
    expect(compileToWireText([])).toBe('')
  })
})

describe('importFromWireText', () => {
  it('round-trips well-formed wire to a stable string', () => {
    const wire = '你好 @folder:`/a/b/xhs_covers` 世界 @image:`/c/d/pic.jpg`'
    const doc = importFromWireText(wire)
    expect(compileToWireText(doc)).toBe(wire)
  })

  it('imports malformed @file: with a space after the colon', () => {
    const doc = importFromWireText('@file: Desktop/sage/xhs_covers 封面图')

    const attach = doc.find(isAttachmentToken)
    expect(attach).toMatchObject({ kind: 'file', path: 'Desktop/sage/xhs_covers', displayName: 'xhs_covers' })

    const trailing = doc.filter(isTextToken).map(t => t.value)
    expect(trailing.join('')).toContain('封面图')

    // Compiling the imported document yields the well-formed quoted form.
    expect(compileToWireText(doc)).toBe('@file:`Desktop/sage/xhs_covers` 封面图')
  })

  it('returns an empty document for empty text', () => {
    expect(importFromWireText('')).toEqual([])
  })

  it('imports a single attachment with no surrounding text', () => {
    const doc = importFromWireText('@image:`/x/y.png`')
    expect(doc).toHaveLength(1)
    expect(doc[0]).toMatchObject({ type: 'attachment', kind: 'image', path: '/x/y.png' })
  })

  it('leaves @terminal: and @url: refs as text in v1', () => {
    const doc = importFromWireText('run @terminal:`bash` then see @url:`https://x.dev`')
    expect(doc.every(isTextToken)).toBe(true)
    expect(compileToWireText(doc)).toBe('run @terminal:`bash` then see @url:`https://x.dev`')
  })

  it('assigns a unique id per imported attachment', () => {
    const doc = importFromWireText('@file:`/a` and @file:`/b`')
    const ids = doc.filter(isAttachmentToken).map(t => t.id)
    expect(new Set(ids).size).toBe(ids.length)
  })
})

describe('normalizeDocument', () => {
  it('merges adjacent text tokens', () => {
    const doc: MessageDocument = [
      { type: 'text', value: 'a' },
      { type: 'text', value: 'b' },
      attachment({ id: 'x' }),
      { type: 'text', value: 'c' },
      { type: 'text', value: 'd' }
    ]
    const out = normalizeDocument(doc)
    expect(out.filter(isTextToken).map(t => t.value)).toEqual(['ab', 'cd'])
  })

  it('drops empty text tokens at the boundaries but keeps them between attachments', () => {
    const doc: MessageDocument = [
      { type: 'text', value: '' },
      attachment({ id: 'a', path: '/tmp/a' }),
      { type: 'text', value: '' },
      attachment({ id: 'b', path: '/tmp/b' }),
      { type: 'text', value: '' }
    ]
    const out = normalizeDocument(doc)
    expect(out).toHaveLength(3)
    expect(out[0]).toMatchObject({ type: 'attachment', id: 'a' })
    expect(out[1]).toMatchObject({ type: 'text', value: '' })
    expect(out[2]).toMatchObject({ type: 'attachment', id: 'b' })
  })

  it('does not mutate the input array', () => {
    const doc: MessageDocument = [
      { type: 'text', value: 'a' },
      { type: 'text', value: 'b' }
    ]
    const copy = JSON.parse(JSON.stringify(doc))
    normalizeDocument(doc)
    expect(doc).toEqual(copy)
  })
})

describe('attach routing', () => {
  it('collects absolute OS image paths as attach candidates', () => {
    const doc: MessageDocument = [
      attachment({ id: 'img-abs', kind: 'image', path: '/Users/k/temp/pic.jpg' }),
      attachment({ id: 'img-rel', kind: 'image', path: 'rel/pic.jpg' }),
      attachment({ id: 'file-abs', kind: 'file', path: '/Users/k/doc.txt' }),
      attachment({ id: 'folder-abs', kind: 'folder', path: '/Users/k/dir' })
    ]
    const candidates = collectAttachCandidates(doc)
    expect(candidates.map(t => t.id)).toEqual(['img-abs'])
  })

  it('rewrites a token path by stable id and leaves others untouched', () => {
    const doc: MessageDocument = [
      attachment({ id: 'i1', kind: 'image', path: '/local/a.jpg' }),
      attachment({ id: 'i2', kind: 'image', path: '/local/b.jpg' })
    ]
    const out = updateTokenPath(doc, 'i1', '.hermes/attachments/a.jpg')
    expect((out[0] as AttachmentToken).path).toBe('.hermes/attachments/a.jpg')
    expect((out[1] as AttachmentToken).path).toBe('/local/b.jpg')
    // Original untouched (immutable).
    expect((doc[0] as AttachmentToken).path).toBe('/local/a.jpg')
  })
})

describe('mergeAttachmentPillsIntoDocument', () => {
  it('prepends pill attachments not already inline and dedupes by path', async () => {
    const { mergeAttachmentPillsIntoDocument } = await import('./merge-pills')

    const doc: MessageDocument = [
      { type: 'text', value: 'hello' },
      attachment({ kind: 'folder', path: '/a/dir', displayName: 'dir' })
    ]

    const merged = mergeAttachmentPillsIntoDocument(doc, [
      { id: 'p1', kind: 'folder', label: 'dir', path: '/a/dir' },
      { id: 'p2', kind: 'image', label: 'pic.jpg', path: '/a/pic.jpg' }
    ])

    expect(merged.filter(isAttachmentToken).map(t => t.path)).toEqual(['/a/pic.jpg', '/a/dir'])
  })

  it('treats relative inline folder and absolute pill as the same ref', async () => {
    const { mergeAttachmentPillsIntoDocument } = await import('./merge-pills')

    const cwd = '/home/user'
    const doc: MessageDocument = [
      { type: 'text', value: 'hello ' },
      attachment({ kind: 'folder', path: 'Desktop/sage/xhs_covers', displayName: 'xhs_covers' }),
      { type: 'text', value: ' ' }
    ]

    const merged = mergeAttachmentPillsIntoDocument(
      doc,
      [
        { id: 'folder:xhs', kind: 'folder', label: 'xhs_covers', path: '/home/user/Desktop/sage/xhs_covers' },
        { id: 'image:pic', kind: 'image', label: 'pic.jpg', path: '/home/user/tmp/pic.jpg' }
      ],
      cwd
    )

    expect(merged.filter(isAttachmentToken).map(t => t.path)).toEqual([
      '/home/user/tmp/pic.jpg',
      '/home/user/Desktop/sage/xhs_covers'
    ])
    expect(compileToWireText(merged).match(/@folder:/g)?.length).toBe(1)
  })

  it('matches inline folder tail when session cwd differs from the absolute pill path', async () => {
    const { mergeAttachmentPillsIntoDocument } = await import('./merge-pills')

    const doc: MessageDocument = [
      { type: 'text', value: 'hello ' },
      attachment({ kind: 'folder', path: 'Desktop/sage/xhs_covers', displayName: 'xhs_covers' }),
      { type: 'text', value: ' ' }
    ]

    const merged = mergeAttachmentPillsIntoDocument(
      doc,
      [
        { id: 'folder:xhs', kind: 'folder', label: 'xhs_covers', path: '/home/user/Desktop/sage/xhs_covers' },
        { id: 'image:pic', kind: 'image', label: 'pic.jpg', path: '/home/user/tmp/pic.jpg' }
      ],
      '/repo/apps/desktop'
    )

    expect(merged.filter(isAttachmentToken).map(t => t.path)).toEqual([
      '/home/user/tmp/pic.jpg',
      '/home/user/Desktop/sage/xhs_covers'
    ])
    expect(compileToWireText(merged).match(/@folder:/g)?.length).toBe(1)
  })
})

describe('dedupeAttachmentTokens', () => {
  it('collapses relative and absolute paths with the same cwd', async () => {
    const { dedupeAttachmentTokens } = await import('./validate')

    const cwd = '/home/user'
    const doc: MessageDocument = [
      attachment({ id: 'abs', kind: 'folder', path: '/home/user/Desktop/sage/xhs_covers' }),
      { type: 'text', value: 'hello' },
      attachment({ id: 'rel', kind: 'folder', path: 'Desktop/sage/xhs_covers' })
    ]

    const out = dedupeAttachmentTokens(doc, cwd)

    expect(out.filter(isAttachmentToken)).toHaveLength(1)
    expect((out[0] as AttachmentToken).path).toBe('/home/user/Desktop/sage/xhs_covers')
  })
})

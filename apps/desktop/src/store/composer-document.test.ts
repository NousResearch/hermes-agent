import { afterEach, describe, expect, it } from 'vitest'

import { isAttachmentToken, isTextToken } from '@/lib/message-document'

import {
  $composerDocument,
  appendTextToken,
  clearComposerDocument,
  createAttachmentToken,
  insertAttachmentToken,
  setComposerDocument
} from './composer-document'

afterEach(() => {
  clearComposerDocument()
})

describe('$composerDocument helpers', () => {
  it('normalizes on set (merges adjacent text, trims boundary empties)', () => {
    setComposerDocument([
      { type: 'text', value: '' },
      { type: 'text', value: 'a' },
      { type: 'text', value: 'b' }
    ])
    expect($composerDocument.get()).toEqual([{ type: 'text', value: 'ab' }])
  })

  it('appendTextToken merges into the trailing text token', () => {
    setComposerDocument([{ type: 'text', value: 'hi ' }])
    appendTextToken('there')
    expect($composerDocument.get()).toEqual([{ type: 'text', value: 'hi there' }])
  })

  it('appendTextToken ignores empty input', () => {
    setComposerDocument([{ type: 'text', value: 'x' }])
    appendTextToken('')
    expect($composerDocument.get()).toEqual([{ type: 'text', value: 'x' }])
  })

  it('insertAttachmentToken places a token at the given index', () => {
    setComposerDocument([{ type: 'text', value: 'before after' }])
    const token = createAttachmentToken({ kind: 'folder', path: '/a/dir', displayName: 'dir' })
    insertAttachmentToken(token, 1)

    const doc = $composerDocument.get()
    expect(doc).toHaveLength(2)
    expect(isTextToken(doc[0]!)).toBe(true)
    expect(isAttachmentToken(doc[1]!)).toBe(true)
  })

  it('createAttachmentToken assigns a stable unique id when omitted', () => {
    const a = createAttachmentToken({ kind: 'image', path: '/a.png', displayName: 'a.png' })
    const b = createAttachmentToken({ kind: 'image', path: '/b.png', displayName: 'b.png' })
    expect(a.id).not.toBe(b.id)
    expect(a.type).toBe('attachment')
  })
})

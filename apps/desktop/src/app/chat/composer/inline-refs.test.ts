import { describe, expect, it } from 'vitest'

import { droppedFileInlineRef, splitDroppedFilesForComposer } from './inline-refs'

describe('splitDroppedFilesForComposer', () => {
  it('routes native OS image drops to attachments instead of inline @file refs', () => {
    const drop = {
      file: new File(['png'], 'screenshot.png', { type: 'image/png' }),
      path: '/Users/seiyeong/Library/Containers/me.damir.dropover-mac/Data/tmp/Screenshots/screenshot.png'
    }

    const { attachmentCandidates, inlineRefCandidates } = splitDroppedFilesForComposer([drop])

    expect(attachmentCandidates).toEqual([drop])
    expect(inlineRefCandidates).toEqual([])
    expect(inlineRefCandidates.map(candidate => droppedFileInlineRef(candidate, '/Users/seiyeong/sywork'))).toEqual([])
  })

  it('keeps in-app path drags as inline context refs', () => {
    const drop = {
      path: '/Users/seiyeong/sywork/10-projects/spec.md'
    }

    const { attachmentCandidates, inlineRefCandidates } = splitDroppedFilesForComposer([drop])

    expect(attachmentCandidates).toEqual([])
    expect(inlineRefCandidates).toEqual([drop])
    expect(inlineRefCandidates.map(candidate => droppedFileInlineRef(candidate, '/Users/seiyeong/sywork'))).toEqual([
      '@file:10-projects/spec.md'
    ])
  })
})

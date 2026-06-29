import { describe, expect, it } from 'vitest'

import type { ComposerAttachment } from '@/store/composer'

import {
  attachmentDisplayText,
  coerceThinkingText,
  optimisticAttachmentRef,
  parseCommandDispatch
} from './chat-runtime'

const DATA_URL = 'data:image/png;base64,iVBORw0KGgoAAAANS'

function attachment(overrides: Partial<ComposerAttachment> & Pick<ComposerAttachment, 'kind'>): ComposerAttachment {
  return { id: 'a', label: 'file.png', ...overrides }
}

describe('optimisticAttachmentRef', () => {
  it('renders an image from its in-hand base64 preview (no @image: path ref)', () => {
    const ref = optimisticAttachmentRef(attachment({ kind: 'image', detail: '/tmp/shot.png', previewUrl: DATA_URL }))

    // The raw data URL flows through extractEmbeddedImages → inline thumbnail,
    // dodging the remote /api/media 403 an @image:<localpath> ref would hit.
    expect(ref).toBe(DATA_URL)
  })

  it('falls back to an @image: path ref when no preview is available', () => {
    expect(optimisticAttachmentRef(attachment({ kind: 'image', detail: '/tmp/shot.png' }))).toBe('@image:/tmp/shot.png')
  })

  it('ignores a non-data preview url and uses the path ref', () => {
    const ref = optimisticAttachmentRef(
      attachment({ kind: 'image', detail: '/tmp/shot.png', previewUrl: 'https://example.com/x.png' })
    )

    expect(ref).toBe('@image:/tmp/shot.png')
  })

  it('passes non-image attachments straight through to attachmentDisplayText', () => {
    expect(optimisticAttachmentRef(attachment({ kind: 'file', refText: '@file:src/a.ts', previewUrl: DATA_URL }))).toBe(
      '@file:src/a.ts'
    )
  })

  // Session switches / draft restores can leave undefined|null holes in the
  // composer attachments array. AttachmentList already filters them (#49624),
  // but the submit path maps the same array through these helpers — an unguarded
  // hole threw "Cannot read properties of undefined (reading 'refText')",
  // crashing the chat surface (blank pane). The helpers must no-op on holes.
  it('returns null for an undefined attachment instead of throwing', () => {
    expect(() => optimisticAttachmentRef(undefined as unknown as ComposerAttachment)).not.toThrow()
    expect(optimisticAttachmentRef(undefined as unknown as ComposerAttachment)).toBeNull()
  })

  it('returns null for a null attachment instead of throwing', () => {
    expect(optimisticAttachmentRef(null as unknown as ComposerAttachment)).toBeNull()
  })
})

describe('attachmentDisplayText', () => {
  it('returns null for undefined|null instead of reading .kind/.refText on a hole', () => {
    expect(() => attachmentDisplayText(undefined as unknown as ComposerAttachment)).not.toThrow()
    expect(attachmentDisplayText(undefined as unknown as ComposerAttachment)).toBeNull()
    expect(attachmentDisplayText(null as unknown as ComposerAttachment)).toBeNull()
  })

  it('still resolves a normal file ref', () => {
    expect(attachmentDisplayText(attachment({ kind: 'file', refText: '@file:src/a.ts' }))).toBe('@file:src/a.ts')
  })
})

describe('coerceThinkingText', () => {
  it('strips streaming status prefixes from thinking deltas', () => {
    expect(coerceThinkingText("◉_◉ processing... checking the user's request")).toBe("checking the user's request")
    expect(coerceThinkingText('(¬‿¬) analyzing... reading the file')).toBe('reading the file')
  })

  it('drops empty thinking rewrite placeholder text', () => {
    expect(
      coerceThinkingText(
        "◉_◉ processing... I don't see any current rewritten thinking or next thinking to process. Could you provide the thinking content you'd like me to rewrite?"
      )
    ).toBe('')
  })
})

describe('parseCommandDispatch', () => {
  it('keeps the notice on a send directive (e.g. /goal set)', () => {
    // The backend's /goal set returns {type:send, notice:"⊙ Goal set …", message}.
    // Dropping the notice made /goal look like it did nothing in the desktop app.
    const parsed = parseCommandDispatch({ type: 'send', notice: '⊙ Goal set', message: 'do the thing' })

    expect(parsed).toEqual({ type: 'send', message: 'do the thing', notice: '⊙ Goal set' })
  })

  it('keeps message-only send directives working (no notice)', () => {
    expect(parseCommandDispatch({ type: 'send', message: 'hi' })).toEqual({
      type: 'send',
      message: 'hi',
      notice: undefined
    })
  })

  it('parses a prefill directive with its notice (e.g. /undo)', () => {
    const parsed = parseCommandDispatch({ type: 'prefill', notice: 'backed up 1 turn', message: 'edit me' })

    expect(parsed).toEqual({ type: 'prefill', message: 'edit me', notice: 'backed up 1 turn' })
  })

  it('rejects a prefill directive missing its message', () => {
    expect(parseCommandDispatch({ type: 'prefill', notice: 'x' })).toBeNull()
  })
})

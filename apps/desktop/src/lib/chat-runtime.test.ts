import { describe, expect, it } from 'vitest'

import { type ChatMessage, liveUserCorrectionInsertionIndex, textPart } from '@/lib/chat-messages'
import type { ComposerAttachment } from '@/store/composer'

import {
  attachmentDisplayText,
  attachmentId,
  coerceThinkingText,
  messageCreatedAt,
  optimisticAttachmentRef,
  parseCommandDispatch,
  parseSlashCommand
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

describe('parseSlashCommand', () => {
  it('parses a single-line command', () => {
    expect(parseSlashCommand('/some-skill do something')).toEqual({
      arg: 'do something',
      name: 'some-skill'
    })
  })

  it('keeps a multiline arg intact instead of failing the whole parse (#41323)', () => {
    expect(parseSlashCommand('/goal Write a Python script\nthat prints Hello World')).toEqual({
      arg: 'Write a Python script\nthat prints Hello World',
      name: 'goal'
    })
  })

  it('parses a skill command with a long pasted multi-paragraph context (#55510)', () => {
    const context = 'summarize this:\n\nparagraph one\nparagraph two\n\nparagraph three'

    expect(parseSlashCommand(`/some-skill ${context}`)).toEqual({
      arg: context,
      name: 'some-skill'
    })
  })

  it('takes the name across a newline boundary like the CLI and gateway (split on any whitespace)', () => {
    expect(parseSlashCommand('/goal\npasted block')).toEqual({ arg: 'pasted block', name: 'goal' })
  })

  it('keeps truly empty slash input empty', () => {
    expect(parseSlashCommand('/')).toEqual({ arg: '', name: '' })
    expect(parseSlashCommand('/   ')).toEqual({ arg: '', name: '' })
  })

  it('does not treat text after horizontal whitespace as a command name (CLI parity)', () => {
    expect(parseSlashCommand('/ some words')).toEqual({ arg: '', name: '' })
  })
})

describe('attachmentId', () => {
  it('normalizes a trailing slash on a url so a re-attach dedupes (#59305 P2)', () => {
    expect(attachmentId('url', 'https://example.com/a')).toBe(attachmentId('url', 'https://example.com/a/'))
  })

  it('falls back to the trimmed raw value for a malformed url instead of throwing', () => {
    expect(() => attachmentId('url', 'not a url')).not.toThrow()
    expect(attachmentId('url', '  not a url  ')).toBe(attachmentId('url', 'not a url'))
  })

  it('normalizes backslash path separators so a Windows and posix path dedupe', () => {
    expect(attachmentId('file', 'a\\b.ts')).toBe(attachmentId('file', 'a/b.ts'))
  })

  it('normalizes a trailing slash on a folder path', () => {
    expect(attachmentId('folder', 'src/app/')).toBe(attachmentId('folder', 'src/app'))
  })

  it('does not collapse a bare root path to an empty id', () => {
    expect(attachmentId('folder', '/')).toBe('folder:/')
  })

  it('keeps distinct urls distinct', () => {
    expect(attachmentId('url', 'https://example.com/a')).not.toBe(attachmentId('url', 'https://example.com/b'))
  })
})

describe('messageCreatedAt', () => {
  const NOW = Date.UTC(2026, 6, 28, 18, 0, 0)

  it('reads the authoritative Unix-seconds timestamp (not ms)', () => {
    // 1785282262s → July 2026, not the 1970 epoch a *1000-less read would give.
    expect(messageCreatedAt({ timestamp: 1785282262 }, NOW).getFullYear()).toBe(2026)
  })

  it('falls back to now — never digs digits out of the id → "20663d ago" (1970)', () => {
    // The old fallback did `new Date(Number(id.match(/\d+/)))`, so a session-style
    // id like 20260728_184420_05e697 parsed to 20260728 *ms* = Jan 1970, showing
    // as an absurd 20663-day age. A timestamp-less message is freshly created.
    expect(messageCreatedAt({ timestamp: undefined }, NOW).getTime()).toBe(NOW)
  })

  it('treats a zero / non-finite timestamp as absent', () => {
    expect(messageCreatedAt({ timestamp: 0 }, NOW).getTime()).toBe(NOW)
    expect(messageCreatedAt({ timestamp: Number.NaN }, NOW).getTime()).toBe(NOW)
  })
})

describe('liveUserCorrectionInsertionIndex', () => {
  const message = (id: string, role: ChatMessage['role'], text: string): ChatMessage => ({
    id,
    role,
    parts: [textPart(text)]
  })

  it('inserts before the active streaming assistant when streamId is current', () => {
    const messages = [
      message('u1', 'user', 'first prompt'),
      message('a1', 'assistant', 'first answer'),
      message('u2', 'user', 'second prompt'),
      message('stream-current', 'assistant', 'partial answer')
    ]

    expect(liveUserCorrectionInsertionIndex(messages, 'stream-current')).toBe(3)
  })

  it('appends to the live tail when streamId is missing or stale', () => {
    const messages = [
      message('u1', 'user', 'first prompt'),
      message('a1', 'assistant', 'first answer'),
      message('u2', 'user', 'second prompt'),
      message('a2', 'assistant', 'second answer')
    ]

    expect(liveUserCorrectionInsertionIndex(messages, null)).toBe(4)
    expect(liveUserCorrectionInsertionIndex(messages, 'missing-stream')).toBe(4)
  })
})

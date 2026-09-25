import { describe, expect, it } from 'vitest'

import type { ChatMessage, ChatMessagePart } from '@/lib/chat-messages'
import type { ComposerAttachment } from '@/store/composer'

import {
  attachmentDisplayText,
  attachmentId,
  coalesceToolOnlyAssistants,
  coerceThinkingText,
  createToolMergeCache,
  messageCreatedAt,
  optimisticAttachmentRef,
  toRuntimeMessage
} from './chat-runtime'

const DATA_URL = 'data:image/png;base64,iVBORw0KGgoAAAANS'
const THUMB_URL = 'data:image/png;base64,dGh1bWI='

function attachment(overrides: Partial<ComposerAttachment> & Pick<ComposerAttachment, 'kind'>): ComposerAttachment {
  return { id: 'a', label: 'file.png', ...overrides }
}

describe('optimisticAttachmentRef', () => {
  it('renders a path-backed image through the same @image: ref as a reloaded turn (#93204)', () => {
    const ref = optimisticAttachmentRef(attachment({ kind: 'image', detail: '/tmp/shot.png', previewUrl: DATA_URL }))

    // DirectiveImage paints a bounded thumbnail inline and hands the full file
    // to the lightbox, so the in-flight bubble matches the reloaded turn instead
    // of freezing on a 512px thumbnail. Remote gateways resolve the same path
    // over the authenticated media API (no /api/media 403).
    expect(ref).toBe('@image:/tmp/shot.png')
  })

  it('prefers the path ref even when a downscaled thumbnail is present (#93204)', () => {
    const ref = optimisticAttachmentRef(
      attachment({ kind: 'image', path: '/tmp/shot.png', previewUrl: DATA_URL, thumbnailUrl: THUMB_URL })
    )

    // The thumbnail no longer caps fidelity: the path lets the lightbox load the
    // original. Full bytes are read on demand and for upload.
    expect(ref).toBe('@image:/tmp/shot.png')
  })

  it('emits the path ref for a freshly attached image before its thumbnail resolves (#93204)', () => {
    // Previously this returned null (waiting on the resize); now the path drives
    // an @image: ref that DirectiveImage renders bounded-inline immediately.
    expect(optimisticAttachmentRef(attachment({ kind: 'image', detail: '/tmp/shot.png' }))).toBe('@image:/tmp/shot.png')
  })

  it('emits the path ref regardless of a non-data preview url (#93204)', () => {
    const ref = optimisticAttachmentRef(
      attachment({ kind: 'image', detail: '/tmp/shot.png', previewUrl: 'https://example.com/x.png' })
    )

    expect(ref).toBe('@image:/tmp/shot.png')
  })

  it('falls back to the bounded thumbnail for a path-less pasted image', () => {
    // No filesystem path to rehydrate from (raw clipboard bytes): keep the
    // inline thumbnail so the bubble still renders something.
    const ref = optimisticAttachmentRef(attachment({ kind: 'image', previewUrl: DATA_URL, thumbnailUrl: THUMB_URL }))

    expect(ref).toBe(THUMB_URL)
  })

  it('falls back to a data preview url when a path-less image has no thumbnail', () => {
    const ref = optimisticAttachmentRef(attachment({ kind: 'image', previewUrl: DATA_URL }))

    expect(ref).toBe(DATA_URL)
  })

  it('returns null for a path-less image with no renderable inline source', () => {
    expect(optimisticAttachmentRef(attachment({ kind: 'image', previewUrl: 'https://example.com/x.png' }))).toBeNull()
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
})

describe('attachmentDisplayText', () => {
  it('returns null for undefined|null instead of reading .kind/.refText on a hole', () => {
    expect(() => attachmentDisplayText(undefined as unknown as ComposerAttachment)).not.toThrow()
    expect(attachmentDisplayText(undefined as unknown as ComposerAttachment)).toBeNull()
    expect(attachmentDisplayText(null as unknown as ComposerAttachment)).toBeNull()
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

describe('toRuntimeMessage timeline metadata', () => {
  it('does not expose a fabricated visible timestamp for timestamp-less history', () => {
    const runtime = toRuntimeMessage({
      id: 'old-message',
      parts: [{ text: 'old', type: 'text' }],
      role: 'assistant'
    })

    expect((runtime.metadata?.custom as { timelineTimestamp?: number }).timelineTimestamp).toBeUndefined()
  })

  it('forwards turnStats on the assistant custom metadata', () => {
    const turnStats = { durationS: 12, input: 100, output: 20, cacheRead: 80, cacheWrite: 0, calls: 2 }
    const runtime = toRuntimeMessage({
      id: 'with-stats',
      parts: [{ text: 'ok', type: 'text' }],
      role: 'assistant',
      turnStats
    })

    expect((runtime.metadata?.custom as { turnStats?: typeof turnStats }).turnStats).toEqual(turnStats)
  })
})

describe('coalesceToolOnlyAssistants toolCallId uniqueness', () => {
  // Regression contract for #87857: two individually-clean assistant rows can
  // share a toolCallId (structural carry-over re-attaching a cached row's tool
  // calls while the same turn also exists as a committed row). Folding them
  // used to manufacture ONE message carrying the id twice — the exact shape
  // that makes assistant-ui's useResources throw and crash-loop the pane.
  const tool = (toolCallId: string): ChatMessagePart =>
    ({ type: 'tool-call', toolCallId, toolName: 'terminal', args: {} as never, argsText: '' }) as ChatMessagePart

  const assistant = (id: string, parts: ChatMessagePart[]): ChatMessage =>
    ({ id, role: 'assistant', parts }) as unknown as ChatMessage

  it('drops the copy the predecessor already carries, keeps the new call', () => {
    const merged = coalesceToolOnlyAssistants(
      [
        assistant('committed-49-assistant', [
          { type: 'text', text: 'working' } as ChatMessagePart,
          tool('call-a'),
          tool('call-b')
        ]),
        assistant('assistant-stream-49', [tool('call-b'), tool('call-c')])
      ],
      createToolMergeCache()
    )

    expect(merged).toHaveLength(1)

    const ids = merged[0].parts
      .filter(part => part.type === 'tool-call')
      .map(part => (part as { toolCallId: string }).toolCallId)

    expect(ids).toEqual(['call-a', 'call-b', 'call-c'])
  })

  it('folds a clean follow-up unchanged', () => {
    const merged = coalesceToolOnlyAssistants(
      [
        assistant('a1', [{ type: 'text', text: 'ok' } as ChatMessagePart, tool('call-a')]),
        assistant('a2', [tool('call-b')])
      ],
      createToolMergeCache()
    )

    expect(merged).toHaveLength(1)

    const ids = merged[0].parts
      .filter(part => part.type === 'tool-call')
      .map(part => (part as { toolCallId: string }).toolCallId)

    expect(ids).toEqual(['call-a', 'call-b'])
  })
})

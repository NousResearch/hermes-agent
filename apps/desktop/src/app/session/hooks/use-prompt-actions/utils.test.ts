import type { AppendMessage } from '@assistant-ui/react'
import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import {
  appendText,
  base64FromDataUrl,
  friendlyRemoteAttachError,
  imageFilenameFromPath,
  inlineErrorMessage,
  isSessionBusyError,
  isSessionIdCandidate,
  isSessionNotFoundError,
  slashStatusText,
  visibleUserIndexAtOrdinal,
  visibleUserOrdinal,
  withFileAttachRetry
} from './utils'

describe('isSessionIdCandidate', () => {
  it('accepts the timestamped and hex id forms', () => {
    expect(isSessionIdCandidate('20260101_120000_abc123')).toBe(true)
    expect(isSessionIdCandidate('a'.repeat(32))).toBe(true)
  })

  it('rejects arbitrary text', () => {
    expect(isSessionIdCandidate('hello world')).toBe(false)
    expect(isSessionIdCandidate('abc')).toBe(false)
  })
})

describe('inlineErrorMessage', () => {
  it('unwraps an electron remote-method error', () => {
    expect(inlineErrorMessage(new Error("Error invoking remote method 'x': Error: boom"), 'fallback')).toBe('boom')
  })

  it('strips a leading Error: prefix', () => {
    expect(inlineErrorMessage(new Error('Error: nope'), 'fallback')).toBe('nope')
  })

  it('falls back for non-error, non-string input', () => {
    expect(inlineErrorMessage(undefined, 'fallback')).toBe('fallback')
  })
})

describe('session error classifiers', () => {
  it('detects not-found and busy errors', () => {
    expect(isSessionNotFoundError(new Error('Session not found'))).toBe(true)
    expect(isSessionBusyError(new Error('session busy'))).toBe(true)
    expect(isSessionNotFoundError(new Error('other'))).toBe(false)
    expect(isSessionBusyError(new Error('other'))).toBe(false)
  })
})

describe('base64FromDataUrl', () => {
  it('returns the part after the comma', () => {
    expect(base64FromDataUrl('data:image/png;base64,AAAA')).toBe('AAAA')
  })

  it('returns empty when there is no comma', () => {
    expect(base64FromDataUrl('nope')).toBe('')
  })
})

describe('imageFilenameFromPath', () => {
  it('takes the last path segment', () => {
    expect(imageFilenameFromPath('/a/b/c.png')).toBe('c.png')
    expect(imageFilenameFromPath('C:\\a\\b\\d.jpg')).toBe('d.jpg')
  })

  it('defaults when the path is empty', () => {
    expect(imageFilenameFromPath('')).toBe('image.png')
  })
})

describe('friendlyRemoteAttachError', () => {
  it('rewrites a too-large error with the parsed cap', () => {
    const err = friendlyRemoteAttachError(new Error('file is too large (20 bytes; limit 16777216 bytes)'), 'pic.png')
    expect(err.message).toBe('pic.png is too large to upload to the remote gateway (max 16 MB).')
  })

  it('rewrites a missing chunk RPC as a backend update requirement', () => {
    const err = friendlyRemoteAttachError(
      new Error('method not found: file.attach.begin'),
      'deck.pptx'
    )
    expect(err.message).toBe(
      'The remote Hermes backend is too old for large-file uploads. Update the backend and try again.'
    )
  })

  it('passes unrelated errors through', () => {
    const original = new Error('something else')
    expect(friendlyRemoteAttachError(original, 'pic.png')).toBe(original)
  })
})

describe('slashStatusText', () => {
  it('joins command and trimmed output', () => {
    expect(slashStatusText('/model', '  gpt  ')).toBe('slash:/model\ngpt')
  })

  it('omits empty output', () => {
    expect(slashStatusText('/clear', '   ')).toBe('slash:/clear')
  })
})

describe('appendText', () => {
  it('concatenates text parts and trims', () => {
    const message = {
      content: [
        { type: 'text', text: ' a' },
        { type: 'text', text: 'b ' }
      ]
    } as unknown as AppendMessage

    expect(appendText(message)).toBe('ab')
  })
})

describe('file attachment retry', () => {
  it('retries transient gateway failures and returns the eventual result', async () => {
    let calls = 0
    const result = await withFileAttachRetry(
      async () => {
        calls += 1
        if (calls < 3) {
          throw new Error('request timed out: file.attach.chunk')
        }
        return 'ok'
      },
      [0, 0]
    )

    expect(result).toBe('ok')
    expect(calls).toBe(3)
  })

  it('does not retry permanent validation errors', async () => {
    let calls = 0
    await expect(
      withFileAttachRetry(
        async () => {
          calls += 1
          throw new Error('unexpected offset')
        },
        [0, 0]
      )
    ).rejects.toThrow('unexpected offset')
    expect(calls).toBe(1)
  })
})

describe('visible message ordinals', () => {
  const messages = [
    { role: 'user', hidden: false },
    { role: 'assistant' },
    { role: 'user', hidden: true },
    { role: 'user', hidden: false }
  ] as ChatMessage[]

  it('counts visible user messages before an index', () => {
    expect(visibleUserOrdinal(messages, messages.length)).toBe(2)
  })

  it('maps an ordinal back to a message index, skipping hidden', () => {
    expect(visibleUserIndexAtOrdinal(messages, 1)).toBe(3)
    expect(visibleUserIndexAtOrdinal(messages, 5)).toBe(-1)
  })
})

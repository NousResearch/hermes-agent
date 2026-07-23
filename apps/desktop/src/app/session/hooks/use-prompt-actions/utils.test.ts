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
  renderRpcResult,
  slashStatusText,
  visibleUserIndexAtOrdinal,
  visibleUserOrdinal
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

  it('passes non-cap errors through', () => {
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

describe('visible user ordinals', () => {
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

describe('renderRpcResult', () => {
  describe('session.compress (summary shape)', () => {
    it('renders the summary headline with token line and note', () => {
      expect(
        renderRpcResult(
          {
            summary: {
              headline: 'Compressed: 280 → 120 messages',
              token_line: 'Approx request size: ~126,575 → ~30,000 tokens',
              note: 'Removed 8 older turns',
              noop: false
            }
          },
          'compress'
        )
      ).toBe(
        [
          '✓ Compressed: 280 → 120 messages',
          '  Approx request size: ~126,575 → ~30,000 tokens',
          '  Removed 8 older turns'
        ].join('\n')
      )
    })

    it('drops the checkmark when the summary is a noop', () => {
      expect(
        renderRpcResult(
          { summary: { headline: 'Already compressed', note: 'No new turns since last compress', noop: true } },
          'compress'
        )
      ).toBe('Already compressed\n  No new turns since last compress')
    })
  })

  describe('session.steer', () => {
    it('reports a queued steer with the original text', () => {
      expect(renderRpcResult({ status: 'queued', text: 'skip the docs' }, 'steer')).toBe(
        'Steered · "skip the docs" queued for next tool call'
      )
    })

    it('reports a rejected steer without echoing user text', () => {
      expect(renderRpcResult({ status: 'rejected', text: 'whatever' }, 'steer')).toBe(
        'Steer rejected — agent declined input'
      )
    })
  })

  describe('process.stop', () => {
    it('reports the numeric number of stopped processes', () => {
      expect(renderRpcResult({ killed: 2 }, 'stop')).toBe('Stopped 2 background processes.')
    })

    it('reports nothing-to-stop when the numeric count is zero', () => {
      expect(renderRpcResult({ killed: 0 }, 'stop')).toBe('No background processes to stop.')
    })
  })

  describe('session.save', () => {
    it('echoes the saved file path', () => {
      expect(renderRpcResult({ file: '/home/user/.hermes/sessions/saved/x.json' }, 'save')).toBe(
        'Saved transcript to /home/user/.hermes/sessions/saved/x.json'
      )
    })
  })

  describe('session.status', () => {
    it('passes through the multi-line plain-text output verbatim', () => {
      const output = 'Hermes TUI Status\n\nSession ID: s-1\nModel: nous-hermes-3 (unknown)'
      expect(renderRpcResult({ output }, 'status')).toBe(output)
    })
  })

  describe('session.usage', () => {
    it('formats calls / input / output / total with thousands separators', () => {
      expect(renderRpcResult({ calls: 12, input: 1_234_567, output: 89_012, total: 1_323_579 }, 'usage')).toBe(
        'Usage: 12 calls · 1,234,567 in / 89,012 out · 1,323,579 total'
      )
    })

    it('appends credits_lines when present', () => {
      const body = renderRpcResult(
        {
          calls: 1,
          input: 10,
          output: 20,
          total: 30,
          credits_lines: ['Nous credits: 8,420 remaining', 'Resets: 2026-08-01']
        },
        'usage'
      )

      expect(body.split('\n')).toEqual([
        'Usage: 1 calls · 10 in / 20 out · 30 total',
        'Nous credits: 8,420 remaining',
        'Resets: 2026-08-01'
      ])
    })
  })

  describe('agents.list', () => {
    it('reports no running tasks when the array is empty', () => {
      expect(renderRpcResult({ processes: [] }, 'agents')).toBe('No background tasks running.')
    })

    it('formats each process with status, command, and metadata', () => {
      expect(
        renderRpcResult(
          {
            processes: [
              { session_id: 's-1', command: 'npm test', status: 'running', uptime: 42 },
              { session_id: 's-2', command: 'vitest', status: 'completed' }
            ]
          },
          'agents'
        )
      ).toBe(['• [running] npm test (42s · s-1)', '• [completed] vitest (s-2)'].join('\n'))
    })
  })

  describe('fallback', () => {
    it('serialises unknown shapes as JSON so we never lose data', () => {
      expect(renderRpcResult({ custom: 'value', nested: { a: 1 } }, 'mystery')).toBe(
        '/mystery: {"custom":"value","nested":{"a":1}}'
      )
    })

    it('returns an empty string for null and primitive payloads', () => {
      expect(renderRpcResult(null, 'x')).toBe('')
      expect(renderRpcResult(undefined, 'x')).toBe('')
      expect(renderRpcResult('plain string', 'x')).toBe('')
      expect(renderRpcResult(42, 'x')).toBe('')
    })
  })
})

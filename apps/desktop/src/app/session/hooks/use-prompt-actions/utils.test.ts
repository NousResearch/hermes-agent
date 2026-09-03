import type { AppendMessage } from '@assistant-ui/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import { rememberDesktopCommandsCatalog } from '@/lib/desktop-slash-commands'

import {
  acquireSubmitInFlight,
  appendText,
  base64FromDataUrl,
  clearSessionRecentlyInterrupted,
  clearSubmitInFlight,
  friendlyRemoteAttachError,
  type GatewayRequest,
  imageFilenameFromPath,
  inlineErrorMessage,
  isSessionBusyError,
  isSessionIdCandidate,
  isSessionNotFoundError,
  isSessionRecentlyInterrupted,
  isSubmitInFlight,
  isTargetSessionBusy,
  markSessionRecentlyInterrupted,
  readFileDataUrlForAttach,
  RECENT_INTERRUPT_COOLDOWN_MS,
  releaseSubmitInFlight,
  renderCommandsCatalog,
  renderRpcResult,
  SessionRecoveryAborted,
  shouldInterruptBeforeRewind,
  slashStatusText,
  SUBMIT_IN_FLIGHT_TTL_MS,
  visibleUserIndexAtOrdinal,
  visibleUserOrdinal,
  withSessionNotFoundResume
} from './utils'

describe('desktop command catalog', () => {
  it('/help all includes unavailable commands omitted from backend categories', () => {
    const catalog = {
      categories: [
        {
          name: 'Commands',
          pairs: [['/help', 'Show help']] as [string, string][]
        }
      ],
      canon: {
        '/remote-alias': '/remote-only'
      },
      commands: {
        '/restart': { desktop: 'terminal' },
        '/remote-only': { desktop: 'terminal' }
      }
    }

    const copy = {} as Parameters<typeof renderCommandsCatalog>[1]

    const all = renderCommandsCatalog(catalog, copy, { includeUnavailable: true })
    const rowCount = (command: string) => all.match(new RegExp(`^${command}\\s`, 'gm'))?.length ?? 0

    for (const command of [
      '/restart',
      '/approve',
      '/deny',
      '/sethome',
      '/set-home',
      '/remote-only',
      '/model',
      '/resume'
    ]) {
      expect(rowCount(command), command).toBe(1)
    }

    for (const alias of [
      '/fork',
      '/compact',
      '/sessions',
      '/switch',
      '/reset',
      '/commands',
      '/learning',
      '/memory-graph',
      '/generate-pet',
      '/remote-alias'
    ]) {
      expect(rowCount(alias), alias).toBe(1)
    }

    expect(all).toContain('/restart is only available in the terminal interface')
    expect(all).toContain('/approve is only used from messaging platforms')
    expect(all).toContain('/remote-only is only available in the terminal interface')
    expect(all).toMatch(/^\/remote-alias\s+\/remote-only is only available in the terminal interface\.$/m)
    expect(renderCommandsCatalog(catalog, copy)).not.toContain('/restart')
    expect(renderCommandsCatalog(catalog, copy)).not.toContain('/commands')
  })

  it('merges categorized commands with pairs-only skills without duplicates', () => {
    const copy = {
      desktopCommands: 'Desktop commands',
      skillCommandsAvailable: (count: number) => `${count} skill commands available`
    } as Parameters<typeof renderCommandsCatalog>[1]

    const output = renderCommandsCatalog(
      {
        categories: [{ name: 'Info', pairs: [['/help', 'Show help']] }],
        pairs: [
          ['/help', 'Show help'],
          ['/docx', 'Work with Word documents']
        ],
        skills: { '/docx': { origin: 'bundled', usage: 1 } },
        skill_count: 1
      },
      copy
    )

    expect(output.match(/^\/help\s/gm)).toHaveLength(1)
    expect(output.match(/^\/docx\s/gm)).toHaveLength(1)
    expect(output).toContain('Work with Word documents')
  })

  it('uses the exact catalog being rendered for dynamic availability guidance', () => {
    rememberDesktopCommandsCatalog({ commands: { '/dynamic': { desktop: 'settings' } } })

    const output = renderCommandsCatalog(
      {
        commands: { '/dynamic': { desktop: 'terminal' } },
        pairs: [['/dynamic', 'Dynamic backend command']]
      },
      { desktopCommands: 'Desktop commands' } as Parameters<typeof renderCommandsCatalog>[1],
      { includeUnavailable: true }
    )

    expect(output).toContain('/dynamic is only available in the terminal interface')
    expect(output).not.toContain('/dynamic is managed from the desktop sidebar')
  })
})

afterEach(() => {
  rememberDesktopCommandsCatalog(undefined)
  clearSessionRecentlyInterrupted()
  clearSubmitInFlight()
})

describe('recent interrupt cooldown', () => {
  it('is true within the cooldown and false after expiry', () => {
    const sessionId = 'sess-cooldown'
    const t0 = 1_000_000

    markSessionRecentlyInterrupted(sessionId, t0)

    expect(isSessionRecentlyInterrupted(sessionId, t0)).toBe(true)
    expect(isSessionRecentlyInterrupted(sessionId, t0 + RECENT_INTERRUPT_COOLDOWN_MS - 1)).toBe(true)
    expect(isSessionRecentlyInterrupted(sessionId, t0 + RECENT_INTERRUPT_COOLDOWN_MS)).toBe(false)
  })

  it('returns false after mark + elapsed past cooldown', () => {
    const sessionId = 'sess-elapsed'
    const t0 = 5_000_000

    markSessionRecentlyInterrupted(sessionId, t0)
    expect(isSessionRecentlyInterrupted(sessionId, t0 + RECENT_INTERRUPT_COOLDOWN_MS + 1)).toBe(false)
  })

  it('shouldInterruptBeforeRewind is true when recently interrupted even if not busy', () => {
    const sessionId = 'sess-edit-after-stop'
    const t0 = 9_000_000

    markSessionRecentlyInterrupted(sessionId, t0)

    expect(shouldInterruptBeforeRewind({ busy: false, sessionId, now: t0 + 500 })).toBe(true)
    expect(shouldInterruptBeforeRewind({ busy: false, sessionId, now: t0 + RECENT_INTERRUPT_COOLDOWN_MS + 1 })).toBe(
      false
    )
  })

  it('shouldInterruptBeforeRewind stays false for idle sessions with no recent interrupt', () => {
    expect(shouldInterruptBeforeRewind({ busy: false, sessionId: 'idle-sess' })).toBe(false)
    expect(shouldInterruptBeforeRewind({ busy: true, sessionId: 'busy-sess' })).toBe(true)
  })
})

describe('submit in-flight TTL', () => {
  it('blocks a second acquire while fresh and frees after TTL without explicit release', () => {
    const key = 'lock-ttl'
    const t0 = 2_000_000

    expect(acquireSubmitInFlight(key, t0)).toBe(true)
    expect(isSubmitInFlight(key, t0 + 1)).toBe(true)
    expect(acquireSubmitInFlight(key, t0 + 1)).toBe(false)

    expect(isSubmitInFlight(key, t0 + SUBMIT_IN_FLIGHT_TTL_MS)).toBe(false)
    expect(acquireSubmitInFlight(key, t0 + SUBMIT_IN_FLIGHT_TTL_MS)).toBe(true)
  })

  it('release clears the lock immediately', () => {
    const key = 'lock-release'
    const t0 = 3_000_000

    expect(acquireSubmitInFlight(key, t0)).toBe(true)
    releaseSubmitInFlight(key)
    expect(isSubmitInFlight(key, t0 + 1)).toBe(false)
    expect(acquireSubmitInFlight(key, t0 + 1)).toBe(true)
  })
})

describe('isTargetSessionBusy', () => {
  it('reads the target session slice, not the leftover foreground flag', () => {
    expect(isTargetSessionBusy({ a: { busy: true }, b: { busy: false } }, 'b', true)).toBe(false)
    expect(isTargetSessionBusy({ a: { busy: true } }, 'b', true)).toBe(false)
  })

  it('uses the focused draft flag only when there is no session id', () => {
    expect(isTargetSessionBusy({}, null, true)).toBe(true)
    expect(isTargetSessionBusy({}, null, false)).toBe(false)
  })
})

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

describe('withSessionNotFoundResume', () => {
  const STORED = 'stored-1'
  const DEAD = 'rt-dead'
  const FRESH = 'rt-fresh'

  // Profile resolution is injected, so these tests never reach the REST layer.
  // Before this was a dependency the helper called resolveStoredSession ->
  // getSession() and its coverage silently depended on $sessions/$profiles
  // state left behind by whichever test file ran first.
  const deps = (overrides: Record<string, unknown> = {}) => ({
    requestGateway: vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        return { session_id: FRESH }
      }

      throw new Error(`unexpected ${method}`)
    }) as unknown as GatewayRequest,
    resolveProfile: vi.fn(async () => undefined),
    ...overrides
  })

  it('returns the first-call result without resuming when the RPC succeeds', async () => {
    const d = deps()
    const call = vi.fn(async (sid: string) => `ok:${sid}`)

    expect(await withSessionNotFoundResume(DEAD, STORED, call, d)).toEqual({
      recovered: false,
      result: `ok:${DEAD}`,
      sessionId: DEAD
    })
    expect(call).toHaveBeenCalledTimes(1)
    expect(d.requestGateway).not.toHaveBeenCalled()
  })

  // The whole bug class: every session-scoped RPC recovers identically. Before
  // consolidation only prompt.submit did, so attach/compress/rewind surfaced a
  // raw "session not found" after sleep while plain text worked.
  it.each([
    ['image.attach_bytes', 'attach an image'],
    ['file.attach', 'attach a file'],
    ['session.compress', 'run /compress'],
    ['prompt.submit', 'submit a rewind'],
    ['session.interrupt', 'stop a turn']
  ])('resumes and retries once so %s can %s after a stale drop', async rpc => {
    const d = deps()
    let attempts = 0

    const call = vi.fn(async (sid: string) => {
      attempts += 1

      if (attempts === 1) {
        throw new Error(`${rpc} failed: session not found`)
      }

      return `ok:${sid}`
    })

    expect(await withSessionNotFoundResume(DEAD, STORED, call, d)).toEqual({
      recovered: true,
      result: `ok:${FRESH}`,
      sessionId: FRESH
    })
    expect(call.mock.calls.map(c => c[0])).toEqual([DEAD, FRESH])
  })

  it('resumes on the session-owning profile so recovery cannot fork the conversation', async () => {
    const d = deps({ resolveProfile: vi.fn(async () => 'work') })
    let first = true

    await withSessionNotFoundResume(
      DEAD,
      STORED,
      async (sid: string) => {
        if (first) {
          first = false
          throw new Error('session not found')
        }

        return sid
      },
      d
    )

    expect(d.requestGateway).toHaveBeenCalledWith(
      'session.resume',
      expect.objectContaining({ session_id: STORED, source: 'desktop', omit_messages: true, profile: 'work' })
    )
  })

  it('publishes the recovered id exactly once', async () => {
    const onRecovered = vi.fn()
    const d = deps({ onRecovered })
    let first = true

    await withSessionNotFoundResume(
      DEAD,
      STORED,
      async (sid: string) => {
        if (first) {
          first = false
          throw new Error('session not found')
        }

        return sid
      },
      d
    )

    expect(onRecovered).toHaveBeenCalledExactlyOnceWith(FRESH)
  })

  it('aborts instead of retrying when the user moved on during the resume', async () => {
    const onRecovered = vi.fn()
    const d = deps({ driftReason: () => 'selection:a->b', onRecovered })

    const call = vi.fn(async () => {
      throw new Error('session not found')
    })

    await expect(withSessionNotFoundResume(DEAD, STORED, call, d)).rejects.toThrow(SessionRecoveryAborted)
    // Retry suppressed and nothing published: landing it would run against a
    // session the user is no longer looking at.
    expect(call).toHaveBeenCalledTimes(1)
    expect(onRecovered).not.toHaveBeenCalled()
  })

  it('rethrows the original error when the resume itself 404s (never-persisted draft)', async () => {
    const d = deps({
      requestGateway: vi.fn(async () => {
        throw new Error('resume failed: session not found')
      }) as unknown as GatewayRequest
    })

    const call = vi.fn(async () => {
      throw new Error('prompt.submit failed: original symptom')
    })

    // The ORIGINAL error, not the secondary resume failure — the double-404
    // otherwise reported the confusing inner message.
    await expect(withSessionNotFoundResume(DEAD, STORED, call, d)).rejects.toThrow('original symptom')
  })

  it('rethrows when no stored session id is available to resume', async () => {
    const d = deps()

    const call = vi.fn(async () => {
      throw new Error('session not found')
    })

    await expect(withSessionNotFoundResume(DEAD, null, call, d)).rejects.toThrow('session not found')
    expect(d.requestGateway).not.toHaveBeenCalled()
  })

  it('does not recover a gateway timeout unless the caller opts in', async () => {
    const d = deps()

    const timeout = vi.fn(async () => {
      throw new Error('request timed out: session.compress')
    })

    // /compress is legitimately LLM-slow; retrying its timeout as a dead
    // session would double a minutes-long call.
    await expect(withSessionNotFoundResume(DEAD, STORED, timeout, d)).rejects.toThrow('request timed out')
    expect(d.requestGateway).not.toHaveBeenCalled()

    // prompt.submit opts in: a starved backend loop is indistinguishable from
    // a dead runtime client-side (#55578).
    const optIn = deps()
    let first = true

    expect(
      await withSessionNotFoundResume(
        DEAD,
        STORED,
        async (sid: string) => {
          if (first) {
            first = false
            throw new Error('request timed out: prompt.submit')
          }

          return `ok:${sid}`
        },
        optIn,
        { alsoTimeout: true }
      )
    ).toMatchObject({ recovered: true, sessionId: FRESH })
  })

  it('gives up after one retry so a genuinely broken session still surfaces', async () => {
    const d = deps()

    const call = vi.fn(async () => {
      throw new Error('session not found')
    })

    await expect(withSessionNotFoundResume(DEAD, STORED, call, d)).rejects.toThrow('session not found')
    expect(call).toHaveBeenCalledTimes(2)
    expect(d.requestGateway).toHaveBeenCalledTimes(1)
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

describe('readFileDataUrlForAttach', () => {
  it('prefers the attachment-specific desktop reader over the preview reader', async () => {
    const previewReader = vi.fn(async () => 'preview')
    const attachmentReader = vi.fn(async () => 'data:application/zip;base64,UEs=')
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileDataUrl: previewReader, readFileDataUrlForAttach: attachmentReader }
    })

    await expect(readFileDataUrlForAttach('/tmp/archive.zip')).resolves.toBe('data:application/zip;base64,UEs=')
    expect(attachmentReader).toHaveBeenCalledWith('/tmp/archive.zip')
    expect(previewReader).not.toHaveBeenCalled()
  })

  it('falls back to the preview reader on older shells', async () => {
    const previewReader = vi.fn(async () => 'data:text/plain;base64,YQ==')
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileDataUrl: previewReader }
    })

    await expect(readFileDataUrlForAttach('/tmp/note.txt')).resolves.toBe('data:text/plain;base64,YQ==')
    expect(previewReader).toHaveBeenCalledWith('/tmp/note.txt')
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

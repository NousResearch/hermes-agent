import { afterEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'
import type { SessionInfo, SessionMessage } from '@/types/hermes'

import { artifactImageSrc, collectArtifactsForSession } from './artifact-utils'

function makeSession(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    ended_at: null,
    id: 'session-1',
    input_tokens: 0,
    is_active: false,
    last_active: 1000,
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    source: null,
    started_at: 1000,
    title: 'Session',
    tool_call_count: 0,
    ...overrides
  }
}

describe('collectArtifactsForSession', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.clearAllMocks()
    $connection.set(null)
  })

  it('indexes plain https links from assistant text', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: 'Reference: https://example.com/docs/getting-started',
        role: 'assistant',
        timestamp: 2000
      }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]).toMatchObject({
      href: 'https://example.com/docs/getting-started',
      kind: 'link',
      value: 'https://example.com/docs/getting-started'
    })
  })

  it('indexes http links present in tool JSON payloads', () => {
    const messages: SessionMessage[] = [
      {
        content: JSON.stringify({ source_url: 'https://example.com/changelog/latest' }),
        role: 'tool',
        timestamp: 3000
      }
    ]

    const artifacts = collectArtifactsForSession(makeSession({ id: 'session-2' }), messages)

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]).toMatchObject({
      href: 'https://example.com/changelog/latest',
      kind: 'link',
      value: 'https://example.com/changelog/latest'
    })
  })

  it('normalizes session-level unix-second timestamps to epoch milliseconds', () => {
    const session = makeSession({ last_active: 1_700_000_000, started_at: 1_699_000_000 })
    const artifacts = collectArtifactsForSession(session, [
      {
        content: 'Reference: https://example.com/status',
        role: 'assistant'
      }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]?.timestamp).toBe(1_700_000_000_000)
  })

  it('keeps message timestamps that are already in milliseconds', () => {
    const artifacts = collectArtifactsForSession(makeSession({ last_active: 1_700_000_000 }), [
      {
        content: 'Reference: https://example.com/docs',
        role: 'assistant',
        timestamp: 1_700_000_123_456
      }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]?.timestamp).toBe(1_700_000_123_456)
  })

  it('converts fractional unix-second message timestamps to milliseconds', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: 'Reference: https://example.com/fractional',
        role: 'assistant',
        timestamp: 1_700_000_000.125
      }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]?.timestamp).toBe(1_700_000_000_125)
  })

  it('falls back to the session timestamp when a message timestamp is non-finite', () => {
    const session = makeSession({ last_active: 1_700_000_000 })

    for (const timestamp of [Number.NaN, Number.POSITIVE_INFINITY]) {
      const artifacts = collectArtifactsForSession(session, [
        {
          content: 'Reference: https://example.com/fallback',
          role: 'assistant',
          timestamp
        }
      ])

      expect(artifacts).toHaveLength(1)
      expect(artifacts[0]?.timestamp).toBe(1_700_000_000_000)
    }
  })

  it('treats numeric epochs before 1973 as seconds at the unit boundary', () => {
    const before1973 = 94_694_399
    const millisecondsAt1973 = 94_694_400_000
    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: 'Reference: https://example.com/boundary',
        role: 'assistant',
        timestamp: before1973
      }
    ])

    expect(artifacts[0]?.timestamp).toBe(before1973 * 1000)

    const boundaryArtifacts = collectArtifactsForSession(makeSession({ id: 'session-boundary' }), [
      {
        content: 'Reference: https://example.com/boundary-ms',
        role: 'assistant',
        timestamp: millisecondsAt1973
      }
    ])

    expect(boundaryArtifacts[0]?.timestamp).toBe(millisecondsAt1973)
  })

  it('resolves remote image artifact thumbnails through the desktop fs bridge', async () => {
    const api = vi.fn(async ({ path }: { path: string }) => {
      if (path.startsWith('/api/fs/read-data-url?')) {
        return { dataUrl: 'data:image/jpeg;base64,cmVtb3Rl' }
      }

      throw new Error(`unexpected path ${path}`)
    })

    vi.stubGlobal('window', { hermesDesktop: { api } })
    $connection.set({ baseUrl: 'https://gw', mode: 'remote', token: 'secret' } as never)

    const path = '/Users/me/.hermes/skills/work-esab/references/images/manual-step03.jpeg'
    const downloadHref = `https://gw/api/files/download?path=${encodeURIComponent(path)}&token=secret`

    await expect(artifactImageSrc(path, downloadHref)).resolves.toBe('data:image/jpeg;base64,cmVtb3Rl')

    expect(api).toHaveBeenCalledWith({
      path: '/api/fs/read-data-url?path=%2FUsers%2Fme%2F.hermes%2Fskills%2Fwork-esab%2Freferences%2Fimages%2Fmanual-step03.jpeg'
    })
  })
})

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

  it('normalizes Unix-second timestamps before rendering artifacts', () => {
    const seconds = 1_785_144_916.737_952

    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: 'Reference: https://example.com/from-seconds',
        role: 'assistant',
        timestamp: seconds
      }
    ])

    expect(artifacts[0]?.timestamp).toBe(seconds * 1000)
    expect(new Date(artifacts[0]?.timestamp ?? 0).getUTCFullYear()).toBe(2026)
  })

  it('preserves timestamps that are already milliseconds', () => {
    const milliseconds = 1_785_144_916_737

    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: 'Reference: https://example.com/from-milliseconds',
        role: 'assistant',
        timestamp: milliseconds
      }
    ])

    expect(artifacts[0]?.timestamp).toBe(milliseconds)
  })

  it('interprets ambiguous values below the millisecond threshold as Unix seconds', () => {
    const ambiguousTimestamp = 42_000_000_000

    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: 'Reference: https://example.com/from-ambiguous-timestamp',
        role: 'assistant',
        timestamp: ambiguousTimestamp
      }
    ])

    expect(artifacts[0]?.timestamp).toBe(ambiguousTimestamp * 1000)
  })

  it('does not multiply the Date.now fallback', () => {
    const now = 1_785_144_916_737

    vi.spyOn(Date, 'now').mockReturnValue(now)

    const artifacts = collectArtifactsForSession(makeSession({ last_active: 0, started_at: 0 }), [
      {
        content: 'Reference: https://example.com/from-now',
        role: 'assistant'
      }
    ])

    expect(artifacts[0]?.timestamp).toBe(now)
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

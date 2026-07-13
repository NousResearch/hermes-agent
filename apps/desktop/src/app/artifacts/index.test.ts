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

  it('uses the supplied localized title for untitled sessions', () => {
    const artifacts = collectArtifactsForSession(
      makeSession({ title: null, preview: null }),
      [
        {
          content: 'Reference: https://example.com/report',
          role: 'assistant',
          timestamp: 2000
        }
      ],
      'جلسة بلا عنوان'
    )

    expect(artifacts[0]?.sessionTitle).toBe('جلسة بلا عنوان')
  })

  it('ignores generated path templates that are not concrete files', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: 'Input frames: /Users/me/video/frame_%05d.png',
        role: 'tool',
        timestamp: 2000
      }
    ])

    expect(artifacts).toHaveLength(0)
  })

  it('does not treat an entire multiline tool output as one path', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: JSON.stringify({
          output:
            '/dev/disk3s1s1 926Gi 16Gi 248Gi 6% /\\nMach Virtual Memory Statistics\\ncheck_p01.png'
        }),
        role: 'tool',
        timestamp: 2000
      }
    ])

    expect(artifacts).toHaveLength(0)
  })

  it('ignores web-root paths in HTML attributes while keeping absolute web URLs', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content:
          '<meta property="og:image" content="https://agents.md/og.png"><link rel="icon" href="/public/og.png">',
        role: 'tool',
        timestamp: 2000
      }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]).toMatchObject({
      href: 'https://agents.md/og.png',
      kind: 'image',
      value: 'https://agents.md/og.png'
    })
  })

  it('ignores relative references inside fetched web documents', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: JSON.stringify({
          output:
            '====================\\nURL: https://raw.githubusercontent.com/agentsmd/agents.md/main/README.md\\n====================\\n![AGENTS.md logo](./public/og.png)\\n[AGENTS.md](https://agents.md)'
        }),
        role: 'tool',
        timestamp: 2000
      }
    ])

    expect(artifacts.map(artifact => artifact.value)).not.toContain('./public/og.png')
    expect(artifacts.some(artifact => artifact.value === 'https://agents.md')).toBe(true)
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

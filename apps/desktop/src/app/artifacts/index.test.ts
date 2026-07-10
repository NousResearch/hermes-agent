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

  it('keeps an unprovenanced remote filesystem image on the typed fallback', async () => {
    const api = vi.fn(async ({ path }: { path: string }) => {
      if (path.startsWith('/api/fs/read-data-url?')) {
        return { dataUrl: 'data:image/jpeg;base64,cmVtb3Rl' }
      }

      throw new Error(`unexpected path ${path}`)
    })

    vi.stubGlobal('window', { hermesDesktop: { api } })
    $connection.set({ baseUrl: 'https://gw', mode: 'remote', token: 'secret' } as never)

    const path = '/Users/me/.hermes/skills/work-esab/references/images/manual-step03.jpeg'
    await expect(artifactImageSrc(path, `file://${path}`)).resolves.toBe('')
    expect(api).not.toHaveBeenCalled()
  })

  it('keeps ambiguous historical /images routes on the typed fallback', async () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      { content: 'Generated /images/generated/chart.png', role: 'assistant', timestamp: 2000 }
    ])

    expect(artifacts[0]?.href).toBe('')
    await expect(artifactImageSrc('/images/generated/chart.png', '')).resolves.toBe('')
  })

  it.each([
    ['https://example.com/output/report.pdf?download=1', 'file'],
    ['./src/widget.tsx', 'file'],
    ['/tmp/photo.avif', 'image']
  ] as const)('keeps collector and preview extension classification aligned for %s', (value, kind) => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      { content: `Generated ${value}`, role: 'assistant', timestamp: 2000 }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]?.kind).toBe(kind)
  })

  it('keeps embedded image labels and identity bounded while preserving the render source', async () => {
    const value = 'data:image/png;base64,c21hbGw='

    const artifacts = collectArtifactsForSession(makeSession(), [
      { content: `![Embedded](${value})`, role: 'assistant', timestamp: 2000 }
    ])

    expect(artifacts[0]).toMatchObject({ href: '', kind: 'image', label: 'data:image' })
    expect(artifacts[0]?.id).not.toContain('c21hbGw=')
    await expect(artifactImageSrc(value, '')).resolves.toBe(value)
  })

  it('strips escaped line suffixes and rejects route-template false positives', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: 'Generated /tmp/widget.tsx\\n92|const widget = true and /products/${product.id}',
        role: 'assistant',
        timestamp: 2000
      }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]).toMatchObject({ kind: 'file', label: 'widget.tsx', value: '/tmp/widget.tsx' })
  })
})

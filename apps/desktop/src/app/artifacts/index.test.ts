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

  it('keeps extensionless Markdown images in the click-to-load image path', () => {
    const [artifact] = collectArtifactsForSession(makeSession(), [
      {
        content: '![Tracking image](https://tracker.example/render?id=42)',
        role: 'assistant',
        timestamp: 2000
      }
    ])

    expect(artifact).toMatchObject({
      href: 'https://tracker.example/render?id=42',
      imageCandidate: true,
      kind: 'image',
      previewable: false
    })
  })

  it('promotes a previously seen extensionless link when a later message marks it as an image', () => {
    const url = 'https://tracker.example/render?id=42'

    const artifacts = collectArtifactsForSession(makeSession(), [
      { content: `See ${url}`, role: 'assistant', timestamp: 1000 },
      { content: `![Tracking image](${url})`, role: 'assistant', timestamp: 2000 }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]).toMatchObject({ imageCandidate: true, kind: 'image', value: url })
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

  it('previews a remote image emitted by the authenticated image generation tool', async () => {
    const api = vi.fn(async ({ path }: { path: string }) => {
      if (path.startsWith('/api/fs/read-data-url?')) {
        return { dataUrl: 'data:image/png;base64,dHJ1c3RlZA==' }
      }

      throw new Error(`unexpected path ${path}`)
    })

    const path = '/home/remote/.hermes/images/generated.png'

    const artifacts = collectArtifactsForSession(makeSession({ profile: 'work' }), [
      {
        content: JSON.stringify({ image: path, success: true }),
        role: 'tool',
        timestamp: 2000,
        tool_name: 'image_generate'
      }
    ])

    vi.stubGlobal('window', { hermesDesktop: { api } })
    $connection.set({ mode: 'local' } as never)

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]?.previewable).toBe(true)
    await expect(
      artifactImageSrc(path, `file://${path}`, artifacts[0]?.previewable, artifacts[0]?.profile)
    ).resolves.toBe('data:image/png;base64,dHJ1c3RlZA==')
    expect(api).toHaveBeenCalledWith({
      path: '/api/fs/read-data-url?path=%2Fhome%2Fremote%2F.hermes%2Fimages%2Fgenerated.png',
      profile: 'work'
    })
  })

  it('does not trust a path-shaped image field from an unrelated tool result', () => {
    const [artifact] = collectArtifactsForSession(makeSession(), [
      {
        content: JSON.stringify({ image: '/tmp/private.png' }),
        role: 'tool',
        timestamp: 2000,
        tool_name: 'read_file'
      }
    ])

    expect(artifact?.previewable).toBe(false)
  })

  it.each([
    ['missing success', { image: '/tmp/private.png' }],
    ['explicit failure', { image: '/tmp/private.png', success: false }],
    ['error-bearing success', { error: 'denied', image: '/tmp/private.png', success: true }]
  ])('does not trust image generation output with %s', (_label, result) => {
    const [artifact] = collectArtifactsForSession(makeSession(), [
      {
        content: JSON.stringify(result),
        role: 'tool',
        timestamp: 2000,
        tool_name: 'image_generate'
      }
    ])

    expect(artifact?.previewable).toBe(false)
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
    ['/tmp/photo.avif', 'image'],
    ['/models/weights.gguf', 'file'],
    ['/models/adapter.safetensors', 'file'],
    ['/analysis/results.ipynb', 'file'],
    ['/archives/model.tar.zst', 'file']
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
      { content: `![Campaign board](${value})`, role: 'assistant', timestamp: 2000 }
    ])

    expect(artifacts[0]).toMatchObject({ href: '', kind: 'image', label: 'Campaign board' })
    expect(artifacts[0]?.id).not.toContain('c21hbGw=')
    await expect(artifactImageSrc(value, '')).resolves.toBe(value)
  })

  it('keeps same-id sessions from different profiles distinct', () => {
    const messages: SessionMessage[] = [
      { content: 'Generated /tmp/shared-report.pdf', role: 'assistant', timestamp: 2000 }
    ]

    const local = collectArtifactsForSession(makeSession({ id: 'shared', profile: 'default' }), messages)
    const remote = collectArtifactsForSession(makeSession({ id: 'shared', profile: 'work' }), messages)

    expect(local[0]?.profile).toBe('default')
    expect(remote[0]?.profile).toBe('work')
    expect(local[0]?.id).not.toBe(remote[0]?.id)
  })

  it('uses the newest occurrence timestamp for a repeated artifact', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      { content: 'Generated /tmp/report.pdf', role: 'assistant', timestamp: 1000 },
      { content: 'Updated /tmp/report.pdf', role: 'assistant', timestamp: 9000 }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]?.timestamp).toBe(9000)
  })

  it('upgrades a repeated embedded image from a generic label to a useful caption', () => {
    const value = 'data:image/png;base64,c21hbGw='

    const artifacts = collectArtifactsForSession(makeSession(), [
      { content: `![](${value})`, role: 'assistant', timestamp: 1000 },
      { content: `![Final campaign board](${value})`, role: 'assistant', timestamp: 2000 }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]?.label).toBe('Final campaign board')
    expect(artifacts[0]?.timestamp).toBe(2000)
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

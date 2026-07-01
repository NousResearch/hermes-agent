import { describe, expect, it } from 'vitest'

import type { SessionInfo, SessionMessage } from '@/types/hermes'

import { collectArtifactsForSession } from './index'

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

  it('normalizes epoch-second message timestamps for sorting and display', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: 'Reference: https://example.com/report.pdf',
        role: 'assistant',
        timestamp: 1_781_774_001.5943704
      }
    ])

    expect(artifacts[0]?.timestamp).toBe(1_781_774_001_594.3704)
  })

  it('leaves millisecond message timestamps unchanged', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: 'Reference: https://example.com/report.pdf',
        role: 'assistant',
        timestamp: 1_781_774_001_594
      }
    ])

    expect(artifacts[0]?.timestamp).toBe(1_781_774_001_594)
  })

  it('does not promote browser page image assets to artifacts', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: JSON.stringify({
          snapshot:
            'link "ad" https://example.com/workspace-advertising/banner.gif and text https://example.com/article'
        }),
        role: 'tool',
        timestamp: 4000,
        tool_name: 'browser_snapshot'
      }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]).toMatchObject({
      href: 'https://example.com/article',
      kind: 'link',
      value: 'https://example.com/article'
    })
  })

  it('keeps generated remote images as artifacts', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: JSON.stringify({ image: 'https://cdn.example.com/generated/cat.png', success: true }),
        role: 'tool',
        timestamp: 5000,
        tool_name: 'image_generate'
      }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]).toMatchObject({
      href: 'https://cdn.example.com/generated/cat.png',
      kind: 'image',
      value: 'https://cdn.example.com/generated/cat.png'
    })
  })
})

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
        timestamp: 1_780_732_996
      }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]).toMatchObject({
      href: 'https://example.com/docs/getting-started',
      kind: 'link',
      timestamp: 1_780_732_996_000,
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

  it('does not treat arbitrary source paths as artifacts', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: 'Touched files: /Users/dev/hermes/apps/desktop/src/app/artifacts/index.tsx',
        role: 'assistant',
        timestamp: 4000
      }
    ])

    expect(artifacts).toHaveLength(0)
  })

  it('keeps real file outputs with supported extensions', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: 'Wrote report to /tmp/hermes-artifacts/session-summary.pdf',
        role: 'assistant',
        timestamp: 5000
      }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]).toMatchObject({
      href: 'file:///tmp/hermes-artifacts/session-summary.pdf',
      kind: 'file',
      value: '/tmp/hermes-artifacts/session-summary.pdf'
    })
  })
})

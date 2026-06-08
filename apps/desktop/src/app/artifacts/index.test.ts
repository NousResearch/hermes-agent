import { describe, expect, it } from 'vitest'

import type { SessionInfo, SessionMessage } from '@/types/hermes'

import { collectArtifactsForSession, formatArtifactTime } from './index'

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
})

describe('formatArtifactTime', () => {
  it('treats small values as epoch seconds and converts to ms', () => {
    // 1780936770 = 2026-06-08 ~19:39 UTC (epoch seconds)
    const result = formatArtifactTime(1780936770)
    // Should NOT be Jan 1970 — should contain a June date
    expect(result).not.toMatch(/Jan/)
    expect(result).toMatch(/Jun/)
  })

  it('passes through values already in milliseconds', () => {
    // 1781036951103 = epoch milliseconds (already > 1e12)
    const result = formatArtifactTime(1781036951103)
    expect(result).not.toMatch(/Jan/)
    expect(result).toMatch(/Jun/)
  })

  it('handles fractional epoch seconds', () => {
    // 1780936770.62 — Python time.time() returns floats
    const result = formatArtifactTime(1780936770.62)
    expect(result).toMatch(/Jun/)
  })
})

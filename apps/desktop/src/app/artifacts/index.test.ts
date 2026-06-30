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
})

describe('collectArtifactsForSession origin tagging', () => {
  it('marks write_file tool results as generated', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: JSON.stringify({ path: '/out/report.md' }),
        name: 'write_file',
        role: 'tool',
        timestamp: 2000
      }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]).toMatchObject({ kind: 'file', origin: 'generated', value: '/out/report.md' })
  })

  it('marks patch tool results as generated (resolved_path)', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: JSON.stringify({ resolved_path: '/src/app.ts' }),
        name: 'patch',
        role: 'tool',
        timestamp: 2000
      }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]).toMatchObject({ origin: 'generated', value: '/src/app.ts' })
  })

  it('marks every image_generate output (host/sandbox/visible) as generated', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: JSON.stringify({
          agent_visible_image: 'https://cdn.example.com/y.png',
          host_image: '/i/out.png',
          image: '/sb/out.png'
        }),
        name: 'image_generate',
        role: 'tool',
        timestamp: 2000
      }
    ])

    expect(artifacts).toHaveLength(3)
    expect(artifacts.every(artifact => artifact.origin === 'generated')).toBe(true)
  })

  it('marks text_to_speech tool results as generated', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: JSON.stringify({ file_path: '/audio/v.mp3' }),
        name: 'text_to_speech',
        role: 'tool',
        timestamp: 2000
      }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]).toMatchObject({ origin: 'generated', value: '/audio/v.mp3' })
  })

  it('resolves the producing tool from tool_name when name is absent', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: JSON.stringify({ path: '/out/x.txt' }),
        role: 'tool',
        timestamp: 2000,
        tool_name: 'write_file'
      }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]).toMatchObject({ origin: 'generated', value: '/out/x.txt' })
  })

  it('marks read_file results and prose mentions as referenced', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: JSON.stringify({ content: 'hello', path: '/src/x.json' }),
        name: 'read_file',
        role: 'tool',
        timestamp: 2000
      }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]).toMatchObject({ origin: 'referenced', value: '/src/x.json' })
  })

  it('upgrades a prior referenced mention to generated when the tool later produces it', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      { content: 'Editing /out/report.md now', role: 'assistant', timestamp: 1000 },
      {
        content: JSON.stringify({ path: '/out/report.md' }),
        name: 'write_file',
        role: 'tool',
        timestamp: 2000
      }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]).toMatchObject({ origin: 'generated', value: '/out/report.md' })
  })

  it('never downgrades a generated artifact to referenced', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: JSON.stringify({ path: '/out/report.md' }),
        name: 'write_file',
        role: 'tool',
        timestamp: 1000
      },
      { content: 'See /out/report.md', role: 'assistant', timestamp: 2000 }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]).toMatchObject({ origin: 'generated', value: '/out/report.md' })
  })

  it('does not mark a failed generation ({ success: false }) as generated', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: JSON.stringify({ image: '/i/fail.png', success: false }),
        name: 'image_generate',
        role: 'tool',
        timestamp: 2000
      }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]).toMatchObject({ origin: 'referenced', value: '/i/fail.png' })
  })

  it('marks edit_file tool results as generated', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: JSON.stringify({ resolved_path: '/src/edited.ts' }),
        name: 'edit_file',
        role: 'tool',
        timestamp: 2000
      }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]).toMatchObject({ origin: 'generated', value: '/src/edited.ts' })
  })

  it('does not mark a failed generation ({ error }) as generated', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      {
        content: JSON.stringify({ error: 'permission denied: /out/blocked.md' }),
        name: 'write_file',
        role: 'tool',
        timestamp: 2000
      }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]).toMatchObject({ origin: 'referenced', value: '/out/blocked.md' })
  })

  it('adopts the producing tool-result timestamp when upgrading to generated', () => {
    const artifacts = collectArtifactsForSession(makeSession(), [
      { content: 'Editing /out/report.md now', role: 'assistant', timestamp: 1000 },
      {
        content: JSON.stringify({ path: '/out/report.md' }),
        name: 'write_file',
        role: 'tool',
        timestamp: 2000
      }
    ])

    expect(artifacts).toHaveLength(1)
    expect(artifacts[0]).toMatchObject({ origin: 'generated', timestamp: 2000, value: '/out/report.md' })
  })
})

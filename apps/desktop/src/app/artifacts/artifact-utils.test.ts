import { describe, expect, it } from 'vitest'

import type { SessionInfo, SessionMessage } from '@/types/hermes'

import { collectArtifactsForSession } from './artifact-utils'

const session = (profile: string): SessionInfo => ({
  archived: false,
  cwd: null,
  ended_at: null,
  id: 'shared-id',
  input_tokens: 0,
  is_active: false,
  last_active: 0,
  message_count: 1,
  model: null,
  output_tokens: 0,
  preview: null,
  profile,
  source: null,
  started_at: 1,
  title: `${profile} chat`,
  tool_call_count: 0
})

const messages: SessionMessage[] = [
  {
    content: '[report](https://example.com/report.pdf)',
    role: 'assistant',
    timestamp: 123
  }
]

describe('collectArtifactsForSession profile identity', () => {
  it('retains the owning profile and includes it in the artifact id', () => {
    const alpha = collectArtifactsForSession(session('alpha'), messages)[0]
    const beta = collectArtifactsForSession(session('beta'), messages)[0]

    expect(alpha).toMatchObject({ profile: 'alpha', sessionId: 'shared-id' })
    expect(beta).toMatchObject({ profile: 'beta', sessionId: 'shared-id' })
    expect(alpha.id).not.toBe(beta.id)
  })
})

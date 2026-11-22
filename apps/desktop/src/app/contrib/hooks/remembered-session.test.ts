import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { resolveRememberedSessionId } from './remembered-session'

const session = (overrides: Partial<SessionInfo>): SessionInfo =>
  ({
    ended_at: null,
    id: 'session',
    input_tokens: 0,
    is_active: false,
    last_active: 0,
    message_count: 0,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'tui',
    started_at: 0,
    title: null,
    tool_call_count: 0,
    ...overrides
  }) as SessionInfo

describe('resolveRememberedSessionId', () => {
  it('repairs a remembered delegate child to its parent', async () => {
    await expect(
      resolveRememberedSessionId('child', async () =>
        session({ id: 'child', parent_session_id: 'parent', source: 'subagent' })
      )
    ).resolves.toBe('parent')
  })

  it('clears an orphaned delegate child instead of reopening it', async () => {
    await expect(
      resolveRememberedSessionId('child', async () => session({ id: 'child', source: 'subagent' }))
    ).resolves.toBeNull()
  })

  it('keeps normal sessions', async () => {
    await expect(
      resolveRememberedSessionId('normal', async () => session({ id: 'normal', source: 'tui' }))
    ).resolves.toBe('normal')
  })

  it('keeps /branch children: parenthood is not the discriminator, source is', async () => {
    await expect(
      resolveRememberedSessionId('branch', async () =>
        session({ id: 'branch', parent_session_id: 'parent', source: 'tui' })
      )
    ).resolves.toBe('branch')
  })
})

import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/hermes'

import { resolveRememberedSessionId } from './remembered-session'

describe('resolveRememberedSessionId', () => {
  it('repairs a remembered delegate child to its parent without using the sidebar list', async () => {
    await expect(
      resolveRememberedSessionId('child', async () =>
        ({ id: 'child', source: 'subagent', parent_session_id: 'parent' }) as SessionInfo
      )
    ).resolves.toBe('parent')
  })

  it('clears an orphaned delegate child instead of reopening it', async () => {
    await expect(
      resolveRememberedSessionId('child', async () => ({ id: 'child', source: 'subagent' }) as SessionInfo)
    ).resolves.toBeNull()
  })

  it('keeps normal and branch sessions', async () => {
    await expect(
      resolveRememberedSessionId('branch', async () =>
        ({ id: 'branch', source: 'tui', parent_session_id: 'parent' }) as SessionInfo
      )
    ).resolves.toBe('branch')
  })
})

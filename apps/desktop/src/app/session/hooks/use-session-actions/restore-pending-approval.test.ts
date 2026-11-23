import { afterEach, describe, expect, it } from 'vitest'

import { clearAllPrompts, sessionApprovalRequests } from '@/store/prompts'
import type { SessionResumeResult } from '@/types/hermes'

import { restorePendingApproval } from './restore-pending-approval'

const resumed = (pending: Record<string, unknown>) => ({ pending_approval: pending }) as unknown as SessionResumeResult

afterEach(() => {
  clearAllPrompts()
})

describe('restorePendingApproval', () => {
  it('restores nothing for a folder request that names no folder', () => {
    expect(restorePendingApproval(resumed({ kind: 'access', request_id: 'sandbox-card-1' }), 's1')).toBe(false)
    expect(sessionApprovalRequests('s1').get()).toEqual([])
  })

  it('restores a folder card with its kind, folder, access level and risk line', () => {
    const pending = {
      access: 'read',
      choices: ['once', 'always', 'deny'],
      command: '',
      description: 'Allow read access to C:\\ref',
      kind: 'access',
      path: 'C:\\ref',
      request_id: 'sandbox-card-1',
      risk: 'The agent can read every file in this folder.'
    }

    // The return value says whether the session's TURN waits on the approval. A folder card marks
    // its session by being parked, so it does not raise the turn's flag (which nothing would lower
    // once the card is answered).
    expect(restorePendingApproval(resumed(pending), 's1')).toBe(false)
    expect(sessionApprovalRequests('s1').get()).toEqual([
      expect.objectContaining({
        kind: 'access',
        requestId: 'sandbox-card-1',
        risk: 'The agent can read every file in this folder.',
        target: { access: 'read', path: 'C:\\ref' }
      })
    ])
  })

  it('restores a command approval and says that the turn waits on it', () => {
    const pending = { command: 'rm -rf build', description: 'delete the build folder', request_id: 'apr-1' }

    expect(restorePendingApproval(resumed(pending), 's1')).toBe(true)
    expect(sessionApprovalRequests('s1').get()).toEqual([
      expect.objectContaining({ command: 'rm -rf build', requestId: 'apr-1' })
    ])
  })
})

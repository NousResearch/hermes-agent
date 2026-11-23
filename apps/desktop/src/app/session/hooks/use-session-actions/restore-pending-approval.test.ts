import { afterEach, describe, expect, it, vi } from 'vitest'

import { $gateway } from '@/store/gateway'
import { clearAllPrompts, sessionApprovalRequests } from '@/store/prompts'
import type { SessionResumeResult } from '@/types/hermes'

import { restorePendingApproval } from './restore-pending-approval'

const resumed = (pending: Record<string, unknown>) => ({ pending_approval: pending }) as unknown as SessionResumeResult

afterEach(() => {
  clearAllPrompts()
  $gateway.set(null)
})

describe('restorePendingApproval', () => {
  it('restores nothing for a folder request that names no folder', () => {
    expect(restorePendingApproval(resumed({ kind: 'access', request_id: 'sandbox-card-1' }), 's1')).toBe(false)
    expect(sessionApprovalRequests('s1').get()).toEqual([])
  })

  it('still restores the cards the sender lists when the snapshot entry names no folder', async () => {
    // A snapshot carries one pending approval. An entry that names no folder parks no card, and
    // the sender's `approval.pending` list is still asked for the cards that do name one.
    $gateway.set({
      request: async (method: string) =>
        method === 'approval.pending'
          ? { approvals: [{ access: 'read', kind: 'access', path: '/srv/data', request_id: 'sandbox-card-2' }] }
          : {}
    } as never)

    expect(restorePendingApproval(resumed({ kind: 'access', request_id: 'sandbox-card-1' }), 's1')).toBe(false)

    await vi.waitFor(() =>
      expect(sessionApprovalRequests('s1').get()).toMatchObject([
        { kind: 'access', requestId: 'sandbox-card-2', target: { access: 'read', path: '/srv/data' } }
      ])
    )
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

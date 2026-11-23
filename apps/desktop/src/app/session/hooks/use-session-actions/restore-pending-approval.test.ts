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
})

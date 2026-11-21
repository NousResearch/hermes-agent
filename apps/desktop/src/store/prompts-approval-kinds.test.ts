import { afterEach, describe, expect, it } from 'vitest'

import { canonicalPromptParams } from '@/api/canonical-protocol'

import {
  approvalKindFields,
  clearAllPrompts,
  networkTargetName,
  replayPendingApproval,
  sessionApprovalRequests,
  sessionCommandApprovalPending,
  setApprovalRequest
} from './prompts'
import { resetBackgroundPollingGuard } from './runtime-gone'

afterEach(() => {
  clearAllPrompts()
  resetBackgroundPollingGuard()
})

const kinds = (sessionId: string) =>
  sessionApprovalRequests(sessionId)
    .get()
    .map(request => request.kind ?? 'command')

function parkMixedQueue(sessionId = 's1') {
  setApprovalRequest({ command: 'rm -rf build', description: 'd', requestId: 'cmd', sessionId })
  setApprovalRequest({
    command: '',
    description: 'Allow read and write access to C:\\docs',
    kind: 'access',
    requestId: 'folder',
    sessionId,
    target: { access: 'read_write', path: 'C:\\docs' }
  })
  setApprovalRequest({
    command: '',
    description: 'Allow the sandbox to reach example.com',
    kind: 'network',
    requestId: 'site',
    sessionId,
    target: { host: 'example.com' }
  })
}

describe('approvalKindFields', () => {
  it('reads a payload without a kind, or with an unknown one, as a command', () => {
    expect(approvalKindFields({ command: 'ls' })).toEqual({})
    expect(approvalKindFields({ kind: 'command', path: '/x' })).toEqual({})
    expect(approvalKindFields({ kind: 'printer', path: '/x' })).toEqual({})
    expect(approvalKindFields(null)).toEqual({})
  })

  it('reads a folder target and its risk line', () => {
    expect(
      approvalKindFields({
        access: 'read_write',
        kind: 'access',
        path: ' C:\\Users\\me\\docs ',
        risk: ' Holds tax returns '
      })
    ).toEqual({
      kind: 'access',
      risk: 'Holds tax returns',
      target: { access: 'read_write', path: 'C:\\Users\\me\\docs' }
    })
  })

  it('drops an access level outside the contract and keeps the path', () => {
    expect(approvalKindFields({ access: 'rw', kind: 'access', path: '/srv/data' })).toEqual({
      kind: 'access',
      target: { path: '/srv/data' }
    })
  })

  it('keeps the kind when the target is missing, so the card still shows its own copy', () => {
    expect(approvalKindFields({ kind: 'network', risk: 'Unknown host' })).toEqual({
      kind: 'network',
      risk: 'Unknown host'
    })
  })

  it('reads a site target and its optional label', () => {
    expect(
      approvalKindFields({ host: 'api.keenable.ai', kind: 'network', label: ' Web search (api.keenable.ai) ' }).target
    ).toEqual({ host: 'api.keenable.ai', label: 'Web search (api.keenable.ai)' })
    expect(approvalKindFields({ host: 'example.com', kind: 'network', label: '  ' }).target).toEqual({
      host: 'example.com'
    })
  })
})

describe('schedule approvals', () => {
  it('a canonical prompt\'s approval_kind reaches the card as kind "schedule"', () => {
    const params = canonicalPromptParams('approval', 's1', {
      approval_kind: 'schedule',
      choices: ['once', 'session', 'always', 'deny'],
      command: '<cronjob_manage> (schedule a new job)',
      description: 'Schedule "Water reminder" (every 2h): remind me',
      prompt_id: 'p1'
    })

    expect(approvalKindFields(params)).toEqual({ kind: 'schedule' })
    expect(approvalKindFields(canonicalPromptParams('approval', 's1', { command: 'rm -rf x', prompt_id: 'p2' }))).toEqual({})
  })
})

describe('networkTargetName', () => {
  it('names a site by its label, else by its host', () => {
    expect(networkTargetName({ host: 'example.com' })).toBe('example.com')
    expect(networkTargetName({ host: 'api.keenable.ai', label: 'Web search (api.keenable.ai)' })).toBe(
      'Web search (api.keenable.ai)'
    )
  })
})

describe('turn-end prompt clear', () => {
  it('drops command approvals and keeps folder and site cards', () => {
    parkMixedQueue()

    clearAllPrompts('s1', 'turn-end')

    expect(kinds('s1')).toEqual(['access', 'network'])
    expect(sessionCommandApprovalPending('s1').get()).toBe(false)
  })

  it('removes a command-only queue entirely', () => {
    setApprovalRequest({ command: 'rm -rf build', description: 'd', requestId: 'cmd', sessionId: 's1' })

    clearAllPrompts('s1', 'turn-end')

    expect(sessionApprovalRequests('s1').get()).toEqual([])
  })

  it('still clears every kind on stop, close and reclaim (the default mode)', () => {
    parkMixedQueue()

    clearAllPrompts('s1')

    expect(sessionApprovalRequests('s1').get()).toEqual([])
  })

  it('leaves other sessions alone', () => {
    parkMixedQueue('s1')
    parkMixedQueue('s2')

    clearAllPrompts('s1', 'turn-end')

    expect(kinds('s2')).toEqual(['command', 'access', 'network'])
    expect(sessionCommandApprovalPending('s2').get()).toBe(true)
  })
})

describe('approval.pending replay', () => {
  it('carries kind, target and risk from the pending list', async () => {
    const gateway = {
      request: async (method: string) =>
        method === 'approval.pending'
          ? {
              approvals: [
                {
                  access: 'read',
                  choices: ['once', 'always', 'deny'],
                  command: '',
                  description: 'Allow read access to C:\\ref',
                  kind: 'access',
                  path: 'C:\\ref',
                  request_id: 'sandbox-card-1',
                  risk: 'The agent can read files in this folder.'
                },
                {
                  choices: ['once', 'always', 'deny'],
                  command: '',
                  description: 'Allow the sandbox to reach Web search (api.keenable.ai)',
                  host: 'api.keenable.ai',
                  kind: 'network',
                  label: 'Web search (api.keenable.ai)',
                  request_id: 'sandbox-card-2'
                }
              ]
            }
          : { acknowledged: true }
    }

    await replayPendingApproval(gateway, 's1')

    const [folder, site] = sessionApprovalRequests('s1').get()

    expect(folder).toMatchObject({
      kind: 'access',
      requestId: 'sandbox-card-1',
      risk: 'The agent can read files in this folder.',
      target: { access: 'read', path: 'C:\\ref' }
    })
    expect(site).toMatchObject({
      kind: 'network',
      requestId: 'sandbox-card-2',
      target: { host: 'api.keenable.ai', label: 'Web search (api.keenable.ai)' }
    })
  })
})

import { afterEach, describe, expect, it } from 'vitest'

import {
  approvalKindFields,
  approvalNamesNothing,
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

  // The kind stays without a target so that the request is refused as a folder or site request.
  // Read as a command it would become a command card with nothing to run.
  it('keeps the kind when the target is missing, so the request is refused and never read as a command', () => {
    expect(approvalNamesNothing({ kind: 'network' })).toBe(true)
    expect(approvalNamesNothing({ kind: 'network', target: { host: 'example.com' } })).toBe(false)
    expect(approvalNamesNothing({})).toBe(false)
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

  it('shows hidden and bidirectional characters in a folder, a site, its label and the risk line as escapes', () => {
    // U+202E reorders the text after it and U+200B draws nothing: the name on the card would
    // differ from the folder or site the answer is for.
    expect(approvalKindFields({ kind: 'access', path: 'C:\\docs\\a\u202Etxt.exe', risk: 'Safe\u200B.' })).toEqual({
      kind: 'access',
      risk: 'Safe\\u200b.',
      target: { path: 'C:\\docs\\a\\u202etxt.exe' }
    })
    expect(
      approvalKindFields({ host: 'evil.example\u202Emoc.buhtig', kind: 'network', label: 'Web\u2066 search\u001B[2K' })
        .target
    ).toEqual({ host: 'evil.example\\u202emoc.buhtig', label: 'Web\\u2066 search\\x1b[2K' })
  })

  it('leaves accented letters, CJK and an emoji in each of the four fields as they came', () => {
    const folder = {
      path: 'C:\\Users\\zoë\\Fotos días 📁\\日本語',
      risk: 'L’agent peut lire ce dossier — 読み取りのみ。'
    }

    const site = { host: 'bücher.example', label: 'Recherche web 🔎 (bücher.example)' }

    expect(approvalKindFields({ kind: 'access', ...folder })).toEqual({
      kind: 'access',
      risk: folder.risk,
      target: { path: folder.path }
    })
    expect(approvalKindFields({ kind: 'network', ...site }).target).toEqual(site)
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
  it('parks nothing for a pending entry that names no folder, and does not acknowledge it', async () => {
    const calls: string[] = []

    const gateway = {
      request: async (method: string) => {
        calls.push(method)

        return method === 'approval.pending'
          ? { approvals: [{ kind: 'access', request_id: 'sandbox-card-1' }] }
          : { acknowledged: true }
      }
    }

    await replayPendingApproval(gateway, 's1')

    expect(sessionApprovalRequests('s1').get()).toEqual([])
    expect(calls).toEqual(['approval.pending'])
  })

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

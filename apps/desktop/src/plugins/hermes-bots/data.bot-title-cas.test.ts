/**
 * #120853: a stale Desktop appearance save must not revert a newer server
 * bot title. Two-client lost update, derived from the issue's candidate
 * reproduction: client A holds revision N (title "Old name"), the server
 * moves to N+1 (title "New name"), then A saves only a section change from
 * its stale snapshot. The server title must survive.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $botMeta, botMetaWriteAt, saveBotMeta } from './data'
import { mergeServerMeta } from './profile-ops'
import type { RosterRow } from './types'

const { hostMock, storageMock } = vi.hoisted(() => ({
  hostMock: {
    agents: undefined as unknown,
    profileRoutes: undefined as unknown,
    request: vi.fn(),
    requestProfile: vi.fn(),
    state: { connectionId: { get: () => 'local' }, profile: { get: () => 'default' } }
  },
  storageMock: { get: vi.fn(), remove: vi.fn(), set: vi.fn() }
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  return {
    atom,
    forgetSessionUnread: vi.fn(),
    host: hostMock,
    queryClient: { invalidateQueries: vi.fn() },
    useQuery: vi.fn(),
    useValue: vi.fn()
  }
})

vi.mock('./shared', () => ({ getPluginCtx: () => ({ storage: storageMock }), ID: 'hermes-bots' }))
vi.mock('./avatar-image', () => ({ isBackfilledFacePng: () => false }))
vi.mock('./canonical-chat', () => ({ ensureBotMetadata: vi.fn() }))

/** Fake gateway implementing tui_gateway/methods_profiles._configure_ui_meta
 *  semantics: whole-namespace replace, per-key CAS only when expected
 *  revisions are supplied, revision bumped per write. */
const fakeServer = {
  meta: {} as Record<string, unknown>,
  rev: 0
}

function serverRow(): RosterRow {
  return {
    name: 'bot1',
    ui_meta: { 'hermes-bots': { ...fakeServer.meta } },
    ui_meta_revisions: { 'hermes-bots': fakeServer.rev }
  } as unknown as RosterRow
}

beforeEach(() => {
  vi.clearAllMocks()
  $botMeta.set({})
  botMetaWriteAt.clear()
  storageMock.get.mockResolvedValue(null)
  storageMock.set.mockResolvedValue(undefined)
  storageMock.remove.mockResolvedValue(undefined)

  fakeServer.meta = { color: '#38bdf8', sectionId: 's1', sectionName: 'Alpha', shape: 'cloud', title: 'Old name' }
  fakeServer.rev = 43

  hostMock.request.mockImplementation(async (method: string, params: Record<string, unknown>) => {
    if (method === 'profiles.list') {
      return { profiles: [serverRow()] }
    }

    if (method === 'profiles.configure') {
      const expected = (params.ui_meta_expected_revisions as Record<string, number> | undefined)?.['hermes-bots']

      if (expected !== undefined && expected !== fakeServer.rev) {
        return {
          applied: {
            ui_meta: false,
            ui_meta_conflicts: { 'hermes-bots': { actual: fakeServer.rev, expected } },
            ui_meta_revisions: { 'hermes-bots': fakeServer.rev }
          }
        }
      }

      fakeServer.meta = { ...(params.ui_meta as Record<string, Record<string, unknown>>)['hermes-bots'] }
      fakeServer.rev += 1

      return { applied: { ui_meta: true, ui_meta_revisions: { 'hermes-bots': fakeServer.rev } } }
    }

    return {}
  })
})

describe('stale appearance save vs newer server title (#120853)', () => {
  it('preserves the newer server title when a stale client saves only a section', async () => {
    const seen: Array<Record<string, unknown>> = []
    const impl = hostMock.request.getMockImplementation()

    hostMock.request.mockImplementation(async (method: string, params: Record<string, unknown>) => {
      if (method === 'profiles.configure') {
        seen.push(structuredClone(params))
      }

      return impl!(method, params)
    })

    // Client A reads revision 43 ("Old name").
    mergeServerMeta([serverRow()])
    expect($botMeta.get().bot1.title).toBe('Old name')

    // Client B corrects the title server-side: revision 44 ("New name").
    fakeServer.meta = { ...fakeServer.meta, title: 'New name' }
    fakeServer.rev = 44

    // Client A, still on its stale snapshot, moves only the section.
    const result = await saveBotMeta('bot1', { sectionId: 's2', sectionName: 'Beta' })

    expect(result.serverPersisted).toBe(true)
    expect(fakeServer.meta.title).toBe('New name')
    expect(fakeServer.meta.sectionId).toBe('s2')
    expect($botMeta.get().bot1.title).toBe('New name')
    // The stale first attempt guarded on 43 (rejected), the retry on 44.
    expect(seen.map(body => (body.ui_meta_expected_revisions as Record<string, number> | undefined)?.['hermes-bots'])).toEqual([
      43,
      44
    ])
  })
})

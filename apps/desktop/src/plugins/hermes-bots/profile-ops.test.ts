/**
 * Duplicating a bot: clone the profile (config/skills/SOUL/memory via
 * `clone_from`) and copy the LOOK, but never the things that belong to the
 * original — its canonical-chat pointer and its creation stamp.
 *
 * The name search is the other half. Candidates are `<base>-2`, `-3`, … and
 * the BASE is truncated to fit, never the suffix (#19): slicing the joined
 * string chops the "-2" off a max-length name, so the candidate collides with
 * the base forever and the search runs out at -99.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $botMeta } from './data'
import { duplicateBot } from './profile-ops'
import type { RosterRow } from './types'

const { ensureBotMetadataMock, hostMock, storageMock } = vi.hoisted(() => ({
  ensureBotMetadataMock: vi.fn(),
  hostMock: {
    request: vi.fn(),
    requestProfile: vi.fn(),
    state: { connectionId: { get: () => 'local' }, focusedSessionOwner: null, profile: { get: () => 'default' } }
  },
  storageMock: { get: vi.fn(), set: vi.fn() }
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
vi.mock('./canonical-chat', () => ({ ensureBotMetadata: ensureBotMetadataMock }))

const calls: Array<{ method: string; params: Record<string, unknown> }> = []

beforeEach(() => {
  vi.clearAllMocks()
  calls.length = 0
  $botMeta.set({})
  storageMock.set.mockResolvedValue(undefined)
  hostMock.request.mockImplementation(async (method: string, params: Record<string, unknown>) => {
    calls.push({ method, params: structuredClone(params ?? {}) })

    return { ok: true }
  })
})

describe('duplicating a bot', () => {
  it('copies the look but neither the chat pointer nor the creation stamp', async () => {
    $botMeta.set({
      researcher: {
        chat: 'sess-source-forever',
        color: '#f97316',
        created: 1_700_000_000_000,
        image: 'data:image/png;base64,xx',
        shape: 'circle',
        title: 'Researcher'
      }
    })

    const name = await duplicateBot({ description: 'finds things', name: 'researcher' } as RosterRow, [
      { name: 'researcher' } as RosterRow
    ])

    expect(name).toBe('researcher-2')

    const clone = $botMeta.get()['researcher-2']

    expect(clone).toMatchObject({
      color: '#f97316',
      image: 'data:image/png;base64,xx',
      shape: 'circle',
      title: 'Researcher (copy)'
    })
    expect(clone.chat).toBeUndefined()
    expect(clone.created).toBeUndefined()

    expect(calls.find(call => call.method === 'profiles.create')?.params).toMatchObject({
      clone_from: 'researcher',
      name: 'researcher-2'
    })

    const configure = calls.filter(call => call.method === 'profiles.configure').at(-1)
    const uiMeta = (configure?.params.ui_meta as Record<string, Record<string, unknown>>)['hermes-bots']

    expect(uiMeta.title).toBe('Researcher (copy)')
    expect(uiMeta.chat).toBeUndefined()
    expect(uiMeta.created).toBeUndefined()
  })

  it('duplicates the look of a bot that never had a pointer', async () => {
    $botMeta.set({ painter: { color: '#38bdf8', shape: 'cloud', title: 'Painter' } })

    const name = await duplicateBot({ name: 'painter' } as RosterRow, [{ name: 'painter' } as RosterRow])

    expect($botMeta.get()[name]).toMatchObject({ shape: 'cloud', title: 'Painter (copy)' })
    expect($botMeta.get()[name].chat).toBeUndefined()
  })

  it('walks past taken suffixes to the first free slot', async () => {
    const roster = ['ops', 'ops-2', 'ops-3'].map(name => ({ name }) as RosterRow)

    expect(await duplicateBot({ name: 'ops' } as RosterRow, roster)).toBe('ops-4')
  })

  it('truncates the BASE so a max-length name still gets a distinct suffix (#19)', async () => {
    const base = 'b'.repeat(64)

    const name = await duplicateBot({ name: base } as RosterRow, [{ name: base } as RosterRow])

    expect(name).toHaveLength(64)
    expect(name.endsWith('-2')).toBe(true)
    expect(name).not.toBe(base)
  })

  it('ensures the source bot has its metadata before cloning', async () => {
    // clone_from copies the profile dir; the source's Bot Chat has to exist
    // first or the clone inherits a half-built profile.
    await duplicateBot({ name: 'ops' } as RosterRow, [])

    expect(ensureBotMetadataMock).toHaveBeenCalledTimes(1)
  })

  it('only collides against rows on the SAME connection', async () => {
    // A same-named bot on another gateway is a different agent entirely.
    const bot = {
      connectionId: 'vera',
      name: 'ops',
      route: { connectionId: 'vera', mode: 'remote', profile: 'ops', targetProfile: 'ops' },
      sourceScoped: true
    } as RosterRow

    const elsewhere = {
      connectionId: 'other',
      name: 'ops-2',
      route: { connectionId: 'other', mode: 'remote', profile: 'ops-2', targetProfile: 'ops-2' },
      sourceScoped: true
    } as RosterRow

    hostMock.requestProfile.mockResolvedValue({ ok: true })

    expect(await duplicateBot(bot, [bot, elsewhere])).toBe('ops-2')
  })
})

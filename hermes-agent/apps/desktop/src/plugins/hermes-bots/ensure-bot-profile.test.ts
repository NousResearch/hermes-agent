/**
 * Edit profile and Duplicate read a bot's description from its own backend (#139548).
 *
 * The roster carries a description only for rows a poll can read: an SSH connection is never
 * dialed by the roster poll, so its rows arrive with none. `ensureBotProfile` asks the bot's own
 * backend before those actions seed the editor or the copy.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { RosterRow } from './types'

const { requestForBotMock } = vi.hoisted(() => ({ requestForBotMock: vi.fn() }))

vi.mock('@hermes/plugin-sdk', () => ({ host: {} }))

vi.mock('./routing', () => ({
  backendTargetProfile: (route: { targetProfile?: string } | null, name: string) => route?.targetProfile ?? name,
  botConnectionRoute: (bot: RosterRow) => bot.route ?? null,
  botRosterMeta: () => ({}),
  botWorkspaceOwnerKey: () => 'bot:x',
  requestForBot: requestForBotMock
}))

vi.mock('./data', () => ({
  $botMeta: { get: () => ({}), set: vi.fn() },
  botMetaKey: (bot: { name?: string }) => bot?.name ?? '',
  botOwner: vi.fn(),
  persistBotMetaSnapshot: vi.fn()
}))

vi.mock('./shared', () => ({ getPluginCtx: () => null }))

const { ensureBotProfile } = await import('./canonical-chat')

const sshRow: RosterRow = {
  name: 'scout',
  connectionId: 'spark',
  connectionKind: 'ssh',
  sourceScoped: true,
  route: { connectionId: 'spark', mode: 'remote', profile: 'scout', targetProfile: 'scout' }
}

beforeEach(() => requestForBotMock.mockReset())

describe('ensureBotProfile', () => {
  it("fills a source row's description from its own backend, including a cleared one", async () => {
    requestForBotMock.mockResolvedValue({
      profiles: [
        { name: 'default', description: 'Main' },
        { name: 'scout', description: 'Research Scout' }
      ]
    })

    expect(await ensureBotProfile(sshRow)).toEqual({ ...sshRow, description: 'Research Scout' })
    expect(requestForBotMock).toHaveBeenCalledWith(sshRow, 'profiles.list', {})

    requestForBotMock.mockResolvedValue({ profiles: [{ name: 'scout', description: '' }] })

    expect((await ensureBotProfile({ ...sshRow, description: 'stale' })).description).toBe('')
  })

  it('keeps the roster value when the backend omits the field, and never reads for a plain local row', async () => {
    requestForBotMock.mockResolvedValue({ profiles: [{ name: 'scout' }] })

    expect(await ensureBotProfile({ ...sshRow, description: 'Roster text' })).toMatchObject({
      description: 'Roster text'
    })

    const local = { name: 'leo', description: 'Creative Director' } as RosterRow

    expect(await ensureBotProfile(local)).toBe(local)
    expect(requestForBotMock).toHaveBeenCalledTimes(1)
  })
})

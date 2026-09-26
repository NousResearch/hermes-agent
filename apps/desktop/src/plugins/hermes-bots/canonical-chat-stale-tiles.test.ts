/**
 * RED for hermes-agent#120810: a Bot Chat compression leaves two tabs
 * captioned with the bot name.
 *
 * openStoredBotChat (via openBotCanonicalChat) must run the same
 * owner-scoped stale-"Bot Chat" probe the roster click runs
 * (roster-actions focusExistingBotTab) against the registry id + resolved
 * tip BEFORE host.openSession, so the activity-refresh and reclaim opens
 * (which call openBotCanonicalChat directly) discard a tile keyed to the
 * old segment instead of leaving it beside the tip.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'

const { focusMock, hostMock, requestForBotMock } = vi.hoisted(() => ({
  focusMock: vi.fn(),
  hostMock: { focusOpenWorkspaceSession: vi.fn(), openSession: vi.fn() } as any,
  requestForBotMock: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', () => ({
  BOT_CHAT_SESSION_HYDRATION_TIMEOUT_MS: 15_000,
  host: hostMock
}))

vi.mock('./routing', () => ({
  backendTargetProfile: (route: { targetProfile?: string } | null, name: string) => route?.targetProfile ?? name,
  botConnectionRoute: () => null,
  botRosterMeta: () => ({}),
  botWorkspaceOwnerKey: (bot: { connectionId?: string; name?: string } | null) =>
    `bot:${bot?.connectionId ? `${bot.connectionId}::` : ''}${bot?.name || 'default'}`,
  requestForBot: requestForBotMock
}))

vi.mock('./data', () => ({
  $botMeta: { get: () => ({}), set: vi.fn() },
  botMetaKey: (bot: { name?: string }) => bot?.name ?? '',
  botOwner: (owner: string) => ({ bot: { name: owner }, key: owner, name: owner, route: null }),
  persistBotMetaSnapshot: vi.fn()
}))

vi.mock('./shared', () => ({ getPluginCtx: () => null }))

async function loadModule() {
  vi.resetModules()
  return import('./canonical-chat')
}

beforeEach(() => {
  vi.clearAllMocks()
  hostMock.openSession.mockResolvedValue(undefined)
  hostMock.focusOpenWorkspaceSession = focusMock
  focusMock.mockReturnValue(null)
  requestForBotMock.mockImplementation(async (_bot: unknown, method: string) => {
    if (method === 'session.list') {
      return { sessions: [{ id: 'root-1', resolved_id: 'tip-9', root_title: 'Bot Chat', title: 'Bot Chat' }] }
    }
    return {}
  })
})

describe('120810: compressed Bot Chat discards the stale tile', () => {
  it('probes owner-scoped stale Bot Chat tiles before opening the tip', async () => {
    const { openBotCanonicalChat } = await loadModule()
    await openBotCanonicalChat('ops')

    expect(focusMock).toHaveBeenCalledTimes(1)
    const [ownerKey, probe, onlyIds] = focusMock.mock.calls[0]
    expect(ownerKey).toBe('bot:ops')
    expect(onlyIds).toEqual(expect.arrayContaining(['root-1', 'tip-9']))
    // Stale: Bot Chat-titled tile at an id that is neither registry nor tip.
    expect(probe({ storedSessionId: 'old-segment', workspaceTabTitle: 'Bot Chat' })).toBe(true)
    // Live tip and registry row are never stale.
    expect(probe({ storedSessionId: 'tip-9', workspaceTabTitle: 'Bot Chat' })).toBe(false)
    expect(probe({ storedSessionId: 'root-1', workspaceTabTitle: 'Bot Chat' })).toBe(false)
    // User + threads are never stale.
    expect(probe({ storedSessionId: 'side-1', workspaceTabTitle: 'Bot Chat' })).toBe(true)
    expect(probe({ storedSessionId: 'side-1', workspaceTabTitle: 'my thread' })).toBe(false)
    // Tip opens after the probe.
    expect(hostMock.openSession.mock.calls[0][0]).toBe('tip-9')
  })
})

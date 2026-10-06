import { beforeEach, describe, expect, it, vi } from 'vitest'

const {
  createCanonicalChat,
  requestForBot,
  saveSelectedRosterBot,
  setBotsWorkspaceOwner,
  requestGatewayForAgent,
  requestGatewayForProfile
} = vi.hoisted(() => ({
  createCanonicalChat: vi.fn(async () => 'canonical-session'),
  requestForBot: vi.fn(async (_bot: unknown, method: string) =>
    method === 'session.resume' ? { session_id: 'runtime-session' } : { ok: true }
  ),
  saveSelectedRosterBot: vi.fn(),
  setBotsWorkspaceOwner: vi.fn(),
  requestGatewayForAgent: vi.fn(async () => ({ ok: true })),
  requestGatewayForProfile: vi.fn(async () => ({ ok: true }))
}))

vi.mock('@/plugins/hermes-bots/bot-state', () => ({ saveSelectedRosterBot }))
vi.mock('@/plugins/hermes-bots/canonical-chat', () => ({ createCanonicalChat }))
vi.mock('@/plugins/hermes-bots/routing', () => ({
  botWorkspaceOwnerKey: (bot: { name: string }) => `bot:${bot.name}`,
  requestForBot,
  setBotsWorkspaceOwner
}))
vi.mock('@/store/gateway', () => ({
  activeGatewayConnectionId: () => 'local',
  requestGatewayForAgent,
  requestGatewayForProfile
}))

const { $foregroundBotProfile, kickoffInstalledBot, openInstalledBot, requestBotMarketplace } = await import('./marketplace-actions')

beforeEach(() => {
  $foregroundBotProfile.set('')
  vi.clearAllMocks()
})

describe('Bot Marketplace routing and kickoff', () => {
  it('keeps passive inventory/status reads in the background while explicit installs are foreground', async () => {
    await requestBotMarketplace('bots.status', { profile: 'source' }, { connectionId: 'homelab', profile: 'source' })
    await requestBotMarketplace('bots.install', { profile: 'source' }, { connectionId: 'homelab', profile: 'source' })

    expect(requestGatewayForAgent).toHaveBeenNthCalledWith(
      1,
      'homelab',
      'source',
      'bots.status',
      { profile: 'source' },
      undefined,
      undefined,
      { spawnPriority: 'background' }
    )
    expect(requestGatewayForAgent).toHaveBeenNthCalledWith(
      2,
      'homelab',
      'source',
      'bots.install',
      { profile: 'source' },
      undefined,
      undefined,
      { spawnPriority: 'foreground' }
    )
    expect(requestGatewayForProfile).not.toHaveBeenCalled()
  })

  it('opens an installed bot without submitting the starter again', async () => {
    await openInstalledBot('research-assistant', { connectionId: 'homelab', profile: 'research-assistant' })

    const owner = expect.objectContaining({
      name: 'research-assistant',
      connectionId: 'homelab',
      route: expect.objectContaining({ profile: 'research-assistant', targetProfile: 'research-assistant' })
    })

    expect(saveSelectedRosterBot).toHaveBeenCalledWith(owner)
    expect(createCanonicalChat).toHaveBeenCalledWith(owner)
  })

  it('coalesces concurrent starter submits but permits a later backend-authorized retry', async () => {
    let resolveCanonical!: (value: string) => void
    createCanonicalChat.mockImplementationOnce(() => new Promise(resolve => { resolveCanonical = resolve }))
    const scope = { connectionId: 'local', profile: 'default' }
    const first = kickoffInstalledBot('research-assistant', scope, 'Ask what we are researching.')
    await vi.waitFor(() => expect(createCanonicalChat).toHaveBeenCalledTimes(1))
    await Promise.resolve()
    const concurrent = kickoffInstalledBot('research-assistant', scope, 'Ask what we are researching.')
    resolveCanonical('canonical-session')
    await Promise.all([first, concurrent])
    await kickoffInstalledBot('research-assistant', scope, 'Ask what we are researching.')

    const owner = expect.objectContaining({
      name: 'research-assistant',
      connectionId: 'local',
      sourceScoped: true,
      route: expect.objectContaining({ profile: 'research-assistant', targetProfile: 'research-assistant' })
    })

    expect(saveSelectedRosterBot).toHaveBeenCalledWith(owner)
    expect($foregroundBotProfile.get()).toBe('research-assistant')
    expect(setBotsWorkspaceOwner).toHaveBeenCalledWith('bot:research-assistant', owner)
    expect(createCanonicalChat).toHaveBeenCalledTimes(2)
    expect(createCanonicalChat).toHaveBeenCalledWith(owner)
    expect(requestForBot).toHaveBeenCalledWith(owner, 'session.resume', {
      session_id: 'canonical-session',
      profile: 'research-assistant',
      omit_messages: true
    })
    expect(requestForBot).toHaveBeenCalledWith(owner, 'prompt.submit', {
      session_id: 'runtime-session',
      text: 'Ask what we are researching.'
    })
    expect(requestForBot.mock.calls.filter(([, method]) => method === 'prompt.submit')).toHaveLength(2)
  })
})

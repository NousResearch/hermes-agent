import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopAgentRoster } from '@/global'

const gatewayMocks = vi.hoisted(() => ({ requestGatewayForAgent: vi.fn() }))

vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestGatewayForAgent: gatewayMocks.requestGatewayForAgent
}))

import {
  deliverDirectBot,
  directBotCompletions,
  parseDirectBotInvocation,
  resolveDirectBotTarget
} from './direct-bot-routing'

const roster: DesktopAgentRoster = {
  agents: [
    {
      connectionId: 'laptop', connectionKind: 'local', connectionLabel: 'Laptop',
      handle: 'research-laptop', profile: 'research'
    },
    {
      connectionId: 'server', connectionKind: 'ssh', connectionLabel: 'Server',
      handle: 'research-server', profile: 'research'
    },
    {
      connectionId: 'server', connectionKind: 'ssh', connectionLabel: 'Server',
      handle: 'production', profile: 'production'
    }
  ],
  sources: [
    { connectionId: 'laptop', kind: 'local', label: 'Laptop', reachable: true },
    { connectionId: 'server', kind: 'ssh', label: 'Server', reachable: true }
  ]
}

describe('direct Bot routing', () => {
  beforeEach(() => {
    gatewayMocks.requestGatewayForAgent.mockReset()
  })

  it('preserves the prompt after the source-qualified handle', () => {
    expect(parseDirectBotInvocation('research-server Compare A and B\nwith citations')).toEqual({
      target: 'research-server', prompt: 'Compare A and B\nwith citations'
    })
  })

  it('rejects duplicate profile names and resolves their unique handles', () => {
    expect(() => resolveDirectBotTarget(roster, 'research')).toThrow(/ambiguous.*@research-laptop.*@research-server/i)
    expect(resolveDirectBotTarget(roster, '@research-server')).toMatchObject({ connectionId: 'server', profile: 'research' })
  })

  it('offers source-qualified handles as command completions', () => {
    expect(directBotCompletions(roster, 'research').map(item => item.text)).toEqual([
      '/to research-laptop ', '/to research-server '
    ])
  })

  it('cold-routes a structured delivery through the target source and correlates its reply', async () => {
    gatewayMocks.requestGatewayForAgent.mockResolvedValue({ reply: '  finished by research  ' })

    await expect(
      deliverDirectBot(
        roster,
        { target: 'research-server', prompt: 'Compare A and B' },
        { connectionId: 'laptop', profile: 'coordinator' }
      )
    ).resolves.toMatchObject({ bot: { handle: 'research-server' }, reply: 'finished by research' })

    expect(gatewayMocks.requestGatewayForAgent).toHaveBeenCalledWith(
      'server',
      'research',
      'bot_relay.deliver',
      {
        profile: 'research',
        message: 'Compare A and B',
        from_profile: 'coordinator',
        from_handle: 'coordinator',
        from_connection: 'laptop'
      },
      1_500_000
    )
  })

  it('reports unreachable routes before dispatch and preserves authorization failures', async () => {
    const unreachable: DesktopAgentRoster = {
      ...roster,
      sources: roster.sources.map(source =>
        source.connectionId === 'server' ? { ...source, reachable: false, error: 'source offline' } : source
      )
    }

    await expect(
      deliverDirectBot(
        unreachable,
        { target: 'production', prompt: 'Deploy' },
        { connectionId: 'laptop', profile: 'coordinator' }
      )
    ).rejects.toThrow(/unreachable.*source offline/i)
    expect(gatewayMocks.requestGatewayForAgent).not.toHaveBeenCalled()

    gatewayMocks.requestGatewayForAgent.mockRejectedValue(new Error('approval denied by target profile'))
    await expect(
      deliverDirectBot(
        roster,
        { target: 'production', prompt: 'Deploy' },
        { connectionId: 'laptop', profile: 'coordinator' }
      )
    ).rejects.toThrow('approval denied by target profile')
  })
})
import { beforeEach, describe, expect, it, vi } from 'vitest'

const requestForOwnedSession = vi.fn()
const requestGatewayForAgent = vi.fn()
const ambientRequest = vi.fn()

vi.mock('@/store/session-states', () => ({
  requestForOwnedSession: (...args: unknown[]) => requestForOwnedSession(...args)
}))

vi.mock('@/store/gateway', () => ({
  requestGatewayForAgent: (...args: unknown[]) => requestGatewayForAgent(...args)
}))

describe('respondOnOwnedGateway', () => {
  beforeEach(() => {
    requestForOwnedSession.mockReset()
    requestGatewayForAgent.mockReset()
    ambientRequest.mockReset()
  })

  it('answers on the owning session gateway before the ambient profile', async () => {
    const { respondOnOwnedGateway } = await import('./desktop-bridge-respond')
    requestForOwnedSession.mockResolvedValue(undefined)

    await respondOnOwnedGateway(
      { session_id: 'rosie-bot-chat', profile: 'rosie' } as never,
      'preview.read.respond',
      { request_id: 'req-1', text: '{}' },
      ambientRequest
    )

    expect(requestForOwnedSession).toHaveBeenCalledWith(
      'rosie-bot-chat',
      ambientRequest,
      'preview.read.respond',
      { request_id: 'req-1', text: '{}' }
    )
    expect(requestGatewayForAgent).not.toHaveBeenCalled()
  })

  it('falls through to the event profile when the owner is unknown', async () => {
    const { respondOnOwnedGateway } = await import('./desktop-bridge-respond')
    requestForOwnedSession.mockRejectedValue(new Error('unknown owner'))
    requestGatewayForAgent.mockResolvedValue(undefined)

    await respondOnOwnedGateway(
      { session_id: 'fresh-runtime', profile: 'rosie', connectionId: 'local' } as never,
      'preview.act.respond',
      { request_id: 'req-2', text: '{}' },
      ambientRequest
    )

    expect(requestGatewayForAgent).toHaveBeenCalledWith(
      'local',
      'rosie',
      'preview.act.respond',
      { request_id: 'req-2', text: '{}' }
    )
  })
})

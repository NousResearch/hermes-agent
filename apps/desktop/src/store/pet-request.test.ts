import { afterEach, describe, expect, it, vi } from 'vitest'

import { requestGatewayForAgent } from './gateway'
import { requestPet } from './pet'
import type { GatewayRequest } from './pet-gallery'
import { $activeGatewayProfile } from './profile'
import { $activeSessionId, $selectedStoredSessionId, $sessions } from './session'

vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn(async () => ({ enabled: false }))
}))

const previous = {
  profile: $activeGatewayProfile.get(),
  active: $activeSessionId.get(),
  selected: $selectedStoredSessionId.get(),
  sessions: $sessions.get()
}

afterEach(() => {
  $activeSessionId.set(previous.active)
  $selectedStoredSessionId.set(previous.selected)
  $sessions.set(previous.sessions)
  $activeGatewayProfile.set(previous.profile)
  vi.clearAllMocks()
})

describe('pet RPC owner isolation', () => {
  it('routes to the visible bot exact connection instead of ambient', async () => {
    $activeGatewayProfile.set('solara')
    $sessions.set([
      {
        id: 'gm-chat',
        profile: 'grand-master',
        connection_id: 'remote-family',
        ended_at: null,
        input_tokens: 0,
        is_active: false,
        last_active: 0,
        message_count: 0,
        model: null,
        output_tokens: 0,
        preview: null,
        source: 'desktop',
        started_at: 0,
        title: 'Bot Chat',
        tool_call_count: 0
      }
    ])
    $selectedStoredSessionId.set('gm-chat')
    const ambient = vi.fn() as unknown as GatewayRequest
    await requestPet(ambient, 'pet.info')
    expect(requestGatewayForAgent).toHaveBeenCalledWith('remote-family', 'grand-master', 'pet.info', {
      profile: 'grand-master'
    })
    expect(ambient).not.toHaveBeenCalled()
  })

  it('rejects an old pet response after the visible owner changes', async () => {
    $activeGatewayProfile.set('solara')
    $sessions.set([])
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    let resolve!: (value: { enabled: boolean }) => void

    const response = new Promise<{ enabled: boolean }>(done => {
      resolve = done
    })

    const ambient = vi.fn(() => response) as unknown as GatewayRequest
    const result = requestPet(ambient, 'pet.info')
    const rejected = expect(result).rejects.toThrow('Pet owner changed')
    $activeGatewayProfile.set('aura')
    resolve({ enabled: true })
    await rejected
  })
})

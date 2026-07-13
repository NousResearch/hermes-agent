// @vitest-environment jsdom

import type { ConnectionState, GatewayEvent } from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { createdGateways, MockGateway } = vi.hoisted(() => {
  const createdGateways: MockGateway[] = []

  class MockGateway {
    connectionState: ConnectionState = 'closed'
    stateCallbacks = new Set<(state: ConnectionState) => void>()

    constructor() {
      createdGateways.push(this)
    }

    close(): void {
      this.connectionState = 'closed'
      this.emitState('closed')
    }

    async connect(): Promise<void> {
      this.connectionState = 'open'
      this.emitState('open')
    }

    onEvent(_callback: (event: GatewayEvent) => void): () => void {
      return () => {}
    }

    onState(callback: (state: ConnectionState) => void): () => void {
      this.stateCallbacks.add(callback)

      return () => void this.stateCallbacks.delete(callback)
    }

    emitState(state: ConnectionState): void {
      this.connectionState = state

      for (const callback of this.stateCallbacks) {
        callback(state)
      }
    }
  }

  return { createdGateways, MockGateway }
})

vi.mock('@/hermes', () => ({ HermesGateway: MockGateway }))

import {
  $gateway,
  $gatewayStatesByProfile,
  activeGateway,
  closeSecondaryGateways,
  ensureGatewayForProfile,
  gatewayForProfile,
  reportPrimaryGatewayState,
  setPrimaryGateway
} from './gateway'
import { $gatewayState } from './session'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

describe('gatewayForProfile', () => {
  beforeEach(() => {
    createdGateways.length = 0
    desktopWindow.hermesDesktop = {
      getConnection: vi.fn().mockResolvedValue({ baseUrl: '', mode: 'local', profile: 'work' }),
      getGatewayWsUrl: vi.fn().mockResolvedValue('ws://127.0.0.1:8642/ws'),
      touchBackend: vi.fn().mockResolvedValue({ ok: true })
    } as unknown as Window['hermesDesktop']
  })

  afterEach(() => {
    closeSecondaryGateways()
    setPrimaryGateway(null)
    $gateway.set(null)

    if (initialHermesDesktop) {
      desktopWindow.hermesDesktop = initialHermesDesktop
    } else {
      delete desktopWindow.hermesDesktop
    }
  })

  it('returns normalized primary and existing secondary connections without switching the active gateway', async () => {
    const primary = new MockGateway()
    primary.connectionState = 'open'
    setPrimaryGateway(primary as never, 'default')
    $gateway.set(primary as never)

    await ensureGatewayForProfile(' work ')
    const secondary = activeGateway()
    await ensureGatewayForProfile('default')

    expect(gatewayForProfile(' default ')).toBe(primary)
    expect(gatewayForProfile('work')).toBe(secondary)
    expect(activeGateway()).toBe(primary)
    expect($gateway.get()).toBe(primary)
  })

  it('does not create a connection for an unknown profile', () => {
    expect(gatewayForProfile('missing')).toBeNull()
    expect(createdGateways).toHaveLength(0)
  })

  it('tracks primary and secondary connection state independently by normalized profile', async () => {
    const primary = new MockGateway()
    primary.connectionState = 'connecting'
    setPrimaryGateway(primary as never, ' default ')
    reportPrimaryGatewayState('connecting')

    await ensureGatewayForProfile(' work ')
    const secondary = gatewayForProfile('work') as unknown as InstanceType<typeof MockGateway>

    expect($gatewayStatesByProfile.get()).toEqual({ default: 'connecting', work: 'open' })

    secondary.emitState('error')
    reportPrimaryGatewayState('open')

    expect($gatewayStatesByProfile.get()).toEqual({ default: 'open', work: 'error' })
  })

  it('does not let background state transitions contaminate the primary profile entry', async () => {
    const primary = new MockGateway()
    primary.connectionState = 'open'
    setPrimaryGateway(primary as never, 'default')
    reportPrimaryGatewayState('open')

    await ensureGatewayForProfile('work')
    const secondary = gatewayForProfile('work') as unknown as InstanceType<typeof MockGateway>
    await ensureGatewayForProfile('default')
    secondary.emitState('closed')

    expect($gatewayStatesByProfile.get()).toEqual(expect.objectContaining({ default: 'open', work: 'closed' }))
    expect($gatewayState.get()).toBe('open')
  })
})

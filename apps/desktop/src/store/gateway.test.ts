// @vitest-environment jsdom

import type { ConnectionState, GatewayEvent } from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { createdGateways, MockGateway } = vi.hoisted(() => {
  const createdGateways: MockGateway[] = []

  class MockGateway {
    connectionState: ConnectionState = 'closed'

    constructor() {
      createdGateways.push(this)
    }

    close(): void {
      this.connectionState = 'closed'
    }

    async connect(): Promise<void> {
      this.connectionState = 'open'
    }

    onEvent(_callback: (event: GatewayEvent) => void): () => void {
      return () => {}
    }

    onState(_callback: (state: ConnectionState) => void): () => void {
      return () => {}
    }
  }

  return { createdGateways, MockGateway }
})

vi.mock('@/hermes', () => ({ HermesGateway: MockGateway }))

import {
  $gateway,
  activeGateway,
  closeSecondaryGateways,
  ensureGatewayForProfile,
  gatewayForProfile,
  setPrimaryGateway
} from './gateway'

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
})

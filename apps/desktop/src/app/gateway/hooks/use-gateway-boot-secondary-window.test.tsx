// @vitest-environment jsdom
// @vitest-environment-options {"url":"http://localhost/?win=secondary&profile=app_factory#/session-123"}

import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useGatewayBoot } from './use-gateway-boot'

function Harness() {
  useGatewayBoot({
    handleGatewayEvent: () => undefined,
    onConnectionReady: () => undefined,
    onGatewayReady: () => undefined,
    refreshHermesConfig: async () => undefined,
    refreshSessions: async () => undefined
  })

  return null
}

describe('useGatewayBoot secondary-window profile routing', () => {
  afterEach(() => {
    cleanup()
    delete (window as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('selects the backend profile encoded in the secondary-window URL', () => {
    // Keep the connection pending: this test covers the first boot hop from
    // the real window URL parser through useGatewayBoot to the Electron bridge.
    const getConnection = vi.fn(() => new Promise<never>(() => undefined))

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = {
      getConnection,
      getBootProgress: vi.fn(() => new Promise<never>(() => undefined)),
      onBackendExit: vi.fn(() => () => undefined),
      onBootProgress: vi.fn(() => () => undefined)
    }

    render(<Harness />)

    expect(getConnection).toHaveBeenCalledOnce()
    expect(getConnection).toHaveBeenCalledWith('app_factory')
  })
})

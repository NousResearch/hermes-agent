import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { getLogs } from '@/hermes'
import { en } from '@/i18n/en'
import { $interfaceMode } from '@/store/interface-mode'

const mocks = vi.hoisted(() => ({
  notifyError: vi.fn(),
  reconnectGateway: vi.fn<() => Promise<void>>()
}))

vi.mock('@/components/ui/tooltip', () => ({
  Tip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock('@/hermes', async importOriginal => {
  const actual = await importOriginal<typeof HermesApi>()

  return { ...actual, getLogs: vi.fn().mockResolvedValue({ lines: [] }) }
})

vi.mock('@/store/gateway-reconnect', () => ({
  reconnectGateway: mocks.reconnectGateway
}))

vi.mock('@/store/notifications', () => ({
  notifyError: mocks.notifyError
}))

vi.mock('@/store/system-actions', () => ({
  runGatewayRestart: vi.fn()
}))

import { PortalGatewayMenuPanel as GatewayMenuPanel } from './portal-gateway-menu-panel'

const renderPanel = (gatewayState: string) =>
  render(
    <GatewayMenuPanel
      gatewayState={gatewayState}
      inferenceStatus={null}
      onClose={vi.fn()}
      onOpenSystem={vi.fn()}
      statusSnapshot={null}
    />
  )

describe('GatewayMenuPanel reconnect action', () => {
  beforeEach(() => {
    mocks.reconnectGateway.mockReset().mockResolvedValue(undefined)
    mocks.notifyError.mockReset()
    $interfaceMode.set('advanced')
    vi.mocked(getLogs).mockClear()
  })

  afterEach(() => {
    cleanup()
    $interfaceMode.set('advanced')
  })

  it('shows reconnect while disconnected and disables it in flight', async () => {
    let finish: (() => void) | undefined
    mocks.reconnectGateway.mockImplementation(
      () =>
        new Promise<void>(resolve => {
          finish = resolve
        })
    )

    renderPanel('closed')

    const reconnect = screen.getByRole('button', { name: en.shell.gatewayMenu.reconnectGateway })
    fireEvent.click(reconnect)
    fireEvent.click(reconnect)

    expect(mocks.reconnectGateway).toHaveBeenCalledOnce()
    expect((reconnect as HTMLButtonElement).disabled).toBe(true)

    await act(async () => finish?.())
  })

  it('allows recovery when an open socket stops delivering', async () => {
    renderPanel('open')
    await act(async () => undefined)

    const reconnect = screen.getByRole('button', { name: en.shell.gatewayMenu.reconnectGateway })
    await act(async () => fireEvent.click(reconnect))
    expect(mocks.reconnectGateway).toHaveBeenCalledOnce()
  })

  it('does not read logs in Simple mode until activity is expanded', async () => {
    $interfaceMode.set('simple')
    renderPanel('open')
    expect(getLogs).not.toHaveBeenCalled()
    expect(screen.queryByRole('button', { name: en.commandCenter.restartGateway })).toBeNull()
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Recent activity' })))
    expect(getLogs).toHaveBeenCalledOnce()
    expect(screen.getByRole('button', { name: 'Recent activity' }).getAttribute('aria-expanded')).toBe('true')
  })

  it('never equates an open connection or failed readiness probe with usable inference', () => {
    const props = {
      gatewayState: 'open',
      onClose: vi.fn(),
      onOpenSystem: vi.fn(),
      statusSnapshot: null,
      inferenceStatus: { ready: false, reason: null, checksDisagree: false, source: 'fallback' as const }
    }

    const view = render(<GatewayMenuPanel {...props} />)
    expect(screen.getByText('Connected')).toBeTruthy()
    expect(screen.getByText('Checking inference')).toBeTruthy()
    view.rerender(
      <GatewayMenuPanel {...props} inferenceStatus={{ ...props.inferenceStatus, source: 'runtime_check' }} />
    )
    expect(screen.getByText('Inference not ready')).toBeTruthy()
  })
})

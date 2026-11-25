import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { getLogs } from '@/hermes'
import { en } from '@/i18n/en'
import { setInterfaceMode } from '@/store/interface-mode'
import { setDocumentHidden } from '@/test/window-state'
import type { StatusResponse } from '@/types/hermes'

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

const flushAsync = () => act(async () => { await vi.advanceTimersByTimeAsync(0) })

describe('GatewayMenuPanel reconnect action', () => {
  beforeEach(() => {
    mocks.reconnectGateway.mockReset().mockResolvedValue(undefined)
    mocks.notifyError.mockReset()
    setInterfaceMode('advanced')
    vi.mocked(getLogs).mockClear()
  })

  afterEach(() => {
    cleanup()
    setInterfaceMode('advanced')
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
    setInterfaceMode('simple')
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

describe('GatewayMenuPanel log tail polling', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    setDocumentHidden(false)
    vi.spyOn(document, 'hasFocus').mockReturnValue(true)
    setInterfaceMode('advanced')
    vi.mocked(getLogs).mockClear()
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.mocked(document.hasFocus).mockRestore()
    setDocumentHidden(false)
    setInterfaceMode('advanced')
  })

  it('pauses the 3s tail while unfocused or hidden, and catches up on return', async () => {
    renderPanel('open')
    await flushAsync()
    expect(getLogs).toHaveBeenCalledOnce()

    // Another app in front (or an occluded window macOS still calls visible).
    vi.mocked(document.hasFocus).mockReturnValue(false)

    await act(async () => { await vi.advanceTimersByTimeAsync(3_000 * 3) })
    expect(getLogs).toHaveBeenCalledOnce()

    setDocumentHidden(true)
    vi.mocked(document.hasFocus).mockReturnValue(true)

    await act(async () => { await vi.advanceTimersByTimeAsync(3_000 * 3) })
    expect(getLogs).toHaveBeenCalledOnce()

    // Back on screen: one catch-up read, without waiting out the interval.
    setDocumentHidden(false)
    document.dispatchEvent(new Event('visibilitychange'))
    await flushAsync()
    expect(getLogs).toHaveBeenCalledTimes(2)

    await act(async () => { await vi.advanceTimersByTimeAsync(3_000) })
    expect(getLogs).toHaveBeenCalledTimes(3)
  })

  it('catches up on focus while the window stays visible', async () => {
    renderPanel('open')
    await flushAsync()
    expect(getLogs).toHaveBeenCalledOnce()

    vi.mocked(document.hasFocus).mockReturnValue(false)
    await act(async () => { await vi.advanceTimersByTimeAsync(3_000) })
    expect(getLogs).toHaveBeenCalledOnce()

    vi.mocked(document.hasFocus).mockReturnValue(true)
    window.dispatchEvent(new Event('focus'))
    await flushAsync()
    expect(getLogs).toHaveBeenCalledTimes(2)
  })
})

describe('GatewayMenuPanel status snapshot loading', () => {
  // A complete (if minimal) snapshot: the hook hands the panel this shape, never a partial one.
  const connectedSnapshot: StatusResponse = {
    active_sessions: 0,
    config_path: '',
    config_version: 1,
    env_path: '',
    gateway_exit_reason: null,
    gateway_health_url: null,
    gateway_pid: null,
    gateway_platforms: {},
    gateway_running: true,
    gateway_state: 'running',
    gateway_updated_at: null,
    hermes_home: '',
    latest_config_version: 1,
    release_date: '',
    version: '1.0.0'
  }

  beforeEach(() => {
    setInterfaceMode('advanced')
  })

  afterEach(() => {
    cleanup()
    setInterfaceMode('advanced')
  })

  it('holds platform placeholders while connected with no snapshot, then shows the real rows', async () => {
    const props = { inferenceStatus: null, onClose: vi.fn(), onOpenSystem: vi.fn() }
    const view = render(<GatewayMenuPanel {...props} gatewayState="open" statusSnapshot={null} />)
    await act(async () => undefined)

    expect(screen.getByText(en.shell.gatewayMenu.messagingPlatforms)).toBeTruthy()
    expect(view.container.querySelectorAll('[data-slot="skeleton"]')).toHaveLength(4)

    view.rerender(
      <GatewayMenuPanel
        {...props}
        gatewayState="open"
        statusSnapshot={{ ...connectedSnapshot, gateway_platforms: { discord: { state: 'connected', updated_at: '' } } }}
      />
    )

    expect(view.container.querySelectorAll('[data-slot="skeleton"]')).toHaveLength(0)
    expect(screen.getByText('discord')).toBeTruthy()
  })

  it('shows no placeholders while the gateway is offline or still connecting', () => {
    for (const gatewayState of ['closed', 'connecting']) {
      const view = render(
        <GatewayMenuPanel
          gatewayState={gatewayState}
          inferenceStatus={null}
          onClose={vi.fn()}
          onOpenSystem={vi.fn()}
          statusSnapshot={null}
        />
      )

      expect(view.container.querySelectorAll('[data-slot="skeleton"]')).toHaveLength(0)
      view.unmount()
    }
  })
})
